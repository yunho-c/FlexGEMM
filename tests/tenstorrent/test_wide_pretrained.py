"""Optional checkpoint-backed regressions for wide Wormhole convolutions.

No checkpoint weights are distributed in this repository. Set
FLEX_GEMM_GENRECON_CHECKPOINT / FLEX_GEMM_GENRECON_TEXTURE_CHECKPOINT to the
shape / texture decoder safetensors, or use the local GenRecon directory below.
Missing weights cause an explicit skip. The shape LayerNorm+SiLU fixture retains
a known elementwise tolerance failure, independently reported by the wide probe.
"""

import os
import math
from pathlib import Path
import torch
import pytest

from flex_gemm.tenstorrent import (
    build_subm_neighbor_map, prepare_explicit, prepare_implicit, sparse_conv_reference,
)
from test_explicit import device


@pytest.fixture(scope="module", params=[
    pytest.param(("shape", "gaussian", 129), id="shape-gaussian"),
    pytest.param(("tex", "norm_silu", 513), id="texture-norm-silu"),
    pytest.param(("shape", "norm_silu", 513), id="shape-norm-silu-known-error",
                 marks=pytest.mark.xfail(strict=True, raises=AssertionError,
                     reason="Known shape output [305,2295]: error 0.033185 exceeds bound 0.032373")),
])
def wide_problem(request):
    safetensors = pytest.importorskip("safetensors")
    model, activation, points = request.param
    variable = "FLEX_GEMM_GENRECON_CHECKPOINT" if model == "shape" else "FLEX_GEMM_GENRECON_TEXTURE_CHECKPOINT"
    path = Path(os.environ.get(variable, str(Path.home() /
        f"models/genrecon/trellis2/ckpts/{model}_dec_next_dc_f16c32_fp16.safetensors")))
    if not path.is_file():
        pytest.skip(f"GenRecon {model}-decoder checkpoint unavailable")
    threads = torch.get_num_threads()
    torch.set_num_threads(16)
    try:
        with safetensors.safe_open(path, framework="pt", device="cpu") as checkpoint:
            weight = checkpoint.get_tensor("blocks.0.4.conv1.weight").bfloat16()
            bias = checkpoint.get_tensor("blocks.0.4.conv1.bias").bfloat16()
            nw = checkpoint.get_tensor("blocks.0.4.norm1.weight").float()
            nb = checkpoint.get_tensor("blocks.0.4.norm1.bias").float()
        assert tuple(weight.shape) == (4096, 3, 3, 3, 1024)
        generator = torch.Generator().manual_seed(123)
        side = math.ceil((points / 0.4 / 2) ** (1 / 3))
        flat = torch.randperm(2 * side**3, generator=generator)[:points]
        coords = torch.stack((flat // side**3, flat // side**2 % side,
                              flat // side % side, flat % side), dim=1)
        x = torch.randn(points, 1024, generator=generator)
        if activation == "norm_silu":
            x = torch.nn.functional.silu(torch.nn.functional.layer_norm(x, (1024,), nw, nb, eps=1e-6))
        x = x.bfloat16()
        mapping = build_subm_neighbor_map(coords, (2, side, side, side))
        expected = sparse_conv_reference(x, weight, mapping, bias)
        yield x, weight, bias, mapping, expected
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("stage", ["explicit", "implicit", "masked", "split_k"])
def test_wide_pretrained_device(wide_problem, device, stage):
    import ttnn
    x, weight, bias, mapping, expected = wide_problem
    if stage == "explicit":
        p = prepare_explicit(x, weight, mapping, device=device, bias=bias)
        result = p.run()
    else:
        p = prepare_implicit(x, weight, mapping, device=device, bias=bias,
                              masked=stage in ("masked", "split_k"))
        result = p.run(split_k=4 if stage == "split_k" else 1)
    actual = ttnn.to_torch(result).float()
    assert torch.isfinite(actual).all()
    assert (actual - expected).norm() / expected.norm() <= 0.01
    # HiFi4 previously passed global L2 while failing this check by ~4 or ~8
    # at a single output. Do not replace the elementwise assertion with PCC.
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)
