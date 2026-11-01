# FlexGEMM on Tenstorrent

This port follows the [FlexGEMM progression](https://jeffreyxiang.github.io/en/blogs/flexgemm):
explicit GEMM → implicit GEMM → masking → split-K. The starting contract is
`Y[m,o] = sum(v,c) X[map[m,v],c] W[o,v,c] + bias[o]`, with missing neighbors
contributing zero. The upstream `[Co,Kw,Kh,Kd,Ci]` weights and `0xffffffff`
neighbor sentinel are accepted, as are `[Co,V,Ci]` weights and `-1`.

## Environment and installation

Validated on 2026-11-01 using the locally installed stack:

- Wormhole B0, one visible device, 8×8 compute/storage grid.
- TT-Metal checkout `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`.
- Python 3.10, PyTorch `2.11.0+cpu`, KMD 2.3.0, firmware 19.11.0.
- `TT_METAL_HOME=/home/user/tt-metal`; its `python_env` provides TTNN.

Use a working TT-Metal environment matching the above API. From this checkout:

```bash
# --no-deps avoids installing the legacy CUDA/Triton dependencies from pyproject.
BUILD_TARGET=tenstorrent python -m pip install -e . --no-build-isolation --no-deps
python -m pytest tests/tenstorrent -q
```

TTNN is provided by TT-Metal, not installed from an arbitrary PyPI version.
In environments without pip, the verified equivalent installation command is
`BUILD_TARGET=tenstorrent uv pip install --python "$VIRTUAL_ENV/bin/python" -e . --no-build-isolation --no-deps`.
Package imports load legacy CUDA/Triton modules only when requested.
The CPU reference and topology builder are usable without TTNN.

## Stage 1: explicit GEMM

`explicit_gemm` materializes the neighborhood-expanded BF16 matrix on the CPU,
uploads it and flattened weights, and runs `ttnn.matmul` with FP32 destination
accumulation and optional device bias addition. Results remain on the device.
This establishes the mathematical baseline; host expansion and transfers are
real costs and must be included in end-to-end timing.

```python
import torch
import ttnn
from flex_gemm.tenstorrent import build_subm_neighbor_map, explicit_gemm

coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
mapping = build_subm_neighbor_map(coords, (1, 4, 4, 4))
x = torch.randn(2, 32).bfloat16()
w = torch.randn(64, 3, 3, 3, 32).bfloat16()
device = ttnn.open_device(device_id=0)
try:
    y = explicit_gemm(x, w, mapping, device=device)
    print(ttnn.to_torch(y).shape)  # [2,64]
finally:
    ttnn.close_device(device)
```

Validation: `python -m pytest tests/tenstorrent/test_explicit.py -q` → **7 passed**.
Four hardware cases cover tile-aligned and ragged dimensions, 27 offsets,
missing neighbors, unequal input/output point counts, bias, and a scalar case.
The independent FP32 oracle sums per-offset products; its coordinate/weight
conventions are checked against `torch.nn.functional.conv3d` for two batches.
Hardware checks use elementwise `atol=rtol=0.025` on BF16-rounded inputs.

## Stage 2: implicit GEMM

`prepare_implicit(..., device=device, bias=None)` uploads row-major features,
transposed neighbor indices, and tiled weights. Its `run()` method launches
custom reader, compute, and writer kernels through `ttnn.generic_op`. The
convenience wrapper `implicit_gemm` includes preparation. The reader gathers
32 feature rows directly into the four 16×16 faces of each L1 BF16 tile, filling
missing rows with zeros. Two input tiles per operand are buffered; no expanded
matrix is written to DRAM. Channel padding is applied separately at each kernel
offset. Output tiles are distributed over available cores with FP32 accumulation.

The [TT-Metal single-core matmul example](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/programming_examples/matmul/matmul_single_core)
documents the reader/compute/writer pipeline used here. Our reader implements the
sparse gather, and output tile ownership extends the pipeline to multiple cores.

Validation: `python -m pytest tests/tenstorrent/test_implicit.py -q` → **7 passed**
on Wormhole. Six shapes are each run twice, including ragged channels/rows, 27
offsets, all-missing rows, zero input points, and 81 output tiles scheduled across
64 cores. A seventh test changes feature values and device addresses on repeated
same-shape launches to check program-cache correctness. Elementwise comparison
uses the same FP32 oracle and tolerance as stage 1.

## Remaining progression

1. Masking: sort by neighborhood pattern, skip inactive block/offset products,
   and restore original output order.
2. Split-K: distribute reduction ranges to independent workers, then reduce
   partial outputs on the device and apply bias once.

Current scope is forward sparse GEMM. Host coordinate preprocessing is explicit;
device coordinate hashing, training/backward, automatic algorithm tuning, and
performance parity with the CUDA backend are not yet implemented. Empty output
sets and autograd inputs are rejected rather than silently falling back to CPU.
