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

## Remaining progression

1. Implicit: fuse sparse row gathering into a Metal reader feeding Tensix matmul;
   keep only tile-sized expansion in L1, with no DRAM im2col tensor.
2. Masking: sort by neighborhood pattern, skip inactive block/offset products,
   and restore original output order.
3. Split-K: distribute reduction ranges to independent workers, then reduce
   partial outputs on the device and apply bias once.

Current scope is forward sparse GEMM. Host coordinate preprocessing is explicit;
device coordinate hashing, training/backward, automatic algorithm tuning, and
performance parity with the CUDA backend are not yet implemented. Empty output
sets and autograd inputs are rejected rather than silently falling back to CPU.
