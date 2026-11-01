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

## Stage 3: masking

Use `masked_implicit_gemm(...)` or `prepare_implicit(..., masked=True).run()`.
Host preprocessing builds 32-bit neighbor masks, sorts rows by inverse Gray code
(the upstream ordering), and ORs masks over each 32-row output block. The reader
skips inactive offsets entirely, including feature and weight reads. It streams
the exact number of products through a metadata circular buffer; compute threads
receive the same count via `read_tile_value`'s mailbox synchronization. An empty
block sends one zero product to initialize the accumulator deterministically.
The writer scatters tile rows back to original order on device, followed by a
device tilize operation. Volumes above 32 are rejected in masked mode.

Validation: masking plus implicit regression tests → **14 passed** on Wormhole.
Five masked hardware cases, each repeated, cover Gray grouping, randomly sparse
maps, bit 31, completely empty blocks, channel tails, bias, and >64 output tiles.
A host test demonstrates a structured map shrinking from 9 block-offset products
to 2 (plus one zero initialization product for the empty block). Row order is
checked against the original unsorted reference, with no CPU output permutation.

## Stage 4: split-K

Use `split_k_gemm(..., split_k=2, masked=True)` or reuse a prepared problem with
`prepared.run(split_k=2)`. Both masked and unmasked implicit GEMM support it.
The reduction is partitioned over `V * ceil(Ci/32)` channel tiles with boundaries
`floor(total_tiles * s / split_k)`. This permits uneven splits and boundaries
inside a kernel offset while covering each product exactly once. Workers own
independent `(split, output_tile)` tasks; there are no floating-point atomics.
Partials are written as FP32 in `[split_k,M_padded,Co_padded]` storage. A TTNN
device sum reduces axis 0, then casts to BF16 and adds bias once. Masked writers
restore original row order separately for each split. Empty masked splits use
the same deterministic zero initialization as stage 3.

The workspace costs `4 * split_k * M_padded * Co_padded` bytes. Split count must
be an integer from 1 through the number of channel tiles in the full reduction.
Split-K trades extra workspace and a reduction launch for more core parallelism;
it is not guaranteed to be faster. There is no automatic split-count selection.

Validation: `python -m pytest tests/tenstorrent/test_splitk.py -q` → **9 passed**.
Eight hardware combinations cover masked/unmasked execution, ragged channels,
2/3/5/7-way splitting, offset-internal partition boundaries, empty splits, bias,
and small output workloads. Each runs split → unsplit → split on the same device
operands and compares every element with the FP32 sparse reference.

## Limitations

Current scope is forward sparse GEMM. Host coordinate preprocessing is explicit;
device coordinate hashing, training/backward, automatic algorithm tuning, and
performance parity with the CUDA backend are not yet implemented. Empty output
sets and autograd inputs are rejected rather than silently falling back to CPU.
