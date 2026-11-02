# FlexGEMM on Tenstorrent

This port follows the [FlexGEMM progression](https://jeffreyxiang.github.io/en/blogs/flexgemm):
explicit GEMM → implicit GEMM → masking → split-K. The starting contract is
`Y[m,o] = sum(v,c) X[map[m,v],c] W[o,v,c] + bias[o]`, with missing neighbors
contributing zero. The upstream `[Co,Kw,Kh,Kd,Ci]` weights and `0xffffffff`
neighbor sentinel are accepted, as are `[Co,V,Ci]` weights and `-1`.

The [1024 → 4096 GenRecon validation](tenstorrent-wide-validation.md) covers real
shape/texture checkpoint weights and documents the precision fixes and one
remaining shape-activation tolerance miss. This remains an experimental backend.

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
`BUILD_TARGET=tenstorrent uv pip install --python "$(command -v python)" -e . --no-build-isolation --no-deps`.
Package imports load legacy CUDA/Triton modules only when requested.
The CPU reference and topology builder are usable without TTNN.

## Stage 1: explicit GEMM

`explicit_gemm` materializes the neighborhood-expanded BF16 matrix on the CPU,
uploads it and flattened weights, and runs `ttnn.matmul` with FP32 destination
accumulation and optional device bias addition. Results remain on the device.
On Wormhole, matmul uses HiFi3 and disables L1 packer accumulation to avoid
the numerical outliers found in the wide-layer validation.
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

Validation: `python -m pytest tests/tenstorrent/test_implicit.py -q` → **8 passed**
on Wormhole. Six shapes are each run twice, including ragged channels/rows, 27
offsets, all-missing rows, zero input points, and 81 output tiles scheduled across
64 cores. Two cache cases change feature values and device addresses on repeated
same-shape launches, with and without masking. Elementwise comparison
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

Validation: masking plus implicit regression tests → **15 passed** on Wormhole.
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
Partials are written as FP32 in `[split_k,M_padded,Co_padded]` storage. Device
slices and `split_k-1` FP32 additions reduce the partials, followed by a BF16
cast and one bias addition. The installed SDK's `ttnn.sum(dim=0)` lost low bits
in wide-layer validation, so this path avoids that lowering. Masked writers
restore original row order separately for each split. Empty masked splits use
the same deterministic zero initialization as stage 3.

The workspace costs `4 * split_k * M_padded * Co_padded` bytes. Split count must
be an integer from 1 through the number of channel tiles in the full reduction.
Split-K trades extra workspace and reduction launches for more core parallelism;
it is not guaranteed to be faster. There is no automatic split-count selection.

Validation: `python -m pytest tests/tenstorrent/test_splitk.py -q` → **13 passed**.
Eight hardware combinations cover masked/unmasked execution, ragged channels,
2/3/5/7-way splitting, offset-internal partition boundaries, empty splits, bias,
and small output workloads. Each runs split → unsplit → split on the same device
operands and compares every element with the FP32 sparse reference.
Four additional numerical cases check integer and fractional cancellation in
FP32 partial storage and reduction; split-count validation is also covered.

## Limitations

Current scope is forward sparse GEMM. Host coordinate preprocessing is explicit;
device coordinate hashing, training/backward, automatic algorithm tuning, and
performance parity with the CUDA backend are not yet implemented. Empty output
sets and autograd inputs are rejected rather than silently falling back to CPU.

## Coordinate API and reproducible comparison

`flex_gemm.tenstorrent.sparse_submanifold_conv3d` accepts the upstream coordinate,
shape, and weight conventions and returns `(device_features, cpu_neighbor_map)`:

```python
from flex_gemm.tenstorrent import sparse_submanifold_conv3d

y, cache = sparse_submanifold_conv3d(
    x, coords, (1, 32, 4, 4, 4), w, device=device,
    algorithm="masked_implicit_gemm_splitk", split_k=4,
)
```

Supported algorithms are `explicit_gemm`, `implicit_gemm`, `masked_implicit_gemm`,
`implicit_gemm_splitk`, and `masked_implicit_gemm_splitk`. Dilation is supported.
The returned map can be passed as `neighbor_map=cache` only while the topology,
kernel shape, and dilation remain unchanged. For repeated layers with the same
operands, `prepare_explicit` and `prepare_implicit` return reusable device buffers.

For network integration, prepare weights and topology independently and pass
device features directly between convolutions:

```python
from flex_gemm.tenstorrent import (
    prepare_sparse_weights, prepare_sparse_topology, sparse_conv_device,
)

weights = prepare_sparse_weights(w.detach(), bias=None, device=device)
topology = prepare_sparse_topology(cache, x.shape[0], device=device, masked=True)
x_device = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device)
y_device = sparse_conv_device(x_device, weights, topology,
                               algorithm="masked_implicit_gemm")
```

`sparse_conv_device` supports the same five algorithms. Features stay on device;
the host prepares only weights and integer metadata. Explicit GEMM gathers
`row_chunk` output rows at a time on device (default 256), with access to every
input neighbor, and reuses the packed weights. Its expansion workspace therefore
scales with `row_chunk * V * ceil(Ci/32)*32`, not the whole output row count.
Each chunk's output is retained until device concatenation, so output storage
still scales with the complete result. Chunk indices are cached in the topology.
No host feature download or re-upload is needed between layers.

The feature input contract remains BF16. `fp32_output=True` retains FP32 output
through reduction and bias for clients that need residual correction or later
FP32 arithmetic; it does not make BF16 inputs or HiFi3 multiplication exact.
Prepared topology must match the actual coordinate layout, not merely its size.
Masked and unmasked implicit algorithms require their corresponding topology;
explicit GEMM can consume either. Empty outputs should be handled by the network
without issuing a device convolution. Fifteen hardware tests cover chaining,
changed input values with cached maps/weights, ragged channels, row chunks that
cross tile boundaries, and FP32 cancellation across reduction tiles.

### Opt-in four-product implicit convolution

`digit_implicit_gemm` uses the validated integer-digit decomposition with four
fused gather/matmul calls on Wormhole. It avoids materializing the two expanded
activation matrices in DRAM. The existing explicit digit path remains available
as `digit_explicit_gemm`; neither changes the default algorithm.

```python
from flex_gemm.tenstorrent import (
    prepare_sparse_digit_weights, prepare_sparse_topology,
    prepare_sparse_digit_implicit, sparse_conv_device,
)

weights = prepare_sparse_digit_weights(weight, bias=bias, device=device)
topology = prepare_sparse_topology(mapping, input_points, device=device)
cp = (weight.shape[-1] + 31) // 32 * 32
chunk = max(32, 1048576 // cp // 32 * 32)
plan = prepare_sparse_digit_implicit(weights, topology, row_chunk=chunk)
print(plan.info["path"], plan.info["reason"])
output = sparse_conv_device(features_device, weights, topology,
    algorithm="digit_implicit_gemm", row_chunk=chunk)
```

Preparation retains only weights, integer topology and shape-dependent plans.
Feature scale and both digit tables are recomputed for every forward. A topology
object owns its cached plans, so distinct same-shaped coordinate layouts do not
share indices. Reuse requires immutable weights and the actual same topology.
Calling the preparation function explicitly separates its cost from warm execution;
the forward also prepares lazily when needed.

Unsupported architectures, kernel volumes above 32, non-tile-aligned row chunks,
and plans exceeding the 900 KiB per-core circular-buffer budget fall back to
`digit_explicit_gemm`. The returned policy reports the selected path and reason.
Both paths support BF16 or `fp32_output=True` results, with original row order.
Small shapes and some wide outputs can be slower on the implicit path; this is
an opt-in algorithm, not a universal performance dispatch policy.

The K6 program builder and kernels derive from Apache-2.0 code at
`changh95/bevfusion-p150`, revision `300095a42e61075a3d5c9916620344c1067200b8`.
The package includes `K6-LICENSE` and `K6-NOTICE`. See the
[autoresearch ledger](tenstorrent-autoresearch.md) for fixed numerical gates,
workspace measurements, matched timing evidence and remaining application work.

### Progression benchmark

```bash
python examples/tenstorrent_progression.py \
    --points 256 --ci 64 --co 64 --split-k 4 --repeat 5 \
    --output /tmp/tt-progression.json
```

The script samples 256 unique voxels from an 8³ grid, builds a real 3×3×3
submanifold topology, and verifies every stage against FP32 on BF16-rounded
inputs. Preparation includes host expansion/sorting and device uploads. Warm
execution measures host wall time including dispatch and device synchronization,
excluding preparation, JIT warmup, and output download. These are **not device
profiler kernel timings**. The common coordinate builder is timed separately.

Initial run before the wide-layer precision fixes:
[raw results](results/tenstorrent-wormhole-256x64.json). These historical timings
used HiFi4, enabled L1 packer accumulation, and the former axis-0 split sum.
For measurements of the current implementation, see the
[wide-layer report](tenstorrent-wide-validation.md#timing-and-memory).

| Stage | Preparation (ms) | Warm median (ms) | Max absolute error |
|---|---:|---:|---:|
| Explicit | 11.391 | 0.423 | 0.01415 |
| Implicit | 2.164 | 0.741 | 0.01415 |
| Masked | 4.829 | 1.237 | 0.01415 |
| Masked split-K (4) | 3.465 | 1.580 | 0.01315 |

This prototype reproduces the algorithms but does **not** reproduce the CUDA
speedups. The optimized TTNN dense GEMM baseline wins warm execution on this
small case. Sorting reduced active block offsets from 216 to 198; its writer
scatter/tilize overhead and split-K's reduction outweigh the saved work here.
The explicit activation expansion is 864 KiB, versus 32 KiB of source features
and 27 KiB of indices for implicit GEMM (common weights excluded). Four FP32
split partials add 256 KiB. Improving gather coalescing, reusing input tiles
across output-channel tiles, and profiling device execution are future tuning
work; current timings should not be extrapolated to other shapes or devices.

The coordinate integration tests compare all five algorithm choices, with
non-unit dilation and two batches, directly to dense PyTorch conv3d. Installation
was also checked by building a pure-Python wheel and inspecting that all three
Metal kernel source files are packaged. The wheel was extracted into a fresh
temporary directory outside the checkout; a masked split-K hardware smoke test
compiled those bundled sources and passed against the independent reference.

Final validation command:

```bash
TT_METAL_WATCHER=1 python -m pytest tests/tenstorrent -q \
    --junitxml=docs/results/tenstorrent-resident-tests.xml
```

**75 passed, 4 expected failures**, on the hardware above with Watcher access
checks enabled and both external GenRecon checkpoints available.
[JUnit results](results/tenstorrent-resident-tests.xml) include all four stages,
coordinate integration, sentinel/bounds validation, exact cancellation checks,
batched topology lookup against a coordinate oracle (including anisotropic
kernels, dilation, batches, boundaries and int64-overflow fallback),
pretrained wide layers, and device-resident convolution chaining. Explicit output chunks are concatenated
in groups of at most four: a 17-input TTNN concat stalled on the pinned SDK,
and both BF16/FP32 regressions now cover that case. The four strict expected failures preserve the
known shape LayerNorm+SiLU tolerance miss described in the wide-layer report;
this is not an all-green numerical validation. Twelve checkpoint-dependent
cases skip explicitly when their external weights are unavailable.
Cache tests keep old device operands alive to force distinct addresses, change
both feature values and mask/permutation data, and assert program-cache entry
counts remain constant across same-shape invocations.
A cancellation case produces split partials `257` and `-256`: the exact residual
`1` survives FP32 workspace/reduction, whereas BF16 partial storage would lose it.

Host topology construction uses sorted integer lattice keys and batched lookup,
with exact Python-coordinate lookup retained when a logical lattice would
overflow int64 keys. This removes one Tensor write per voxel/kernel offset.
On a dense 32³ CPU fixture with 16 threads, construction fell from 6.311 seconds
to 0.057 seconds with identical maps; this is a host-preparation measurement,
not a convolution speedup. See [topology benchmark](results/tenstorrent-topology.json).
It also checks that bias is applied once, in both masked and unmasked modes.
The fractional residual `257.125 - 256 - 1 + 0.03125 = 0.15625` is checked
exactly for four- and seven-way FP32 reductions.
