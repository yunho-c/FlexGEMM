# GenRecon 1024 → 4096 convolution validation

The real shape and texture decoder projections execute on Wormhole B0, but the
numerical validation is **not completely clean**. After correcting three
precision issues, 28 of 32 wide-layer cases pass the original numerical gates.
The other four reproduce one small elementwise tolerance miss in the shape
decoder's LayerNorm+SiLU fixture, once through each algorithm. All 32 cases
complete, produce finite outputs, and reproduce the first output bitwise on the
last repetition. No crashes, allocation failures, or Watcher access errors were
observed. This supports further experimental integration, not a default-backend
or full-decoder accuracy claim.

## Scope and method

Validation date: 2026-11-01. Hardware and SDK match
[the port documentation](tenstorrent.md): Wormhole B0, 8×8 grid,
TT-Metal `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`, PyTorch `2.11.0+cpu`.
The final runs use HiFi3 with FP32 destination accumulation, disable TTNN
matmul's L1 packer accumulation on Wormhole, and reduce split partials through
FP32 device additions. BF16 output packing and the separate bias addition remain.

Both local GenRecon checkpoints supply `blocks.0.4.conv1.weight` with shape
`[4096,3,3,3,1024]` and its bias:

- `shape_dec_next_dc_f16c32_fp16.safetensors`
- `tex_dec_next_dc_f16c32_fp16.safetensors`

Checkpoint weights are not distributed here. Raw reports record tensor SHA256
digests, source-file digests, SDK commit, seed, and device settings. Their
`source_commit` is the base commit before this change; `source_sha256` identifies
the exact working source used for each run.

Coordinates and activations are **synthetic**, not captured decoder activations.
Seed 123 samples unique, unsorted coordinates from two batches on a cubic grid,
with side length `ceil((points / 0.4 / 2) ** (1/3))`. Gaussian features are tested
at 129, 513, and 1025 active voxels. A second fixture applies the checkpoint's
`blocks.0.4.norm1` affine LayerNorm (`eps=1e-6`) and SiLU to Gaussian features at
513 voxels. Both fixtures build the actual 3×3×3 submanifold neighbor map.

Every fixture runs explicit GEMM, implicit GEMM, masked implicit GEMM, and masked
split-K with four splits. Each runs once plus three warm repetitions with
`TT_METAL_WATCHER=1`. Acceptance requires all of:

- Correct output shape and finite values.
- Every element satisfies `abs(actual-reference) <= 0.025 + 0.025*abs(reference)`.
- Relative L2 error ≤ 1% and Pearson correlation ≥ 0.9999.
- Bitwise equality between the first and last output.

The independent reference is FP32 per-offset gather/matmul on **BF16-rounded**
features, weights, and bias. The gate has not been relaxed. Reports also compare
against FP32 computation on the original FP16 checkpoint weights and unrounded
synthetic features, and separately measure BF16 conversion error. Those are
diagnostics, not interchangeable references. A BF16 epilogue model is also
recorded without changing acceptance.

## Final results

| Weights | Synthetic activation | Active voxels | Numerical cases passed | Raw report |
|---|---|---:|---:|---|
| Shape | Gaussian | 129, 513, 1025 | 12/12 | [JSON](results/tenstorrent-wide/shape-gaussian.json) |
| Texture | Gaussian | 129, 513, 1025 | 12/12 | [JSON](results/tenstorrent-wide/texture-gaussian.json) |
| Shape | Learned LayerNorm + SiLU | 513 | 0/4 | [JSON](results/tenstorrent-wide/shape-norm-silu.json) |
| Texture | Learned LayerNorm + SiLU | 513 | 4/4 | [JSON](results/tenstorrent-wide/texture-norm-silu.json) |

Relative L2 error against the rounded reference is approximately 0.237–0.243%
across all cases; all pass the global L2 and correlation gates. The remaining
shape failure is at row 305, output channel 2295, for all four algorithms:

| Quantity | Value |
|---|---:|
| Device output | -0.26171875 |
| FP32 reference | -0.2949037551879883 |
| Absolute error | 0.03318500518798828 |
| Allowed absolute error at this element | 0.03237259387969971 |

That is one failing element out of 2,101,248 per algorithm, with an error about
2.5% above the allowed bound. It persists with L1 packer accumulation disabled
and in a diagnostic that retains FP32 output through bias addition. It therefore
cannot be dismissed solely as the extra BF16 rounding before bias. The precise
arithmetic source and its effect on real decoder activations remain unresolved.

The hardware regression suite reports **52 passed, 4 expected failures** in
16.60 seconds under Watcher. The four strict `xfail` cases retain this shape
fixture; an unexpected pass requires updating the expectation. There are no
unavailable-checkpoint skips in this run. Twelve cases require the external
checkpoints and explicitly skip when they are absent. See
[JUnit results](results/tenstorrent-tests.xml) and
[checkpoint regression tests](../tests/tenstorrent/test_wide_pretrained.py).

## Precision issues exposed and corrected

1. **Rare large matmul outliers with HiFi4 + FP32 accumulation.** The initial
   shape runs completed but had approximately -4 or -8 errors at individual
   outputs, despite acceptable global L2. This matches the behavior described
   in [Tenstorrent issue #38306](https://github.com/tenstorrent/tt-metal/issues/38306).
   Selecting HiFi3 on Wormhole removes those observed outliers in the custom
   kernels. See [initial results](results/tenstorrent-wide/before-hifi4-shape.json).
2. **TTNN axis-0 sum lost low bits from FP32 split partials.** HiFi3 alone left
   small split-K failures: [intermediate results](results/tenstorrent-wide/hifi3-with-axis0-sum.json).
   Downloading the same raw partials isolated an additional maximum error of
   0.11990356 in `ttnn.sum(dim=0)`; tilization was exact. Sequential FP32
   `ttnn.add` operations matched the CPU sum of those partials exactly in this
   experiment. Split-K now uses device slices and `S-1` FP32 additions before
   the final BF16 cast. [Diagnostic results](results/tenstorrent-wide/reduction-diagnostic.json).
   New exact regressions preserve the residual `257.125 - 256 - 1 + 0.03125`.
3. **A remaining explicit matmul outlier with L1 packer accumulation.** The
   texture LayerNorm+SiLU fixture produced `1.515625` instead of `5.522267818`
   at row 77, channel 1112, even with HiFi3. Disabling `packer_l1_acc` removes
   this observed error for both BF16 and FP32-output diagnostics. It is now
   disabled on Wormhole; other architectures retain the prior setting.
   [Diagnostic results](results/tenstorrent-wide/packer-diagnostic.json).

The diagnostic JSON files compare intermediate tensors/configurations; they are
not additional acceptance runs. These workarounds are validated on the pinned
SDK and this device. Other architectures have not been hardware-tested here.

## Timing and memory

A separate run disables Watcher and uses five warm repetitions, shape weights,
Gaussian activations, and 129 voxels. All four numerical checks pass.
[Raw timing results](results/tenstorrent-wide/timing.json):

| Algorithm | Preparation (ms) | Warm median execution (ms) |
|---|---:|---:|
| Explicit | 538.80 | 1.847 |
| Implicit | 410.21 | 38.937 |
| Masked | 390.81 | 39.746 |
| Masked split-K (4) | 390.41 | 43.129 |

Warm time is host wall time including dispatch and synchronization, excluding
preparation, initial compilation, and output download. Preparation includes
packing and uploading the 216 MiB BF16 weight tensor, plus features and metadata;
it is paid again when a new prepared object is created. Coordinate construction
is recorded separately. These are not device profiler kernel timings, and the
Watcher runs' much higher latencies should not be used for performance claims.

The optimized TTNN explicit baseline is still substantially faster than these
custom sparse kernels for this fixture. It performs dense GEMM on expanded
active neighborhoods, not convolution over the full spatial grid. The current
custom kernel repeatedly gathers features and reads weights for each output
tile; input reuse and reader throughput need profiling and tuning.

Each report records live DRAM allocation before preparation, after preparation,
with the output, and after release. These snapshots include runtime/cache
allocations and **do not measure transient peak memory**. In the largest
1025-voxel case, padded explicit activations occupy 55.6875 MiB; four FP32 split
partials require 66 MiB in addition to resident operands, layout-conversion
buffers, and reduction scratch. Successful execution at this size does not
establish full-decoder memory capacity.

## Reproduction and integration decision

Use the TT-Metal environment described in [tenstorrent.md](tenstorrent.md), with
`safetensors` available. From the repository root, select either checkpoint:

```bash
TT_METAL_WATCHER=1 python examples/tenstorrent_wide_conv_probe.py \
    --checkpoint /path/to/shape_dec_next_dc_f16c32_fp16.safetensors \
    --points 129 513 1025 --activation gaussian --output /tmp/wide-shape.json

TT_METAL_WATCHER=1 python examples/tenstorrent_wide_conv_probe.py \
    --checkpoint /path/to/shape_dec_next_dc_f16c32_fp16.safetensors \
    --points 513 --activation norm_silu --output /tmp/wide-shape-norm.json
```

The second command currently exits 1 and preserves its four numerical failures.
Output paths must be fresh. The script serializes device use through the same
`/tmp/genrecon-tt-pjrt.lock` as the GenRecon experiments. For timing, unset
`TT_METAL_WATCHER`, use `--points 129 --repeat 5`, and choose a fresh report path.
The pytest fixtures accept `FLEX_GEMM_GENRECON_CHECKPOINT` and
`FLEX_GEMM_GENRECON_TEXTURE_CHECKPOINT` for nondefault checkpoint locations.

Begin only an experimental adapter with the existing reference path available.
Before adopting this backend, replay captured GenRecon layer inputs through the
wide convolution and C2S boundary, measure sign/topology changes and decoder
quality, and resolve or explicitly assess the remaining tolerance miss. Device
tensor chaining and end-to-end memory/timing also remain unvalidated. The
current tests establish that the wide operations run; they do not yet establish
that a complete shape or texture decoder produces acceptable reconstructions.
