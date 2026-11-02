# Tenstorrent sparse-convolution autoresearch

The objective is competitive sparse-convolution performance on Tenstorrent,
verified under a fixed numerical and residency contract. Improving on one
prototype is not sufficient evidence for a state-of-the-art claim. Published
comparators, multiple workloads, independent confirmation, and representative
application measurements remain required.

Comparator discovery also needs to distinguish hardware and numerical scopes.
The [Tenstorrent model catalog](https://docs.tenstorrent.com/tt-awesome/models/bevfusion/)
lists a community BEVFusion-L bundle with a sparse 3D encoder on p150. That is
a lead to inspect, not a measured n150 comparator or evidence of equivalent
operator accuracy. No claim that this branch is the only TT implementation is
made. GPU engines such as [SpConv](https://github.com/traveller59/spconv) and
[Spira](https://github.com/SPIN-Research-Group/Spira) provide additional comparison
leads; published results on different shapes/hardware are not substituted for
measurements on this corpus.

## Contract and evaluator

The versioned contract is `benchmarks/tenstorrent/contract.json`. The first track
uses BF16 device features/weights, HiFi3 on Wormhole, FP32 accumulation and tiled
BF16 device outputs in original row order. All required gather, layout, bias,
permutation and reduction operations are timed. Static weights and geometry may
persist; gathered feature values may not. Two feature variants on the same
topology check reuse. The second is an explicitly synthetic perturbation.

The independent CPU evaluator implements a coordinate-dictionary oracle and
per-offset FP32 reference. Corpus fixtures are materialized before evaluating a
candidate, with file digests. Candidate source, evaluator, contract and corpus
digests are recorded and checked for mutation. References and thresholds are
not candidate search parameters. The existing wide LayerNorm/SiLU tolerance
miss is included as a numerical failure, not converted to successful xfail.

This is currently a BF16 operator track. The separate GenRecon accurate decoder
uses multiple BF16 digit products with FP32 outputs; accelerating this track
does not by itself establish accurate full-decoder performance or topology
parity. That integration and an FP32-output contract remain required work.

## Prepare and run

Run from this checkout with the pinned TT-Metal Python environment. Preparation
is CPU-only and does not acquire the board. Use a fresh output directory:

```bash
python -m benchmarks.tenstorrent.prepare \
  --output /home/user/tt-scratchpad/flexgemm-autoresearch/corpus-v2 \
  --real-capture /home/user/tt-scratchpad/genrecon/sparse-decoder-poster-bounded/shape_wide_input.pt \
  --checkpoint /home/user/models/genrecon/trellis2/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors

python -m benchmarks.tenstorrent.run \
  --corpus /home/user/tt-scratchpad/flexgemm-autoresearch/corpus-v2 \
  --case tune-surface-256-64x64 \
  --config '{"algorithm":"explicit_gemm","row_chunk":256}' \
  --output /tmp/flexgemm-baseline-256.json
```

The runner holds `/tmp/genrecon-tt-pjrt.lock` through worker process teardown,
including interpreter exit handlers. It never resets another user's runtime.
Compilation/preparation and execution have separate timeouts. A timeout records
that device recovery may be necessary and does not automatically retry. There
are separate `timing`, `correctness` (Watcher), and `profile` modes; lean timing
rejects instrumentation enabled in the environment. The runner's `.runner.json`
records terminal process status; an evaluation JSON alone is not proof that the
process has exited. Logs sit beside the output. Fresh report paths are required.

Before using the hardware, coordinate with other active conversations in
addition to honoring the lock. The other GenRecon conversation completed its
pipeline and was idle before iteration 004 began; workers still acquire the
shared lock separately for each trial.

## Corpus and score

Five tuning and five validation fixtures span scattered, surface and clustered
topologies, ragged channels, 35–4096 voxels, and 17–256 channels. They use distinct
seeds. The real poster fixture contains all 4,490 source/output voxels, all
neighbors, captured 1024-channel decoder activations, and the pretrained
1024→4096 weights. It is not a cropped spatial problem. The known numerical
regression is a separate mandatory case. Large tensors remain outside git;
the committed manifest contains their hashes and provenance.

Current coverage is not a held-out scene benchmark. Iteration 001 adds full-size
narrow/late-stage captures and measured layer occurrence counts from the poster
shape decoder. Additional independent scenes, complete stage timings and
accurate-output integration are still outstanding. Synthetic
geometric-mean speedup describes coverage; it is not an application score.

Iteration 011 adds an independent public LiDAR geometry corpus from
[Autoware's test cloud in the pinned BEVFusion bundle](https://huggingface.co/changh95/bevfusion-p150/blob/300095a42e61075a3d5c9916620344c1067200b8/code/tt_bevfusion/samples/README.md).
Independent voxelization and the coordinate-dictionary oracle reproduce its
21,359 voxels and 112,747 submanifold neighbor interactions. Six fixed channel
shapes use synthetic BF16 features, weights and biases, including ragged
channels. This supplies different scene geometry, not captured BEVFusion
activations or model performance. The manifest records source hashes, pinned
URLs, Apache-2.0 attribution, voxel parameters and operand seeds. Once used to
guide further changes, these cases are validation data rather than held-out
data. Independent full-application scenes remain outstanding.

Promote a performance candidate only after all applicable correctness cases
pass and alternating incumbent/candidate measurements exceed observed timing
noise. Baseline-against-baseline repeats establish that noise. A fixture known
to fail cannot be silently dropped from the promotion report.

## Resource accounting and SoL

`common.work_accounting` computes valid neighbor interactions and useful FLOPs,
the baseline kernels' tile products, channel/row padding, mask unions, empty
split initialization products, selected cores and per-core work. It models
requested feature/weight/map payloads for the custom kernels. These are not
measured DRAM/NoC bus bytes. For TTNN explicit matmul the issued count is a
logical padded-GEMM estimate; internal physical scheduling is unavailable.

Split partial storage and FP32-add payloads are separate fields. Live allocator
snapshots are not peaks. The runner rejects mandatory buffers above the 1 GiB
workspace limit; passing that lower-bound check alone does not establish
compliance with the full cap.

The new `--mode memory` run captures actual allocator events across a warmed
forward. It reports the inclusive peak of all new aligned DRAM allocations and
a temporary-only bound that excludes the final output, as specified by the
original contract. Only the final live allocation at the returned output's
address is eligible for exclusion: earlier allocations reusing that address
still count. Exclude only the padded output tensor's bytes, retaining any extra
backing-buffer capacity and bank-alignment padding in the temporary bound.
An absent, ambiguous or undersized output allocation invalidates the trace.
Both bounds add the entire device L1 capacity. Resident inputs, weights and
topology predate capture. Iterations 002–006 used the stricter inclusive bound;
their original reports remain unchanged. Each promoted candidate/case requires
its own capture. The compressed allocation trace is saved beside the report.

Accounting models distinguish baseline, tiled gather and four-product digit
implementations. The latter also distinguishes concatenation from direct output
tile copies when evaluating an older candidate checkout. Structural kernel
changes require an updated model; missing measurements are null.
Iteration 002 establishes initial dense/gather empirical
references and device wait timing. Hardware counters and a complete memory/
communication lower bound remain outstanding.

`sol.py` now provides the separate arithmetic-only capacity reference. The
[pinned SDK's GEMM report](https://github.com/tenstorrent/tt-metal/blob/9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md)
specifies 48 ideal cycles per 32×32×32 HiFi3 tile product. At the captured
1 GHz clock with the available 8×8 grid, this gives 87.381 TFLOP/s and a
6.558 ms useful-arithmetic bound for the 4,490-row wide layer. This excludes
indexing, communication and launch overhead; it is not an attainable latency
prediction. Memory bounds and measured empirical rates remain separate.

## Screening, confirmation and diagnostics

`sweep.py --screen` runs one baseline and the configured candidates per case.
Without `--screen`, it brackets each candidate with baseline trials, shuffles
candidate order reproducibly, and repeats the brackets. The supplied parameter
set includes an unchanged baseline control. Failed or missing cases remain in
the summary and invalidate aggregate scores. A consistent screening win needs
at least three rounds and an improvement beyond both the observed variation
and a prespecified 2% margin. This is a screening heuristic, not a statistical
confidence interval or automatic promotion decision.

```bash
python -m benchmarks.tenstorrent.sweep \
  --corpus /home/user/tt-scratchpad/flexgemm-autoresearch/corpus-v2 \
  --cases poster-shape-wide-4490 \
  --candidates benchmarks/tenstorrent/configs/row_chunks.json \
  --output /tmp/flexgemm-chunk-screen --screen --repeat 7

python -m benchmarks.tenstorrent.run \
  --corpus /home/user/tt-scratchpad/flexgemm-autoresearch/corpus-v2 \
  --case poster-shape-wide-4490 --workload calibration --config '{}' \
  --output /tmp/flexgemm-calibration.json

python -m benchmarks.tenstorrent.run \
  --corpus /home/user/tt-scratchpad/flexgemm-autoresearch/corpus-v2 \
  --case shape-norm-silu-513-known-failure --workload precision \
  --config '{"algorithm":"explicit_gemm","row_chunk":256}' \
  --output /tmp/flexgemm-precision.json
```

Calibration uses resident dense matmul, DRAM clone, tilize/untilize, and the
captured embedding gather. Logical payload rates are not bus counters or hard
ceilings. The validated dense reference uses BF16 small-integer inputs and
FP32 output; its output format differs from the BF16 operator track. The
precision diagnostic separately checks the raw GEMM, device bias
addition, host FP32 bias addition after download, and one final BF16 conversion.
It is unscored: successful execution means only that the diagnostic completed.

Custom profiler regions identify implicit reader/compute/writer spans and
accumulate reader buffer-wait and gather-plus-weight time. The SDK's existing
compute CB wait counters retain their two accumulation slots. These instrumented
runs are separate from lean latency measurements.

## Iteration ledger

Each completed iteration is committed and pushed on `research/tt-sol`, including
discarded experiments' evidence. Record evidence, prediction, result and
interpretation. Keep benchmark/evaluator changes separate from kernel changes.

### 000 — evaluation foundation

- Evidence: prepared explicit benchmarks reuse expanded features, whereas
  implicit execution gathers each call. The current reader also repeats input
  gathers across output-channel tiles.
- Prediction: fair resident timings and sparse-work accounting will distinguish
  actual operator improvements from work moved into preparation.
- Work: fixed contract, CPU corpus/goldens, integrity checks, changing-feature
  evaluation, complete-operator worker, serialized runner, timeout handling and
  baseline arithmetic/traffic accounting.
- Validation: independent oracle against dense convolution and reordered
  coordinates; rejection of scaled, stale, nonfinite and incorrectly shaped
  results; split/padding/empty-mask accounting; child failure and timeout handling.
  CPU topology/validation/masking regressions also pass.
- Hardware status: pending the other conversation's device release. No latency
  improvement or state-of-the-art performance is claimed for this iteration.

### 001 — hardware evaluator smoke and real shape corpus

- Evidence: iteration 000's corpus/evaluator passes CPU checks. A coordinated
  short hardware window permits testing evaluation and profiler extraction.
- Prediction: including fresh feature gather changes comparisons with the old
  prepared-explicit timings; the known numerical failure must still be rejected.
- Result: resident explicit and implicit both pass on surface-256. Screening
  medians are 1.006 ms explicit and 0.601 ms implicit. Full 4,490-row explicit
  execution with row_chunk=256 passes at 58.088 ms. These use 7, 3 and 3 timed
  repetitions respectively and are not paired promotion measurements.
- The original LayerNorm+SiLU error reproduces exactly at [305,2295]; its report
  fails. The second feature variant passes. No tolerance was changed.
- Device profiling produced 13,440 records covering 21 program runs, with zero
  unmatched start/end events. Program durations use the critical span across
  cores/RISCs, not a sum of overlapping times. The raw CSV and parsed report are
  committed. This first capture has no accumulated waits/hardware counters;
  `TT_METAL_PROFILER_SUM=1` was identified in the pinned SDK and enabled for the
  next profiling run. Operator identity remains null until correlated.
- Independently replayed the full CPU shape decoder on the saved poster latent.
  Captured nine distinct convolution signatures across all 40 invocations,
  from 4,490 to 1,711,800 voxels and 64–4096 channels. Preserved all source and
  output rows. Recorded occurrences (4,1,17,1,9,1,5,1,1), file/checkpoint/source
  digests, and input provenance. The capture took 99.5 seconds and left source
  files unchanged. Full BF16-rounded reference pairs are materialized for each
  capture; 1,024 output rows per map are cross-checked against an independent
  coordinate dictionary, with all input neighbors available.
- Interpretation: the evaluation path works on hardware and rejects the known
  error. The expanded corpus prevents optimizing only tiny synthetic cases.
  No kernel/configuration has been promoted yet. Next: explicit row-chunk
  screening, paired confirmations, wait profiling and resource calibration.

### 002 — wait attribution, calibration and workspace capture

- Evidence: baseline profiling had no operation identity or accumulated waits;
  live allocation snapshots could not verify temporary workspace.
- Work: named reader/compute/writer regions, reader wait/gather accumulation,
  serialized configuration screening and bracketed comparisons, arithmetic
  capacity references, allocator-event capture and unscored precision probes.
  Vectorized the arithmetic model; all nine real shape models now run without
  per-block Python loops (the largest takes about 1.75 s on this CPU).
- Validation: 10 CPU evaluator tests pass. Hardware memory capture on the
  surface-256 explicit case records seven DRAM allocations, a 1,830,912-byte
  peak of new aligned DRAM allocations, and a 91,013,120-byte conservative
  workspace bound after adding all device L1 capacity. This verifies only that
  case/configuration; larger and changed candidates still require capture.
- Profile: all seven custom implicit programs were identified, with zero
  unmatched events. The last two spans are 262.4–262.8 us. Reader gather-plus-
  weight time is about 238 us median per core, reader buffer wait about 3 us,
  and compute input-buffer wait about 251 us. These overlapping intervals
  support a data-supply hypothesis; they must not be added together.
- Calibration: an initial BF16-output small-integer dense test failed exact
  equality and is retained as rejected calibration evidence. Repeating with
  FP32 output passes exact equality: 27.899 TFLOP/s for 1024×4096×2048 and
  23.970 TFLOP/s for 256×27648×4096. These are host-completed operator rates,
  not hardware maxima, and their output format differs from the BF16 track.
  Exact-copy tests pass at 131.0 GB/s logical DRAM-clone payload, 138.3 GB/s
  tilize and 124.0 GB/s untilize. Captured gathers measure 61.3 and 78.3 GB/s
  logical payload for 256 and 2048 output rows. None are bus counter readings.
- Precision: the known failure survives FP32 GEMM output plus host FP32 bias
  addition, so postponing output rounding alone does not fix it. Raw-output
  tolerance checks pass because their reference has a different magnitude;
  the original bias-inclusive acceptance gate still fails. No threshold changed.
- Interpretation: measurement paths now work on hardware. No performance
  candidate has been promoted. Next: the chunk/gather screen, paired repeats,
  full correctness coverage and per-candidate memory capture.

### 003 — chunk screening and discarded direct tiled gather

- Hypothesis: the small-case wait profile and large-case traffic model suggest
  reducing data-supply overhead. Test larger explicit row chunks and a new
  gather that writes tiles directly, then calls the existing TTNN matmul.
- The tiled-gather implementation passed five Watcher-enabled tests covering
  missing neighbors, ragged channels, changing-feature reuse, device chaining,
  FP32 residual preservation and 17 chunks with cross-chunk neighbors. The
  candidate patch and tests are archived in `iteration-003/tiled-gather.patch`.
- Full wide-layer screening, seven repetitions per configuration:

| Algorithm / row chunk | Median ms | Correctness |
| --- | ---: | --- |
| Explicit / 256 | 58.120 | Pass |
| Unchanged baseline control / 256 | 58.089 | Pass |
| Explicit / 512 | 43.454 | Pass |
| Explicit / 1024 | 38.401 | Pass |
| Explicit / 2048 | 42.072 | Pass |
| Explicit / 8192 | 43.212 | Pass |
| Direct tiled gather / 1024 | 38.404 | Pass |
| Direct tiled gather / 2048 | 42.389 | Pass |
| Direct tiled gather / 8192 | 42.977 | Pass |
| Implicit | 1419.650 | Pass |
| Masked implicit | 1505.015 | Pass |
| Masked split-K=4 | 1517.056 | Fail: 1/2 elements in feature variants 0/1 |

- Three independent A/B/A rounds, eleven repetitions per trial, confirm
  explicit row_chunk=1024 against 256. Candidate medians are 38.412, 38.515
  and 38.454 ms; bracketed speedups are 1.5134×, 1.5095× and 1.5106×.
  Baseline drift stays below 0.135%; the maximum noise proxy is 1.978%.
  All outputs, feature-reuse checks and integrity checks pass for this case.
- Decision: retain row_chunk=1024 as a configuration lead. Discard direct tiled
  gather for this workload: it does not improve on the best ordinary gather.
  Its active API/kernel changes are removed; the complete patch and raw evidence
  remain reproducible. The `tiled_gather_v1` accounting model describes that
  archived experiment, not a currently exposed device algorithm.
- Limits: the regression fixture, all synthetic/real cases and workspace bound
  have not yet been revalidated for row_chunk=1024. The known original numerical
  failure remains unresolved. The default is unchanged and no global candidate
  or state-of-the-art claim is promoted. Device workers exited normally and the
  shared board was released by 07:10 UTC.

### 004 — preserve FP32 bias residuals and screen a cheaper accuracy path

- Evidence: the precision probe confirms that the known error originates in
  the floating-point GEMM, before bias. It also exposes a separate loss in the
  SDK's row-broadcast bias addition, even when the result storage is FP32.
- Reproduction: a 37×33 BF16 input with values 256, 1 and 0.125, unit weights,
  and exactly representable BF16 bias -256 should produce 1.125. All five
  algorithms instead produce 1.0 with FP32 output. The corrected reproduction
  fails for all five before the change; its raw report is retained.
- Change: for requested FP32 output, convert the bias to FP32 and explicitly
  expand it to the result shape, selecting the equal-shaped SFPU addition
  path. All 22 resident tests pass with Watcher enabled, including all five
  new cancellation tests and the existing BF16, reuse, chaining and split-K
  checks. This currently adds an output-sized temporary bias tensor; fusion or
  bounded expansion remains a performance/memory improvement opportunity.
- The original BF16 wide-layer regression remains unresolved by this fix.
  Changing to FP32 GEMM output or row_chunk=1024 does not remove its one failure.
- CPU-only approximation screen: two signed base-256 digits, a global feature
  power-of-two scale and per-output weight scales pass the unchanged BF16
  tolerances for both feature variants across all nine full decoder shapes.
  This is evidence to try four integer products on the device, not proof of
  device correctness, full-decoder numerical equivalence, or runtime improvement.
  `python -m benchmarks.tenstorrent.digit_study --corpus ... --base 256
  --digits 2 --output ...` reproduces the study with source/fixture digests.

### 005 — four integer products restore numerical correctness

- Baseline audit: explicit row_chunk=1024 passes only 13 of the 21 existing
  synthetic, regression and real-shape cases. Eight cases fail the unchanged
  numerical gate. Several real cases contain repeatable isolated errors near
  4, 8 or 16, beyond ordinary BF16 rounding; repeated outputs are bitwise equal.
- Candidate: `prepare_sparse_digit_weights` plus `digit_explicit_gemm` use two
  signed base-256 digits for each BF16 operand. Immutable weights have per-output
  power-of-two scales. Each forward recomputes a global feature scale and both
  feature digit tensors on the device. Two gathers feed four FP32-output GEMMs;
  equal-shaped FP32 arithmetic reconstructs the result, adds bias, and rounds to
  BF16 once. Feature-dependent work is timed and never cached across calls.
- All 21 cases pass numerical, changing-feature, layout, input-mutation and
  source-integrity checks. Six focused Watcher tests pass, covering chaining,
  reuse, ragged dimensions, bias cancellation and many chunks; 11 CPU evaluator
  tests pass. This is a BF16-accuracy operator result, not FP32-exact inference.
- Resource gate: 20/21 cases pass the conservative workspace bound. The
  388,659-row 128→512 layer fails at 1,681,338,368 bytes against the 1 GiB cap.
  Its numerical checks pass. The allocator trace identifies the final ragged
  tiled concat: the SDK retains tiled inputs, untilized inputs, row-major output
  and retilized output together. The report remains failed; no limit changed.
- Screening medians, three repetitions: 141.118 ms for 4,490 rows at 1024→4096,
  1,218.730 ms for 388,659 rows at 128→512, and 2,802.346 ms for 1,711,800 rows
  at 64→64. This candidate is slower than ordinary BF16 GEMM and is retained
  as an experimental accuracy path. It is not a promoted performance winner.
- Raw reports, test reports and exact DRAM allocation/deallocation events are
  archived under `iteration-005`. The compact event archives reproduce the
  workspace bound; `allocation-evidence.json` also records full trace digests
  and external paths. Next: remove the ragged-concat memory overhead, then
  optimize the valid path and test the new full-sampler topology.

### 006 — avoid the ragged tiled-concat fallback

- Pad only the final row chunk to a tile boundary when every earlier chunk is
  aligned, concatenate in tiled layout, then trim the logical result. Interior
  unaligned chunks retain the original path so no rows are inserted between
  chunks. Ragged channel counts and FP32 output remain supported.
- Eight Watcher tests pass. The prior 388,659-row 128→512 failure now passes
  numerical, mutation, integrity and workspace gates. The inclusive conservative
  bound falls from 1,681,338,368 to 889,204,736 bytes. Median latency is
  1,209.721 ms over seven repetitions; no paired performance claim is made.
- Added a second full CPU decoder capture from the completed sampler's
  6,181-row latent. It contains nine signatures and 40 layer invocations,
  reaching 2,389,000 CPU-decoder output rows. Both feature variants have fixed
  BF16-input reference outputs and sampled independent topology checks. The
  input is from the same poster scene, not an independent held-out scene.
  CPU-generated topology is not asserted equal to the BF16 device pipeline.
- The new 551,044-row 128→512 case passes all numerical checks but still fails
  the conservative workspace bound at 1,223,438,336 bytes. Its screening median
  is 1,728.331 ms. The evidence remains failed and the cap is unchanged.
- Decision: retain this local memory fix. Larger outputs need bounded output
  assembly rather than repeated full-output concatenation. Full-suite reruns,
  paired performance confirmation and application integration remain required
  before promotion; the default algorithm is unchanged.

### 007 — bounded output assembly and larger row chunks

- For tile-aligned row chunks, allocate the final tiled output once and copy
  each completed chunk directly to disjoint tile positions. The data-movement
  kernel batches eight tiles per core through bounded L1 scratch. It supports
  BF16 and FP32 storage, ragged tails/channels and changing tensor addresses.
  Unaligned interior chunks retain the concatenation path.
- The fixed contract has always excluded the final output from temporary
  workspace. The evaluator now identifies its final live allocation, preserves
  earlier lifetimes at reused addresses, and reports inclusive and temporary
  peaks separately. Missing, ambiguous and undersized output identities are
  rejected; tests cover those cases. Only padded tensor bytes are excluded, so
  oversized backing allocations cannot hide workspace. A final CPU replay of
  all 31 captured traces, including the initial probe, verifies this stricter
  accounting and preserves every pass. `memory-reaudit.json` records both bounds
  and source/trace digests. The 1 GiB cap and numerical gates are unchanged,
  and all earlier failed reports remain available.
- Tested a bounded-expansion configuration policy: row_chunk is
  `floor_to_32(1048576 / padded_Ci)`, giving 1024/2048/4096/8192/16384 rows for
  Ci=1024/512/256/128/64. This is a recorded experiment configuration, not a
  new API default. Every gather still accesses the complete input topology.
- All 30 prepared cases pass numerical, changing-feature, dtype/layout,
  input-mutation, source-integrity and workspace checks. Coverage includes both
  real topologies, both feature variants, the original regression and all ten
  synthetic fixtures. Maximum temporary-workspace bound is 1,007,091,200 bytes
  on the 2,389,000-row case, including the entire device L1 capacity.
- Three-repetition screening medians include 139.543 ms for the 4,490-row
  1024→4096 layer, 953.455 ms for 388,659 rows at 128→512, 2,332.506 ms for
  1,711,800 rows at 64→64, and 3,225.740 ms for 2,389,000 rows at 64→64.
  The occurrence-weighted proxy is 12.954 s for the original shape corpus and
  18.221 s for the new topology. These use representative layers, not complete
  decoder inference, and are not paired application speedups.
- The arithmetic model now charges all four products and direct output-copy
  traffic, while retaining the previous model for older candidate checkouts.
  The wide layer issues 4.088 TFLOP in 139.543 ms (29.29 issued TFLOP/s), but
  the 388,659-row narrow layer reaches only 1.84 issued TFLOP/s. This points to
  a different bottleneck in narrow layers; profiling gather, feature-digit
  preparation and matmul separately is the next diagnostic step.
- Three independent A/B/A rounds, eleven repetitions per trial, confirm the
  388,659-row 128→512 improvement against iteration 006 at row_chunk=1024.
  Candidate medians are 953.430, 953.468 and 953.484 ms. Bracketed speedups are
  1.26890×, 1.26869× and 1.26885×; maximum baseline drift is 0.0176% and the
  maximum noise proxy is 0.0634%. All nine trials pass. This combines direct
  output assembly with row_chunk=8192; it does not isolate the two changes.
- Ten focused Watcher tests and 12 CPU evaluator tests pass. The Watcher suite
  covers device chaining and changing-feature reuse with both aligned and
  unaligned chunks, ragged channels, FP32 cancellation and many output chunks.
  Decision: retain the memory fix and this confirmed case-specific performance
  lead. Default dispatch, full-decoder accuracy/runtime, independent scenes and
  external comparator measurements remain outstanding; no global or SOTA
  promotion is claimed. All device workers exited normally.

### 008 — reject reduced-product accuracy shortcuts

- Tested two GEMMs by decomposing only weights, two by decomposing only
  features, and three by omitting the low-digit × low-digit term. The reference,
  fidelity and tolerances stayed fixed. Ten weight-only and 20 alternative
  Watcher tests pass, but the mandatory real/regression gates reject all three.
- Weight-only decomposition passes the 513-row regression at 10.076 ms and the
  4,490-row wide layer at 74.635 ms. On 388,659 rows at 128→512 it runs in
  496.154 ms but has three failing elements in feature variant 0, with a maximum
  absolute error of 8.026. Variant 1 passes. It is not an eligible global winner.
- Feature-only decomposition passes the regression at 13.572 ms and the wide
  layer at 94.725 ms, but fails one element in variant 0 of the 388,659-row wide
  layer (771.463 ms). Preserving either operand alone is insufficient.
- Three products pass that late wide layer at 857.466 ms, but fail 18/1 elements
  in the regression's two feature variants and one element in variant 0 of the
  4,490-row wide layer. The supposedly small omitted term cannot be ignored
  under this contract.
- Decision: restore the four-product incumbent. Exact candidate patches,
  base-commit metadata, tests and successful/failed trial reports are archived
  under `iteration-008`. None of these screening timings is a promoted speedup.
  Next: measure the passing implementation's component costs and optimize the
  expensive operations while preserving all four products.

### 009 — attribute costs in the passing implementation

- Added `run --workload components` to instrument the actual candidate's
  top-level TTNN calls with synchronization. Nested calls are not double-counted;
  the separate feature-preparation stage overlaps operation timings. Reports
  include unwrapped forwards before/after instrumentation, both feature values,
  numerical checks, input/source integrity, raw call samples and environment.
  This is a serialized diagnostic, not a promotion score or device-only timing.
- For 388,659 rows at 128→128, unwrapped median is 747.279 ms and serialized
  median is 1,156.604 ms. Mean accumulated operation time per forward is
  218.777 ms matmul, 217.051 ms embedding, 86.975 ms repeat, and 81.086 ms layout
  conversion. Feature-digit preparation spans 151.208/164.437 ms; these stage
  numbers overlap the operation numbers. The global maximum itself is 6.905 ms.
- For 4,490 rows at 1024→4096, unwrapped median is 145.374 ms and serialized
  median is 176.164 ms. Matmul accounts for 91.312 ms, embedding 15.445 ms,
  repeat 12.032 ms and layout conversion 8.382 ms. Both diagnostic runs pass
  all numerical and integrity checks and exit normally.
- Decision: do not optimize the global reduction first. Preserve all four
  products and target repeated scale/bias expansion, gather/layout overhead,
  and shape-specific matmul configuration. Serialized costs guide hypotheses;
  subsequent changes still require the ordinary full-operator evaluator and
  paired measurements.

### 010 — broadcast FP32 scale/bias tiles without full expansion

- Iteration 009 attributed 86.975 ms of serialized narrow-layer time to repeat
  operations. Prepare one immutable 32-row scale/bias tile with the weights;
  reshape aligned result chunks into batches of tiles and broadcast across the
  batch dimension. Equal H/W dimensions preserve the FP32 SFPU path. Ragged
  chunks retain explicit expansion. Feature normalization also broadcasts its
  power-of-two scale directly: its BF16 values are exactly representable in
  the FPU input format. All four integer products and numerical gates remain.
- All 30 corpus cases pass both feature variants, input/source integrity,
  repeatability, output format and allocator-traced workspace checks. Maximum
  temporary bound remains 1,007,091,200 bytes. Ten Watcher tests and 12 CPU
  evaluator tests pass; the FP32 cancellation test exercises an aligned chunk
  and a ragged tail. The accounting label is `digit_explicit_v3`; useful and
  issued arithmetic counts are unchanged.
- Three A/B/A rounds with eleven repetitions confirm the 388,659-row 128→128
  improvement against iteration 009 with the same row_chunk=8192. Candidate
  medians are 689.891, 689.900 and 689.849 ms versus approximately 747.5 ms.
  Speedups are 1.083326×, 1.083646× and 1.083452×; maximum noise is 0.1136%
  and maximum baseline drift is 0.0240%. All nine trials pass.
- The 4,490-row 1024→4096 comparison is inconclusive under the preset noise
  rule. All three rounds are faster (1.058887×, 1.065016× and 1.056358×), but
  the minimum 5.6358% gain is below the campaign's 5.7510% noise estimate.
  All nine trials pass numerically; retain the failed performance decision
  rather than interpreting these runs as a confirmed wide-layer win.
- Full-corpus three-repetition screening gives occurrence-weighted proxies of
  12.037 s and 16.924 s for the two poster topologies. These are unpaired sums
  of representative layers, not decoder timings or independent scenes.
- A separate post-change narrow-layer component diagnostic passes: repeat
  time falls to 1.058 ms per serialized forward, while matmul (218.134 ms) and
  embedding (216.197 ms) remain similar. Unwrapped median is 689.668 ms.
  The diagnostic changes overlap and synchronization, so its operation times
  are evidence for the mechanism, not additive savings in the lean score.
- Decision: retain the optimization on the confirmed narrow-layer gain and
  full contract coverage. The wide gain, application benefit and SOTA status
  remain unconfirmed. Reports are unchanged in `iteration-010`; its compact
  allocation-event bundle reproduces all 30 original workspace bounds and
  records the full-trace hashes. Every worker exited normally.

### 011 — gather both feature digits directly into tiles

- Revisit iteration 003's direct tiled gather for the four-product algorithm.
  Both high/low feature tables now gather into matmul tiles directly, using the
  resident original-order topology. Each invocation reads fresh feature values;
  missing neighbors and padded rows remain zero. Unaligned multi-chunk calls
  retain embedding plus tilization. Numerical decomposition, four products,
  HiFi3 fidelity and tolerances are unchanged. The accounting model is
  `digit_explicit_v4` and charges both gathers' requested payloads.
- All 30 existing corpus cases and six new independent LiDAR geometry cases
  pass both feature variants, repeatability, input/source integrity, output
  dtype/layout and allocator-traced workspace gates. The LiDAR corpus uses
  21,359 real voxels and synthetic BF16 operands at six channel configurations;
  it is not BEVFusion inference or captured model activations. Its preparer,
  pinned source attribution and unchanged manifest are included. Thirteen
  Watcher tests and 12 CPU evaluator tests pass, including exact gather checks
  for missing/repeated neighbors, chunk boundaries and ragged channels/rows.
- Three A/B/A rounds with eleven repetitions confirm the 388,659-row 128→128
  gain against iteration 010 at the same row_chunk=8192. Candidate medians are
  552.117, 552.114 and 552.229 ms versus approximately 689.9 ms. Speedups are
  1.249600×, 1.249482× and 1.249355×; maximum baseline drift is 0.0236% and
  maximum noise is 0.1151%. All nine trials pass.
- The 4,490-row 1024→4096 comparison remains inconclusive: round speedups are
  1.001734×, 0.999601× and 1.009230×, with maximum noise 3.4229%. All nine
  trials pass numerically, but this is not a confirmed wide-layer improvement.
- Maximum temporary-workspace bound remains 1,007,091,200 bytes. On the
  confirmed narrow layer it falls from 514,855,936 to 426,944,512 bytes. All
  36 archived compact allocation-event replays reproduce the original bounds,
  allocation counts and final-output exclusions; full-trace hashes are retained.
- Full-corpus three-repetition screening gives occurrence-weighted proxies of
  9.881 s and 13.863 s for the two poster topologies. The largest 64-channel
  cases screen at 1,249.609 ms (1,711,800 rows) and 1,739.282 ms (2,389,000
  rows). These are unpaired operator measurements, not full-decoder speedups.
- A separate component diagnostic passes at 552.180 ms unwrapped. Embedding
  calls disappear and serialized layout conversion falls from 80.179 to
  10.371 ms per forward; generic operations, which include the new gathers,
  take 300.424 ms. Matmul remains approximately 217.731 ms and feature-digit
  preparation spans 121.111/126.125 ms. The stage overlaps operation timings;
  synchronized diagnostics cannot be summed into savings in the lean score.
- Decision: retain the change for the confirmed narrow-layer gain and expanded
  correctness coverage. Default dispatch, application-level benefit, matched
  external comparisons and SOTA status remain unconfirmed. All 61 device
  workers exited normally. Unchanged reports are in `iteration-011`.

### 012 — preserve allocator evidence with cheaper memory capture

- Disable TTNN's Python argument/return sidecar during memory capture, as
  recommended by the pinned SDK for memory-only tracking. Keep C++ graph capture
  and its allocator events. No kernel, reference, fidelity or tolerance changes.
- Five representative probes (tiny, wide, narrow, LiDAR and the largest layer)
  reproduce iteration 011's allocation lifetime/size sequences and workspace
  bounds exactly, including which final allocation is excluded. The largest
  remains 1,007,091,200 bytes. A CPU replay verifies event ordering, reused
  addresses, allocation counts and exclusion identity from committed bundles.
- The 2,389,000-row complete worker takes 57.982 s versus 434.998 s in the
  archived run, with 2.982 s spent in the new capture. The narrow worker takes
  23.474 s versus 71.223 s. These are observed evaluator-overhead reductions
  across separate runs, not paired operator/application speedups.
- Add an explicit tensor gate for allocated storage on the expected device,
  DRAM, BF16 and tiled layout in both main and component evaluators. Input
  preparation must pass too. Previously dtype/layout were explicit and memory
  reports identified DRAM output allocations, but lean timings lacked a
  direct residency gate. No prior candidate residency violation was found.
- Full evaluator rejection tests return numerically correct L1 and host
  outputs from isolated candidate checkouts. Both fail the new gate as intended;
  the exact patches and failed reports are preserved. One focused Watcher test
  also rejects wrong dtype/layout, 12 CPU evaluator tests pass, and the component
  evaluator passes its shared-gate integration check. All processes exit normally.
- Decision: retain the stricter gate and cheaper capture. This iteration uses
  five representative cases to verify evaluator behavior; iteration 011's
  36-case kernel validation is unchanged. Reports, compact allocator events and
  CPU replay are in `iteration-012`. Next: tune integer-product matmul blocking
  while preserving all four products, then measure actual decoder integration.

### 013 — tune tall integer-product matmul blocking

- Select an explicit Wormhole 1D multicast program for suitable tall digit
  GEMMs, using a maximum six-tile K block, SDK-compatible FP32 subblocks and
  a conservative 1 MiB CB estimate. Other shapes retain SDK selection. All
  four products, HiFi3, FP32 accumulation and the numerical contract stay fixed.
- Three eleven-repeat A/B/A rounds against iteration 012 confirm the
  388,659-row 128→128 gain: 536.647/536.753/536.498 ms versus approximately
  552.0 ms, or 1.028786×/1.028540×/1.028775×. Maximum noise is 0.1756% and
  baseline drift 0.0161%. Six K-limit screens and all exact patches are retained.
- All 36 cases pass correctness, residency, integrity and workspace gates;
  maximum workspace remains 1,007,091,200 bytes. Fourteen Watcher tests and
  12 CPU evaluator tests pass. A new test covers a partially occupied core
  grid with ragged channels, missing/repeated neighbors and a one-row tail.
- Unpaired occurrence-weighted proxies are 9.603 s and 13.505 s versus 9.881 s
  and 13.863 s in iteration 011. Some shapes screen slower: largest 64→64
  cases by about 1.6–1.7%, LiDAR 16→16 by 4.5%, and LiDAR 17→35 by 2.5%.
  A paired small-case control is inconclusive (ratios 0.8683/0.9937/2.0851,
  maximum baseline drift 41.0%). No general per-shape or application gain.
- Component diagnostics pass at 536.533 ms unwrapped; serialized matmul is
  203.357 ms versus iteration 011's 217.731 ms. Generic operations remain
  about 301.752 ms. These synchronized timings cannot be added into lean savings.
- Decision: retain in the opt-in digit algorithm, with default dispatch and
  SOTA status unchanged. All 65 device workers exit normally. Unchanged
  reports, patches, allocation-event replays and drivers are in iteration-013.
  Shape-specific regression confirmation and application/external comparisons
  remain required before broader adoption.

### 014 — reject ordinary single-product HiFi4 substitution

- Run a separate unscored fidelity study with the unchanged external BF16
  operand goldens and accuracy thresholds. The fixed HiFi3 contract and all
  incumbent kernels remain unchanged. Higher fidelity alone is insufficient.
- Ordinary HiFi3 fails the known regression at [305,2295] on variant 0. HiFi4
  fails there and at [344,2508] on variant 1. An implicit kernel that keeps
  each result tile in DEST without intermediate spill fails at the same
  coordinates, so intermediate reload alone does not explain the failures.
- Stage downloads locate the error before bias/conversion; those device
  operations match host operations exactly in the diagnostic. Storing the
  identical BF16-rounded weights as FP32 does not fix it. Independent CPU
  float64 dot products agree with the unchanged final goldens at both positions.
  The low-level arithmetic cause remains unresolved.
- Decision: reject this simplification and keep iteration 013's four products.
  Five normal numerical-failure exits are preserved with scripts, source and
  fixture digests in iteration-014. No full sweep, workspace, paired timing,
  application or SOTA claim is made for a candidate that fails accuracy.

### 015 — fuse fresh feature digit extraction

- Replace the Wormhole feature conversion/rounding/layout/concat sequence with
  a bounded SFPU program that directly writes both row-major digit tables.
  Global scale calculation, four products, HiFi3, FP32 accumulation, matmul
  policy and fixed numerical gates remain unchanged. No feature values persist.
- All 36 cases pass full correctness, residency, integrity and workspace gates.
  Five new Watcher tests establish raw-bit equality with the unfused helper,
  including signed zero, ties, extreme scales, padding and chunk boundaries.
  Fourteen existing digit Watcher tests and 12 CPU evaluator tests also pass.
- Three eleven-repeat A/B/A rounds confirm 128→128 at 388,659 rows:
  499.510/499.258/499.593 ms, speedups 1.074352×/1.075079×/1.074263×,
  maximum noise 0.1303%. At 2,389,000 rows, 64→64 takes
  1562.558/1562.359/1562.528 ms versus approximately 1768.0 ms,
  speedups 1.131544×/1.131569×/1.131533×, maximum noise 0.0541%.
- Wide 1024→4096 remains inconclusive (minimum gain 2.1163%, maximum noise
  3.5549%). A three-round 51-repeat small control confirms 35-row 17→45:
  speedups 1.189213×/1.230731×/1.208479×, maximum noise 6.8188%. This does
  not resolve all other small shapes that screened slower in unpaired runs.
- Maximum workspace falls 16.6%, from 1,007,091,200 to 839,528,448 bytes.
  All 37 committed allocator-event replays match their original bounds and
  output exclusions. All 77 device workers exit normally without recovery.
- Unpaired occurrence-weighted proxies are 8.992 s and 12.599 s versus
  9.603 s and 13.505 s. The component diagnostic passes at 499.028 ms;
  synchronized feature preparation is 17.029/17.245 ms versus iteration 013's
  119.565/113.783 ms. These stage timings overlap operation measurements and
  cannot be added into lean savings. Neither proxy is a full decoder result.
- Decision: retain in the opt-in algorithm, with default dispatch unchanged.
  Reports, exact patch, compact allocation events and drivers are in iteration-015.
  Prepared decoder integration and K6 comparator probes have not yet run;
  application benefit, matched external comparisons and SOTA remain pending.

The additional CPU capture tools are:

```bash
python -m benchmarks.tenstorrent.capture_genrecon \
  --checkpoint /home/user/models/genrecon/trellis2/ckpts/shape_dec_next_dc_f16c32_fp16 \
  --latent /home/user/tt-scratchpad/genrecon/sparse-decoder-poster-bounded/decoder_input.npz \
  --output /home/user/tt-scratchpad/flexgemm-autoresearch/poster-shape-captures-v1

python -m benchmarks.tenstorrent.prepare_captures \
  --captures /home/user/tt-scratchpad/flexgemm-autoresearch/poster-shape-captures-v1 \
  --output /home/user/tt-scratchpad/flexgemm-autoresearch/poster-shape-corpus-v1
```

This corpus supports an occurrence-weighted latency proxy. It uses one
representative set of values/weights per repeated shape, so it does not replace
the complete decoder's numerical or runtime validation.
