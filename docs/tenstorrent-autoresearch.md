# Tenstorrent sparse-convolution autoresearch

The objective is competitive sparse-convolution performance on Tenstorrent,
verified under a fixed numerical and residency contract. Improving on one
prototype is not sufficient evidence for a state-of-the-art claim. Published
comparators, multiple workloads, independent confirmation, and representative
application measurements remain required.

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
addition to honoring the lock. The current GenRecon pipeline has been granted
uninterrupted board use until its owner announces completion.

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
workspace limit, but complete peak-workspace verification is still pending;
passing this lower-bound check does not prove compliance with the full cap.

The new `--mode memory` run captures actual allocator events across a warmed
forward. Its conservative bound counts all new aligned DRAM allocations,
including the final output, and adds the entire device L1 capacity. Resident
inputs, weights and topology predate capture. The smoke capture passed on
hardware in iteration 002; each promoted candidate/case still requires its own
capture. The compressed allocation trace is saved beside the report.

The accounting model is tagged `baseline_v1`. Structural kernel changes require
an updated model; do not continue reporting obsolete traffic counts. Missing
measurements are null. Iteration 002 establishes initial dense/gather empirical
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
