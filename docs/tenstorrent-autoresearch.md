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

Current coverage is not a held-out scene benchmark. Additional independent
scenes, captured narrow/late-stage layers, layer occurrence counts, complete
stage timings and accurate-output integration are still outstanding. Synthetic
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

The accounting model is tagged `baseline_v1`. Structural kernel changes require
an updated model; do not continue reporting obsolete traffic counts. Missing
measurements are null. Fixed workload SoL bounds, calibrated dense/gather
reference rates, device wait timing and hardware-counter capture are the next
measurement tasks. No numerical SoL utilization is claimed before calibration.

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
