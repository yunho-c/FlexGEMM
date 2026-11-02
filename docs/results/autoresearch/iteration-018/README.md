# Iteration 018: integrate opt-in four-product implicit convolution

Add `digit_implicit_gemm` to the resident API with explicit fallback. All 36
fixed fixtures pass: 24 use K6 and 12 use the existing explicit digit path,
including the known 513-row regression. Keep this path opt-in. Small cases and
some wide outputs can regress; complete decoder benefit and SOTA remain pending.

The implementation reuses the existing fresh feature digit extraction,
four-product reconstruction, FP32 scale/bias and output-copy helpers. It adds
two statically prepared FP32 encodings of the BF16 integer weight digits and
per-topology index/plan caches. Different same-shaped topologies never share
indices; weights and topology contain no cached feature values. The optional
`prepare_sparse_digit_implicit` function makes static preparation explicit.
Both BF16 and FP32 output modes are supported.

The internal K6 builder is trimmed from the pinned Apache-2.0 prototype,
fixed to HiFi3 and stripped of environment-controlled debug paths and model
loading. The four kernel files preserve iteration017's arithmetic. The
physical-coordinate multicast guard and actual per-core CB bound remain.
Non-Wormhole devices, volumes above 32, unaligned row chunks and CB plans above
900 KiB use explicit digit GEMM and report a reason. Runtime files and upstream
license notices are included in the built wheel.

Three eleven-repeat A/B/A rounds against the unchanged iteration015 kernel
sources, using the same native evaluator, give:

| Case | Candidate medians, ms | Round speedups | Maximum noise |
| --- | --- | --- | --- |
| 388,659 rows, 128→128 | 172.081 / 171.639 / 171.648 | 2.900530 / 2.909645 / 2.908647 | 0.8494% |
| 2,389,000 rows, 64→64 | 458.533 / 462.537 / 459.105 | 3.407418 / 3.378234 / 3.403709 | 3.0839% |
| Wide fallback, 4,490 rows, 1024→4096 | 129.340 / 130.145 / 129.397 | 0.997758 / 0.992303 / 0.999134 | 2.4281% |

The fallback control is reported separately; it is not a claimed new kernel
speedup. Timings include fresh feature processing, gathering, all four products,
native temporary outputs, reconstruction, final assembly and synchronization.
Both feature variants, input immutability and repeated results are checked by
the unchanged native numerical protocol. Preparation is reported separately.
No concurrent hardware worker or CPU-heavy analysis was launched during this
paired campaign.

All 36 full-validation cases pass residency, dtype/layout, source/fixture integrity,
repeatability, numerical and workspace gates. Maximum workspace is 726,282,240
bytes, excluding only the identified final output's padded bytes and including
whole-device L1 capacity. All 36 allocator-event replays reproduce the recorded
bounds. All 64 supervised device workers exit normally without timeout or reset.

Fifteen Watcher tests cover changed features/weights, separate same-shaped
layouts, immutable input, missing/repeated/out-of-chunk neighbors, FP32
cancellation, tap-chunk partial reload, ragged rows/channels, many chunks,
and explicit fallback. Thirteen CPU evaluator tests pass. The wheel check
compares eight new runtime/header/license files byte-for-byte with source.

The evaluator adds included-header hashes, static-plan reporting and a new
independent structural work model for this algorithm. Existing numerical gates,
reference outputs, timing boundaries and baseline work models remain unchanged.
The model counts four fused products with empty-tap skips, no DRAM im2col,
FP32 weight payloads, repeated index reads and unused BF16 output tables. These
are requested payloads, not measured DRAM/NoC bus traffic or peak workspace.

The unpaired occurrence-weighted proxies are 6.367 s for the first poster
topology and 8.848 s for the full-sampler topology. These are not measured decoder latencies. All unpaired cases,
including regressions, remain in `unpaired-comparison.json`. The independently
sourced LiDAR geometry still has synthetic features/weights, and the two decoder
topologies are from the same poster.

`archive.json` hashes unchanged reports and the exact source patch; `verify.py`
checks those hashes, all 36 workspace replays, source/header coverage, dispatch
counts, test counts and paired comparisons without a device. Full traces and
wheel artifacts remain at their recorded scratch paths and hashes. An isolated
matched decoder adapter is prepared for the next experiment; original GenRecon
sources are unchanged. Its strict saved CPU quality gates will be preserved.
