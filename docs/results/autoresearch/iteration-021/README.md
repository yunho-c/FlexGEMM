# Iteration 021: omit unused native BF16 product tables

The implicit digit path now requests only each K6 call's FP32 product. It omits
the unused BF16 row-major output table, sentinel row, conversion, untilization
and writes. The four products, FP32 reconstruction, scale/bias and final output
format are unchanged. The internal builder retains its table-output mode for
direct controls; public algorithms and defaults do not change.

Tap planning deliberately uses the original conservative CB bound. Reported
actual CB bytes exclude the omitted table buffers. All 36 fixtures retain the
same dispatch and tap grouping: 24 implicit and 12 explicit fallbacks. HiFi3,
operand rounding, immutable CPU goldens and numerical/workspace gates are fixed.

Three A/B/A rounds per case, eleven timed repetitions per trial, compare the
candidate with frozen commit `7bda9c4` under the same native evaluator:

| Case group | Candidate medians, ms | Round speedups | Maximum noise | Result |
| --- | --- | --- | ---: | --- |
| narrow | 167.218 / 167.272 / 167.280 | 1.026517 / 1.026451 / 1.026341 | 0.2351% | confirmed gain |
| largest | 445.499 / 447.120 / 456.828 | 1.054251 / 1.040333 / 1.028688 | 5.7875% | inconclusive |
| wide-feasible | 632.766 / 633.058 / 633.114 | 1.028286 / 1.027944 / 1.028248 | 0.1930% | confirmed gain |
| fallback | 129.587 / 129.070 / 129.468 | 0.992354 / 1.001601 / 1.000159 | 1.3651% | control |

The narrow case is 388,659 rows at 128→128; largest is 2,389,000 at 64→64;
wide-feasible is 388,659 at 128→512; fallback is 4,490 at 1024→4096. All rounds,
including noisy ones, are retained. The largest case does not meet the fixed
noise-aware gain rule; positive ratios alone are insufficient. Fallback behavior
is a control, not a new kernel speedup. No external CPU-heavy analysis or competing
hardware worker was launched during the paired campaign.

All 36 full-validation cases pass numerical, residency, dtype/layout, input
immutability, repeatability, source/fixture integrity and workspace checks.
Thirty-six allocator replays reproduce a maximum bound of 726,282,240 bytes,
excluding only the identified final output's padded bytes and including whole
device L1 capacity. This maximum is unchanged from iteration018.

Eighteen Watcher tests pass. Three new cases compare 32 individual FP32 products
bitwise against the table-producing path across changed features, ragged rows,
missing/repeated neighbors, multiple row chunks and partial-sum reloads. The
existing convolution tests cover both output dtypes, cancellation and fallback.
Thirteen CPU evaluator tests pass, including preservation of the old work model.

The additional output-mode work model differs only in its label and zero unused
BF16 table-write payload. Other arithmetic and requested traffic fields match
iteration018 on every fixture. The largest case removes 1,223,255,040 requested
BF16 write bytes; this is not a measured DRAM/NoC bus quantity.

A complete decoder run through the unchanged iteration020 stage-cache adapter
has bitwise-identical saved shape/texture outputs and subdivision guides: all
12 records match. Repeated warm outputs/guides also match. Its original strict
FP32 CPU parity gates still fail, as in both earlier application baselines.
The application timing is unpaired and carries no new speedup claim.

All 74 device workers complete normally: 73 return success and the application
returns its preserved numerical-failure status. No timeout or reset occurs.
`archive.json` hashes unchanged reports, source patch and compact allocator events;
`verify.py` checks those hashes, dispatch/tap/work-model equivalence, all workspace
replays, paired statistics and application equality without opening the device.
Full traces and tensor artifacts remain at their recorded scratch paths/hashes.

Decision: retain the FP32-only output path. Next investigate device wait regions
and host dispatch gaps before sharing gathers and weights across the four products.
Independent-scene coverage, stronger external comparisons and SOTA remain pending.
