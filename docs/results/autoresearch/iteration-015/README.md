# Iteration 015: fused feature digit extraction

Fuse fresh BF16 feature digit extraction into one SFPU program on Wormhole.
The reader loads input tiles and the current device inverse scale; compute
performs the existing round/subtract/multiply sequence in FP32; the writer
stores both final row-major BF16 digit tables directly. It explicitly zeroes
channel padding and sentinel rows. Scratch is bounded per core. This removes
full-tensor conversion intermediates and concatenation. Global scale calculation,
four products, HiFi3, FP32 accumulation, matmul policy and tolerances stay fixed.
The unfused helper remains available as an oracle and for other architectures.

Three A/B/A rounds against iteration 014 (`180e5c6`, same kernels as 013) give:

| Case | Candidate medians, ms | Round speedups | Maximum noise | Decision |
| --- | --- | --- | --- | --- |
| 388,659 rows, 128→128 | 499.510 / 499.258 / 499.593 | 1.074352 / 1.075079 / 1.074263 | 0.1303% | Confirmed |
| 2,389,000 rows, 64→64 | 1562.558 / 1562.359 / 1562.528 | 1.131544 / 1.131569 / 1.131533 | 0.0541% | Confirmed |
| 4,490 rows, 1024→4096 | 129.231 / 129.233 / 128.351 | 1.021163 / 1.036283 / 1.038919 | 3.5549% | Inconclusive |
| 35 rows, 17→45 | 2.891 / 2.804 / 2.803 | 1.189213 / 1.230731 / 1.208479 | 6.8188% | Confirmed |

The first three cases use eleven repetitions per trial. The small control uses
51 to investigate unstable short-run screening; all three gains exceed its
larger measured noise. These are completed-operator timings, including fresh
feature work, layouts, gathers, reconstruction, output assembly and synchronization.

All 36 fixtures pass numerical, repeatability, input/source integrity, device,
DRAM, dtype/layout and workspace gates. Maximum temporary workspace falls from
1,007,091,200 to 839,528,448 bytes (16.6%). The narrow case remains 426,944,512
bytes; the largest case benefits most from removing concatenation. Bounds come
from allocator lifetimes, exclude only the identified final output's padded
bytes, and include whole-device L1 capacity.

Five new Watcher tests compare the raw BF16/FP32 bit patterns of both tables
and scale against the unfused helper, including signed zero, rounding ties,
extreme scales, ragged channels/rows, a core boundary and a conversion-chunk
boundary. Fourteen existing digit Watcher tests and 12 CPU evaluator tests pass.
The initial value-equality test run is also retained; the later raw-bit run is
the stronger oracle. All 77 device workers exit normally without recovery.

Unpaired occurrence-weighted operator proxies are 8.992 s and 12.599 s, versus
iteration 013's 9.603 s and 13.505 s. Several small shapes screen slower in those
short unpaired runs; their records remain in `unpaired-overview.json`. The
confirmed 35-row control does not resolve every other small shape or establish
a universal speedup. These proxies are not decoder latency measurements.

A separate diagnostic passes at 499.028 ms unwrapped. Serialized feature
preparation is 17.029/17.245 ms versus iteration 013's 119.565/113.783 ms.
Matmul is 201.486 ms and generic operations 307.903 ms. Stage and operation
measurements overlap and instrumentation changes synchronization; they cannot
be summed into savings in the lean score.

Retain the fused helper in the opt-in digit algorithm. Default dispatch,
complete decoder benefit, matched external comparisons and SOTA remain pending.
`archive.json` hashes 168 unchanged reports/patches and a compact event bundle.
`verify.py` replays 37 memory captures (36 full-validation cases plus the screen),
checks final-output exclusions, hashes and normal worker exits without a device.
Full traces remain at the paths and hashes in the reports. Drivers record the
original local paths and test-oracle refinement. Source digests identify the
exact tested implementation before this commit.
