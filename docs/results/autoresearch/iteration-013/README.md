# Iteration 013: shape-selected integer matmul blocking

Select an explicit Wormhole 1D multicast matmul program for tall four-product
digit GEMMs. The maximum K block is six tiles, subject to divisibility and a
conservative 1 MiB circular-buffer estimate. Wide/short or infeasible shapes
retain the SDK default. The estimate does not replace compiler checks or the
allocator-derived workspace bound. HiFi3, FP32 accumulation, all four products,
fixtures and numerical tolerances are unchanged from iteration 012 (`2ed860d`).

Six maximum-K screens (2, 6, 9, 12, 18, 27) selected six. Exact patches and
all screening reports are retained, including the initial K18/1x4 variant.
Three independent A/B/A rounds with eleven repetitions confirm the
388,659-row 128→128 improvement at row_chunk=8192. Candidate medians are
536.647, 536.753 and 536.498 ms versus approximately 552.0 ms. Round speedups
are 1.028786×, 1.028540× and 1.028775×, exceeding the 2%/noise gate. Maximum
noise is 0.1756% and baseline drift 0.0161%.

All 36 cases pass numerical, repeatability, input/source integrity, expected
device/DRAM/BF16/tiled output and workspace checks. Maximum temporary workspace
is 1,007,091,200 bytes. Fourteen Watcher tests and 12 CPU evaluator tests pass.
The new Watcher case exercises 65 row tiles across 64 cores, a one-row final
chunk, ragged channels and missing/repeated neighbors with fresh features.

Unpaired full-corpus screening gives occurrence-weighted operator proxies of
9.603 s and 13.505 s versus iteration 011's 9.881 s and 13.863 s. Some shapes
screen slower: the two largest 64→64 cases by about 1.6–1.7%, LiDAR 16→16 by
4.5%, and LiDAR 17→35 by 2.5%. Small synthetic cases are also variable. A
three-round small-case control, which uses the SDK default in both versions,
has speedup ratios 0.8683, 0.9937 and 2.0851 and up to 41.0% baseline drift.
It establishes neither a regression nor a gain. These results remain visible
in `unpaired-overview.json` and `confirm-small`; no universal speedup is claimed.

The component diagnostic passes at 536.533 ms unwrapped. Serialized matmul
time is 203.357 ms versus iteration 011's 217.731 ms; generic operations remain
approximately 301.752 ms. Instrumentation changes synchronization and overlap,
so these times are not additive savings or the promotion score.

Retain the program policy in the opt-in digit algorithm for the confirmed
narrow-layer gain. Default dispatch, complete decoder benefit, external
comparisons and SOTA status remain unconfirmed. All 65 device worker reports
record normal exit without recovery. Screening regressions need shape-specific
confirmation before broader dispatch adoption.

`archive.json` hashes 152 unchanged reports/patches and a compact DRAM event
bundle. Full traces remain at the paths and hashes in each workspace report.
Run `python docs/results/autoresearch/iteration-013/verify.py` to check hashes,
replay all 36 bounds and audit terminal worker results without opening a device.
The experiment drivers preserve their original local paths. Measurement reports
identify the base commit plus actual source/evaluator digests before this commit.
