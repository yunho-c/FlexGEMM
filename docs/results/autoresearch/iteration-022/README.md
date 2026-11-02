# Iteration 022: attribute K6 waits before changing weight storage

This iteration adds a diagnostic profiler and leaves production kernels unchanged
at `ec0fe26`. It measures first, middle and final row chunks from three real decoder
layers, using both fixed feature variants and three warmed product samples per
selection. Separate generated kernel copies measure selected compute waits,
gather backpressure/read barriers and weight read/multicast waits. A base copy
adds only whole-kernel zones. All timings are diagnostic; none is a promotion score.

| Layer | Rows per chunk | Weight wait | Gather wait |
| --- | ---: | ---: | ---: |
| 388,659 rows, 128→128 | 8,192 | 52.53% | 5.80% |
| 388,659 rows, 128→512 | 8,192 | 47.04% | 2.37% |
| 2,389,000 rows, 64→64 | 16,384 | 28.14% | 31.99% |

Each percentage is the median across 18 product samples of the median per-core
selected wait divided by that core's UNPACK kernel scope. Weight and gather waits
come from separate workers. These are RISC wall-time observations, not hardware
utilization counters or an attainable speedup ceiling. The larger 64-channel case
has a materially different balance from the two 128-channel cases.

Weight multicast receivers also show substantial handshake waits. Sender-only
read-barrier observations support investigating weight transfers, but do not
separate DRAM, NoC and scheduling effects. Gather read barriers include local
zero copies and index prefetches. Built-in aggregate CB waits include selected
waits and must never be added to them. Math issue-region times do not measure
Tensor Engine active cycles.

Two complete narrow-operator captures each contain all 650 programs, including
192 K6 products. Host calls take 168.343 / 168.957 ms; the union of device kernel
spans covers 164.986 / 164.985 ms. Gaps between program spans total only
0.405 / 0.407 ms. The K6 program spans total 118.717 / 118.692 ms. Device work is
the next priority for this case; these observations do not establish that host
dispatch is negligible for other shapes or execution modes. Profiler overhead
and host synchronization remain included. The operator captures are not a new
benchmark or speedup claim.

All 16 completed workers pass 456 checks: 184 comparisons against the existing
BF16-rounded CPU goldens, including 92 bitwise native/instrumented operator
comparisons; 270 bitwise product comparisons; and two captured complete-operator
repeat comparisons. The original inputs and production sources remain unchanged.
All 272 selected CSV segments have balanced start/end records. CPU tests pass
15/15, including regression guards for the accumulator collision described below.
This is diagnostic coverage of selected layers, not a replacement for the full
36-case promotion suite or the original strict FP32 full-decoder gates.

The initial probe exits normally with a collection failure: this SDK requires
`TT_METAL_PROFILER_MID_RUN_DUMP=1` for a CSV before device close. Inspection also
finds that the SDK owns UNPACK accumulator slot 1 and PACK slot 2. The corrected
worker uses only UNPACK slot 2 for either selected weight or gather waits in
separate runs; MATH and the data-movement RISCs use their own slots. The failed
report, log and original harness are retained. All 17 workers terminate without
a timeout or device reset. One short CPU analysis pass overlaps the diagnostic
campaign; there is no claim of isolated lean host timing here.

`archive.json` hashes the original compressed reports/logs and selected raw CSVs.
`verify.py` regenerates instrumentation, replays every selected CSV, recomputes
all interval unions and scope statistics, and checks source/fixture evidence
without opening the device. Full warmup CSVs remain in scratch storage.

Decision: next test BF16 storage/transfers for the already-BF16 integer weights,
keeping arithmetic order and tap grouping fixed for the first comparison. Current
weights are carried as FP32, so this can reduce requested weight payload without
first implementing a larger four-product fusion. Format-dependent arithmetic
must still pass individual product controls, full correctness/workspace gates,
paired timing and decoder comparisons. Shared gathers remain relevant to the
largest case. Independent-scene coverage, matched external comparisons and SOTA
are still pending.
