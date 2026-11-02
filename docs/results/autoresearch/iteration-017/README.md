# Iteration 017: four-product K6 prototype

The four-product K6 prototype passes all 24 feasible fixtures and confirms
large completed-operator improvements on two decoder-derived cases. Keep it
as an integration candidate; production kernels and default dispatch remain
unchanged. Twelve wide fixtures are unsupported, and several other shapes
screen slower. This is not a full-decoder or SOTA result.

The prototype calls the iteration-016 K6 port four times, using the incumbent
fresh feature digit tables and immutable integer weight digits encoded in
FP32. Each call gathers directly into L1 and returns an FP32 product. The
existing reconstruction order, scale/bias epilogue, final BF16 conversion and
output-copy helper remain. HiFi3 and all external BF16 golden tolerances stay
fixed. The four unused native BF16 table outputs remain inside timed forwards.
Static weights and integer topology are prepared before timing; feature values
are recomputed on every call.

Three clean eleven-repeat A/B/A rounds give:

| Case | Prototype medians, ms | Speedups | Maximum noise |
| --- | --- | --- | --- |
| 388,659 rows, 128→128 | 171.735 / 171.663 / 171.680 | 2.908429 / 2.910213 / 2.908000 | 0.2249% |
| 2,389,000 rows, 64→64 | 479.340 / 484.331 / 479.125 | 3.259543 / 3.225936 / 3.260971 | 3.2865% |

Baselines are approximately 499.5 ms and 1562.4 ms respectively. Candidate
forwards include all feature preparation, four gather/matmul calls, temporary
native outputs, reconstruction, layout/copy work and synchronization. Each
trial uses both fresh feature variants, checks every measured output against
the same immutable CPU goldens and verifies repeatability and input integrity.
The prototype harness captures workspace before timing. It is a separate
harness whose exact source and dependency commit are recorded, not a change
to the production evaluator or its numerical gate.

The first narrow bracket overlapped CPU-only structural-model analysis. That
round is retained as diagnostic and was excluded prospectively before its
comparison completed, regardless of its result. The clean confirmation uses
original rounds 1 and 2 plus one additional bracket; see
`timing-interference-note.json` and `clean-narrow-confirmation.json`.

All six LiDAR-geometry, ten synthetic and eight feasible decoder-topology
fixtures pass. This fixes the seven failed-numerics cases in iteration 016's
ordinary K6 study without changing tolerances. LiDAR feature/weight values are
synthetic; the two decoder topologies are from the same poster. The twelve
unsupported wide cases remain listed, including the known 513-row regression.
There is no fallback in this prototype and no claim of full 36-case coverage.
Unpaired screening retains all results: small cases and both 128→512 decoder
shapes are slower. A shape-aware path and fallback require separate validation.

The stronger product diagnostic initially assumed integer inputs whose sums
fit FP32 would produce exact CPU integer dots. Both the FP32-weight and
BF16-weight K6 variants fail that oracle on a ragged case (maximum difference
20). Those failed reports remain unchanged. A follow-up compares all four
products directly with the incumbent HiFi3 matmul: all 48 match bit-for-bit,
including empty neighborhoods, repeated and out-of-chunk neighbors, padding,
row chunks, tap-chunk partial reload and two feature variants. Thirty-two also
match exact CPU integer sums; sixteen do not, with identical deviations in
the incumbent. Two separate HiFi4 controls do not remove those deviations.
The low-level cause is unresolved; representability alone was an insufficient
oracle assumption. No fixed convolution accuracy gate was relaxed, and the
BF16-weight variation was not adopted.

All 33 captured forwards meet the 1 GiB workspace limit; the maximum bound is
726,282,240 bytes. Bounds exclude only identified final-output padded bytes
and include whole-device L1 capacity. All 50 workers exit normally: 48 successes
and two retained exact-integer oracle failures, with no timeout or recovery.
`verify.py` checks archived hashes, source versions, all allocator replays,
exit classifications, matched operands and the clean timing comparisons.
`structural-work.json` models custom matmul calls and requested payloads; it
is not measured DRAM/NoC traffic or a complete I/O lower bound.

Next, integrate an opt-in path with explicit fallback, protected header hashes,
correct work accounting and full 36-case validation, then test complete decoder
behavior. A later combined kernel could share feature gathers and eliminate
unused native outputs. All comparator-derived sources retain Apache-2.0
attribution; see `PORT-NOTICE` and the embedded license files. Archived scripts
retain their original local paths and source hashes for provenance.
