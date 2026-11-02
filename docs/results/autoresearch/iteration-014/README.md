# Iteration 014: reject a single-product higher-fidelity substitution

Five diagnostic workers tested whether ordinary single-product matmul could
replace the four-product integer decomposition on the existing regression case.
This was a separate, unscored fidelity study suggested by the planning document;
the fixed HiFi3 contract, BF16 operands, CPU goldens and tolerances are unchanged.
The incumbent remains iteration 013 (`ed9ca83`). No candidate kernel changed.

The ordinary HiFi3 control fails at row 305, channel 2295 for feature variant 0.
HiFi4 also fails there and at row 344, channel 2508 for variant 1. A second
implementation using the existing implicit kernel keeps each output tile in
DEST throughout its K reduction, without intermediate spill/reload. It fails
at the same coordinates with HiFi4. Intermediate reload alone therefore does
not explain these failures.

A stage diagnostic downloads raw FP32 matmul, FP32 biased and BF16 outputs.
Device bias addition and conversion agree exactly with the corresponding host
operations in this run. The raw matmul already contains the error: at variant
1's failing coordinate it returns -5.209697 versus a bias-subtracted reference
of -4.208978. Variant 0's smaller raw error becomes unacceptable after bias
moves the result closer to zero. The raw-reference comparison is diagnostic;
acceptance always uses the unchanged final golden and thresholds.

Storing the same BF16-rounded weights as FP32 leaves these stage-probe failures
unchanged. Independent CPU float64 dot products at both coordinates agree
closely with the immutable float32 goldens; neither reference was replaced.
The controls narrow the problem to the multiplication/accumulation path but
do not uniquely identify its low-level cause. Exact values are in the reports.

Decision: reject the proposed substitution and retain the validated four-product
HiFi3 algorithm. No full-corpus, workspace or paired performance sweep follows
because the prerequisite accuracy check fails. Timings in the ordinary probes
are unscored, and stage probes include downloads and have no timing score.
All five workers exit normally with code 1 for numerical failure, without
timeout or recovery. These are preserved failures, not successful candidates.

The four probe scripts are archived with their original local dependency path
(`/home/user/GitHub/flexgemm-evaluator`, commit `2ed860d`). Source, script, fixture,
contract and SDK revision provenance are recorded in their reports. The implicit
probe overrides fidelity only in its private process. `archive.json` hashes the
unchanged reports and scripts. The earlier planning metadata describes the
proposal before execution; `decision.json` records the final rejection.
