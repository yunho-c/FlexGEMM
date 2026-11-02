# Iteration 019: complete decoder integration diagnostic

The opt-in four-product implicit path produces bitwise-identical saved shape,
texture and subdivision-guide tensors to the explicit digit baseline on the
4,490-voxel poster latent. Both implementations fail the original saved FP32
CPU decoder parity gates. This experiment establishes matched BF16 application
behavior and a modest latency benefit; it does not establish CPU parity or SOTA.

An isolated GenRecon adapter changes only sparse convolutions. Dense linear,
normalization, residual, subdivision and C2S operators remain inherited. The
original GenRecon checkout is unchanged. The baseline is FlexGEMM iteration015
(`8c1c896`); the candidate is iteration018 (`facc5e2`). Both use the same adapter,
checkpoints, inputs, source checks, per-layer row chunks and topology-cache policy.

Run one A/B/A bracket, with two warm complete calls per decoder in each worker:

| Decoder | Baseline A median, s | Candidate B median, s | Baseline C median, s | Bracket ratio | Baseline drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shape | 11.445487 | 10.068519 | 11.432286 | 1.136104 | 0.1154% |
| Texture | 10.953361 | 10.049999 | 10.936736 | 1.089060 | 0.1519% |

These are diagnostic observations from one scene and one bracket, rather than
the three-bracket operator promotion protocol. Complete-call timing includes
CPU boundary transfers, topology construction, all model operations and final
synchronization. Initial model/weight preparation, quality comparisons and
artifact I/O are outside the timed calls. First-use calls, which can include
compilation and lazy implicit weight/index preparation, are recorded separately;
their asymmetric compiler-cache history is not used for speedup claims.

The 262,144-row topology-cache limit is preserved. Large final-stage maps are
rebuilt even in warm calls, so these are not fully cached-geometry timings.
Measured topology time is 2.96–3.57 seconds per warm call. Each candidate decoder
executes 16 implicit and 24 explicit-fallback sparse convolutions. Model inputs
and features are not cached. All workers own the shared device lock through exit;
no competing hardware worker or external CPU-heavy analysis was launched during
timing. Output comparisons run only after all three workers finish.

Across B-versus-A, B-versus-C and C-versus-A, all 36 saved output/guide records
have exact coordinates and bitwise-identical features. This covers fixed-guide
and predicted-guide shape/texture outputs plus all four subdivision stages for
both guide modes. Each worker's additional warm outputs and predicted guides
also repeat exactly. Independently predicted geometry has 1,724,221 voxels in all
three runs, versus 1,711,800 in the saved CPU reference.

The original CPU gates remain relative L2 ≤0.0001, PCC ≥0.999999, exact coordinates,
exact subdivision signs and exact geometry edge signs. All three workers report
`numerical_failure`: fixed-guide shape relative L2 is 0.014234 with 10,159 edge
sign mismatches; fixed-guide texture relative L2 is 0.007241; predicted coordinates
differ from CPU; the separately captured wide convolution also fails its existing
FP32-reference elementwise gate. Matched-baseline equality does not replace any
of these failures. Operator BF16-rounded golden validation from iteration018
remains a separate contract.

All three workers complete normally with numerical-failure return code 1, no
timeout and no reset. Input, checkpoint, integration, FlexGEMM/header and GenRecon
source checks pass. Live allocator snapshots are retained; this experiment does
not measure a full-model transient memory peak. Full tensor artifacts remain at
their scratch paths, with SHA256 hashes and sizes in `comparison.json`.

`verify.py` checks the archived reports, original gate failures, source coverage,
dispatch counts, repeated outputs and reconstructed timing ratios without using
the device. `compare.py` records the separate CPU tensor comparison procedure.
The next application experiment should test bounded topology reuse during a
stage, including its memory lifetime and new-geometry cost. Further kernel
profiling, independent scenes and matched external comparisons remain necessary.
