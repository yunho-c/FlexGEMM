# Iteration 020: bounded topology reuse within a decoder forward

Retaining one large integer topology during its stage reduces complete decoder
latency while preserving every saved BF16 output and subdivision guide bitwise.
The experiment changes only the isolated application adapter's cache policy;
FlexGEMM kernels and the original GenRecon checkout remain unchanged.

Both sides use iteration018's `digit_implicit_gemm` kernel sources at commit
`a2b31cc`. The baseline preserves GenRecon's 262,144-row persistent-cache limit.
The candidate adds one temporary entry with exact coordinates and batch,
spatial-shape, kernel, dilation, stage and device checks. It retains integer
indices and plans, never features. It releases the entry on layout/device change,
before C2S expansion, on clear, and in a `finally` block on forward return/error.
Masked layouts use the original cache.

Each retained entry is limited to 256 MiB of host tensor storage and 256 MiB of
device metadata between convolutions. Host accounting includes the owned map and
coordinate tensor storage, not Python object overhead or process RSS. Device
accounting deduplicates actual integer buffer addresses and uses their page count,
aligned page size and DRAM bank count. A Watcher test checks this formula against
allocator events. These limits describe retained cache metadata, not a full-model
transient memory bound.

One A/B/A bracket, with three warm complete calls per decoder, gives:

| Decoder | Baseline A median, s | Candidate B median, s | Baseline C median, s | Bracket ratio | Baseline drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shape | 10.145796 | 8.715254 | 10.420886 | 1.179924 | 2.6751% |
| Texture | 10.158420 | 8.586000 | 10.383023 | 1.196217 | 2.1868% |

The ratio uses the mean of the two baseline medians divided by the candidate
median. This is a diagnostic result from one poster latent, not a broad promotion
or SOTA result. Timing includes boundary transfers, topology construction, all
model work and final synchronization. Static initial preparation, CPU quality
comparisons and artifact I/O are excluded. First-use timings are retained
separately. No external CPU-heavy analysis or competing hardware job was launched
during the bracket; tensor comparisons run afterward.

Each warm candidate forward builds two large maps and reuses the 389,307-row map
five times. It releases that map before C2S. The 1,724,221-row map requires
400,019,272 bytes of host tensor storage and is rejected for retention. Maximum
retained storage across all ten candidate forwards is 90,319,224 host bytes and
99,440,640 aligned device bytes. No large entry remains after a call. Warm topology
time falls from 2.96–3.59 seconds in the baseline to 1.70–1.87 seconds in the candidate.
Small persistent maps retain the original policy; large maps are rebuilt on the
next forward, so warm timing is not a fully cached-geometry measurement.

All 36 matched saved output/guide records across B/A, B/C and C/A comparisons are
bitwise equal with exact coordinates and signs. All additional warm outputs and
guides repeat exactly. The original saved CPU parity gates still fail in every
run, as in iteration019: the same BF16 decoder has numerical and predicted-geometry
differences from FP32 CPU. The matched comparison does not replace those gates.

Three Watcher tests pass, covering actual allocator accounting, fresh features,
decoder equivalence and release before C2S. Two CPU tests cover coordinate/stage/
device invalidation, budget rejection and an injected error after retention.
All four device workers exit normally: one passing Watcher process and three
normal numerical-failure application exits, with no timeout or reset.

`archive.json` hashes the original reports, logs, tests and harness. `verify.py`
recomputes retained metadata bounds, checks all ten lifetimes, preserved gate
failures, exact matched comparisons, tests and diagnostic timing ratios. Tensor
artifacts remain at their scratch paths with hashes in `comparison.json`.

Decision: retain the isolated cache prototype for broader application testing.
Next kernel work should remove the four unused native BF16 product tables and
their conversion/write traffic, then investigate shared gathering across the
four products. Independent scenes, stronger comparisons and CPU decoder parity
remain outstanding; this experiment does not change the library's defaults.
