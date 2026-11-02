# Iteration 023: compact BF16 weight transfers

K6 now reads and multicasts the existing BF16 integer digit weights directly,
using 2048-byte weight tiles instead of an FP32 encoding with 4096-byte tiles.
The builder retains FP32 support for direct controls. FP32 accumulation, HiFi3,
four-product arithmetic, reconstruction, scale/bias and output assembly are fixed.
The wrapper shares already-prepared weight tensors instead of creating FP32 copies.

Tap planning and eligibility deliberately retain the original conservative bounds.
All 36 fixtures preserve their original tap grouping and 24/12 implicit/fallback
dispatch. Reported actual CB bytes account for compact weight storage. The new
work-model variant changes only its name and halves requested weight bytes; all
other fields match iteration021. These payload counts are not DRAM/NoC bus counters.

Three A/B/A rounds, eleven timed repetitions per trial, compare with frozen
commit `d88bbf2` (iteration022, production kernels from iteration021):

| Group | Candidate medians, ms | Round speedups | Maximum noise | Decision |
| --- | --- | --- | ---: | --- |
| narrow | 149.242 / 149.271 / 148.939 | 1.121281 / 1.120447 / 1.123374 | 1.1527% | confirmed gain |
| largest | 456.906 / 445.845 / 452.530 | 1.002577 / 1.010059 / 0.998664 | 6.2870% | neutral within noise |
| wide-feasible | 463.086 / 463.362 / 463.004 | 1.367173 / 1.365991 / 1.367340 | 0.2432% | confirmed gain |
| fallback | 129.563 / 129.977 / 129.384 | 0.993476 / 0.998062 / 0.999186 | 1.8862% | control: neutral |

Narrow is 388,659 rows at 128→128; wide-feasible is the same row count at 128→512;
largest is 2,389,000 at 64→64; fallback is 4,490 at 1024→4096. The narrow and wide
cases exceed the fixed noise-aware rule in all three rounds. The largest case
shows no confirmed gain; retain all its noisy/neutral rounds. Fallback remains a
control. No external CPU-heavy analysis or competing hardware worker was launched
during this paired campaign.

All 36 fixtures pass the original numerical, residency, layout, dtype, input
immutability, repeatability, source/fixture integrity and workspace checks.
Thirty-six allocator replays preserve the maximum 726,282,240-byte workspace
bound. Twenty-two Watcher tests pass, including 40 bitwise BF16-versus-FP32-weight
product comparisons and the prior output-mode controls. Fifteen CPU tests pass,
including preservation of old accounting and profiler accumulator ownership.

The unchanged stage-cache decoder adapter produces bitwise-identical saved
shape/texture outputs and subdivision guides: all 12 records match iteration021.
Its original strict FP32 CPU gates still fail. The additional application run
is unpaired and carries no new decoder-speedup claim.

Five post-confirmation diagnostic workers add 74 CSV segments. Compared with
iteration022's unpaired profiles, median per-core weight waits decrease from
317.8 to 168.5 μs (narrow), 1199.2 to 590.8 μs (wide), and 81.6 to 53.5 μs (largest).
The largest selected gather wait increases from 93.7 to 112.1 μs. These scopes
are overlapping RISC observations, not additive bottleneck percentages or a
prediction of complete-operator speedup.

Two complete largest-operator captures contain all 1924 programs, including 584
K6 products. Device kernel-span unions are about 371 ms: about 63 ms precedes the
first K6 call (feature preparation and scale), 199 ms belongs to K6, and 109 ms
to loop reconstruction/epilogue/assembly. The three category unions do not overlap.
Host calls measure 446 and 908 ms; the supervisor repeatedly parses a 238 MB
report after the first sample and can perturb scheduling. Consequently these host
times and inter-program gaps are diagnostic only. They cannot support a lean
host-overhead or speedup claim. The profiler's program capacity is increased only
for this diagnostic worker; promotion runs use the unchanged lean environment.

All 79 hardware workers terminate without timeout or reset: 78 return success,
and the decoder preserves its numerical-failure status. `archive.json` hashes
unchanged reports and exact source snapshots. The verifier replays all allocator
events, bracket statistics and diagnostic CSVs, including phase interval unions.
Large tensor artifacts and full warmup traces remain at their recorded scratch
paths and hashes.

Decision: retain compact weights. Next test caching immutable per-device core
ranges/runtime argument maps, which the current Python builder reconstructs for
every product call. This is a hypothesis to test with lean complete-operator
A/B/A, not an established explanation of the largest result. Fused reconstruction,
compact-weight-aware tap grouping and shared gathers remain further candidates.
Independent-scene coverage, stronger matched external comparisons and SOTA remain
pending.
