# Iteration 016: external K6 portability and accuracy study

Retain iteration 015. The adapted K6 kernel passes 17 of 24 feasible fixtures
but fails the unchanged numerical gate on seven decoder-derived fixtures.
Twelve wider fixtures exceed its per-core circular-buffer budget. These
results reject the ordinary K6 implementation as a broad replacement and
motivate testing its fused data movement with four-product arithmetic.

The comparator is the Apache-2.0 K6 source from
[changh95/bevfusion-p150 at revision 300095a](https://huggingface.co/changh95/bevfusion-p150/tree/300095a42e61075a3d5c9916620344c1067200b8).
Its native target is Blackhole p150 with HiFi4. This study explicitly selects
HiFi3 on Wormhole and uses exactly the immutable fixtures' BF16-rounded
feature, weight and bias values. The weights and bias are encoded in FP32
as required by K6. Residual and ReLU are disabled. Inputs and outputs obey
the tiled BF16 DRAM contract. All fresh input conversion, sentinel padding,
gather, arithmetic, output conversion and chunk assembly are timed; initial
upload, static topology/weight preparation, compilation and downloads are not.
The native model's published latency is not a matched comparison here.

The initial Watcher probe failed to compile because the pinned SDK has
`copy_tile_init`, not `copy_init`. The explicit API port then passed a new
Watcher probe. A physical-coordinate guard permits native row multicast only
for contiguous compute columns. On this device it remained enabled, as did
split gather. No arithmetic change or numerical relaxation was made.

| Corpus | Passed | Numerical failures | Unsupported |
| --- | ---: | ---: | ---: |
| Synthetic and regression corpus | 10 | 0 | 2 |
| Independent LiDAR geometry | 6 | 0 | 0 |
| First poster decoder topology | 0 | 4 | 5 |
| Full-sampler poster topology | 1 | 3 | 5 |
| Total | 17 | 7 | 12 |

The LiDAR fixtures use real geometry with synthetic feature/weight values;
they are not a BEVFusion inference result. The two decoder topologies come
from the same poster and are not independent scenes. The largest case,
2,389,000 rows at 64→64, passes at a three-sample median of 135.252 ms.
The 388,659-row 128→128 case takes 45.550 ms but fails accuracy and has no
qualifying speedup. Several failures contain rare errors near powers of two,
consistent with the need to retain the numerical checks despite small global
relative errors. Their low-level cause is not established by this experiment.

These are unpaired diagnostic timings. `unpaired-comparison.json` links the
earlier iteration-015 measurements and suppresses ratios for failed cases.
It does not establish a controlled speedup, full-decoder benefit or SOTA.
All 25 memory captures, including failed-numerics cases, meet the 1 GiB
workspace bound; the maximum is 700,768,256 bytes. Residency, repeatability,
source/fixture integrity and input-immutability checks pass. All 26 workers
exit normally: 18 successful exits, seven numerical-failure exits and the
initial compilation-failure exit. There are no timeouts or device resets.

`archive.json` hashes unchanged reports, port sources, original source excerpts,
license notices, drivers and the compact allocator-event bundle. The embedded
provenance status describes preparation time; final outcomes are recorded in
`campaign-summary.json` and `decision.json`. `verify.py` validates hashes,
source versions, exit classifications, comparison operands and all 25 workspace
replays without a device. Full traces remain at the paths and hashes in reports.
The archived scripts retain their original local paths for provenance.

Next, test K6's FP32 result path with the existing four-product integer digits
and reconstruction. Wide layers require channel partitioning or the incumbent
fallback. Production kernels and default dispatch are unchanged in this iteration.
