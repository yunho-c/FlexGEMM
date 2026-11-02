# Iteration 012: cheaper memory capture and explicit residency gates

The evaluator disables unused Python argument/return recording immediately
after beginning TTNN graph capture. It still captures C++ allocator events.
The pinned SDK recommends this for memory-only capture in `ttnn/ttnn/graph.py`.
Candidate kernels, CPU goldens, fidelity, tolerances and workspace cap are
unchanged from iteration 011 (`b6fc69a`).

Five representative cases pass the expanded numerical/residency gates and
reproduce the previous DRAM allocation/deallocation lifetime and size sequences,
workspace bounds, and final-output exclusions exactly. The cases span tiny,
wide, narrow, independent LiDAR geometry, and the largest 2,389,000-row layer.
Addresses and device IDs are normalized; allocation identities, reused-address
lifetimes, ordering and all other allocation parameters must match.

The largest complete worker took 57.982 seconds versus 434.998 seconds in the
archived iteration011 run. Its new capture took 2.982 seconds. These are
observed evaluator overhead measurements across separate runs, not a paired
kernel or application speedup. This iteration does not rerun all 36 cases;
the kernel sources remain identical to the iteration011 validation.

Both main and component evaluators now check allocated device storage on the
expected device, DRAM residency, BF16 dtype and tiled layout for inputs/outputs.
Two deliberately invalid candidates return numerically correct L1/host tensors.
Both are rejected by the full evaluator, with normal process exit code 1 and
no device recovery. Their patches and failed reports are retained. One Watcher
test additionally covers wrong dtype/layout; 12 CPU evaluator tests pass. A
component diagnostic exercises the shared gate successfully.

`archive.json` hashes the unchanged experiment reports and compact DRAM event
bundle. Full traces remain at the paths and hashes in each workspace report.
`experiment-driver.py` records the exact experiment with its original local
paths. To verify the archived comparisons and bounds without a device, run:

```bash
python docs/results/autoresearch/iteration-012/verify.py
```

Measurements precede this commit; reports identify base commit `b6fc69a`, the
actual evaluator hashes, and unchanged candidate source hashes. This is an
evaluator improvement, with no new operator or SOTA performance claim.
