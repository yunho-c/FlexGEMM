# Iteration 011: direct tiled gather for both feature digits

Retain the change for the confirmed 1.249× narrow-layer gain and 36/36 passing
BF16 operator cases. The wide-layer result remains within measurement noise.
See `decision.json` and the iteration ledger for measurements and scope.

The six new cases use independent real LiDAR geometry with synthetic BF16
operands. They do not measure BEVFusion inference. The unchanged corpus
manifest is `../autoware-lidar-corpus-manifest.json`; the tested preparer is
`benchmarks/tenstorrent/prepare_lidar.py`. Its source hash matches the manifest.
The preparer was copied from scratch after all device campaigns completed;
no evaluator or candidate source was changed while measurements ran.

`archive.json` records hashes of the unchanged reports copied from scratch.
Measurements precede this commit: their candidate metadata identifies base
commit `34a92cb` with a dirty tree and records the actual candidate source hash.
The frozen incumbent uses that commit without these changes. The contract and
CPU goldens are unchanged.

`allocation-events.json.gz` is a JSON object keyed by report path, containing
the original DRAM allocation/deallocation nodes in order. Their `counter`
fields retain their positions in the full traces. The archive records each
original compressed trace's hash and the replayed workspace bound. Full traces
remain at the scratch paths recorded in the reports. All 36 compact replays
match the hardware report's peak, allocation count and output exclusion.

For example, from the repository root:

```python
import gzip, json
from pathlib import Path
from benchmarks.tenstorrent.memory import workspace_bound

root = Path("docs/results/autoresearch/iteration-011")
name = "full-validation/n2389000-64x64-k3x3x3.json"
events = json.loads(gzip.decompress((root / "allocation-events.json.gz").read_bytes()))[name]
w = json.loads((root / name).read_text())["workspace"]
out = w["excluded_final_output"]
replay = workspace_bound(events, dram_banks=w["dram_banks"],
    l1_capacity_bytes=w["all_l1_capacity_bytes"], limit_bytes=w["workspace_limit_bytes"],
    output_address=out["address"], expected_output_bytes=out["expected_padded_bytes"])
assert replay["workspace_upper_bound_bytes"] == w["workspace_upper_bound_bytes"]
assert events[replay["excluded_final_output"]["event_index"]]["counter"] == out["event_index"]
```

The `narrow-components` report is a serialized diagnostic, separate from the
lean paired campaigns. Its synchronized operation times change overlap and
must not be treated as additive savings in full-operator latency.
