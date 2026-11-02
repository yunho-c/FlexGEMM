"""Verify archived iteration013 evidence without opening a device."""

import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
from benchmarks.tenstorrent.memory import workspace_bound

archive = json.loads((ROOT / "archive.json").read_text())
for entry in archive["files"] + [archive["allocation_event_bundle"]]:
    assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
bundle = json.loads(gzip.decompress((ROOT / "allocation-events.json.gz").read_bytes()))
assert len(bundle) == 36
maximum = 0
for name, events in bundle.items():
    report = json.loads((ROOT / name).read_text())
    assert report["status"] == "passed"
    w = report["workspace"]
    o = w["excluded_final_output"]
    replay = workspace_bound(events, dram_banks=w["dram_banks"],
        l1_capacity_bytes=w["all_l1_capacity_bytes"], limit_bytes=w["workspace_limit_bytes"],
        output_address=o["address"], expected_output_bytes=o["expected_padded_bytes"])
    assert replay["trace_valid"] and replay["within_limit"]
    for key in ("new_dram_peak_bytes", "dram_allocations", "inclusive_upper_bound_bytes",
                "temporary_dram_peak_bytes", "workspace_upper_bound_bytes"):
        assert replay[key] == w[key], (name, key)
    assert events[replay["excluded_final_output"]["event_index"]]["counter"] == o["event_index"]
    maximum = max(maximum, replay["workspace_upper_bound_bytes"])
runners = list(ROOT.rglob("*.runner.json"))
assert len(runners) == 65
for path in runners:
    report = json.loads(path.read_text())
    assert report["returncode"] == 0 and not report["device_recovery_required"]
    assert report["timeout"] is None
print(f"Verified {len(archive['files'])} reports/patches, 36 memory replays, "
      f"{len(runners)} normal device exits; maximum workspace {maximum} bytes")
