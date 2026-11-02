"""Verify iteration 016 evidence and allocator bounds without opening a device."""

import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO))
from benchmarks.tenstorrent.memory import workspace_bound


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


archive = read(ROOT / "archive.json")
assert digest(REPO / "benchmarks/tenstorrent/memory.py") == archive["memory_source_sha256"]
for entry in archive["files"] + [archive["allocation_event_bundle"]]:
    assert digest(ROOT / entry["path"]) == entry["sha256"], entry["path"]

summary = read(ROOT / "campaign-summary.json")
assert summary["status"] == "complete"
assert len(summary["trials"]) == 24 and len(summary["unsupported"]) == 12
assert sum(t["status"] == "passed_probe_gates" for t in summary["trials"]) == 17
assert sum(t["status"] == "numerical_or_integrity_failure" for t in summary["trials"]) == 7
for trial in summary["trials"]:
    report = read(ROOT / "hifi3-cases" / (trial["case"] + ".json"))
    assert report["status"] == trial["status"]
    assert report["fidelity"] == "HiFi3"
    for key in ("input_unchanged", "source_unchanged", "fixture_unchanged",
                "contract_unchanged", "probe_unchanged", "comparator_unchanged"):
        assert report[key], (trial["case"], key)
    assert all(c["tensor_contract_valid"] for c in report["checks"])
    assert all(c.get("repeat_bitwise_equal", True) for c in report["checks"])
    passed = all(c["passed"] for c in report["checks"])
    assert passed == (trial["status"] == "passed_probe_gates")
    if not passed:
        assert any(c["elementwise_failures"] > 0 for c in report["checks"])

bundle = json.loads(gzip.decompress((ROOT / "allocation-events.json.gz").read_bytes()))
assert len(bundle) == 25
maximum = 0
for name, events in bundle.items():
    report = read(ROOT / name)
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
assert len(runners) == 26
codes = []
for path in runners:
    runner = read(path)
    report = read(path.with_name(path.name.replace(".runner.json", ".json")))
    expected_code = 0 if report["status"] == "passed_probe_gates" else 1
    assert runner["returncode"] == expected_code
    assert runner["timeout"] is None and not runner["device_recovery_required"]
    codes.append(expected_code)
    initial = path.name == "watcher-small-hifi3.runner.json"
    assert digest(ROOT / ("probe-v0.py" if initial else "k6_probe.py")) == report["probe_sha256"]
    for name, expected in report["comparator_sha256"].items():
        source = ROOT / "comparator" / name
        if initial and name == "adapter.py":
            source = ROOT / "adapter-v0.py"
        elif initial and name == "kernels/k6_compute.cpp":
            source = ROOT / "upstream/k6_compute.cpp"
        assert digest(source) == expected, (path.name, name)
assert codes.count(0) == 18 and codes.count(1) == 8
initial = read(ROOT / "watcher-small-hifi3.json")
assert initial["status"] == "execution_failure" and "copy_init" in initial["error"]
assert read(ROOT / "watcher-small-port-hifi3.json")["status"] == "passed_probe_gates"

for row in read(ROOT / "unpaired-comparison.json")["cases"]:
    baseline_path = ROOT.parent / row["baseline_report"]
    assert digest(baseline_path) == row["baseline_report_sha256"]
    baseline = read(baseline_path)
    report = read(ROOT / "hifi3-cases" / (row["case"] + ".json"))
    assert baseline["case"]["sha256"] == report["fixture_sha256"]
    assert baseline["timing"]["median_ms"] == row["baseline_ms"]
    assert report["timing"]["median_ms"] == row["comparator_ms"]
    expected = row["baseline_ms"] / row["comparator_ms"] if report["status"] == "passed_probe_gates" else None
    assert row["ratio_for_passed_case_only"] == expected
print(f"Verified {len(archive['files'])} archived files, 25 memory replays, "
      f"26 normal exits (18 zero, 8 one); maximum workspace {maximum} bytes")
