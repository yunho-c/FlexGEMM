"""Replay iteration012 bounds and compare allocation lifetimes with iteration011.

Run from the repository root with Python. No device is opened.
"""

import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
from benchmarks.tenstorrent.memory import workspace_bound


def normalized(nodes):
    active, resident, rows = {}, {}, []
    next_id = 0
    for node in nodes:
        p, kind = node["params"], node["node_type"]
        key = (p["device_id"], p["address"])
        if kind == "buffer_allocate":
            assert key not in active
            active[key] = next_id
            next_id += 1
            ordinal = active[key]
        else:
            ordinal = active.pop(key, None)
            if ordinal is None:
                if key not in resident:
                    resident[key] = -len(resident) - 1
                ordinal = resident[key]
        rows.append([kind, ordinal, {k: v for k, v in p.items() if k not in ("device_id", "address")}])
    return rows


archive = json.loads((ROOT / "archive.json").read_text())
for entry in archive["files"] + [archive["allocation_event_bundle"]]:
    assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
bundle = json.loads(gzip.decompress((ROOT / "allocation-events.json.gz").read_bytes()))
previous = ROOT.parent / "iteration-011"
old_bundle = json.loads(gzip.decompress((previous / "allocation-events.json.gz").read_bytes()))
comparisons = json.loads((ROOT / "memory-comparison.json").read_text())["comparisons"]
for row in comparisons:
    name = row["report"]
    old_name = "/".join(Path(row["previous_report"]).parts[-2:])
    events, old_events = bundle[name], old_bundle[old_name]
    after, before = normalized(events), normalized(old_events)
    assert after == before
    assert hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest() == row["normalized_event_sha256"]
    w = json.loads((ROOT / name).read_text())["workspace"]
    old_w = json.loads((previous / old_name).read_text())["workspace"]
    o = w["excluded_final_output"]
    replay = workspace_bound(events, dram_banks=w["dram_banks"],
        l1_capacity_bytes=w["all_l1_capacity_bytes"], limit_bytes=w["workspace_limit_bytes"],
        output_address=o["address"], expected_output_bytes=o["expected_padded_bytes"])
    assert replay["trace_valid"] and replay["within_limit"]
    for k in ("new_dram_peak_bytes", "dram_allocations", "temporary_dram_peak_bytes", "workspace_upper_bound_bytes"):
        assert replay[k] == w[k] == old_w[k]
    output_index = replay["excluded_final_output"]["event_index"]
    old_output_index = next(i for i, e in enumerate(old_events)
                            if e["counter"] == old_w["excluded_final_output"]["event_index"])
    assert events[output_index]["counter"] == o["event_index"]
    assert output_index == old_output_index
    print(row["case"], len(events), "identical DRAM lifetime events; workspace", replay["workspace_upper_bound_bytes"])
