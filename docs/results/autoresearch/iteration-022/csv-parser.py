"""Parse pinned TT-Metal device CSVs without summing overlapping RISC spans."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re
import statistics

from .common import sha256, write_json


def parse_device_csv(path):
    path = Path(path)
    with path.open() as stream:
        header = stream.readline().strip()
        frequency = re.search(r"CHIP_FREQ\[MHz\]:\s*([0-9.]+)", header)
        if not frequency:
            raise ValueError("device CSV lacks clock frequency")
        mhz = float(frequency.group(1))
        rows = [{k.strip(): (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(stream)]
    starts = defaultdict(list)
    runs = defaultdict(list)
    totals = defaultdict(list)
    unmatched_ends = 0
    for row in rows:
        key = tuple(row[k] for k in ("PCIe slot", "run host ID", "core_x", "core_y",
                                    "RISC processor type", "timer_id"))
        if row["type"] == "ZONE_START":
            starts[key].append(int(row["time[cycles since reset]"]))
        elif row["type"] == "ZONE_END":
            if not starts[key]:
                unmatched_ends += 1
                continue
            start = starts[key].pop()
            end = int(row["time[cycles since reset]"])
            if end < start:
                raise ValueError("negative device zone duration")
            runs[key[:2]].append({"core": list(key[2:4]), "risc": key[4], "zone": row["zone name"],
                                  "start": start, "end": end, "duration_us": (end - start) / mhz})
        elif row["type"] == "ZONE_TOTAL":
            # The pinned SDK stores accumulated cycles in data; the timestamp
            # is the enclosing operation's timestamp, not a duration.
            totals[key[:2]].append({"core": list(key[2:4]), "risc": key[4], "zone": row["zone name"],
                                    "total_us": int(row["data"]) / mhz})
    output = []
    for key, spans in sorted(runs.items(), key=lambda item: tuple(map(int, item[0]))):
        kernels = [s for s in spans if s["zone"].endswith("-KERNEL")]
        if not kernels:
            continue
        core_spans = defaultdict(list)
        for span in kernels:
            core_spans[tuple(span["core"])].append(span)
        durations = [(max(s["end"] for s in items) - min(s["start"] for s in items)) / mhz
                     for items in core_spans.values()]
        by_risc = {}
        for risc in sorted({s["risc"] for s in kernels}):
            values = [s["duration_us"] for s in kernels if s["risc"] == risc]
            by_risc[risc] = {"min_us": min(values), "median_us": statistics.median(values),
                             "max_us": max(values)}
        output.append({"device": key[0], "run_host_id": key[1], "cores": len(core_spans),
            "kernel_span_us": (max(s["end"] for s in kernels) - min(s["start"] for s in kernels)) / mhz,
            "core_span_median_us": statistics.median(durations), "core_span_max_us": max(durations),
            "core_max_over_median": max(durations) / statistics.median(durations),
            "by_risc": by_risc, "accumulated_zones": totals[key],
            "custom_zones": [s for s in spans if s["zone"].startswith("FG_")],
            "operator_identity": ("flexgemm_implicit" if any(
                s["zone"] == "FG_IMPLICIT_COMPUTE" for s in spans) else
                "flexgemm_tiled_gather" if any(s["zone"] == "FG_EXPLICIT_GATHER" for s in spans) else None)})
    return {"kind": "measured", "source": str(path), "source_sha256": sha256(path),
            "header": header, "clock_mhz": mhz, "rows": len(rows),
            "unmatched_starts": sum(map(len, starts.values())), "unmatched_ends": unmatched_ends,
            "scope": "per-program device kernel spans; no summation of overlapping RISC/core durations; operator identity requires host correlation or custom zones",
            "runs": output, "hardware_counters": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, parse_device_csv(args.csv))


if __name__ == "__main__":
    main()
