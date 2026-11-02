"""Serialized configuration screening and bracketed A/B/A confirmation.

No result from this tool alone promotes a kernel: full coverage, workspace
verification and application validation remain separate requirements.
"""

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys

from .common import sha256, write_json


def comparison(before, candidate, after):
    """Compare complete operator medians only when all three trials passed."""
    records = (before, candidate, after)
    if any(r.get("status") != "passed" for r in records):
        return {"eligible": False, "statuses": [r.get("status") for r in records]}
    a, b, c = [r["timing"]["median_ms"] for r in records]
    if not all(math.isfinite(x) and x > 0 for x in (a, b, c)):
        return {"eligible": False, "reason": "invalid timing"}
    noise = max(abs(c / a - 1), abs(a / c - 1),
        *(3 * r["timing"]["mad_ms"] / r["timing"]["median_ms"] for r in records))
    return {"eligible": True, "speedup": math.sqrt(a * c) / b,
            "baseline_drift_fraction": max(abs(c / a - 1), abs(a / c - 1)),
            "noise_fraction": noise, "before_ms": a, "candidate_ms": b, "after_ms": c}


def summarize_comparisons(rows, required_cases, required_rounds):
    """Retain failed/missing cases; never improve a score by dropping them."""
    result = {}
    for name in sorted({r["candidate"] for r in rows}):
        cases = {}
        for case in required_cases:
            records = [r["comparison"] for r in rows if r["candidate"] == name and r["case"] == case]
            complete = len(records) == required_rounds and all(r["eligible"] for r in records)
            cases[case] = {"complete_and_correct": complete, "rounds": len(records)}
            if complete:
                ratios = [r["speedup"] for r in records]
                noise = max(r["noise_fraction"] for r in records)
                cases[case].update(median_speedup=statistics.median(ratios),
                    minimum_speedup=min(ratios), maximum_noise_fraction=noise,
                    consistent_screening_win=required_rounds >= 3 and min(ratios) > 1 + max(.02, noise))
        complete = all(c["complete_and_correct"] for c in cases.values())
        result[name] = {"cases": cases, "all_requested_cases_pass": complete,
            "geomean_speedup": math.exp(statistics.mean(math.log(c["median_speedup"])
                for c in cases.values())) if complete else None,
            "promotion": "not_evaluated: requires full contract, workspace and application checks"}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--cases", nargs="+", required=True, help="Explicit case names; use ALL for whole corpus")
    p.add_argument("--candidate-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--baseline-root", type=Path)
    p.add_argument("--baseline", default='{"algorithm":"explicit_gemm","row_chunk":256}')
    p.add_argument("--candidates", type=Path, required=True, help="JSON object mapping names to configurations")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--repeat", type=int, default=11)
    p.add_argument("--screen", action="store_true", help="One baseline + candidates per case, no paired conclusions")
    p.add_argument("--lock-timeout", type=float, default=1)
    args = p.parse_args()
    if args.rounds < 1 or args.repeat < 2:
        p.error("rounds must be positive and repeat at least two")
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    available = {c["name"] for c in manifest["cases"]}
    cases = [c["name"] for c in manifest["cases"]] if args.cases == ["ALL"] else args.cases
    if not cases or len(set(cases)) != len(cases) or set(cases) - available:
        p.error("cases must be unique names in the corpus")
    configs = json.loads(args.candidates.read_text())
    if not configs or "baseline" in configs:
        p.error("provide at least one candidate; 'baseline' is reserved")
    baseline = json.loads(args.baseline)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "screen_only": args.screen, "cases": cases,
        "baseline": baseline, "candidates": configs, "rounds": args.rounds,
        "repeat": args.repeat, "corpus_manifest_sha256": sha256(args.corpus / "manifest.json"),
        "ordering_seed": 20261102, "trials": [], "comparisons": []}
    path = args.output / "summary.json"
    write_json(path, report)

    def trial(case, name, config, root, round_index):
        output = args.output / f"trial-{len(report['trials']):04d}.json"
        command = [sys.executable, "-m", "benchmarks.tenstorrent.run", "--corpus", str(args.corpus),
            "--case", case, "--candidate-root", str(root), "--config", json.dumps(config),
            "--output", str(output), "--repeat", str(args.repeat), "--lock-timeout", str(args.lock_timeout)]
        completed = subprocess.run(command)
        runner = json.loads(output.with_suffix(".runner.json").read_text())
        record = json.loads(output.read_text()) if output.exists() else {"status": "missing"}
        report["trials"].append({"case": case, "candidate": name, "round": round_index,
            "path": output.name, "runner_status": runner["status"], "status": record["status"],
            "timing": record.get("timing")})
        write_json(path, report)
        # Numerical rejection is safe to record and continue. Execution failure,
        # a busy board, or incomplete teardown must stop the whole campaign.
        if runner.get("device_recovery_required") or runner["status"] not in ("passed", "failed") or (
                completed.returncode and record["status"] != "numerical_or_integrity_failure"):
            raise RuntimeError(f"campaign stopped after {output.name}: {runner['status']}/{record['status']}")
        return record

    rng = random.Random(report["ordering_seed"])
    try:
        for case in cases:
            for round_index in range(1 if args.screen else args.rounds):
                order = list(configs)
                rng.shuffle(order)
                before = trial(case, "baseline", baseline, args.baseline_root or args.candidate_root, round_index)
                for name in order:
                    candidate = trial(case, name, configs[name], args.candidate_root, round_index)
                    if not args.screen:
                        after = trial(case, "baseline", baseline, args.baseline_root or args.candidate_root, round_index)
                        report["comparisons"].append({"case": case, "candidate": name, "round": round_index,
                            "comparison": comparison(before, candidate, after)})
                        before = after
                        report["results"] = summarize_comparisons(report["comparisons"], cases, args.rounds)
                    write_json(path, report)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="stopped", error=str(error))
    write_json(path, report)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
