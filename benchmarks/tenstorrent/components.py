"""Serialized operation diagnostics, not a performance promotion score.

Wrap the actual candidate's TTNN calls with device synchronization. This changes
overlap and adds synchronization overhead. Stage timings overlap operation
timings and must not be added to them. Compare with unwrapped full forwards.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from .common import correctness, sha256, summarize, write_json
from .evaluate import source_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--mode", choices=("timing",), default="timing")
    args = p.parse_args()
    if not os.getenv("FLEX_GEMM_EVALUATOR_LOCKED") or args.output.exists() or args.repeat < 2:
        p.error("use the locked runner, a fresh output path, and at least two repeats")
    if any(os.getenv(k, "0") not in ("", "0") for k in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER")):
        p.error("component diagnostics do not combine with other instrumentation")
    torch.set_num_threads(8)
    manifest_path, contract_path = args.corpus / "manifest.json", args.corpus / "contract.json"
    manifest = json.loads(manifest_path.read_text())
    contract = json.loads(contract_path.read_text())
    if sha256(contract_path) != manifest["contract_sha256"]:
        raise ValueError("contract digest mismatch")
    entry = next(c for c in manifest["cases"] if c["name"] == args.case)
    fixture_path = args.corpus / entry["file"]
    if sha256(fixture_path) != entry["sha256"]:
        raise ValueError("fixture digest mismatch")
    fixture = torch.load(fixture_path, weights_only=True, map_location="cpu")
    config = json.loads(args.config)
    if config["algorithm"] != "digit_explicit_gemm":
        p.error("this diagnostic currently targets the four-product candidate")
    root = args.candidate_root.resolve()
    protected = {str(p): sha256(p) for p in [manifest_path, contract_path,
                 *sorted(Path(__file__).parent.glob("*.py"))]}
    report = {"status": "running", "kind": "serialized_component_diagnostic_not_score",
        "case": entry, "config": config, "candidate": source_state(root), "torch": torch.__version__,
        "evaluator_sha256": protected, "contract": contract, "checks": [],
        "unwrapped_ms": [], "instrumented_ms": [], "operation_calls": [], "stage_calls": [],
        "scope": "host wall time with synchronization around each top-level TTNN call; overlap is changed; stage times overlap operation times; not device-only timing or a promotion score"}
    sys.path.insert(0, str(root))
    import flex_gemm.tenstorrent as candidate
    import flex_gemm.tenstorrent.digits as digits
    import ttnn
    if not Path(candidate.__file__).resolve().is_relative_to(root):
        raise ValueError("wrong candidate import")

    def phase(name):
        report["phase"] = name
        write_json(args.output, report)
        print(name, flush=True)

    device = None
    originals = {}
    feature_original = digits._feature_digits
    depth = 0
    variant = 0

    def wrapped(name, function, destination, stage=False):
        def call(*a, **kw):
            nonlocal depth
            if depth and not stage:
                return function(*a, **kw)
            ttnn.synchronize_device(device)
            start = time.perf_counter()
            if not stage:
                depth += 1
            try:
                result = function(*a, **kw)
                ttnn.synchronize_device(device)
            finally:
                if not stage:
                    depth -= 1
            destination.append({"name": name, "variant": variant,
                "ms": (time.perf_counter() - start) * 1000,
                "input_shape": list(a[0].shape) if a and hasattr(a[0], "shape") else None})
            return result
        return call

    try:
        phase("prepare")
        device = ttnn.open_device(device_id=args.device)
        grid = device.compute_with_storage_grid_size()
        report.update(arch=str(device.arch()), available_grid=[grid.x, grid.y])
        if os.getenv("TT_METAL_HOME"):
            report["tt_metal_commit"] = subprocess.check_output(
                ["git", "-C", os.environ["TT_METAL_HOME"], "rev-parse", "HEAD"], text=True).strip()
        weights = candidate.prepare_sparse_digit_weights(fixture["weight"], bias=fixture["bias"], device=device)
        topology = candidate.prepare_sparse_topology(fixture["mapping"], len(fixture["features"][0]), device=device)
        features = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                    for x in fixture["features"]]

        def forward(index, samples=None, check=False):
            ttnn.synchronize_device(device)
            start = time.perf_counter()
            result = candidate.sparse_conv_device(features[index], weights, topology, **config)
            ttnn.synchronize_device(device)
            elapsed = (time.perf_counter() - start) * 1000
            if samples is not None:
                samples.append(elapsed)
            if check:
                metric = correctness(ttnn.to_torch(result), fixture["expected"][index], contract)
                metric.update(variant=index, dtype_valid=result.dtype == ttnn.bfloat16,
                              layout_valid=result.layout == ttnn.TILE_LAYOUT)
                metric["passed"] &= metric["dtype_valid"] and metric["layout_valid"]
                report["checks"].append(metric)

        for _ in range(2):
            for variant in range(len(features)):
                forward(variant)
        phase("measure")
        for variant in range(len(features)):
            forward(variant, report["unwrapped_ms"], check=True)
        names = ("abs", "max", "typecast", "subtract", "ceil", "log2", "maximum", "exp2", "neg",
                 "multiply", "repeat", "slice", "round", "to_layout", "pad", "zeros", "concat",
                 "allocate_tensor_on_device", "embedding", "matmul", "add", "generic_op", "from_torch")
        for name in names:
            originals[name] = getattr(ttnn, name)
            setattr(ttnn, name, wrapped(name, originals[name], report["operation_calls"]))
        digits._feature_digits = wrapped("feature_digits", feature_original, report["stage_calls"], stage=True)
        for i in range(args.repeat):
            variant = i % len(features)
            forward(variant, report["instrumented_ms"], check=True)
        for name, function in originals.items():
            setattr(ttnn, name, function)
        digits._feature_digits = feature_original
        for variant in range(len(features)):
            forward(variant, report["unwrapped_ms"], check=True)
        groups = defaultdict(list)
        for event in report["operation_calls"]:
            groups[event["name"]].append(event["ms"])
        report["operations"] = {name: {"calls": len(values), "sum_ms_per_forward": sum(values) / args.repeat,
            "per_call": summarize(values)} for name, values in groups.items()}
        report["unwrapped"] = summarize(report["unwrapped_ms"])
        report["instrumented"] = summarize(report["instrumented_ms"])
        report["input_unchanged"] = all(torch.equal(ttnn.to_torch(d), h) for d, h in zip(features, fixture["features"]))
        report["fixture_unchanged"] = sha256(fixture_path) == entry["sha256"]
        report["source_unchanged"] = source_state(root)["sha256"] == report["candidate"]["sha256"]
        report["protected_unchanged"] = all(sha256(p) == h for p, h in protected.items())
        report["status"] = "passed" if all(c["passed"] for c in report["checks"]) and all(
            report[k] for k in ("input_unchanged", "fixture_unchanged", "source_unchanged", "protected_unchanged")) else "numerical_or_integrity_failure"
    except Exception as error:
        report.update(status="execution_failure", error=str(error), traceback=traceback.format_exc())
    finally:
        for name, function in originals.items():
            setattr(ttnn, name, function)
        digits._feature_digits = feature_original
        phase("close_device")
        if device is not None:
            ttnn.close_device(device)
        phase("complete")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
