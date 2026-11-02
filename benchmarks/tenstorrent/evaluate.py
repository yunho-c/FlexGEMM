"""Single-process device worker. Invoke through run.py for locking/timeouts."""

import argparse
import gc
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from .common import correctness, sha256, summarize, work_accounting, write_json


def source_state(root):
    paths = sorted(p for p in (root / "flex_gemm" / "tenstorrent").rglob("*")
                   if p.suffix in (".py", ".cpp", ".hpp", ".h"))
    return {"root": str(root), "commit": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)),
        "sha256": {str(p.relative_to(root)): sha256(p) for p in paths}}


def memory(device):
    import ttnn
    result = {}
    for name, kind in (("dram", ttnn.BufferType.DRAM), ("l1", ttnn.BufferType.L1)):
        view = ttnn.get_memory_view(device, kind)
        result[name] = {"allocated_bytes": int(view.num_banks * view.total_bytes_allocated_per_bank),
                        "capacity_bytes": int(view.num_banks * view.total_bytes_per_bank)}
    return result


def tensor_contract(tensor, device):
    """Inspect residency and format without copying or changing the tensor."""
    import ttnn
    on_device = ttnn.is_tensor_storage_on_device(tensor) and tensor.is_allocated()
    actual_device = tensor.device() if on_device else None
    buffer_type = tensor.memory_config().buffer_type
    checks = {"dtype_valid": tensor.dtype == ttnn.bfloat16,
              "layout_valid": tensor.layout == ttnn.TILE_LAYOUT,
              "device_valid": actual_device is not None and actual_device.id() == device.id(),
              "dram_valid": on_device and buffer_type == ttnn.BufferType.DRAM}
    return {**checks, "tensor_contract_valid": all(checks.values()),
            "storage_type": str(tensor.storage_type()), "buffer_type": str(buffer_type),
            "device_id": actual_device.id() if actual_device is not None else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--config", required=True, help="JSON: algorithm and optional row_chunk/split_k")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=11)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--mode", choices=("timing", "correctness", "profile", "memory"), default="timing")
    args = p.parse_args()
    if not os.environ.get("FLEX_GEMM_EVALUATOR_LOCKED"):
        p.error("run through benchmarks.tenstorrent.run to serialize device access")
    if args.repeat < 2:
        p.error("repeat must be at least two to exercise both feature values")
    if args.output.exists():
        p.error("report path must be fresh")
    if args.mode == "timing" and any(os.getenv(k, "0") not in ("", "0") for k in (
            "TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_DPRINT_CORES")):
        p.error("lean timings require Watcher, profiler and debug printing to be disabled")
    torch.set_num_threads(args.threads)
    manifest_path = args.corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    contract_path = args.corpus / "contract.json"
    if sha256(contract_path) != manifest["contract_sha256"]:
        raise ValueError("contract integrity failure")
    contract = json.loads(contract_path.read_text())
    entry = next(c for c in manifest["cases"] if c["name"] == args.case)
    fixture_path = args.corpus / entry["file"]
    if sha256(fixture_path) != entry["sha256"]:
        raise ValueError("fixture integrity failure")
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    config = json.loads(args.config)
    protected = {str(path): sha256(path) for path in [manifest_path, contract_path,
        *sorted(Path(__file__).parent.glob("*.py"))]}
    candidate_root = args.candidate_root.resolve()
    report = {"status": "running", "phase": "import", "case": entry, "config": config,
              "contract": contract, "mode": args.mode, "torch": torch.__version__,
              "candidate": source_state(candidate_root), "evaluator_sha256": protected,
              "timing_scope": contract["timing"], "samples": [], "checks": [],
              "workspace_peak_measured": False, "profiler": None}

    def phase(name):
        report["phase"] = name
        write_json(args.output, report)
        print(name, flush=True)

    phase("import")
    sys.path.insert(0, str(candidate_root))
    import flex_gemm.tenstorrent as candidate
    if not Path(candidate.__file__).resolve().is_relative_to(candidate_root):
        raise ValueError("candidate imported from the wrong checkout")
    import ttnn
    metal_home = os.environ.get("TT_METAL_HOME")
    if metal_home:
        report["tt_metal_commit"] = subprocess.check_output(
            ["git", "-C", metal_home, "rev-parse", "HEAD"], text=True).strip()
    device = None
    try:
        phase("open_device")
        device = ttnn.open_device(device_id=args.device)
        grid = device.compute_with_storage_grid_size()
        report.update(arch=str(device.arch()), available_grid=[grid.x, grid.y])
        from flex_gemm.tenstorrent.ops import _matmul_fidelity
        report["math_fidelity"] = str(_matmul_fidelity(device))
        if device.arch() == ttnn.Arch.WORMHOLE_B0 and _matmul_fidelity(device) != ttnn.MathFidelity.HiFi3:
            raise ValueError("candidate changed the fixed Wormhole fidelity")
        def account(policy=None):
            return work_accounting(fixture["mapping"], entry["ci"], entry["co"],
                algorithm=config["algorithm"], split_k=config.get("split_k", 1),
                row_chunk=config.get("row_chunk", 256), available_cores=grid.x * grid.y,
                digit_output_assembly=getattr(sys.modules.get("flex_gemm.tenstorrent.digits"), "OUTPUT_ASSEMBLY", "concat"),
                digit_epilogue=getattr(sys.modules.get("flex_gemm.tenstorrent.digits"), "EPILOGUE", "expanded"),
                digit_gather=getattr(sys.modules.get("flex_gemm.tenstorrent.digits"), "GATHER", "embedding"),
                digit_implicit_policy=policy)
        report["work"] = None if config["algorithm"] == "digit_implicit_gemm" else account()
        digit_module = sys.modules.get("flex_gemm.tenstorrent.digits")
        if config["algorithm"] in ("digit_explicit_gemm", "digit_implicit_gemm"):
            report["feature_digit_preparation"] = getattr(digit_module, "FEATURE_PREPARATION", "unfused")
        if (config["algorithm"] == "digit_explicit_gemm" and hasattr(digit_module, "matmul_plan")
                and device.arch() == ttnn.Arch.WORMHOLE_B0):
            from collections import Counter
            rows = len(fixture["mapping"])
            chunk = config.get("row_chunk", 256)
            counts = Counter(min(chunk, rows - start) for start in range(0, rows, chunk))
            report["matmul_policy"] = {"kind": "bounded_1d_kblocking_v1",
                "maximum_k_block_tiles": digit_module.MATMUL_K_BLOCK,
                "cb_estimate_limit_bytes": digit_module.MATMUL_L1_ESTIMATE_LIMIT,
                "chunk_plans": [{"rows": count, "chunks": occurrences,
                    "plan": digit_module.matmul_plan(count, entry["volume"] * ((entry["ci"] + 31) // 32 * 32),
                                                    entry["co"], (grid.x, grid.y))}
                    for count, occurrences in counts.items()]}
        mandatory = (report["work"]["split_partial_bytes"] + report["work"]["explicit_expansion_bytes"]
                     if report["work"] is not None else 0)
        if mandatory > contract["workspace_limit_bytes"]:
            raise ValueError("mandatory intermediate storage exceeds workspace cap")
        phase("prepare")
        before = memory(device)
        start = time.perf_counter()
        prepare_weights = (candidate.prepare_sparse_digit_weights if config["algorithm"] in ("digit_explicit_gemm", "digit_implicit_gemm")
                           else candidate.prepare_sparse_weights)
        weights = prepare_weights(fixture["weight"], bias=fixture["bias"], device=device)
        topology = candidate.prepare_sparse_topology(fixture["mapping"], len(fixture["features"][0]),
            device=device, masked=config["algorithm"].startswith("masked"))
        if config["algorithm"] == "digit_implicit_gemm":
            plan = candidate.prepare_sparse_digit_implicit(weights, topology, row_chunk=config.get("row_chunk", 256))
            report["digit_implicit_policy"] = plan.info
        ttnn.synchronize_device(device)
        report["static_prepare_ms"] = (time.perf_counter() - start) * 1000
        if config["algorithm"] == "digit_implicit_gemm":
            report["work"] = account(report["digit_implicit_policy"])
            if report["work"]["explicit_expansion_bytes"] > contract["workspace_limit_bytes"]:
                raise ValueError("fallback expansion exceeds workspace cap")
        features = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=device) for x in fixture["features"]]
        ttnn.synchronize_device(device)
        report["input_contract"] = [tensor_contract(x, device) for x in features]
        if not all(c["tensor_contract_valid"] for c in report["input_contract"]):
            raise ValueError("input tensor residency or format violates the fixed contract")
        report["memory"] = {"before": before, "prepared": memory(device),
                            "scope": "live allocator snapshots, not transient peaks"}

        def execute(index):
            result = candidate.sparse_conv_device(features[index], weights, topology, **config)
            ttnn.synchronize_device(device)
            return result

        def check(result, index, label):
            actual = ttnn.to_torch(result).float()
            metrics = correctness(actual, fixture["expected"][index], contract)
            metrics.update(label=label, feature_variant=index, **tensor_contract(result, device))
            metrics["passed"] &= metrics["tensor_contract_valid"]
            report["checks"].append(metrics)
            return actual

        phase("warmup")
        first = []
        for index in range(len(features)):
            output = execute(index)
            first.append(check(output, index, "warmup"))
            output = None
        # A second invocation catches program-cache and lazy metadata behavior.
        for index in range(len(features)):
            output = execute(index)
            output = None
        if args.mode == "memory":
            from .memory import workspace_bound
            phase("capture_memory")
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            try:
                # end_graph_capture returns C++ events; Python argument/return
                # summaries are unused and can dominate large memory probes.
                ttnn.graph.disable_python_io_recording()
                report["memory_capture"] = {
                    "python_io_recording": ttnn.graph.is_python_io_recording_enabled(),
                    "scope": "C++ graph including allocator events; Python I/O sidecar disabled"}
                capture_start = time.perf_counter()
                output = execute(0)
            finally:
                allocation_trace = ttnn.graph.end_graph_capture()
            report["memory_capture"]["elapsed_ms"] = (time.perf_counter() - capture_start) * 1000
            trace_path = args.output.with_suffix(".allocations.json.gz")
            with gzip.open(trace_path, "wt") as stream:
                json.dump(allocation_trace, stream)
            dram = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
            l1 = ttnn.get_memory_view(device, ttnn.BufferType.L1)
            report["workspace"] = workspace_bound(allocation_trace, dram_banks=dram.num_banks,
                l1_capacity_bytes=l1.num_banks * l1.total_bytes_per_bank,
                limit_bytes=contract["workspace_limit_bytes"], output_address=output.buffer_address(),
                expected_output_bytes=math.prod(output.padded_shape) * 2)
            report["workspace"].update(trace_path=str(trace_path), trace_sha256=sha256(trace_path))
            check(output, 0, "memory_capture")
            output = None
        phase("measure")
        for iteration in range(args.repeat):
            index = iteration % len(features)
            start = time.perf_counter()
            output = execute(index)
            elapsed = (time.perf_counter() - start) * 1000
            report["samples"].append({"variant": index, "ms": elapsed})
            if iteration >= args.repeat - len(features):
                actual = check(output, index, "last_timed")
                report["checks"][-1]["repeat_bitwise_equal"] = bool(torch.equal(first[index], actual))
                report["checks"][-1]["passed"] &= report["checks"][-1]["repeat_bitwise_equal"]
            output = None
        report["timing"] = summarize([s["ms"] for s in report["samples"]])
        report["input_unchanged"] = all(torch.equal(ttnn.to_torch(d), h)
            for d, h in zip(features, fixture["features"]))
        report["memory"]["after_execution"] = memory(device)
        report["useful_tflops"] = report["work"]["useful_flops"] / (report["timing"]["median_ms"] * 1e9)
        report["issued_tflops"] = report["work"]["issued_flops"] / (report["timing"]["median_ms"] * 1e9)
        report["protected_files_unchanged"] = all(sha256(path) == digest for path, digest in protected.items())
        report["fixture_unchanged"] = sha256(fixture_path) == entry["sha256"]
        report["candidate_source_unchanged"] = source_state(candidate_root)["sha256"] == report["candidate"]["sha256"]
        passed = all(c["passed"] for c in report["checks"]) and all(report[k] for k in (
            "input_unchanged", "protected_files_unchanged", "fixture_unchanged", "candidate_source_unchanged"))
        if args.mode == "memory":
            passed &= report["workspace"]["within_limit"]
        report["status"] = "passed" if passed else "numerical_or_integrity_failure"
    except Exception as error:
        report.update(status="execution_failure", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
    finally:
        phase("close_device")
        if device is not None:
            ttnn.close_device(device)
        phase("complete")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
