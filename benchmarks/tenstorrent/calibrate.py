"""Resident TTNN resource references, launched with run --workload calibration.

These are measured implementations, not hardware maxima or bus counters. Every
operation includes allocation, launch and synchronization. Transfers to/from
the host and initial layout preparation are outside the timed region.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from .common import sha256, summarize, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=11)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--mode", choices=("timing", "correctness", "profile"), default="timing")
    args = p.parse_args()
    if not os.environ.get("FLEX_GEMM_EVALUATOR_LOCKED"):
        p.error("launch through run --workload calibration")
    if args.output.exists() or args.repeat < 2:
        p.error("fresh output and at least two repeats required")
    if args.mode == "timing" and any(os.getenv(k, "0") not in ("", "0") for k in (
            "TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_DPRINT_CORES")):
        p.error("lean timing cannot use instrumentation")
    torch.set_num_threads(8)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    entry = next(c for c in manifest["cases"] if c["name"] == args.case)
    fixture_path = args.corpus / entry["file"]
    if sha256(fixture_path) != entry["sha256"]:
        raise ValueError("fixture integrity failure")
    fixture = torch.load(fixture_path, weights_only=True, map_location="cpu")
    config = json.loads(args.config)
    sys.path.insert(0, str(args.candidate_root.resolve()))
    from flex_gemm.tenstorrent.ops import _compute_config, _matmul_fidelity
    import ttnn
    report = {"status": "running", "kind": "empirical_reference", "config": config,
        "case": entry, "mode": args.mode, "operations": [], "hardware_counters": None,
        "timing_scope": "resident operator, host wall time with device synchronization; upload/download excluded",
        "byte_scope": "logical payload, not measured DRAM or NoC bus traffic",
        "source_sha256": sha256(__file__), "torch": torch.__version__}
    metal_home = os.environ.get("TT_METAL_HOME")
    if metal_home:
        report["tt_metal_commit"] = subprocess.check_output(
            ["git", "-C", metal_home, "rev-parse", "HEAD"], text=True).strip()

    def phase(name):
        report["phase"] = name
        write_json(args.output, report)
        print(name, flush=True)

    device = None
    try:
        phase("open_device")
        device = ttnn.open_device(device_id=args.device)
        grid = device.compute_with_storage_grid_size()
        report.update(arch=str(device.arch()), available_grid=[grid.x, grid.y],
            math_fidelity=str(_matmul_fidelity(device)))

        def upload(x, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=layout, device=device)

        def measure(name, operation, expected, **work):
            phase("warmup_" + name)
            for _ in range(2):
                result = operation()
                ttnn.synchronize_device(device)
                result = None
            phase("measure")
            samples = []
            for _ in range(args.repeat):
                start = time.perf_counter()
                result = operation()
                ttnn.synchronize_device(device)
                samples.append((time.perf_counter() - start) * 1000)
                if len(samples) < args.repeat:
                    result = None
            actual = ttnn.to_torch(result)
            exact = bool(torch.equal(actual, expected))
            result = None
            timing = summarize(samples)
            record = {"name": name, "timing": timing, "exact_reference_match": exact, **work}
            record["max_abs_error"] = float((actual.float() - expected.float()).abs().max())
            if not exact:
                indices = (actual != expected).nonzero()[:5]
                record["first_mismatches"] = [{"index": i.tolist(),
                    "actual": float(actual[tuple(i)]), "expected": float(expected[tuple(i)])} for i in indices]
            if "flops" in work:
                record["tflops"] = work["flops"] / (timing["median_ms"] * 1e9)
            if "logical_bytes" in work:
                record["logical_gbps"] = work["logical_bytes"] / (timing["median_ms"] * 1e6)
            report["operations"].append(record)
            write_json(args.output, report)
            if not exact:
                raise ValueError(f"calibration reference mismatch: {name}")

        gen = torch.Generator().manual_seed(20261102)
        # Small integers have exact products at HiFi3, and FP32 accumulation
        # stays exact at these dimensions; only the final BF16 rounding remains.
        for m, k, n in config.get("dense_shapes", [[1024, 4096, 2048], [256, 27648, 4096]]):
            phase("prepare_dense")
            a = torch.randint(-2, 3, (m, k), generator=gen).bfloat16()
            b = torch.randint(-2, 3, (k, n), generator=gen).bfloat16()
            expected = a.float() @ b.float()
            da, db = upload(a), upload(b)
            measure(f"dense_{m}x{k}x{n}", lambda: ttnn.matmul(da, db,
                dtype=ttnn.float32, compute_kernel_config=_compute_config(device)), expected,
                flops=2 * m * k * n, shape=[m, k, n], input_distribution="integers -2 through 2",
                output_dtype="float32 (separate from the BF16-output operator contract)")
            del a, b, da, db, expected
        phase("prepare_layout_and_stream")
        rows, cols = config.get("stream_shape", [65536, 256])
        h = torch.randint(-32, 32, (rows, cols), generator=gen).bfloat16()
        tiled, row_major = upload(h), upload(h, ttnn.ROW_MAJOR_LAYOUT)
        payload = h.numel() * 2 * 2
        measure("dram_clone", lambda: ttnn.clone(tiled), h, logical_bytes=payload, shape=[rows, cols])
        measure("tilize", lambda: ttnn.to_layout(row_major, ttnn.TILE_LAYOUT), h,
                logical_bytes=payload, shape=[rows, cols])
        measure("untilize", lambda: ttnn.to_layout(tiled, ttnn.ROW_MAJOR_LAYOUT), h,
                logical_bytes=payload, shape=[rows, cols])
        del h, tiled, row_major
        phase("prepare_gather")
        x = fixture["features"][0]
        cp = (x.shape[1] + 31) // 32 * 32
        padded = torch.zeros(len(x) + 1, cp, dtype=torch.bfloat16)
        padded[:len(x), :x.shape[1]] = x
        dx = upload(padded, ttnn.ROW_MAJOR_LAYOUT)
        for chunk in config.get("gather_rows", [256, 2048]):
            mapping = fixture["mapping"][:chunk]
            ids = torch.where(mapping < 0, len(x), mapping).long().reshape(1, -1)
            di = ttnn.from_torch(ids.to(torch.uint32), dtype=ttnn.uint32,
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            expected = padded[ids]
            measure(f"embedding_{len(mapping)}", lambda: ttnn.embedding(di, dx,
                layout=ttnn.ROW_MAJOR_LAYOUT), expected,
                logical_bytes=ids.numel() * (4 + 2 * cp * 2),
                rows=len(mapping), channels=cp, volume=mapping.shape[1])
            del di, expected
        report["status"] = "passed"
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
