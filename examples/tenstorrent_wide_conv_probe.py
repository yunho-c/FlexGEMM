"""Validate pretrained GenRecon 1024->4096 sparse convolutions on TT hardware.

Uses real checkpoint weights with explicitly synthetic coordinates/features.
Writes incremental evidence even when a stage fails numerical validation.
Run with TT_METAL_WATCHER=1 to check device accesses; those timings include its
instrumentation and must not be compared to uninstrumented performance runs.
"""

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import ttnn
from safetensors import safe_open

from flex_gemm.tenstorrent import (
    build_subm_neighbor_map, prepare_explicit, prepare_implicit, sparse_conv_reference,
)
from flex_gemm.tenstorrent.ops import _matmul_fidelity


def digest(tensor):
    return hashlib.sha256(memoryview(tensor.contiguous().view(torch.uint8).numpy())).hexdigest()


def metrics(actual, expected):
    actual, expected = actual.double(), expected.double()
    error = actual - expected
    ac, ec = actual - actual.mean(), expected - expected.mean()
    bound = 0.025 + 0.025 * expected.abs()
    finite = bool(torch.isfinite(actual).all())
    failed = (error.abs() > bound) | ~torch.isfinite(actual)
    result = {
        "all_finite": finite,
        "max_abs_error": error.abs().max().item() if finite else None,
        "relative_l2": (error.norm() / expected.norm().clamp_min(1e-30)).item() if finite else None,
        "pcc": ((ac * ec).sum() / (ac.norm() * ec.norm()).clamp_min(1e-30)).item() if finite else None,
        "elementwise_failures_atol_rtol_0.025": int(failed.sum()),
        "max_error_over_elementwise_tolerance": (error.abs() / bound).max().item() if finite else None,
        "elements": actual.numel(),
        "first_failures": [],
    }
    for row, col in failed.nonzero()[:5].tolist():
        value = actual[row, col].item()
        result["first_failures"].append(dict(row=row, channel=col,
            actual=value if math.isfinite(value) else None,
            expected=expected[row, col].item(), tolerance=bound[row, col].item()))
    return result


def accepted(result):
    return (result["all_finite"] and result["relative_l2"] <= 0.01
            and result["pcc"] >= 0.9999
            and result["elementwise_failures_atol_rtol_0.025"] == 0)


def fixture(points, norm_weight, norm_bias, activation, seed):
    # Match the earlier GenRecon sparse-conv probe's two-batch, ~40% occupancy
    # sampling and generator sequence. Rows deliberately remain unsorted.
    generator = torch.Generator().manual_seed(seed)
    side = math.ceil((points / 0.4 / 2) ** (1 / 3))
    flat = torch.randperm(2 * side**3, generator=generator)[:points]
    coords = torch.stack((flat // side**3, flat // side**2 % side,
                          flat // side % side, flat % side), dim=1)
    features = torch.randn(points, 1024, generator=generator)
    if activation == "norm_silu":
        features = torch.nn.functional.silu(torch.nn.functional.layer_norm(
            features, (1024,), norm_weight.float(), norm_bias.float(), eps=1e-6))
    start = time.perf_counter()
    neighbor = build_subm_neighbor_map(coords, (2, side, side, side))
    map_ms = (time.perf_counter() - start) * 1000
    return features, neighbor, side, map_ms


def dram_live_bytes(device):
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    return int(view.num_banks * view.total_bytes_allocated_per_bank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path.home() /
                        "models/genrecon/trellis2/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors")
    parser.add_argument("--layer", default="blocks.0.4.conv1")
    parser.add_argument("--points", type=int, nargs="+", default=[129, 513, 1025])
    parser.add_argument("--activation", choices=["gaussian", "norm_silu"], default="gaussian")
    parser.add_argument("--stages", nargs="+", default=["explicit", "implicit", "masked", "split_k"],
                        choices=["explicit", "implicit", "masked", "split_k", "unmasked_split_k"])
    parser.add_argument("--split-k", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.points + [args.repeat, args.threads]) < 1 or not 1 <= args.split_k <= 864:
        parser.error("points, repeat and threads must be positive; split-k must be in [1,864]")
    if args.output.exists():
        parser.error("use a fresh output path to preserve previous evidence")
    torch.set_num_threads(args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    report = {
        "status": "running", "phase": "load_checkpoint", "source_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__).resolve(), *sorted((root / "flex_gemm/tenstorrent").glob("*.py")),
                                    *sorted((root / "flex_gemm/tenstorrent/kernels").glob("*.cpp"))]},
        "checkpoint": str(args.checkpoint), "layer": args.layer, "torch": torch.__version__,
        "seed": args.seed, "activation": args.activation, "cpu_threads": args.threads,
        "watcher": os.environ.get("TT_METAL_WATCHER", "disabled"),
        "scope": "One pretrained convolution; synthetic two-batch coordinates/activations; not a full decoder",
        "reference": "FP32 per-offset gather/matmul on BF16-rounded operands; original-checkpoint FP32 comparison also recorded",
        "gate": {"finite": True, "atol": 0.025, "rtol": 0.025, "max_relative_l2": 0.01, "min_pcc": 0.9999},
        "timing": "Host wall time; prepare includes packing/upload; warm execution includes dispatch/synchronize, excludes preparation/JIT/output download",
        "memory": "Live DRAM allocator snapshots, not transient peak measurements; workspace estimates listed separately",
        "cases": [],
    }

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    save()
    with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
        weight = checkpoint.get_tensor(args.layer + ".weight")
        bias = checkpoint.get_tensor(args.layer + ".bias")
        block = args.layer.rsplit(".", 1)[0]
        norm_weight = checkpoint.get_tensor(block + ".norm1.weight")
        norm_bias = checkpoint.get_tensor(block + ".norm1.bias")
    if tuple(weight.shape) != (4096, 3, 3, 3, 1024) or tuple(bias.shape) != (4096,):
        raise ValueError("this probe specifically requires 1024->4096, 3x3x3 checkpoint weights")
    report.update(weight_shape=list(weight.shape), checkpoint_dtype=str(weight.dtype),
                  weight_sha256=digest(weight), bias_sha256=digest(bias),
                  norm_weight_sha256=digest(norm_weight), norm_bias_sha256=digest(norm_bias))
    metal_home = os.environ.get("TT_METAL_HOME")
    if metal_home:
        report["tt_metal_commit"] = subprocess.check_output(
            ["git", "-C", metal_home, "rev-parse", "HEAD"], text=True).strip()
    rounded_weight, rounded_bias = weight.bfloat16(), bias.bfloat16()
    report["phase"] = "waiting_for_device"
    save()
    # Share the GenRecon probe lock: two runtimes must not open this device at once.
    with open("/tmp/genrecon-tt-pjrt.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        device = ttnn.open_device(device_id=args.device)
        try:
            report["arch"] = str(device.arch())
            report["matmul_fidelity"] = str(_matmul_fidelity(device))
            for points in args.points:
                report["phase"] = f"reference_{points}"
                save()
                x, neighbor, side, map_ms = fixture(points, norm_weight, norm_bias, args.activation, args.seed)
                xb = x.bfloat16()
                start = time.perf_counter()
                products = sparse_conv_reference(xb, rounded_weight, neighbor)
                reference = products + rounded_bias.float()
                # The current API packs matmul to BF16 before a separate bias add.
                # This diagnostic models that rounding; it does not relax the gate.
                epilogue_reference = (products.bfloat16().float() + rounded_bias.float()).bfloat16().float()
                reference_ms = (time.perf_counter() - start) * 1000
                original_reference = sparse_conv_reference(x, weight, neighbor, bias)
                for stage in args.stages:
                    split = args.split_k if stage.endswith("split_k") else 1
                    masked = stage in ("masked", "split_k")
                    padded_m = (points + 31) // 32 * 32
                    case = {"points": points, "stage": stage, "split_k": split, "spatial_side": side,
                            "mean_neighbors": (neighbor >= 0).sum(1).float().mean().item(),
                            "map_ms": map_ms, "reference_ms": reference_ms,
                            "status": "running", "weight_bytes": weight.numel() * 2,
                            "explicit_expansion_bytes": padded_m * 27 * 1024 * 2 if stage == "explicit" else 0,
                            "split_workspace_bytes": split * padded_m * 4096 * 4 if split > 1 else 0}
                    report["cases"].append(case)
                    report["phase"] = f"{stage}_{points}_prepare"
                    save()
                    print(report["phase"], flush=True)
                    prepared = result = run = None
                    try:
                        gc.collect()
                        ttnn.synchronize_device(device)
                        case["dram_live_before"] = dram_live_bytes(device)
                        start = time.perf_counter()
                        if stage == "explicit":
                            prepared = prepare_explicit(xb, rounded_weight, neighbor, device=device, bias=rounded_bias)
                            run = prepared.run
                        else:
                            prepared = prepare_implicit(xb, rounded_weight, neighbor, device=device,
                                                        bias=rounded_bias, masked=masked)
                            run = lambda: prepared.run(split_k=split)
                        ttnn.synchronize_device(device)
                        case["prepare_ms"] = (time.perf_counter() - start) * 1000
                        case["dram_live_prepared"] = dram_live_bytes(device)
                        if masked:
                            case.update(active_offsets=prepared.schedule.active_offsets,
                                        dense_offsets=prepared.schedule.dense_offsets)
                        report["phase"] = f"{stage}_{points}_first_execution"
                        save()
                        start = time.perf_counter()
                        result = run()
                        ttnn.synchronize_device(device)
                        case["first_execution_ms"] = (time.perf_counter() - start) * 1000
                        case["dram_live_with_output"] = dram_live_bytes(device)
                        actual = ttnn.to_torch(result).float()
                        if tuple(actual.shape) != (points, 4096):
                            raise ValueError(f"incorrect output shape {tuple(actual.shape)}")
                        case["rounded_reference"] = metrics(actual, reference)
                        case["bf16_epilogue_reference"] = metrics(actual, epilogue_reference)
                        case["epilogue_rounding_only"] = metrics(epilogue_reference, reference)
                        case["original_checkpoint_reference"] = metrics(actual, original_reference)
                        case["quantization_only"] = metrics(reference, original_reference)
                        case["warm_ms"] = []
                        report["phase"] = f"{stage}_{points}_warm_execution"
                        save()
                        for _ in range(args.repeat):
                            result = None
                            start = time.perf_counter()
                            result = run()
                            ttnn.synchronize_device(device)
                            case["warm_ms"].append((time.perf_counter() - start) * 1000)
                        again = ttnn.to_torch(result).float()
                        case["repeat_bitwise_equal"] = bool(torch.equal(actual, again))
                        case["warm_median_ms"] = statistics.median(case["warm_ms"])
                        case["status"] = "passed" if accepted(case["rounded_reference"]) and case["repeat_bitwise_equal"] else "numerical_failure"
                        print(f"{stage} N={points}: {case['status']}, "
                              f"relL2={case['rounded_reference']['relative_l2']}, "
                              f"warm={case['warm_median_ms']:.3f} ms", flush=True)
                    except Exception as error:
                        case.update(status="execution_failure", error=f"{type(error).__name__}: {error}")
                        print(case["error"], flush=True)
                    finally:
                        run = result = prepared = None
                        gc.collect()
                        ttnn.synchronize_device(device)
                        case["dram_live_after_release"] = dram_live_bytes(device)
                        save()
            report["phase"] = "complete"
            report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"]) else "failed"
            save()
        finally:
            ttnn.close_device(device)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
