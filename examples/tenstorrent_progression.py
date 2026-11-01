"""Run all four stages on actual voxel topology; emit correctness and timings.

python examples/tenstorrent_progression.py --output /tmp/tt-progression.json
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import torch
import ttnn
from flex_gemm.tenstorrent import (
    build_subm_neighbor_map, prepare_explicit, prepare_implicit, sparse_conv_reference,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--ci", type=int, default=64)
    parser.add_argument("--co", type=int, default=64)
    parser.add_argument("--split-k", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.points <= 512 or min(args.ci, args.co, args.repeat) < 1:
        parser.error("points must be in [1,512]; channels and repeat must be positive")
    torch.manual_seed(42)
    # Sample unique coordinates from an 8^3 grid, preserving random input order.
    ids = torch.randperm(512)[:args.points]
    coords = torch.stack((torch.zeros_like(ids), ids // 64, ids // 8 % 8, ids % 8), dim=1)
    start = time.perf_counter()
    mapping = build_subm_neighbor_map(coords, (1, 8, 8, 8))
    topology_ms = (time.perf_counter() - start) * 1000
    x = torch.randn(args.points, args.ci).bfloat16()
    w = (torch.randn(args.co, 27, args.ci) / (27 * args.ci) ** 0.5).bfloat16()
    bias = torch.randn(args.co).bfloat16()
    golden = sparse_conv_reference(x, w, mapping, bias)
    device = ttnn.open_device(device_id=args.device)
    rows = []
    try:
        for name, masked, split in [("explicit", False, 1), ("implicit", False, 1),
                                     ("masked", True, 1), ("split_k", True, args.split_k)]:
            start = time.perf_counter()
            if name == "explicit":
                prepared = prepare_explicit(x, w, mapping, device=device, bias=bias)
                run = prepared.run
            else:
                prepared = prepare_implicit(x, w, mapping, device=device, bias=bias, masked=masked)
                run = lambda: prepared.run(split_k=split)
            ttnn.synchronize_device(device)
            prepare_ms = (time.perf_counter() - start) * 1000
            result = run()  # Compile/warm outside measured execution.
            ttnn.synchronize_device(device)
            actual = ttnn.to_torch(result).float()
            torch.testing.assert_close(actual, golden, atol=0.025, rtol=0.025)
            samples = []
            for _ in range(args.repeat):
                start = time.perf_counter()
                result = run()
                ttnn.synchronize_device(device)
                samples.append((time.perf_counter() - start) * 1000)
            entry = dict(stage=name, prepare_ms=prepare_ms, warm_median_ms=statistics.median(samples),
                         warm_samples_ms=samples, max_abs_error=(actual - golden).abs().max().item(),
                         relative_l2=((actual - golden).norm() / golden.norm()).item())
            if masked:
                entry.update(active_block_offsets=prepared.schedule.active_offsets,
                             dense_block_offsets=prepared.schedule.dense_offsets)
            rows.append(entry)
        report = dict(arch=str(device.arch()), torch=torch.__version__, seed=42,
                      points=args.points, ci=args.ci, co=args.co, volume=27, split_k=args.split_k,
                      topology_ms=topology_ms, timing="host wall time including dispatch/synchronize; "
                      "warm execution excludes preparation, compilation, and output download", stages=rows)
        metal_home = os.environ.get("TT_METAL_HOME")
        if metal_home:
            report["tt_metal_commit"] = subprocess.check_output(
                ["git", "-C", metal_home, "rev-parse", "HEAD"], text=True).strip()
        serialized = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n")
        print(serialized)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
