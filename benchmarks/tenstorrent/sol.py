"""Fixed arithmetic-capacity reference, distinct from empirical calibration.

The pinned TT-Metal GEMM_FLOPS report specifies 16/32/48/64 ideal cycles per
32x32x32 tile product for LoFi/HiFi2/HiFi3/HiFi4. This is an optimistic matrix
capacity bound. It is not a prediction of sparse-convolution runtime.
"""

import argparse
import json
from pathlib import Path

from .common import sha256, write_json


def arithmetic_reference(useful_flops, *, clock_mhz, available_cores, fidelity_phases=3):
    if min(clock_mhz, available_cores, fidelity_phases) <= 0:
        raise ValueError("clock, cores and phases must be positive")
    per_core_tflops = (2 * 32**3) / (16 * fidelity_phases) * clock_mhz / 1e6
    capacity_tflops = per_core_tflops * available_cores
    return {"kind": "theoretical_matrix_capacity", "useful_flops": useful_flops,
        "clock_mhz": clock_mhz, "available_cores": available_cores,
        "fidelity_phases": fidelity_phases, "capacity_tflops": capacity_tflops,
        "arithmetic_lower_bound_ms": useful_flops / (capacity_tflops * 1e9),
        "memory_lower_bound_ms": None,
        "scope": "optimistic arithmetic-only bound; excludes sparse indexing, communication, dependencies and launch overhead"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--clock-mhz", type=float, required=True, help="Observed profiler clock; record source separately")
    p.add_argument("--cores", type=int, required=True, help="Available compute grid, not chip marketing core count")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = {"manifest_sha256": sha256(args.manifest), "cases": {},
        "source": "https://github.com/tenstorrent/tt-metal/blob/9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md",
        "precision_contract": "BF16 operands, HiFi3, FP32 destination accumulation",
        "note": "A low percentage does not establish a realizable reciprocal speedup. Measured calibration is separate."}
    for case in manifest["cases"]:
        useful = 2 * case["valid_interactions"] * case["ci"] * case["co"]
        report["cases"][case["name"]] = arithmetic_reference(useful,
            clock_mhz=args.clock_mhz, available_cores=args.cores)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
