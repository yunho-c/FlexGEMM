"""CPU error-budget screen for signed digit products; not hardware validation."""

import argparse
import json
import math
from pathlib import Path
import time

import torch

from .common import correctness, reference, sha256, write_json


def reconstruct(values, *, base, digits, per_output=False):
    maximum = values.abs().flatten(1).amax(1) if per_output else values.abs().max()
    exponent = torch.ceil(torch.log2(maximum.clamp_min(1e-30))) - (int(math.log2(base)) - 1)
    scale = torch.exp2(exponent)
    if per_output:
        scale = scale.reshape(-1, *([1] * (values.ndim - 1)))
    residual, result = values / scale, torch.zeros_like(values)
    for i in range(digits):
        digit = residual.round()
        result += digit * (float(base) ** -i)
        residual = (residual - digit) * base
    return result * scale


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--cases", nargs="+", default=["ALL"])
    p.add_argument("--base", type=int, default=256)
    p.add_argument("--digits", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.base not in (64, 128, 256) or args.digits not in (2, 3, 4) or args.output.exists():
        p.error("use base 64/128/256, 2/3/4 digits and a fresh output")
    torch.set_num_threads(8)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    contract = json.loads((args.corpus / "contract.json").read_text())
    cases = manifest["cases"] if args.cases == ["ALL"] else [c for c in manifest["cases"] if c["name"] in args.cases]
    if not cases or (args.cases != ["ALL"] and len(cases) != len(args.cases)):
        p.error("unknown or duplicate case names")
    report = {"status": "running", "kind": "CPU approximation screen, not hardware arithmetic or latency",
        "base": args.base, "digits": args.digits, "source_sha256": sha256(__file__),
        "manifest_sha256": sha256(args.corpus / "manifest.json"), "cases": {}}
    for case in cases:
        start = time.perf_counter()
        path = args.corpus / case["file"]
        if sha256(path) != case["sha256"]:
            raise ValueError("fixture digest mismatch")
        fixture = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        weight = reconstruct(fixture["weight"].float(), base=args.base, digits=args.digits, per_output=True)
        metrics = []
        for x, expected in zip(fixture["features"], fixture["expected"]):
            approximate = reconstruct(x.float(), base=args.base, digits=args.digits)
            output = reference(approximate, weight, fixture["mapping"], fixture["bias"]).bfloat16()
            metrics.append(correctness(output, expected, contract))
        report["cases"][case["name"]] = {"fixture_sha256": case["sha256"], "checks": metrics,
            "seconds": time.perf_counter() - start}
        write_json(args.output, report)
        print(case["name"], [r["passed"] for r in metrics], flush=True)
    report["status"] = "complete"
    report["all_pass"] = all(c["passed"] for v in report["cases"].values() for c in v["checks"])
    write_json(args.output, report)


if __name__ == "__main__":
    main()
