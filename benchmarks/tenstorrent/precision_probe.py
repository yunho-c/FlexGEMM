"""Diagnose GEMM, output rounding and bias separately; never a scored trial."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import traceback

import torch

from .common import correctness, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--mode", default="timing")
    args = p.parse_args()
    if not os.environ.get("FLEX_GEMM_EVALUATOR_LOCKED") or args.output.exists():
        p.error("use the serialized runner and a fresh output")
    torch.set_num_threads(8)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    contract = json.loads((args.corpus / "contract.json").read_text())
    entry = next(c for c in manifest["cases"] if c["name"] == args.case)
    path = args.corpus / entry["file"]
    if sha256(path) != entry["sha256"]:
        raise ValueError("fixture digest mismatch")
    fixture = torch.load(path, weights_only=True, map_location="cpu")
    config = json.loads(args.config)
    sys.path.insert(0, str(args.candidate_root.resolve()))
    import flex_gemm.tenstorrent as candidate
    import ttnn
    report = {"kind": "unscored_precision_diagnostic", "status": "running",
        "case": entry, "config": config, "results": [], "probe_sha256": sha256(__file__)}

    def phase(name):
        report["phase"] = name
        write_json(args.output, report)
        print(name, flush=True)

    device = None
    try:
        phase("open_device")
        device = ttnn.open_device(device_id=args.device)
        phase("prepare")
        weights = candidate.prepare_sparse_weights(fixture["weight"], bias=fixture["bias"], device=device)
        topology = candidate.prepare_sparse_topology(fixture["mapping"], len(fixture["features"][0]),
            device=device, masked=config["algorithm"].startswith("masked"))
        without_bias = replace(weights, bias=None)
        for variant, x in enumerate(fixture["features"]):
            dx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            expected = fixture["expected"][variant]
            raw_expected = expected if fixture["bias"] is None else expected - fixture["bias"].float()
            for fp32 in (False, True):
                phase("measure")
                raw = candidate.sparse_conv_device(dx, without_bias, topology, fp32_output=fp32, **config)
                ttnn.synchronize_device(device)
                host_raw = ttnn.to_torch(raw).float()
                full = raw if weights.bias is None else ttnn.add(raw, weights.bias)
                ttnn.synchronize_device(device)
                host_full = ttnn.to_torch(full).float()
                index = tuple(int(i) for i in torch.unravel_index((host_full - expected).abs().argmax(), expected.shape))
                row = {"variant": variant, "fp32_output": fp32,
                    "raw_dtype": str(raw.dtype), "with_bias_dtype": str(full.dtype),
                    "raw": correctness(host_raw, raw_expected, contract),
                    "with_bias": correctness(host_full, expected, contract),
                    "host_fp32_bias_after_download": correctness(host_raw + (0 if fixture["bias"] is None else fixture["bias"].float()), expected, contract),
                    "largest_error_index": list(index),
                    "largest_error_values": {"raw": float(host_raw[index]), "raw_reference": float(raw_expected[index]),
                        "with_bias": float(host_full[index]), "reference": float(expected[index])}}
                row["failing_values"] = [{"index": at, "raw": float(host_raw[tuple(at)]),
                    "raw_reference": float(raw_expected[tuple(at)]), "with_bias": float(host_full[tuple(at)]),
                    "reference": float(expected[tuple(at)]),
                    "host_fp32_bias": float((host_raw + (0 if fixture["bias"] is None else fixture["bias"].float()))[tuple(at)])}
                    for at in row["with_bias"].get("first_failures", [])]
                if fp32:
                    converted = ttnn.typecast(full, ttnn.bfloat16)
                    ttnn.synchronize_device(device)
                    row["single_final_bf16_rounding"] = correctness(ttnn.to_torch(converted).float(), expected, contract)
                    del converted
                    if weights.bias is not None:
                        precise_bias = ttnn.typecast(weights.bias, ttnn.float32)
                        precise_sum = ttnn.add(raw, precise_bias, dtype=ttnn.float32)
                        row["explicit_fp32_bias"] = correctness(ttnn.to_torch(precise_sum).float(), expected, contract)
                        row["explicit_fp32_bias_dtype"] = str(precise_sum.dtype)
                        del precise_bias, precise_sum
                report["results"].append(row)
                write_json(args.output, report)
                del raw, full, host_raw, host_full
            del dx
        report["status"] = "passed"  # Probe completed; this is not numerical acceptance.
        report["numerical_acceptance"] = "not_applicable: inspect per-stage metrics; no candidate promotion"
    except Exception as error:
        report.update(status="execution_failure", error=str(error), traceback=traceback.format_exc())
    finally:
        phase("close_device")
        if device is not None:
            ttnn.close_device(device)
        phase("complete")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
