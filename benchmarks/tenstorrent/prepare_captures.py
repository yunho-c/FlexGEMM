"""Build full-size hashed evaluation cases from CPU decoder captures."""

import argparse
import json
from pathlib import Path
import time

import torch

from .common import reference, sha256, write_json


def verify_sampled_topology(coords, mapping, kernel, dilation, samples=1024):
    """Check distributed output rows against an independent coordinate oracle.

    Every input row remains available. Sampling is only a topology cross-check,
    never a crop of the benchmark's convolution or reference outputs.
    """
    from itertools import product
    rows = [tuple(r) for r in coords.tolist()]
    lookup = {row: i for i, row in enumerate(rows)}
    offsets = list(product(*(range(-(k // 2), k // 2 + 1) for k in kernel)))
    chosen = torch.linspace(0, len(rows) - 1, min(samples, len(rows))).long().unique().tolist()
    for index in chosen:
        row = rows[index]
        expected = [lookup.get((row[0], *(c + o * d for c, o, d in
                    zip(row[1:], offset, dilation))), -1) for offset in offsets]
        if mapping[index].tolist() != expected:
            raise AssertionError(f"topology oracle mismatch at row {index}")
    return len(chosen)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    capture_manifest = json.loads((args.captures / "manifest.json").read_text())
    if capture_manifest["status"] != "complete" or not capture_manifest["source_unchanged"]:
        raise ValueError("capture must have completed with unchanged source")
    from safetensors import safe_open
    from flex_gemm.tenstorrent.reference import build_subm_neighbor_map
    import inspect
    builder = Path(inspect.getfile(build_subm_neighbor_map))
    checkpoint_path = Path(capture_manifest["checkpoint"])
    if sha256(checkpoint_path) != capture_manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint digest mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    contract = Path(__file__).with_name("contract.json")
    (args.output / "contract.json").write_bytes(contract.read_bytes())
    manifest = {"version": 1, "status": "preparing", "contract_sha256": sha256(contract),
                "capture_manifest_sha256": sha256(args.captures / "manifest.json"),
                "builder_sha256": sha256(builder), "torch": torch.__version__,
                "cases": [], "held_out_real_scenes": "unavailable",
                "application_score_scope": "occurrence-weighted representative shapes from one decoder; layer values/weights vary across occurrences, so this is a latency proxy"}
    for entry in capture_manifest["captures"]:
        start = time.perf_counter()
        path = args.captures / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("capture digest mismatch")
        capture = torch.load(path, weights_only=True)
        coords = capture["coords"]
        x = capture["feats"].bfloat16()
        del capture
        x1 = (x.float() * .5 + .03125).bfloat16()
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            weight = checkpoint.get_tensor(entry["layer"] + ".weight").bfloat16()
            bias_key = entry["layer"] + ".bias"
            bias = checkpoint.get_tensor(bias_key).bfloat16() if bias_key in checkpoint.keys() else None
        mapping = build_subm_neighbor_map(coords, (int(coords[:, 0].max()) + 1, *entry["spatial_shape"]),
                                         tuple(entry["kernel"]), tuple(entry["dilation"]))
        verified = verify_sampled_topology(coords, mapping, entry["kernel"], entry["dilation"])
        print(entry["signature"], "topology verified; computing references", flush=True)
        expected = [reference(value, weight, mapping, bias) for value in (x, x1)]
        output = args.output / entry["file"]
        torch.save({"coords": coords, "features": [x, x1], "weight": weight, "bias": bias,
                    "mapping": mapping, "expected": expected}, output)
        case = dict(name=entry["signature"], split="application", file=output.name, sha256=sha256(output),
                    points=len(coords), ci=entry["ci"], co=entry["co"], volume=mapping.shape[1],
                    occurrences=entry["occurrences"], valid_interactions=int((mapping >= 0).sum()),
                    topology_oracle_rows=verified, prepare_seconds=time.perf_counter() - start,
                    provenance={"kind": "captured_decoder_layer", "layer": entry["layer"],
                                "capture_sha256": entry["sha256"], "scene_input_sha256": capture_manifest["scene_input_sha256"],
                                "checkpoint_sha256": capture_manifest["checkpoint_sha256"]})
        manifest["cases"].append(case)
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps(case), flush=True)
        del x, x1, coords, mapping, expected, weight, bias
    manifest["status"] = "ready"
    write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
