"""Materialize immutable CPU workloads and independent golden outputs.

python -m benchmarks.tenstorrent.prepare --output /path/to/new/corpus
Use --real-capture and --checkpoint to add the captured GenRecon wide layer.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .common import neighbor_oracle, reference, sha256, write_json


def synthetic(points, ci, co, topology, seed):
    gen = torch.Generator().manual_seed(seed)
    if topology == "surface":
        side = math.ceil(math.sqrt(points / 2)) + 2
        grid = torch.cartesian_prod(torch.arange(2), torch.arange(side), torch.arange(side))
        coords = torch.cat((torch.zeros(len(grid), 1, dtype=torch.long), grid), 1)
    else:
        density = .75 if topology == "clustered" else .12
        side = math.ceil((points / density / 2)**(1 / 3))
        ids = torch.arange(2 * side**3)
        coords = torch.stack((ids // side**3, ids // side**2 % side, ids // side % side, ids % side), 1)
    coords = coords[torch.randperm(len(coords), generator=gen)[:points]].int()
    features = torch.randn(points, ci, generator=gen).bfloat16()
    weight = (torch.randn(co, 27, ci, generator=gen) / math.sqrt(27 * ci)).bfloat16()
    bias = (torch.randn(co, generator=gen) * .1).bfloat16()
    return coords, features, weight, bias


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--real-capture", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    if bool(args.real_capture) != bool(args.checkpoint):
        p.error("--real-capture and --checkpoint must be supplied together")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    contract_path = Path(__file__).with_name("contract.json")
    (args.output / "contract.json").write_bytes(contract_path.read_bytes())
    manifest = {"version": 1, "status": "preparing", "contract_sha256": sha256(contract_path),
                "torch": torch.__version__, "cases": [], "held_out_real_scenes": "unavailable"}

    def save_case(name, coords, x, weight, bias, split, provenance):
        start = time.perf_counter()
        mapping = neighbor_oracle(coords)
        # Fresh values on exactly the same topology: data-dependent caches fail.
        x1 = (x.float() * .5 + .03125).bfloat16()
        expected = [reference(a, weight, mapping, bias) for a in (x, x1)]
        path = args.output / (name + ".pt")
        torch.save({"coords": coords, "features": [x, x1], "weight": weight,
                    "bias": bias, "mapping": mapping, "expected": expected}, path)
        entry = {"name": name, "split": split, "file": path.name, "sha256": sha256(path),
                 "points": len(coords), "ci": x.shape[1], "co": weight.shape[0],
                 "volume": 27, "valid_interactions": int((mapping >= 0).sum()),
                 "occurrences": None, "provenance": provenance,
                 "prepare_seconds": time.perf_counter() - start}
        manifest["cases"].append(entry)
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps(entry), flush=True)

    for split, seed in (("tune", 20261102), ("validation", 20261103)):
        for i, (points, ci, co, topology) in enumerate([
            (35, 17, 45, "scattered"), (256, 64, 64, "surface"),
            (1025, 128, 128, "clustered"), (257, 32, 256, "scattered"),
            (4096, 64, 64, "surface"),
        ]):
            values = synthetic(points, ci, co, topology, seed + i)
            save_case(f"{split}-{topology}-{points}-{ci}x{co}", *values, split,
                      {"kind": "synthetic", "seed": seed + i, "topology": topology})
    if args.real_capture:
        from safetensors import safe_open
        capture = torch.load(args.real_capture, map_location="cpu", weights_only=True)
        with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
            weight = checkpoint.get_tensor("blocks.0.4.conv1.weight").bfloat16()
            bias = checkpoint.get_tensor("blocks.0.4.conv1.bias").bfloat16()
            norm_weight = checkpoint.get_tensor("blocks.0.4.norm1.weight").float()
            norm_bias = checkpoint.get_tensor("blocks.0.4.norm1.bias").float()
        save_case("poster-shape-wide-4490", capture["coords"], capture["feats"].bfloat16(), weight,
                  bias, "application", {"kind": "captured_decoder_layer", "capture": str(args.real_capture),
                  "capture_sha256": sha256(args.real_capture), "checkpoint": str(args.checkpoint),
                  "checkpoint_sha256": sha256(args.checkpoint), "layer": "blocks.0.4.conv1",
                  "source_scene": "poster", "feature_variant_1": "synthetic perturbation of captured activations"})
        # Preserve the exact previously reported tolerance miss. It remains a
        # failing gate if reproduced, rather than an xfail interpreted as success.
        generator = torch.Generator().manual_seed(123)
        points = 513
        side = math.ceil((points / .4 / 2) ** (1 / 3))
        ids = torch.randperm(2 * side**3, generator=generator)[:points]
        coords = torch.stack((ids // side**3, ids // side**2 % side, ids // side % side, ids % side), 1)
        x = torch.randn(points, 1024, generator=generator)
        x = torch.nn.functional.silu(torch.nn.functional.layer_norm(
            x, (1024,), norm_weight, norm_bias, eps=1e-6)).bfloat16()
        save_case("shape-norm-silu-513-known-failure", coords, x, weight, bias, "regression",
                  {"kind": "synthetic_pretrained_activation", "seed": 123,
                   "layer": "blocks.0.4.conv1", "checkpoint_sha256": sha256(args.checkpoint),
                   "known_failure": "original feature variant: row 305, channel 2295; retain original tolerances"})
    manifest["status"] = "ready"
    write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
