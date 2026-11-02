"""Capture real full-decoder convolution inputs using CPU reference inference.

The source GenRecon checkout is read-only. Save one full input per distinct
shape/kernel/topology-stage signature, plus every invocation's metadata. Layer
occurrences are measured; representative timings are still only a stage proxy.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .common import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--genrecon", type=Path, default=Path("/home/user/GitHub/GenRecon"))
    p.add_argument("--checkpoint", type=Path, required=True, help="checkpoint stem without suffix")
    p.add_argument("--latent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.genrecon))
    from genrecon.modules.sparse import config, SparseTensor
    config.set_conv_backend("reference")
    from genrecon.modules.sparse.conv.conv import SparseConv3d
    from genrecon.models import __getattr__ as model_class
    from safetensors.torch import load_file
    checkpoint_file = args.checkpoint.with_suffix(".safetensors")
    configuration = json.loads(args.checkpoint.with_suffix(".json").read_text())
    configuration["args"]["use_fp16"] = False
    source_paths = [args.genrecon / "genrecon/modules/sparse/conv/conv_reference.py",
                    args.genrecon / "genrecon/models/sc_vaes/sparse_unet_vae.py"]
    sources = {str(path): sha256(path) for path in source_paths}
    report = {"status": "loading", "scene_input": str(args.latent),
        "scene_input_sha256": sha256(args.latent), "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": sha256(checkpoint_file), "torch": torch.__version__,
        "genrecon_commit": subprocess.check_output(["git", "-C", str(args.genrecon), "rev-parse", "HEAD"], text=True).strip(),
        "sources": sources, "captures": [], "invocations": []}
    write_json(args.output / "manifest.json", report)
    with torch.device("meta"):
        model = model_class(configuration["name"])(**configuration["args"])
    model.load_state_dict(load_file(str(checkpoint_file)), strict=True, assign=True)
    model = model.float().eval()
    with np.load(args.latent, allow_pickle=False) as data:
        coords = torch.from_numpy(data["coords"]).int()
        features = torch.from_numpy(data["shape_feats"]).float()
    seen = set()

    def hook(name):
        def capture(module, inputs):
            x = inputs[0]
            signature = (len(x.coords), module.in_channels, module.out_channels,
                         tuple(module.kernel_size), tuple(module.dilation))
            key = f"n{signature[0]}-{signature[1]}x{signature[2]}-k{'x'.join(map(str,signature[3]))}"
            invocation = {"layer": name, "signature": key, "points": len(x.coords),
                          "ci": module.in_channels, "co": module.out_channels,
                          "kernel": list(module.kernel_size), "dilation": list(module.dilation)}
            report["invocations"].append(invocation)
            if signature not in seen:
                seen.add(signature)
                path = args.output / (key + ".pt")
                torch.save({"coords": x.coords, "feats": x.feats.detach()}, path)
                report["captures"].append(dict(invocation, file=path.name, sha256=sha256(path),
                    feature_dtype=str(x.feats.dtype), spatial_shape=list(x.spatial_shape)))
                print(json.dumps(report["captures"][-1]), flush=True)
            write_json(args.output / "manifest.json", report)
        return capture

    handles = [module.register_forward_pre_hook(hook(name)) for name, module in model.named_modules()
               if isinstance(module, SparseConv3d)]
    report["status"] = "running_cpu_decoder"
    write_json(args.output / "manifest.json", report)
    start = time.perf_counter()
    try:
        with torch.no_grad():
            output, guides = model.forward_features(SparseTensor(features, coords), return_subs=True)
        counts = Counter(row["signature"] for row in report["invocations"])
        for entry in report["captures"]:
            entry["occurrences"] = counts[entry["signature"]]
        report.update(status="complete", seconds=time.perf_counter() - start,
                      output_points=len(output.coords), stage_points=[len(g.coords) for g in guides],
                      source_unchanged=all(sha256(path) == value for path, value in sources.items()))
        if not report["source_unchanged"]:
            report["status"] = "source_changed_during_capture"
    finally:
        for handle in handles:
            handle.remove()
        write_json(args.output / "manifest.json", report)


if __name__ == "__main__":
    main()
