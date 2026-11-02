"""Independent oracle, accounting, integrity, and statistics for evaluation."""

import hashlib
import json
import math
import statistics
from itertools import product
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def neighbor_oracle(coords, kernel=(3, 3, 3), dilation=(1, 1, 1)):
    """Exact coordinate dictionary, independent of candidate topology routines."""
    rows = [tuple(r) for r in coords.tolist()]
    lookup = {row: i for i, row in enumerate(rows)}
    if len(lookup) != len(rows):
        raise ValueError("duplicate coordinates")
    offsets = list(product(*(range(-(k // 2), k // 2 + 1) for k in kernel)))
    return torch.tensor([[lookup.get((row[0], *(c + o * d for c, o, d in
        zip(row[1:], offset, dilation))), -1) for offset in offsets] for row in rows],
        dtype=torch.int64).reshape(len(rows), len(offsets))


def reference(features, weight, mapping, bias=None):
    """FP32 per-offset products on the stored, already BF16-rounded operands."""
    co, ci = weight.shape[0], weight.shape[-1]
    weight = weight.reshape(co, -1, ci)
    output = torch.zeros((len(mapping), co), dtype=torch.float32)
    for offset in range(mapping.shape[1]):
        valid = mapping[:, offset] >= 0
        if valid.any():
            output[valid] += features[mapping[valid, offset]].float() @ weight[:, offset].float().T
    return output if bias is None else output + bias.float()


def correctness(actual, expected, contract):
    if tuple(actual.shape) != tuple(expected.shape):
        return {"passed": False, "shape_match": False, "actual_shape": list(actual.shape)}
    actual, expected = actual.double(), expected.double()
    finite = bool(torch.isfinite(actual).all())
    if not finite:
        return {"passed": False, "shape_match": True, "finite": False}
    error = actual - expected
    bound = contract["atol"] + contract["rtol"] * expected.abs()
    failures = error.abs() > bound
    rel = float(error.norm() / expected.norm().clamp_min(1e-30))
    return {"passed": not bool(failures.any()) and rel <= contract["max_relative_l2"],
            "shape_match": True, "finite": True, "relative_l2": rel,
            "max_abs_error": float(error.abs().max()) if error.numel() else 0.,
            "elementwise_failures": int(failures.sum()),
            "max_error_over_tolerance": float((error.abs() / bound).max()) if error.numel() else 0.,
            "first_failures": failures.nonzero()[:5].tolist()}


def summarize(samples):
    median = statistics.median(samples)
    return {"median_ms": median, "min_ms": min(samples), "max_ms": max(samples),
            "mad_ms": statistics.median(abs(s - median) for s in samples),
            "samples_ms": samples}


def work_accounting(mapping, ci, co, *, algorithm, split_k=1, row_chunk=256, available_cores=64,
                    digit_output_assembly="concat", digit_epilogue="expanded", digit_gather="embedding",
                    digit_implicit_policy=None):
    """Model the baseline kernels' issued work/request payloads, not bus traffic.

    Model changes must accompany structural kernel changes. Counts include the
    dummy multiply issued for an empty masked split, and per-offset channel tails.
    TTNN dense matmul physical scheduling is unavailable: its count is a logical
    padded GEMM estimate. Requested payloads exclude protocol/alignment overhead.
    """
    if algorithm == "digit_implicit_gemm":
        if digit_implicit_policy is None:
            raise ValueError("implicit digit accounting requires the prepared dispatch policy")
        if digit_implicit_policy["path"] == "digit_explicit_gemm":
            result = work_accounting(mapping, ci, co, algorithm="digit_explicit_gemm",
                row_chunk=row_chunk, available_cores=available_cores,
                digit_output_assembly=digit_output_assembly, digit_epilogue=digit_epilogue,
                digit_gather=digit_gather)
            result["dispatch_fallback"] = digit_implicit_policy["reason"]
            return result
        return _digit_implicit_accounting(mapping, ci, co, row_chunk, available_cores, digit_implicit_policy)
    mapping = mapping.to(torch.int64).clone()
    mapping[mapping == 0xffffffff] = -1
    m, volume = mapping.shape
    ct, nt, mt = (ci + 31) // 32, (co + 31) // 32, (m + 31) // 32
    if min(ci, co, m, volume, row_chunk, available_cores) <= 0:
        raise ValueError("dimensions, row_chunk and available_cores must be positive")
    if isinstance(split_k, bool) or not isinstance(split_k, int) or not 1 <= split_k <= volume * ct:
        raise ValueError("invalid split_k")
    edges = int((mapping >= 0).sum())
    useful = 2 * edges * ci * co
    digit_products = algorithm == "digit_explicit_gemm"
    digit_tiled = digit_products and digit_gather == "tiled" and (row_chunk % 32 == 0 or row_chunk >= m)
    fused_gather = algorithm == "tiled_gather_gemm" or digit_tiled
    digit_copy = digit_products and digit_output_assembly == "tile_copy"
    explicit = algorithm in ("explicit_gemm", "tiled_gather_gemm", "digit_explicit_gemm")
    if fused_gather and row_chunk < m and row_chunk % 32:
        raise ValueError("tiled gather chunk must be tile aligned")
    masked = algorithm.startswith("masked")
    split_k = split_k if algorithm.endswith("splitk") else 1
    if masked:
        # Match the fixed baseline Gray ordering, independently of candidate code.
        if volume > 32:
            raise ValueError("masked kernel volume exceeds 32")
        bits = ((mapping >= 0).long() << torch.arange(volume)).sum(1)
        key = bits.clone()
        for shift in (1, 2, 4, 8, 16):
            key ^= key >> shift
        mapping = mapping[torch.argsort(key.to(torch.int32), stable=True)]
    padded = torch.full((mt * 32, volume), -1, dtype=torch.int64)
    padded[:m] = mapping
    valid = (padded >= 0).reshape(mt, 32, volume).sum(1)
    active = valid > 0
    tile_products = dummy = map_reads = feature_bytes = weight_bytes = 0
    task_work = []
    if explicit:
        padded_rows = sum(math.ceil(min(row_chunk, m - start) / 32) * 32
                          for start in range(0, m, row_chunk))
        tile_products = padded_rows // 32 * nt * volume * ct * (4 if digit_products else 1)
        if fused_gather:
            feature_bytes = edges * ct * 64 * (2 if digit_tiled else 1)
            map_reads = padded_rows // 32 * volume * (2 if digit_tiled else 1)
    else:
        for split in range(split_k):
            begin, end = volume * ct * split // split_k, volume * ct * (split + 1) // split_k
            channels = torch.tensor([max(0, min(end, (offset + 1) * ct) - max(begin, offset * ct))
                                     for offset in range(volume)])
            issued_channels = channels.unsqueeze(0).expand(mt, volume)
            if masked:
                issued_channels = issued_channels * active
            counts = issued_channels.sum(1)
            map_reads += nt * int((issued_channels > 0).sum())
            feature_bytes += nt * int((valid * issued_channels).sum()) * 64
            weight_bytes += nt * int(counts.sum()) * 2048
            dummy += nt * int((counts == 0).sum())
            tile_products += nt * int(counts.clamp_min(1).sum())
            task_work.extend(counts.clamp_min(1).repeat_interleave(nt).tolist())
    issued = tile_products * 2 * 32**3
    ncores = min(len(task_work), available_cores) if task_work else None
    core_work = [sum(task_work[len(task_work) * i // ncores:len(task_work) * (i + 1) // ncores])
                 for i in range(ncores)] if ncores else []
    partial_bytes = split_k * mt * 32 * nt * 32 * 4 if split_k > 1 else 0
    # Payload of S-1 FP32 adds: two inputs and one output per add. Slices/casts
    # may add traffic, so this is not total reduction traffic or peak workspace.
    reduction_bytes = (split_k - 1) * mt * 32 * nt * 32 * 12
    return {"kind": "modeled", "model": ("digit_explicit_v4" if digit_tiled else "digit_explicit_v3" if digit_epilogue == "tile_broadcast" else "digit_explicit_v2" if digit_copy else "digit_explicit_v1") if digit_products else "tiled_gather_v1" if fused_gather else "baseline_v1",
            "valid_interactions": edges, "useful_flops": useful, "issued_flops": issued,
            "issued_scope": "logical_padded_gemm" if explicit else "custom_kernel_tile_products",
            "work_efficiency": useful / issued, "dummy_tile_products": dummy,
            "active_block_offsets": int(active.sum()), "dense_block_offsets": mt * volume,
            "requested_feature_bytes": feature_bytes if fused_gather or not explicit else None,
            "requested_weight_bytes": None if explicit else weight_bytes,
            "requested_map_bytes": map_reads * 128 if fused_gather or not explicit else None,
            "gather_output_bytes": padded_rows * volume * ct * 64 * (2 if digit_tiled else 1) if fused_gather else None,
            "output_assembly_copy_bytes": mt * nt * 1024 * 4 if digit_copy and m > row_chunk and row_chunk % 32 == 0 else 0,
            "split_partial_bytes": partial_bytes, "reduction_add_payload_bytes": reduction_bytes,
            "explicit_expansion_bytes": math.ceil(min(row_chunk, m) / 32) * 32 * volume * ct * 32 * 2 * (2 if digit_products else 1) if explicit else 0,
            "available_cores": available_cores, "selected_cores": ncores,
            "core_tile_products": core_work,
            "core_work_max_over_mean": max(core_work) / statistics.mean(core_work) if core_work else None,
            "dram_bus_bytes": None, "noc_bus_bytes": None, "peak_workspace_bytes": None}


def _digit_implicit_accounting(mapping, ci, co, row_chunk, cores, policy):
    """Count four K6 calls and active taps; requested payloads are not bus traffic."""
    mapping = mapping.to(torch.int64).clone()
    mapping[mapping == 0xffffffff] = -1
    m, volume = mapping.shape
    if min(m, volume, ci, co, row_chunk, cores) <= 0 or row_chunk % 32:
        raise ValueError("implicit digit dimensions must be positive and row chunks tile aligned")
    if policy["path"] != "digit_implicit_gemm" or math.prod(policy["grid"]) != cores:
        raise ValueError("implicit digit policy does not match the device grid")
    ct, nt, mt = (ci + 31) // 32, (co + 31) // 32, (m + 31) // 32
    padded = torch.full((mt * 32, volume), -1, dtype=torch.int64)
    padded[:m] = mapping
    active = (padded.reshape(mt, 32, volume) >= 0).any(1)
    edges = int((mapping >= 0).sum())
    tile_products = 4 * int(active.sum()) * ct * nt
    useful, issued = 2 * edges * ci * co, tile_products * 2 * 32**3
    work = [0] * cores
    index_bytes = unused_output = 0
    expected = [(start, min(start + row_chunk, m)) for start in range(0, m, row_chunk)]
    if [(p["start"], p["end"]) for p in policy["chunks"]] != expected:
        raise ValueError("implicit digit chunk plan does not cover the fixture exactly")
    for plan in policy["chunks"]:
        start, end = plan["start"], plan["end"]
        blocks = (end - start + 31) // 32
        for local in range(blocks):
            work[local % cores] += 4 * int(active[start // 32 + local].sum()) * ct * nt
        # Both data-movement RISCs read a full index page for every tap chunk.
        index_bytes += 4 * 2 * blocks * volume * 128 * plan["tap_chunks"]
        unused_output += 4 * (blocks * 32 + 1) * nt * 32 * 2
    senders = policy["grid"][1] if policy["multicast"] else cores
    return {"kind": "modeled", "model": "digit_implicit_k6_v1",
        "valid_interactions": edges, "useful_flops": useful, "issued_flops": issued,
        "issued_scope": "four_custom_kernel_products_with_empty_tap_skips",
        "work_efficiency": useful / issued if issued else None, "dummy_tile_products": 0,
        "active_block_offsets": int(active.sum()), "dense_block_offsets": mt * volume,
        "requested_feature_bytes": 4 * edges * ct * 64,
        "requested_weight_bytes": 4 * len(expected) * senders * volume * ct * nt * 4096,
        "requested_map_bytes": index_bytes, "gather_output_bytes": 0,
        "fp32_product_write_bytes": 4 * mt * nt * 32 * 32 * 4,
        "unused_bf16_table_write_bytes": unused_output,
        "output_assembly_copy_bytes": mt * nt * 32 * 32 * 4,
        "split_partial_bytes": 0, "reduction_add_payload_bytes": 0,
        "explicit_expansion_bytes": 0, "available_cores": cores, "selected_cores": cores,
        "core_tile_products": work,
        "core_work_max_over_mean": max(work) / statistics.mean(work) if issued else None,
        "dram_bus_bytes": None, "noc_bus_bytes": None, "peak_workspace_bytes": None,
        "payload_scope": "requested payloads excluding feature-preparation/reconstruction traffic and on-chip movement"}
