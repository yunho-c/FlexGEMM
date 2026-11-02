"""Conservative workspace bound from actual allocator events.

Resident inputs/weights/topology exist before capture. Count every new DRAM
allocation, including the final output (a conservative overcount), then add
the entire device L1 capacity. Per-bank aligned sizes include allocation padding.
"""


def workspace_bound(trace, *, dram_banks, l1_capacity_bytes, limit_bytes):
    active = {}
    current = peak = allocations = 0
    anomalies = []
    for node in trace:
        kind, params = node.get("node_type"), node.get("params", {})
        if kind not in ("buffer_allocate", "buffer_deallocate") or params.get("type") != "DRAM":
            continue
        key = (params.get("device_id"), params.get("address"))
        if kind == "buffer_allocate":
            if key in active:
                anomalies.append("duplicate live DRAM allocation")
                continue
            if "max_size_per_bank" not in params:
                anomalies.append("missing aligned size")
                size = int(params["size"])
            else:
                size = int(params["max_size_per_bank"]) * dram_banks
            active[key] = size
            current += size
            peak = max(peak, current)
            allocations += 1
        elif key in active:
            current -= active.pop(key)
        # Deallocation of a pre-capture resident buffer cannot reduce the
        # amount of newly allocated workspace, so deliberately ignore it.
    bound = peak + l1_capacity_bytes
    verified = allocations > 0 and not anomalies
    return {"kind": "conservative_bound_from_allocator_trace", "new_dram_peak_bytes": peak,
        "dram_allocations": allocations, "dram_banks": dram_banks,
        "all_l1_capacity_bytes": l1_capacity_bytes, "workspace_upper_bound_bytes": bound,
        "workspace_limit_bytes": limit_bytes, "trace_valid": verified, "anomalies": anomalies,
        "within_limit": verified and bound <= limit_bytes,
        "scope": "new aligned DRAM allocations, including final output, plus whole-device L1 capacity; resident buffers existed before capture"}
