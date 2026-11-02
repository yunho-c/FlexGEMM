"""Conservative workspace bound from actual allocator events.

Resident inputs/weights/topology exist before capture. Report every new DRAM
allocation and a temporary-only bound excluding the identified final output,
as required by the fixed contract. Add the entire device L1 capacity. Per-bank
aligned sizes include padding; reused addresses are distinct allocations.
"""


def workspace_bound(trace, *, dram_banks, l1_capacity_bytes, limit_bytes,
                    output_address=None, expected_output_bytes=None):
    active = {}
    current = peak = allocations = 0
    anomalies = []
    deltas = []
    for index, node in enumerate(trace):
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
            active[key] = (size, index)
            current += size
            deltas.append((index, size))
            peak = max(peak, current)
            allocations += 1
        elif key in active:
            size, _ = active.pop(key)
            current -= size
            deltas.append((index, -size))
        # Deallocation of a pre-capture resident buffer cannot reduce the
        # amount of newly allocated workspace, so deliberately ignore it.
    excluded = None
    if output_address is not None:
        matches = [(key, value) for key, value in active.items()
                   if int(key[1]) == int(output_address)]
        if len(matches) != 1:
            anomalies.append("final output must identify exactly one live new allocation")
        else:
            key, (size, index) = matches[0]
            if expected_output_bytes is None or size < expected_output_bytes:
                anomalies.append("final output allocation is smaller than its padded tensor")
            else:
                excluded = {"device_id": key[0], "address": key[1],
                    "event_index": index, "aligned_bytes": size,
                    "expected_padded_bytes": expected_output_bytes,
                    "excluded_bytes": expected_output_bytes}
    current = temporary_peak = 0
    for index, delta in deltas:
        if excluded is not None and index == excluded["event_index"]:
            # Count bank alignment and any oversized backing allocation as
            # workspace. A small output view cannot hide a large temporary.
            delta -= excluded["excluded_bytes"]
        current += delta
        temporary_peak = max(temporary_peak, current)
    bound = temporary_peak + l1_capacity_bytes
    verified = allocations > 0 and not anomalies
    return {"kind": "conservative_bound_from_allocator_trace", "new_dram_peak_bytes": peak,
        "dram_allocations": allocations, "dram_banks": dram_banks,
        "inclusive_upper_bound_bytes": peak + l1_capacity_bytes,
        "temporary_dram_peak_bytes": temporary_peak, "excluded_final_output": excluded,
        "all_l1_capacity_bytes": l1_capacity_bytes, "workspace_upper_bound_bytes": bound,
        "workspace_limit_bytes": limit_bytes, "trace_valid": verified, "anomalies": anomalies,
        "within_limit": verified and bound <= limit_bytes,
        "scope": ("new aligned DRAM allocations excluding only the final output's padded tensor bytes from its identified live allocation, plus whole-device L1 capacity"
                  if excluded is not None else "new aligned DRAM allocations including final output, plus whole-device L1 capacity")}
