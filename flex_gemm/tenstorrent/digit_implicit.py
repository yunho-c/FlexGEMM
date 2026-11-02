"""Opt-in four-product implicit convolution with a bounded explicit fallback."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .digits import SparseDigitWeights, _copy_output_rows, _feature_digits, _scale_and_bias
from .implicit import _round32


@dataclass
class DigitImplicitPlan:
    info: dict
    chunks: tuple = ()


def index_rows(mapping, input_rows):
    """K6 page order: output-row tile, kernel tap, row within tile."""
    rows, volume = mapping.shape
    padded = torch.full((_round32(rows), volume), input_rows, dtype=torch.int64)
    padded[:rows] = torch.where(mapping < 0, input_rows, mapping)
    return padded.reshape(-1, 32, volume).permute(0, 2, 1).reshape(-1, volume * 32).to(torch.uint32)


def prepare_sparse_digit_implicit(weights, topology, *, row_chunk=256):
    """Prepare only immutable weights, integer indices and shape-dependent plans.

    A topology owns its index/plan cache; different same-shaped topologies never
    share it. Weights own their two K6 layers. Calling this ahead of a forward
    separates static preparation from hot execution. Unsupported shapes retain
    the exact explicit digit implementation and expose the fallback reason.
    """
    import ttnn
    from ._k6 import CB_BUDGET, K6Layer
    if not isinstance(weights, SparseDigitWeights) or topology.masks is not None:
        raise ValueError("implicit digits require digit weights and unmasked topology")
    if (topology.device != weights.device or topology.host_indices.shape[1] != weights.volume):
        raise ValueError("weights and topology must share a device and kernel volume")
    if not isinstance(row_chunk, int) or isinstance(row_chunk, bool) or row_chunk < 1:
        raise ValueError("row_chunk must be a positive integer")
    key = (weights.ci, weights.co, weights.volume, row_chunk)
    if key in topology.digit_plans:
        plan = topology.digit_plans[key]
        if plan.info["path"] == "digit_implicit_gemm" and not weights.implicit_layers:
            _prepare_layers(weights, plan.info["multicast"])
        return plan
    device = weights.device
    grid = device.compute_with_storage_grid_size()
    info = {"path": "digit_implicit_gemm", "reason": None, "row_chunk": row_chunk,
            "grid": [grid.x, grid.y], "cb_budget_bytes": CB_BUDGET, "chunks": [],
            "product_calls": 4, "weight_storage": "FP32 encoding of BF16 integer digits",
            "unused_bf16_output_included": True}

    def fallback(reason):
        info.update(path="digit_explicit_gemm", reason=reason)
        plan = DigitImplicitPlan(info)
        topology.digit_plans[key] = plan
        return plan

    if device.arch() != ttnn.Arch.WORMHOLE_B0:
        return fallback("architecture")
    if weights.volume > 32:
        return fallback("kernel_volume_exceeds_32")
    if row_chunk % 32:
        return fallback("row_chunk_not_tile_aligned")
    layer = K6Layer(_spec(weights))
    rows = len(topology.host_indices)
    for start in range(0, rows, row_chunk):
        end = min(start + row_chunk, rows)
        nbmax = (_round32(end - start) // 32 + grid.x * grid.y - 1) // (grid.x * grid.y)
        tg, nch = layer.plan(nbmax)
        cb_bytes = layer.cb_bytes(tg, nch, nbmax, has_res=False, write_y=True)
        info["chunks"].append({"start": start, "end": end, "tap_chunk": tg,
                               "tap_chunks": nch, "cb_bytes": cb_bytes})
        if cb_bytes > CB_BUDGET:
            return fallback("per_core_cb_budget")
    physical = [[device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
                 for x in range(grid.x)] for y in range(grid.y)]
    multicast = all(all(row[x].x == row[0].x + x and row[x].y == row[0].y
                        for x in range(grid.x)) for row in physical)
    info.update(multicast=multicast, split_gather=True,
                physical_core_rows=[[[c.x, c.y] for c in row] for row in physical])
    _prepare_layers(weights, multicast)
    chunks = []
    for chunk in info["chunks"]:
        start, end = chunk["start"], chunk["end"]
        ids = index_rows(topology.host_indices[start:end], topology.input_points)
        index = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        chunks.append((start, end, index))
    plan = DigitImplicitPlan(info, tuple(chunks))
    topology.digit_plans[key] = plan
    return plan


def _spec(weights):
    return SimpleNamespace(name="flexgemm_integer_product", cin=weights.ci, cout=weights.co,
        cin_pad=_round32(weights.ci), cout_pad=_round32(weights.co),
        kernel_volume=weights.volume, relu=False, out_terms=1)


def _prepare_layers(weights, multicast):
    import ttnn
    from ._k6 import K6Layer
    if weights.implicit_layers:
        return
    spec = _spec(weights)
    zero_bias = ttnn.zeros([32, spec.cout_pad], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=weights.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    layers = []
    for digit in weights.digits:
        layer = K6Layer(spec)
        layer.mcast = multicast
        layer._dev = (ttnn.typecast(digit.weights, ttnn.float32), zero_bias)
        layers.append(layer)
    weights.implicit_layers = tuple(layers)


def sparse_digit_implicit_conv_device(features, weights, topology, *, row_chunk=256, fp32_output=False):
    import ttnn
    plan = prepare_sparse_digit_implicit(weights, topology, row_chunk=row_chunk)
    if plan.info["path"] == "digit_explicit_gemm":
        from .digits import sparse_digit_conv_device
        return sparse_digit_conv_device(features, weights, topology,
                                         row_chunk=row_chunk, fp32_output=fp32_output)
    high, low, feature_scale = _feature_digits(features, _round32(weights.ci))
    scale = ttnn.multiply(ttnn.repeat(feature_scale, [1, weights.co]), weights.scale)
    scale_tile = ttnn.multiply(weights.scale_tile, feature_scale)
    output = ttnn.allocate_tensor_on_device(ttnn.Shape([len(topology.host_indices), weights.co]),
        ttnn.float32 if fp32_output else ttnn.bfloat16, ttnn.TILE_LAYOUT,
        weights.device, ttnn.DRAM_MEMORY_CONFIG)
    for start, end, index in plan.chunks:
        def product(table, digit):
            y, unused = weights.implicit_layers[digit](table, index, identity=None,
                write_y=True, dead_skip=False, table_memory="dram")
            del unused
            return ttnn.reshape(y, [index.shape[0] * 32, _round32(weights.co)])

        top = product(high, 0)
        cross_a, cross_b = product(high, 1), product(low, 0)
        cross = ttnn.add(cross_a, cross_b)
        del cross_a, cross_b
        bottom = product(low, 1)
        result = ttnn.add(ttnn.add(top, ttnn.multiply(cross, 1 / 256)), ttnn.multiply(bottom, 1 / 65536))
        del top, cross, bottom
        result = ttnn.slice(result, [0, 0], [end - start, weights.co])
        result = _scale_and_bias(result, weights, scale, scale_tile)
        if not fp32_output:
            result = ttnn.typecast(result, ttnn.bfloat16)
        _copy_output_rows(result, output, start)
        del result
    return output
