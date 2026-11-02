"""Experimental four-product sparse convolution using signed base-256 digits.

BF16 operands are approximated with two small-integer digits. The ordinary
BF16 reference and tolerances still apply; this is not an FP32-exact mode.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from .implicit import _round32
from .ops import _add_bias, _compute_config

OUTPUT_ASSEMBLY = "tile_copy"
EPILOGUE = "tile_broadcast"


@dataclass
class SparseDigitWeights:
    digits: tuple
    scale: object
    bias: object
    device: object
    ci: int
    co: int
    volume: int
    scale_tile: object
    bias_tile: object


def prepare_sparse_digit_weights(weight, *, device, bias=None):
    """Prepare immutable integer weights and FP32 power-of-two scales."""
    import ttnn
    from .resident import prepare_sparse_weights
    if weight.device.type != "cpu" or weight.ndim not in (3, 5) or not weight.is_floating_point():
        raise ValueError("expected floating-point CPU sparse weights")
    if weight.requires_grad or (bias is not None and bias.requires_grad):
        raise ValueError("digit sparse convolution does not implement autograd")
    if bias is not None and (bias.device.type != "cpu" or bias.shape != (weight.shape[0],) or not bias.is_floating_point()):
        raise ValueError("bias must be a floating-point CPU [Co] tensor")
    rounded = weight.bfloat16().float()
    maximum = rounded.abs().flatten(1).amax(1)
    scale = torch.exp2(torch.ceil(torch.log2(maximum.clamp_min(1e-30))) - 7)
    residual = rounded / scale.reshape(-1, *([1] * (weight.ndim - 1)))
    high = residual.round()
    low = ((residual - high) * 256).round()
    digits = tuple(prepare_sparse_weights(value.bfloat16(), device=device) for value in (high, low))
    sd = ttnn.from_torch(scale[None], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    bd = None if bias is None else ttnn.from_torch(bias.bfloat16().float()[None],
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    st = ttnn.from_torch(scale[None].expand(32, -1).contiguous(),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    bt = None if bias is None else ttnn.from_torch(bias.bfloat16().float()[None].expand(32, -1).contiguous(),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    first = digits[0]
    return SparseDigitWeights(digits, sd, bd, device, first.ci, first.co, first.volume, st, bt)


def _concat(chunks):
    """Bound concat fan-in and release each previous level as it is replaced."""
    import ttnn
    while len(chunks) > 1:
        level = [ttnn.concat(chunks[i:i + 4], dim=0) if len(chunks[i:i + 4]) > 1
                 else chunks[i] for i in range(0, len(chunks), 4)]
        chunks.clear()
        chunks = level
    return chunks[0]


def _concat_outputs(chunks):
    """Keep tiled concat out of its four-copy ragged-row fallback."""
    import ttnn
    rows = sum(part.shape[0] for part in chunks)
    co = chunks[0].shape[1]
    aligned_prefix = all(part.shape[0] % 32 == 0 for part in chunks[:-1])
    if len(chunks) > 1 and aligned_prefix and chunks[-1].shape[0] % 32:
        last = chunks[-1]
        chunks[-1] = ttnn.pad(last, [(0, _round32(last.shape[0]) - last.shape[0]), (0, 0)], 0)
        del last
        return ttnn.slice(_concat(chunks), [0, 0], [rows, co])
    # Non-tile-aligned interior chunks cannot be padded without inserting rows.
    return _concat(chunks)


def _copy_output_rows(chunk, output, start):
    """Copy complete tile rows directly to their final, disjoint positions."""
    import ttnn
    nt = _round32(output.shape[1]) // 32
    tasks = _round32(chunk.shape[0]) // 32 * nt
    grid = output.device().compute_with_storage_grid_size()
    ncores = min(tasks, grid.x * grid.y)
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    runtime = ttnn.RuntimeArgs()
    for i, core in enumerate(cores):
        begin, end = tasks * i // ncores, tasks * (i + 1) // ncores
        runtime[core.x][core.y] = [chunk.buffer_address(), output.buffer_address(),
                                  begin, end - begin, start // 32 * nt]
    page = 2048 if output.dtype == ttnn.bfloat16 else 4096
    compile_args = [page]
    for tensor in (chunk, output):
        compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).parent / "kernels/copy_output_tiles.cpp"),
        core_ranges=ranges, compile_time_args=compile_args, runtime_args=runtime,
        config=ttnn.ReaderConfigDescriptor())
    cb = ttnn.CBDescriptor(total_size=page * 8, core_ranges=ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=output.dtype, page_size=page)])
    ttnn.generic_op([chunk, output], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb], semaphores=[]))


def _feature_digits(features, cp):
    """Recompute digits each call, with bounded FP32 conversion scratch."""
    import ttnn
    maximum = ttnn.typecast(ttnn.max(ttnn.abs(features), keepdim=True), ttnn.float32)
    exponent = ttnn.subtract(ttnn.ceil(ttnn.log2(ttnn.maximum(maximum, 1e-30))), 7.)
    scale, inverse = ttnn.exp2(exponent), ttnn.exp2(ttnn.neg(exponent))
    high_parts, low_parts = [], []
    rows, ci = features.shape
    chunk = max(32, (1048576 // cp // 32) * 32)
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        part = ttnn.typecast(ttnn.slice(features, [start, 0], [end, ci]), ttnn.float32)
        # BF16 values and a power-of-two multiplier are exactly representable
        # through the FPU broadcast path; no expanded FP32 constant is needed.
        residual = ttnn.multiply(part, inverse)
        high = ttnn.round(residual)
        low = ttnn.round(ttnn.multiply(ttnn.subtract(residual, high), 256.))
        for value, parts in ((high, high_parts), (low, low_parts)):
            rm = ttnn.to_layout(ttnn.typecast(value, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
            if cp != ci:
                rm = ttnn.pad(rm, [(0, 0), (0, cp - ci)], 0)
            parts.append(rm)
        del part, residual, high, low, value, rm
    zero = ttnn.zeros([1, cp], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=features.device())
    high_parts.append(zero)
    low_parts.append(zero)
    return _concat(high_parts), _concat(low_parts), scale


def _scale_and_bias(result, weights, scale, scale_tile):
    """Broadcast complete FP32 tiles across batches, keeping H/W equal."""
    import ttnn
    rows, co = result.shape
    if rows % 32:
        result = ttnn.multiply(result, ttnn.repeat(scale, [rows, 1]))
        return _add_bias(result, weights.bias, fp32_output=True)
    # Batch broadcasting retains the SFPU elementwise path. H/W broadcast
    # would instead truncate FP32 operands through the FPU input registers.
    result = ttnn.reshape(result, [rows // 32, 32, co])
    result = ttnn.multiply(result, ttnn.reshape(scale_tile, [1, 32, co]))
    if weights.bias_tile is not None:
        result = ttnn.add(result, ttnn.reshape(weights.bias_tile, [1, 32, co]))
    return ttnn.reshape(result, [rows, co])


def sparse_digit_conv_device(features, weights, topology, *, row_chunk=256, fp32_output=False):
    """Fresh device digit extraction, gather, four products, scale and bias."""
    import ttnn
    if not isinstance(weights, SparseDigitWeights) or topology.masks is not None:
        raise ValueError("digit convolution requires digit weights and unmasked topology")
    cp = _round32(weights.ci)
    high, low, feature_scale = _feature_digits(features, cp)
    scale = ttnn.multiply(ttnn.repeat(feature_scale, [1, weights.co]), weights.scale)
    # Both operands contain only powers of two, so this broadcast is exact.
    scale_tile = ttnn.multiply(weights.scale_tile, feature_scale)
    outputs = []
    m = len(topology.host_indices)
    output = None
    if m > row_chunk and row_chunk % 32 == 0:
        output = ttnn.allocate_tensor_on_device(ttnn.Shape([m, weights.co]),
            ttnn.float32 if fp32_output else ttnn.bfloat16, ttnn.TILE_LAYOUT, weights.device)
    compute = _compute_config(weights.device)
    for start in range(0, m, row_chunk):
        end = min(start + row_chunk, m)
        key = (start, end)
        ids = topology.gathers.get(key)
        if ids is None:
            host = topology.host_indices[start:end]
            host = torch.where(host < 0, topology.input_points, host)
            ids = ttnn.from_torch(host.reshape(1, -1).to(torch.uint32), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=weights.device)
            topology.gathers[key] = ids

        def gather(x):
            expanded = ttnn.embedding(ids, x, layout=ttnn.ROW_MAJOR_LAYOUT)
            return ttnn.to_layout(ttnn.reshape(expanded, [end - start, weights.volume * cp]), ttnn.TILE_LAYOUT)

        a, b = gather(high), gather(low)

        def product(x, digit):
            return ttnn.matmul(x, weights.digits[digit].weights, dtype=ttnn.float32, compute_kernel_config=compute)

        top = product(a, 0)
        cross_a, cross_b = product(a, 1), product(b, 0)
        cross = ttnn.add(cross_a, cross_b)
        del cross_a, cross_b
        bottom = product(b, 1)
        result = ttnn.add(ttnn.add(top, ttnn.multiply(cross, 1 / 256)), ttnn.multiply(bottom, 1 / 65536))
        del a, b, top, cross, bottom
        result = ttnn.slice(result, [0, 0], [end - start, weights.co])
        result = _scale_and_bias(result, weights, scale, scale_tile)
        if not fp32_output:
            result = ttnn.typecast(result, ttnn.bfloat16)
        if output is None:
            outputs.append(result)
        else:
            _copy_output_rows(result, output, start)
        del result
    del high, low
    return _concat_outputs(outputs) if output is None else output
