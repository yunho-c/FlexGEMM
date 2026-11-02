"""Experimental four-product sparse convolution using signed base-256 digits.

BF16 operands are approximated with two small-integer digits. The ordinary
BF16 reference and tolerances still apply; this is not an FP32-exact mode.
"""

from dataclasses import dataclass

import torch

from .implicit import _round32
from .ops import _add_bias, _compute_config


@dataclass
class SparseDigitWeights:
    digits: tuple
    scale: object
    bias: object
    device: object
    ci: int
    co: int
    volume: int


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
    first = digits[0]
    return SparseDigitWeights(digits, sd, bd, device, first.ci, first.co, first.volume)


def _concat(chunks):
    """Bound concat fan-in and release each previous level as it is replaced."""
    import ttnn
    while len(chunks) > 1:
        level = [ttnn.concat(chunks[i:i + 4], dim=0) if len(chunks[i:i + 4]) > 1
                 else chunks[i] for i in range(0, len(chunks), 4)]
        chunks.clear()
        chunks = level
    return chunks[0]


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
        residual = ttnn.multiply(part, ttnn.repeat(inverse, [end - start, ci]))
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


def sparse_digit_conv_device(features, weights, topology, *, row_chunk=256, fp32_output=False):
    """Fresh device digit extraction, gather, four products, scale and bias."""
    import ttnn
    if not isinstance(weights, SparseDigitWeights) or topology.masks is not None:
        raise ValueError("digit convolution requires digit weights and unmasked topology")
    cp = _round32(weights.ci)
    high, low, feature_scale = _feature_digits(features, cp)
    scale = ttnn.multiply(ttnn.repeat(feature_scale, [1, weights.co]), weights.scale)
    outputs = []
    m = len(topology.host_indices)
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
        result = ttnn.multiply(result, ttnn.repeat(scale, [end - start, 1]))
        result = _add_bias(result, weights.bias, fp32_output=True)
        outputs.append(result if fp32_output else ttnn.typecast(result, ttnn.bfloat16))
        del result
    del high, low
    return _concat(outputs)
