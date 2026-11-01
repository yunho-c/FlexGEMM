"""Host topology preparation and an independent sparse-convolution oracle."""

from itertools import product
import math
import operator
import torch


def validate(feats, weight, neighbor_map, bias=None):
    """Normalize FlexGEMM's uint32 sentinel to -1 and validate host inputs.

    Weights are [Co, V, Ci] or [Co, Kw, Kh, Kd, Ci]; the map is [M, V].
    M may differ from the number of input points, allowing arbitrary sparse maps.
    """
    tensors = [feats, weight, neighbor_map] + ([] if bias is None else [bias])
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("preparation expects CPU tensors")
    if any(t.requires_grad for t in tensors):
        raise ValueError("device forward API does not implement autograd")
    if feats.ndim != 2 or weight.ndim not in (3, 5) or neighbor_map.ndim != 2:
        raise ValueError("expected features [N,Ci], weights [Co,V,Ci] or 5D, map [M,V]")
    if not feats.is_floating_point() or not weight.is_floating_point():
        raise ValueError("features and weights must be floating point")
    if neighbor_map.dtype not in (torch.int32, torch.int64, torch.uint32):
        raise ValueError("neighbor map must contain integer indices")
    co, ci = weight.shape[0], weight.shape[-1]
    if ci <= 0 or co <= 0 or feats.shape[1] != ci:
        raise ValueError("channel dimensions must be positive and agree")
    w = weight.reshape(co, -1, ci)
    if w.shape[1] == 0 or neighbor_map.shape[1] != w.shape[1]:
        raise ValueError("kernel volume must be positive and match the neighbor map")
    indices = neighbor_map.to(torch.int64)
    indices = torch.where(indices == 0xffffffff, -1, indices)
    if torch.any((indices < -1) | (indices >= feats.shape[0])):
        raise ValueError("neighbor index is out of range (use -1 or 0xffffffff for missing)")
    if bias is not None and (bias.shape != (co,) or not bias.is_floating_point()):
        raise ValueError("bias must be floating point with shape [Co]")
    return w, indices


def build_subm_neighbor_map(coords, spatial_shape, kernel_size=(3, 3, 3), dilation=(1, 1, 1)):
    """Build a CPU int64 [N,V] map in (b,x,y,z), x/y/z kernel order.

    spatial_shape is (batch, width, height, depth). Odd kernels preserve centers.
    Host preprocessing is intentionally separate from device GEMM timing.
    """
    if coords.device.type != "cpu" or coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("coords must be a CPU [N,4] tensor")
    if coords.dtype not in (torch.int32, torch.int64):
        raise ValueError("coords must be int32 or int64")
    if len(spatial_shape) != 4 or any(s <= 0 for s in spatial_shape):
        raise ValueError("spatial_shape must contain four positive sizes")
    if len(kernel_size) != 3 or any(k <= 0 or k % 2 != 1 for k in kernel_size):
        raise ValueError("kernel_size must contain three positive odd sizes")
    if len(dilation) != 3 or any(d <= 0 for d in dilation):
        raise ValueError("dilation must contain three positive sizes")
    spatial_shape = tuple(map(operator.index, spatial_shape))
    kernel_size = tuple(map(operator.index, kernel_size))
    dilation = tuple(map(operator.index, dilation))
    offsets = list(product(*(range(-(k // 2), k // 2 + 1) for k in kernel_size)))
    if math.prod(spatial_shape) <= torch.iinfo(torch.int64).max:
        # Integer lattice keys preserve exact coordinate equality. Batch the
        # lookups instead of issuing N*V scalar Tensor writes on the host.
        values = coords.to(torch.int64)
        if torch.any(values < 0) or torch.any(values >= torch.tensor(spatial_shape)):
            raise ValueError("coordinate outside spatial_shape")
        keys = values[:, 0]
        for axis in range(1, 4):
            keys = keys * spatial_shape[axis] + values[:, axis]
        sorted_keys, order = keys.sort()
        if torch.any(sorted_keys[1:] == sorted_keys[:-1]):
            raise ValueError("duplicate sparse coordinates")
        result = torch.full((len(coords), len(offsets)), -1, dtype=torch.int64)
        if not len(coords):
            return result
        for v, offset in enumerate(offsets):
            delta = tuple(o * d for o, d in zip(offset, dilation))
            if any(abs(d) >= size for d, size in zip(delta, spatial_shape[1:])):
                continue
            valid = torch.ones(len(coords), dtype=torch.bool)
            for axis, d in enumerate(delta, start=1):
                if d > 0:
                    valid &= values[:, axis] < spatial_shape[axis] - d
                elif d < 0:
                    valid &= values[:, axis] >= -d
            rows = valid.nonzero().flatten()
            shift = (delta[0] * spatial_shape[2] + delta[1]) * spatial_shape[3] + delta[2]
            candidates = keys[rows] + shift
            positions = torch.searchsorted(sorted_keys, candidates)
            found = (positions < len(coords)) & (sorted_keys[positions.clamp_max(len(coords) - 1)] == candidates)
            result[rows[found], v] = order[positions[found]]
        return result
    # Very large logical lattices can overflow int64 keys. Keep exact Python
    # integer coordinate lookup for those uncommon cases.
    rows = [tuple(row) for row in coords.tolist()]
    if len(set(rows)) != len(rows):
        raise ValueError("duplicate sparse coordinates")
    if any(any(c < 0 or c >= s for c, s in zip(row, spatial_shape)) for row in rows):
        raise ValueError("coordinate outside spatial_shape")
    lookup = {row: i for i, row in enumerate(rows)}
    result = torch.full((len(rows), len(offsets)), -1, dtype=torch.int64)
    for i, (b, x, y, z) in enumerate(rows):
        for v, offset in enumerate(offsets):
            q = (b,) + tuple(c + o * d for c, o, d in zip((x, y, z), offset, dilation))
            result[i, v] = lookup.get(q, -1)
    return result


def sparse_conv_reference(feats, weight, neighbor_map, bias=None):
    """FP32 per-offset gather/add oracle; never constructs an im2col matrix."""
    w, indices = validate(feats, weight, neighbor_map, bias)
    out = torch.zeros((indices.shape[0], w.shape[0]), dtype=torch.float32)
    for v in range(w.shape[1]):
        valid = indices[:, v] >= 0
        out[valid] += feats[indices[valid, v]].float() @ w[:, v, :].float().T
    if bias is not None:
        out += bias.float()
    return out
