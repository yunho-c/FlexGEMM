"""Reusable weights/topology for convolutions of device-resident features.

Only integer topology is prepared on the host. Feature gather, convolution,
layout conversion, reduction and bias execute on the device. Explicit GEMM
expands bounded output-row chunks on device, retaining all input neighbors.
"""

from dataclasses import dataclass, field
import torch
from .implicit import PreparedSparseConv, _round32
from .ops import _compute_config, _upload, _add_bias


@dataclass
class SparseWeights:
    weights: object
    bias: object
    device: object
    ci: int
    co: int
    volume: int


@dataclass
class SparseTopology:
    indices: object
    device: object
    host_indices: torch.Tensor
    input_points: int
    masks: object = None
    permutation: object = None
    schedule: object = None
    gathers: dict = field(default_factory=dict)
    digit_plans: dict = field(default_factory=dict)


def prepare_sparse_weights(weight, *, device, bias=None):
    """Pack CPU [Co,V,Ci] / [Co,Kx,Ky,Kz,Ci] weights once for many forwards."""
    if weight.device.type != "cpu" or weight.ndim not in (3, 5) or not weight.is_floating_point():
        raise ValueError("expected floating-point CPU sparse weights")
    if weight.requires_grad or (bias is not None and bias.requires_grad):
        raise ValueError("resident sparse convolution does not implement autograd")
    co, ci = weight.shape[0], weight.shape[-1]
    if min(co, ci) <= 0 or weight.numel() == 0:
        raise ValueError("weight dimensions must be positive")
    w = weight.reshape(co, -1, ci)
    if bias is not None and (bias.device.type != "cpu" or bias.shape != (co,) or not bias.is_floating_point()):
        raise ValueError("bias must be a floating-point CPU [Co] tensor")
    cp, op = _round32(ci), _round32(co)
    padded = torch.zeros((w.shape[1], cp, op), dtype=torch.bfloat16)
    padded[:, :ci, :co] = w.permute(1, 2, 0)
    return SparseWeights(_upload(padded.reshape(-1, op), device),
                         None if bias is None else _upload(bias.reshape(1, co), device),
                         device, ci, co, w.shape[1])


def prepare_sparse_topology(neighbor_map, input_points, *, device, masked=False):
    """Upload integer metadata once for a fixed input/output coordinate layout.

    Cache this object by the actual topology, not just row count. Empty output
    sets are handled by the caller without launching a device convolution.
    """
    import ttnn
    if (neighbor_map.device.type != "cpu" or neighbor_map.ndim != 2
            or neighbor_map.dtype not in (torch.int32, torch.int64, torch.uint32)):
        raise ValueError("neighbor map must be a CPU integer [M,V] tensor")
    if not isinstance(input_points, int) or isinstance(input_points, bool) or input_points < 0:
        raise ValueError("input_points must be a nonnegative integer")
    indices = neighbor_map.to(torch.int64).clone()
    indices[indices == 0xffffffff] = -1
    if min(indices.shape) <= 0:
        raise ValueError("device topology requires positive output rows and kernel volume")
    if torch.any((indices < -1) | (indices >= input_points)):
        raise ValueError("neighbor index is out of range")
    p = SparseTopology(None, device, indices, input_points)
    if masked:
        from .masking import make_mask_schedule
        p.schedule = make_mask_schedule(indices)
        indices = p.schedule.indices
    m, volume = indices.shape
    mp = _round32(m)
    maps = torch.full((volume, mp), 0xffffffff, dtype=torch.int64)
    maps[:, :m] = torch.where(indices.T < 0, 0xffffffff, indices.T)
    p.indices = ttnn.from_torch(maps.to(torch.uint32), dtype=ttnn.uint32,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    if masked:
        mask_rows = torch.zeros((mp // 32, 8), dtype=torch.uint32)
        mask_rows[:, 0] = p.schedule.block_masks.to(torch.uint32)
        p.masks = ttnn.from_torch(mask_rows, dtype=ttnn.uint32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        p.permutation = ttnn.from_torch(p.schedule.permutation.reshape(1, mp).to(torch.uint32),
                                        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return p


def sparse_conv_device(features, weights, topology, *, algorithm="implicit_gemm",
                       split_k=2, row_chunk=256, fp32_output=False):
    """Convolve resident BF16 features using resident weights and cached topology.

    Returns tiled BF16 [M,Co], or FP32 when requested. Device row chunks change output ownership only:
    each chunk can access every input row, including neighbors outside it.
    """
    import ttnn
    choices = {"explicit_gemm", "digit_explicit_gemm", "digit_implicit_gemm", "implicit_gemm", "masked_implicit_gemm",
               "implicit_gemm_splitk", "masked_implicit_gemm_splitk"}
    if algorithm not in choices:
        raise ValueError(f"unknown Tenstorrent algorithm: {algorithm}")
    if features.dtype != ttnn.bfloat16 or tuple(features.shape) != (topology.input_points, weights.ci):
        raise ValueError("features must be a BF16 device [input_points,Ci] tensor")
    if (features.device() != weights.device or topology.device != weights.device
            or topology.host_indices.shape[1] != weights.volume):
        raise ValueError("features, weights and topology must share a device and kernel volume")
    if not isinstance(row_chunk, int) or isinstance(row_chunk, bool) or row_chunk < 1:
        raise ValueError("row_chunk must be a positive integer")
    masked = algorithm.startswith("masked")
    if masked and topology.masks is None:
        raise ValueError("masked algorithms require a masked topology")
    if not masked and topology.masks is not None and algorithm != "explicit_gemm":
        raise ValueError("unmasked algorithms require an unmasked topology")
    if algorithm == "digit_explicit_gemm":
        from .digits import sparse_digit_conv_device
        return sparse_digit_conv_device(features, weights, topology, row_chunk=row_chunk, fp32_output=fp32_output)
    if algorithm == "digit_implicit_gemm":
        from .digit_implicit import sparse_digit_implicit_conv_device
        return sparse_digit_implicit_conv_device(features, weights, topology,
            row_chunk=row_chunk, fp32_output=fp32_output)
    cp = _round32(weights.ci)
    x = ttnn.to_layout(features, ttnn.ROW_MAJOR_LAYOUT)
    if cp != weights.ci:
        x = ttnn.pad(x, [(0, 0), (0, cp - weights.ci)], 0)
    m = topology.host_indices.shape[0]
    if algorithm == "explicit_gemm":
        # Missing neighbors read a real zero row. Per-offset channel padding
        # matches the shared packed weight matrix, including ragged Ci.
        x = ttnn.pad(x, [(0, 1), (0, 0)], 0)
        outputs = []
        for start in range(0, m, row_chunk):
            end = min(m, start + row_chunk)
            key = (start, end)
            ids = topology.gathers.get(key)
            if ids is None:
                host = topology.host_indices[start:end]
                host = torch.where(host < 0, topology.input_points, host)
                ids = ttnn.from_torch(host.reshape(1, -1).to(torch.uint32), dtype=ttnn.uint32,
                                       layout=ttnn.ROW_MAJOR_LAYOUT, device=weights.device)
                topology.gathers[key] = ids
            expanded = ttnn.embedding(ids, x, layout=ttnn.ROW_MAJOR_LAYOUT)
            expanded = ttnn.reshape(expanded, [end - start, weights.volume * cp])
            expanded = ttnn.to_layout(expanded, ttnn.TILE_LAYOUT)
            outputs.append(ttnn.matmul(expanded, weights.weights,
                                         dtype=ttnn.float32 if fp32_output else ttnn.bfloat16,
                                         compute_kernel_config=_compute_config(weights.device)))
        # The pinned Wormhole SDK stalled in row-major concat's NOC reader
        # when handed 17 chunks. A bounded tree avoids the many-input path
        # while retaining all output rows and logarithmic concat depth.
        while len(outputs) > 1:
            outputs = [ttnn.concat(outputs[i:i + 4], dim=0) if len(outputs[i:i + 4]) > 1
                       else outputs[i] for i in range(0, len(outputs), 4)]
        result = outputs[0]
        result = ttnn.slice(result, [0, 0], [m, weights.co])
        return _add_bias(result, weights.bias, fp32_output=fp32_output)
    p = PreparedSparseConv(x, weights.weights, topology.indices, weights.bias,
                           weights.device, m, weights.ci, weights.co, weights.volume,
                           topology.masks, topology.permutation, topology.schedule)
    return p.run(split_k=split_k if algorithm.endswith("splitk") else 1, fp32_output=fp32_output)
