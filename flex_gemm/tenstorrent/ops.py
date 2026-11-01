"""Device entry points for the successive sparse GEMM algorithms."""

import torch
from .reference import validate


def _compute_config(device):
    import ttnn
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True,
    )


def _upload(tensor, device):
    import ttnn
    return ttnn.from_torch(tensor.to(torch.bfloat16).contiguous(), device=device,
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def explicit_gemm(feats, weight, neighbor_map, *, device, bias=None):
    """Materialize host im2col, then run BF16 GEMM and optional bias on Tensix.

    Returns a TTNN device tensor [M,Co]. Inputs are CPU tensors; BF16 rounding
    occurs at upload. The explicit expansion and transfers are part of this API.
    """
    import ttnn
    w, indices = validate(feats, weight, neighbor_map, bias)
    m, v = indices.shape
    co, _, ci = w.shape
    if m == 0:
        raise ValueError("device GEMM requires at least one output row")
    expanded = torch.zeros((m, v, ci), dtype=torch.bfloat16)
    valid = indices >= 0
    expanded[valid] = feats[indices[valid]].to(torch.bfloat16)
    a = _upload(expanded.reshape(m, v * ci), device)
    b = _upload(w.permute(1, 2, 0).reshape(v * ci, co), device)
    result = ttnn.matmul(a, b, compute_kernel_config=_compute_config(device))
    if bias is not None:
        result = ttnn.add(result, _upload(bias.reshape(1, co), device))
    return result
