"""Device entry points for the successive sparse GEMM algorithms."""

import torch
from dataclasses import dataclass
from .reference import validate


def _matmul_fidelity(device):
    import ttnn
    # Wormhole HiFi4 + FP32 destination accumulation can produce rare, large
    # errors (tt-metal#38306). This affects BF16 inputs too. Keep FP32 accumulation
    # but use the vendor-recommended HiFi3 workaround on this architecture.
    return (ttnn.MathFidelity.HiFi3 if device.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.MathFidelity.HiFi4)


def _compute_config(device):
    import ttnn
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=_matmul_fidelity(device),
        math_approx_mode=False, fp32_dest_acc_en=True,
        # The pretrained texture projection reproduced a ~-4 outlier with
        # HiFi3 + packer L1 accumulation too. Disable that path on Wormhole.
        packer_l1_acc=device.arch() != ttnn.Arch.WORMHOLE_B0,
    )


def _upload(tensor, device):
    import ttnn
    return ttnn.from_torch(tensor.to(torch.bfloat16).contiguous(), device=device,
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


@dataclass
class PreparedExplicitConv:
    expanded: object
    weights: object
    bias: object
    device: object

    def run(self):
        import ttnn
        result = ttnn.matmul(self.expanded, self.weights,
                             compute_kernel_config=_compute_config(self.device))
        if self.bias is not None:
            result = ttnn.add(result, self.bias)
        return result


def prepare_explicit(feats, weight, neighbor_map, *, device, bias=None):
    """Materialize host im2col and upload reusable BF16 device operands.

    Inputs are CPU tensors; BF16 rounding occurs at upload.
    """
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
    bdev = None if bias is None else _upload(bias.reshape(1, co), device)
    return PreparedExplicitConv(a, b, bdev, device)


def explicit_gemm(feats, weight, neighbor_map, *, device, bias=None):
    """Host im2col and upload, then Tensix GEMM/bias; returns device [M,Co]."""
    return prepare_explicit(feats, weight, neighbor_map, device=device, bias=bias).run()
