"""Coordinate-facing submanifold convolution using the four GEMM stages."""

from .reference import build_subm_neighbor_map
from .ops import explicit_gemm
from .implicit import prepare_implicit


def sparse_submanifold_conv3d(feats, coords, shape, weight, bias=None, *, device,
                              algorithm="masked_implicit_gemm_splitk", split_k=None,
                              dilation=(1, 1, 1), neighbor_map=None):
    """Return (TTNN output features, CPU neighbor map) in original voxel order.

    Like upstream FlexGEMM, shape is [batch,Ci,width,height,depth], coordinates
    are [N,4] in bxyz order, and weight is [Co,Kw,Kh,Kd,Ci]. Reuse neighbor_map
    only when coordinates, kernel size, spatial shape, and dilation are unchanged.
    This is an inference API; autograd inputs are rejected by preparation.
    An omitted split_k uses at most two nonempty reduction slices.
    """
    algorithms = {"explicit_gemm", "implicit_gemm", "masked_implicit_gemm",
                  "implicit_gemm_splitk", "masked_implicit_gemm_splitk"}
    if algorithm not in algorithms:
        raise ValueError(f"unknown Tenstorrent algorithm: {algorithm}")
    if len(shape) != 5 or weight.ndim != 5 or shape[1] != feats.shape[1]:
        raise ValueError("expected shape [batch,Ci,W,H,D] and weight [Co,Kw,Kh,Kd,Ci]")
    if coords.ndim != 2 or coords.shape != (feats.shape[0], 4):
        raise ValueError("coords must have one [b,x,y,z] row per input feature")
    if neighbor_map is None:
        neighbor_map = build_subm_neighbor_map(coords, (shape[0], *shape[2:]), weight.shape[1:4], dilation)
    if neighbor_map.shape[0] != feats.shape[0]:
        raise ValueError("submanifold neighbor map must have one row per input point")
    if algorithm == "explicit_gemm":
        y = explicit_gemm(feats, weight, neighbor_map, device=device, bias=bias)
    else:
        p = prepare_implicit(feats, weight, neighbor_map, device=device, bias=bias,
                              masked=algorithm.startswith("masked"))
        if split_k is None:
            split_k = min(2, p.volume * ((p.ci + 31) // 32))
        y = p.run(split_k=split_k if algorithm.endswith("splitk") else 1)
    return y, neighbor_map
