import torch
import pytest
from flex_gemm.tenstorrent import sparse_submanifold_conv3d
from test_explicit import device


@pytest.mark.parametrize("algorithm", ["explicit_gemm", "implicit_gemm", "masked_implicit_gemm",
                                       "implicit_gemm_splitk", "masked_implicit_gemm_splitk"])
def test_coordinate_api_against_dense(device, algorithm):
    import ttnn
    torch.manual_seed(16)
    coords = torch.tensor([[0, 0, 0, 0], [0, 2, 0, 0], [0, 2, 1, 1], [1, 2, 0, 0]])
    x = torch.randn(4, 17).bfloat16()
    w = (torch.randn(19, 3, 3, 3, 17) / 20).bfloat16()
    bias = torch.randn(19).bfloat16()
    dense = torch.zeros(2, 17, 4, 3, 2)
    for c, f in zip(coords, x):
        b, i, j, k = c
        dense[b, :, i, j, k] = f.float()
    golden = torch.nn.functional.conv3d(dense, w.float().permute(0, 4, 1, 2, 3),
                                         bias.float(), padding=(2, 1, 1), dilation=(2, 1, 1))
    expected = torch.stack([golden[b, :, i, j, k] for b, i, j, k in coords])
    cache = None
    for _ in range(2):
        y, cache = sparse_submanifold_conv3d(x, coords, dense.shape, w, bias, device=device,
                                             algorithm=algorithm, split_k=5, dilation=(2, 1, 1),
                                             neighbor_map=cache)
        torch.testing.assert_close(ttnn.to_torch(y).float(), expected, atol=0.025, rtol=0.025)
