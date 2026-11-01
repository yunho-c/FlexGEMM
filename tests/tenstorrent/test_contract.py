import torch
import pytest
from flex_gemm.tenstorrent import (
    prepare_implicit, sparse_conv_reference, build_subm_neighbor_map, sparse_submanifold_conv3d,
)
from test_explicit import device


@pytest.mark.parametrize("masked", [False, True])
def test_uint32_sentinel_and_weight_order(device, masked):
    import ttnn
    x = torch.tensor([[1., 2.], [4., 8.]])
    w = torch.arange(1., 13.).reshape(2, 3, 1, 1, 2)
    mapping = torch.tensor([[1, 0xffffffff, 0], [0xffffffff, 0xffffffff, 0xffffffff]], dtype=torch.uint32)
    expected = torch.stack((x[1] @ w[:, 0, 0, 0, :].T + x[0] @ w[:, 2, 0, 0, :].T,
                             torch.zeros(2)))
    p = prepare_implicit(x, w, mapping, device=device, masked=masked)
    for split in (1, 2, 3):
        result = ttnn.to_torch(p.run(split_k=split)).float()
        torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_validation_before_device_access(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "ttnn", None)
    x = torch.ones(2, 3)
    w = torch.ones(4, 1, 3)
    for bad in (torch.tensor([[-2]]), torch.tensor([[2]]), torch.tensor([[0.5]])):
        with pytest.raises(ValueError):
            prepare_implicit(x, w, bad, device=None)
    with pytest.raises(ValueError, match="autograd"):
        sparse_conv_reference(x.requires_grad_(), w, torch.tensor([[0]]))
    with pytest.raises(ValueError, match="duplicate"):
        build_subm_neighbor_map(torch.zeros(2, 4, dtype=torch.long), (1, 2, 2, 2))
    with pytest.raises(ValueError, match="outside"):
        build_subm_neighbor_map(torch.tensor([[0, -1, 0, 0]]), (1, 2, 2, 2))


def test_default_split_with_one_reduction_tile(device):
    import ttnn
    y, _ = sparse_submanifold_conv3d(torch.ones(1, 1), torch.zeros(1, 4, dtype=torch.long),
                                     (1, 1, 1, 1, 1), torch.ones(1, 1, 1, 1, 1), device=device)
    torch.testing.assert_close(ttnn.to_torch(y).float(), torch.ones(1, 1), atol=0, rtol=0)
