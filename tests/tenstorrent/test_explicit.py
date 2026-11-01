import sys
import torch
import pytest
from flex_gemm.tenstorrent import build_subm_neighbor_map, sparse_conv_reference, explicit_gemm


def test_import_does_not_load_cuda_or_triton():
    assert "flex_gemm.kernels" not in sys.modules
    assert "triton" not in sys.modules


def test_neighbors_against_dense_convolution():
    torch.manual_seed(4)
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 1, 1], [1, 1, 0, 0]])
    x = torch.randn(4, 3)
    w = torch.randn(5, 3, 3, 3, 3)
    mapping = build_subm_neighbor_map(coords, (2, 3, 2, 2))
    dense = torch.zeros(2, 3, 3, 2, 2)
    for c, f in zip(coords, x):
        b, i, j, k = c
        dense[b, :, i, j, k] = f
    y = torch.nn.functional.conv3d(dense, w.permute(0, 4, 1, 2, 3), padding=1)
    expected = torch.stack([y[b, :, i, j, k] for b, i, j, k in coords])
    torch.testing.assert_close(sparse_conv_reference(x, w, mapping), expected)


@pytest.fixture(scope="module")
def device():
    ttnn = pytest.importorskip("ttnn")
    if ttnn.GetNumAvailableDevices() == 0:
        pytest.skip("no Tenstorrent device")
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


@pytest.mark.parametrize("m,n,ci,co,v,bias", [
    (32, 32, 32, 32, 1, False), (35, 41, 17, 45, 3, True),
    (65, 70, 33, 64, 27, False), (1, 1, 1, 1, 1, True),
])
def test_explicit_device(device, m, n, ci, co, v, bias):
    import ttnn
    torch.manual_seed(10)
    x = torch.randn(n, ci).bfloat16()
    w = (torch.randn(co, v, ci) / (ci * v) ** 0.5).bfloat16()
    mapping = torch.randint(-1, n, (m, v))
    mapping[0] = -1
    b = torch.randn(co).bfloat16() if bias else None
    expected = sparse_conv_reference(x, w, mapping, b)
    actual = ttnn.to_torch(explicit_gemm(x, w, mapping, device=device, bias=b)).float()
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)


def test_invalid_map():
    with pytest.raises(ValueError, match="out of range"):
        sparse_conv_reference(torch.ones(1, 2), torch.ones(3, 1, 2), torch.tensor([[1]]))
