import torch
import pytest
from flex_gemm.tenstorrent import implicit_gemm, prepare_implicit, sparse_conv_reference
from test_explicit import device


@pytest.mark.parametrize("m,n,ci,co,v,bias", [
    (32, 32, 32, 32, 1, False), (35, 41, 17, 45, 3, True),
    (65, 70, 33, 64, 27, False), (1, 1, 1, 1, 1, True),
    (33, 0, 17, 35, 3, True), (257, 100, 64, 257, 3, False),
])
def test_implicit_device(device, m, n, ci, co, v, bias):
    import ttnn
    torch.manual_seed(12)
    x = torch.randn(n, ci).bfloat16()
    w = (torch.randn(co, v, ci) / (v * ci) ** 0.5).bfloat16()
    mapping = torch.randint(-1, max(0, n), (m, v)) if n else torch.full((m, v), -1)
    mapping[0] = -1
    b = torch.randn(co).bfloat16() if bias else None
    expected = sparse_conv_reference(x, w, mapping, b)
    p = prepare_implicit(x, w, mapping, device=device, bias=b)
    for _ in range(2):
        actual = ttnn.to_torch(p.run()).float()
        torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)


def test_new_addresses_on_cache_hit(device):
    import ttnn
    for value in (1.0, 2.0, -0.5):
        x = torch.full((32, 32), value)
        w = torch.ones(32, 1, 32)
        mapping = torch.arange(32).reshape(32, 1)
        y = ttnn.to_torch(implicit_gemm(x, w, mapping, device=device))
        torch.testing.assert_close(y.float(), torch.full((32, 32), value * 32))
