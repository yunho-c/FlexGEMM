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


@pytest.mark.parametrize("masked", [False, True])
def test_new_addresses_on_cache_hit(device, masked):
    import ttnn
    keep_alive = []
    cache_entries = None
    for i, value in enumerate((1.0, 2.0, -0.5)):
        x = torch.zeros(32, 32)
        x[:, 0] = torch.arange(32) * value
        w = torch.zeros(32, 3, 32)
        w[:, :, 0] = torch.arange(1, 4)
        mapping = torch.arange(32).reshape(32, 1).expand(-1, 3).clone()
        mapping[torch.arange(32), (torch.arange(32) + i) % 3] = -1
        p = prepare_implicit(x, w, mapping, device=device, masked=masked)
        keep_alive.append(p)  # Force distinct input addresses on each invocation.
        y = ttnn.to_torch(p.run())
        torch.testing.assert_close(y.float(), sparse_conv_reference(x, w, mapping), atol=0, rtol=0)
        current = device.num_program_cache_entries()
        if cache_entries is not None:
            assert current == cache_entries
        cache_entries = current
    assert len({p.features.buffer_address() for p in keep_alive}) == 3
