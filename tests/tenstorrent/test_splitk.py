import torch
import pytest
from flex_gemm.tenstorrent import prepare_implicit, sparse_conv_reference
from test_explicit import device


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("m,ci,co,v,splits", [
    (35, 65, 45, 3, 2), (32, 33, 32, 3, 5),
    (1, 128, 1, 27, 7), (65, 17, 65, 3, 3),
])
def test_splitk_device(device, masked, m, ci, co, v, splits):
    import ttnn
    torch.manual_seed(32)
    x = torch.randn(50, ci).bfloat16()
    w = (torch.randn(co, v, ci) / (ci * v) ** 0.5).bfloat16()
    mapping = torch.randint(0, 50, (m, v))
    mapping[torch.rand(m, v) < 0.6] = -1
    if m > 1:
        mapping[0] = -1
    # Offset sparsity makes some split ranges completely empty in masked mode.
    mapping[:, :v // 2] = -1
    bias = torch.randn(co).bfloat16()
    expected = sparse_conv_reference(x, w, mapping, bias)
    p = prepare_implicit(x, w, mapping, device=device, bias=bias, masked=masked)
    for split in (splits, 1, splits):
        y = ttnn.to_torch(p.run(split_k=split)).float()
        torch.testing.assert_close(y, expected, atol=0.025, rtol=0.025)


def test_split_validation(device):
    p = prepare_implicit(torch.ones(1, 1), torch.ones(1, 1, 1), torch.zeros(1, 1, dtype=torch.long),
                         device=device)
    for split in (0, -1, 2, 1.5, True):
        with pytest.raises(ValueError, match="split_k"):
            p.run(split_k=split)


@pytest.mark.parametrize("masked", [False, True])
def test_fp32_partials_preserve_cancellation(device, masked):
    import ttnn
    # The first partial is 257, which rounds to 256 in BF16. Keeping partials
    # in FP32 preserves the residual 1 after the second partial subtracts 256.
    x = torch.zeros(1, 64)
    x[0, 0], x[0, 1], x[0, 32] = 256, 1, -256
    p = prepare_implicit(x, torch.ones(1, 1, 64), torch.tensor([[0]]),
                         device=device, bias=torch.tensor([3.]), masked=masked)
    actual = ttnn.to_torch(p.run(split_k=2)).float()
    torch.testing.assert_close(actual, torch.tensor([[4.]]), atol=0, rtol=0)


@pytest.mark.parametrize("splits", [4, 7])
def test_split_reduction_preserves_low_mantissa_bits(device, splits):
    import ttnn
    from flex_gemm.tenstorrent.implicit import _sum_split_partials
    partials = torch.zeros(splits, 32, 32)
    partials[0], partials[1], partials[2], partials[3] = 257.125, -256, -1, 0.03125
    device_partials = ttnn.from_torch(partials, dtype=ttnn.float32,
                                      layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(_sum_split_partials(device_partials))
    torch.testing.assert_close(actual, torch.full((32, 32), 0.15625), atol=0, rtol=0)
