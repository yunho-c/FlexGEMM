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
