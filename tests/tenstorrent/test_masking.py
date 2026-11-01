import torch
import pytest
from flex_gemm.tenstorrent import prepare_implicit, sparse_conv_reference
from flex_gemm.tenstorrent.masking import make_mask_schedule
from test_explicit import device


def test_gray_schedule():
    # Interleaved patterns become three homogeneous tiles (including all-empty).
    mapping = torch.full((96, 3), -1)
    mapping[::3, 0] = 0
    mapping[1::3, 2] = 1
    s = make_mask_schedule(mapping)
    assert s.active_offsets == 2
    assert s.dense_offsets == 9
    assert s.block_masks.tolist() == [0, 1, 4]
    assert sorted(s.permutation.tolist()) == list(range(96))
    highest_bit = torch.full((2, 32), -1)
    highest_bit[1, 31] = 0
    assert make_mask_schedule(highest_bit).permutation[:2].tolist() == [1, 0]


@pytest.mark.parametrize("m,ci,co,v,mode", [
    (96, 17, 45, 3, "grouped"), (35, 33, 65, 27, "random"),
    (64, 32, 32, 32, "last_bit"), (33, 1, 1, 3, "empty"),
    (257, 33, 257, 3, "random"),
])
def test_masked_device(device, m, ci, co, v, mode):
    import ttnn
    torch.manual_seed(24)
    x = torch.randn(71, ci).bfloat16()
    w = (torch.randn(co, v, ci) / (ci * v) ** 0.5).bfloat16()
    mapping = torch.full((m, v), -1)
    if mode == "grouped":
        mapping[::3, 0] = torch.arange(32)
        mapping[1::3, 2] = torch.arange(32) + 2
    elif mode == "random":
        mapping = torch.randint(-1, 71, (m, v))
        mapping[torch.rand(m, v) < 0.8] = -1
    elif mode == "last_bit":
        mapping[:, -1] = torch.arange(m)
    bias = torch.randn(co).bfloat16()
    expected = sparse_conv_reference(x, w, mapping, bias)
    p = prepare_implicit(x, w, mapping, device=device, bias=bias, masked=True)
    for _ in range(2):
        actual = ttnn.to_torch(p.run()).float()
        torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)


def test_mask_volume_guard():
    with pytest.raises(ValueError, match="volume"):
        make_mask_schedule(torch.zeros(32, 33, dtype=torch.long))
