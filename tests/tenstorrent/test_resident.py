import pytest
import torch
from flex_gemm.tenstorrent import (
    prepare_sparse_weights, prepare_sparse_topology, sparse_conv_device, sparse_conv_reference,
)
from test_explicit import device


@pytest.mark.parametrize("algorithm", ["explicit_gemm", "implicit_gemm", "masked_implicit_gemm",
                                        "implicit_gemm_splitk", "masked_implicit_gemm_splitk"])
@pytest.mark.parametrize("ci,co", [(32, 64), (17, 35)])
def test_resident_chain_and_reuse(device, algorithm, ci, co):
    import ttnn
    torch.manual_seed(73)
    x = torch.randn(37, ci).bfloat16()
    w1 = (torch.randn(co, 3, ci) / (3 * ci)**0.5).bfloat16()
    w2 = (torch.randn(ci, 3, co) / (3 * co)**0.5).bfloat16()
    mapping = torch.randint(-1, 37, (37, 3))
    mapping[0] = -1
    b = torch.randn(co).bfloat16()
    top = prepare_sparse_topology(mapping, 37, device=device, masked=algorithm.startswith("masked"))
    p1 = prepare_sparse_weights(w1, device=device, bias=b)
    p2 = prepare_sparse_weights(w2, device=device)
    # Keep old operands alive and use changed values on the same cached topology.
    for delta in (0, 0.5):
        xx = (x.float() + delta).bfloat16()
        xd = ttnn.from_torch(xx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        y = sparse_conv_device(xd, p1, top, algorithm=algorithm, row_chunk=17)
        expected = sparse_conv_reference(xx, w1, mapping, b)
        yh = ttnn.to_torch(y).float()
        torch.testing.assert_close(yh, expected, atol=0.025, rtol=0.025)
        # The second convolution consumes the first device tensor directly.
        z = sparse_conv_device(y, p2, top, algorithm=algorithm, row_chunk=17)
        expected2 = sparse_conv_reference(yh, w2, mapping)
        torch.testing.assert_close(ttnn.to_torch(z).float(), expected2, atol=0.025, rtol=0.025)


@pytest.mark.parametrize("algorithm", ["explicit_gemm", "implicit_gemm", "masked_implicit_gemm",
                                        "implicit_gemm_splitk", "masked_implicit_gemm_splitk"])
def test_fp32_output_preserves_small_residual(device, algorithm):
    import ttnn
    # A BF16 output would round 257.125 before the later subtraction. Put
    # the fractional term across the channel-tile split boundary as well.
    x = torch.zeros(1, 33).bfloat16()
    x[0, 0], x[0, 1], x[0, 32] = 256., 1., 0.125
    w = torch.ones(1, 1, 33).bfloat16()
    top = prepare_sparse_topology(torch.tensor([[0]]), 1, device=device,
                                   masked=algorithm.startswith("masked"))
    weights = prepare_sparse_weights(w, device=device)
    xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = sparse_conv_device(xd, weights, top, algorithm=algorithm, split_k=2, fp32_output=True)
    assert y.dtype == ttnn.float32
    torch.testing.assert_close(ttnn.to_torch(ttnn.subtract(y, 256.)), torch.tensor([[1.125]]), atol=0, rtol=0)
