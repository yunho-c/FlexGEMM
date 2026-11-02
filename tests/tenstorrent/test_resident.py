import pytest
import torch
from flex_gemm.tenstorrent import (
    prepare_sparse_weights, prepare_sparse_topology, sparse_conv_device, sparse_conv_reference,
    prepare_sparse_digit_weights,
)
from test_explicit import device


@pytest.mark.parametrize("rows,ci", [(1, 1), (35, 17), (65, 33), (2081, 128), (8193, 128)])
def test_fused_feature_digits_match_unfused(device, rows, ci):
    import ttnn
    from flex_gemm.tenstorrent.digits import _feature_digits
    values = torch.tensor([-128., -127.5, -126.5, -1.5, -.5, -.005859375,
        -.001953125, 0., .001953125, .005859375, .5, 1.5, 126.5, 127.5, 128.])
    pattern = values.repeat((rows * ci + len(values) - 1) // len(values))[:rows * ci].reshape(rows, ci)
    cp = (ci + 31) // 32 * 32
    for x in (pattern, pattern * 2.**-80, pattern * 2.**30, torch.zeros_like(pattern)):
        x = x.bfloat16()
        features = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        expected = _feature_digits(features, cp, fused=False)
        actual = _feature_digits(features, cp)
        for a, b in zip(actual, expected):
            ah, bh = ttnn.to_torch(a), ttnn.to_torch(b)
            bits = torch.int16 if ah.dtype == torch.bfloat16 else torch.int32
            torch.testing.assert_close(ah.view(bits), bh.view(bits), atol=0, rtol=0)
        for table in actual[:2]:
            host = ttnn.to_torch(table)
            assert tuple(host.shape) == (rows + 1, cp)
            assert not bool(host[-1].any())
            assert not bool(host[:, ci:].any())
        torch.testing.assert_close(ttnn.to_torch(features), x, atol=0, rtol=0)
        del expected, actual, features


@pytest.mark.parametrize("ci", [17, 33, 128])
def test_tiled_digit_gather_exact_rows_and_padding(device, ci):
    import ttnn
    from flex_gemm.tenstorrent.gather import tiled_gather
    generator = torch.Generator().manual_seed(111)
    cp = (ci + 31) // 32 * 32
    values = torch.zeros(67, cp, dtype=torch.bfloat16)
    values[:, :ci] = torch.randn(67, ci, generator=generator).bfloat16()
    mapping = torch.randint(-1, 67, (65, 3), generator=generator)
    mapping[0] = -1
    mapping[-1] = torch.tensor([66, -1, 0])
    topology = prepare_sparse_topology(mapping, len(values), device=device)
    features = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    zero_appended = torch.cat([values, torch.zeros(1, cp, dtype=torch.bfloat16)])
    expected = zero_appended[torch.where(mapping < 0, len(values), mapping)].reshape(65, -1)
    for start, end in ((0, 32), (32, 64), (64, 65)):
        actual = tiled_gather(features, topology, start, end, ci)
        torch.testing.assert_close(ttnn.to_torch(actual), expected[start:end], atol=0, rtol=0)


def test_digit_blocked_matmul_ragged_core_boundary_and_reuse(device):
    import ttnn
    from flex_gemm.tenstorrent import prepare_sparse_digit_weights
    from flex_gemm.tenstorrent.digits import matmul_plan
    generator = torch.Generator().manual_seed(113)
    rows, ci, co, volume = 2081, 17, 35, 3
    x = torch.randn(rows, ci, generator=generator).bfloat16()
    weight = (torch.randn(co, volume, ci, generator=generator) / 8).bfloat16()
    bias = torch.randn(co, generator=generator).bfloat16()
    mapping = torch.randint(-1, rows, (rows, volume), generator=generator)
    mapping[0] = -1
    mapping[-1] = torch.tensor([rows - 1, -1, 0])
    prepared = prepare_sparse_digit_weights(weight, device=device, bias=bias)
    topology = prepare_sparse_topology(mapping, rows, device=device)
    grid = device.compute_with_storage_grid_size()
    assert matmul_plan(2080, 96, co, (grid.x, grid.y)) is not None
    for features in (x, (x.float() * .5 + .03125).bfloat16()):
        resident = ttnn.from_torch(features, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        result = sparse_conv_device(resident, prepared, topology,
            algorithm="digit_explicit_gemm", row_chunk=2080)
        expected = sparse_conv_reference(features, weight, mapping, bias)
        torch.testing.assert_close(ttnn.to_torch(result).float(), expected, atol=.025, rtol=.025)


@pytest.mark.parametrize("algorithm", ["explicit_gemm", "digit_explicit_gemm", "implicit_gemm", "masked_implicit_gemm",
                                        "implicit_gemm_splitk", "masked_implicit_gemm_splitk"])
@pytest.mark.parametrize("ci,co", [(32, 64), (17, 35)])
@pytest.mark.parametrize("row_chunk", [17, 32])
def test_resident_chain_and_reuse(device, algorithm, ci, co, row_chunk):
    import ttnn
    torch.manual_seed(73)
    x = torch.randn(37, ci).bfloat16()
    w1 = (torch.randn(co, 3, ci) / (3 * ci)**0.5).bfloat16()
    w2 = (torch.randn(ci, 3, co) / (3 * co)**0.5).bfloat16()
    mapping = torch.randint(-1, 37, (37, 3))
    mapping[0] = -1
    b = torch.randn(co).bfloat16()
    top = prepare_sparse_topology(mapping, 37, device=device, masked=algorithm.startswith("masked"))
    prepare = prepare_sparse_digit_weights if algorithm == "digit_explicit_gemm" else prepare_sparse_weights
    p1 = prepare(w1, device=device, bias=b)
    p2 = prepare(w2, device=device)
    # Keep old operands alive and use changed values on the same cached topology.
    for delta in (0, 0.5):
        xx = (x.float() + delta).bfloat16()
        xd = ttnn.from_torch(xx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        y = sparse_conv_device(xd, p1, top, algorithm=algorithm, row_chunk=row_chunk)
        expected = sparse_conv_reference(xx, w1, mapping, b)
        yh = ttnn.to_torch(y).float()
        torch.testing.assert_close(yh, expected, atol=0.025, rtol=0.025)
        # The second convolution consumes the first device tensor directly.
        z = sparse_conv_device(y, p2, top, algorithm=algorithm, row_chunk=row_chunk)
        expected2 = sparse_conv_reference(yh, w2, mapping)
        torch.testing.assert_close(ttnn.to_torch(z).float(), expected2, atol=0.025, rtol=0.025)


@pytest.mark.parametrize("algorithm", ["explicit_gemm", "digit_explicit_gemm", "implicit_gemm", "masked_implicit_gemm",
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
    prepare = prepare_sparse_digit_weights if algorithm == "digit_explicit_gemm" else prepare_sparse_weights
    weights = prepare(w, device=device)
    xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = sparse_conv_device(xd, weights, top, algorithm=algorithm, split_k=2, fp32_output=True)
    assert y.dtype == ttnn.float32
    torch.testing.assert_close(ttnn.to_torch(ttnn.subtract(y, 256.)), torch.tensor([[1.125]]), atol=0, rtol=0)


@pytest.mark.parametrize("algorithm", ["explicit_gemm", "digit_explicit_gemm", "implicit_gemm", "masked_implicit_gemm",
                                        "implicit_gemm_splitk", "masked_implicit_gemm_splitk"])
def test_fp32_bias_preserves_residual_after_cancellation(device, algorithm):
    import ttnn
    x = torch.zeros(37, 33, dtype=torch.bfloat16)
    x[:, 0], x[:, 1], x[:, 32] = 256., 1., .125
    weight = torch.ones(35, 1, 33, dtype=torch.bfloat16)
    bias = torch.full((35,), -256., dtype=torch.bfloat16)
    top = prepare_sparse_topology(torch.arange(37)[:, None], 37, device=device,
                                   masked=algorithm.startswith("masked"))
    prepare = prepare_sparse_digit_weights if algorithm == "digit_explicit_gemm" else prepare_sparse_weights
    weights = prepare(weight, bias=bias, device=device)
    xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    result = sparse_conv_device(xd, weights, top, algorithm=algorithm, row_chunk=32, fp32_output=True)
    assert result.dtype == ttnn.float32
    torch.testing.assert_close(ttnn.to_torch(result), torch.full((37, 35), 1.125), atol=0, rtol=0)


@pytest.mark.parametrize("fp32_output", [False, True])
@pytest.mark.parametrize("algorithm", ["explicit_gemm", "digit_explicit_gemm"])
@pytest.mark.parametrize("co", [35, 512])
def test_explicit_many_row_chunks(device, fp32_output, algorithm, co):
    import ttnn
    # Seventeen chunks, including a ragged final chunk, reproduced a stalled
    # SDK concat reader in the complete decoder. Bounded integer inputs keep
    # every sum in [-128,128], exactly representable in both output formats,
    # while neighbors cross chunk boundaries.
    generator = torch.Generator().manual_seed(96)
    x = torch.randint(-2, 3, (1032, 32), generator=generator).bfloat16()
    w = torch.randint(-2, 3, (co, 1, 32), generator=generator).bfloat16()
    mapping = ((torch.arange(1032) + 1) % 1032)[:, None]
    top = prepare_sparse_topology(mapping, len(x), device=device)
    prepare = prepare_sparse_digit_weights if algorithm == "digit_explicit_gemm" else prepare_sparse_weights
    weights = prepare(w, device=device)
    xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = sparse_conv_device(xd, weights, top, algorithm=algorithm,
                                row_chunk=64, fp32_output=fp32_output)
    expected = sparse_conv_reference(x, w, mapping)
    if not fp32_output:
        expected = expected.bfloat16().float()
    torch.testing.assert_close(ttnn.to_torch(actual).float(), expected, atol=0, rtol=0)
