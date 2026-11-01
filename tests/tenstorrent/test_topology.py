"""Host neighbor-map parity, including lattice boundaries and integer overflow."""

from itertools import product
import pytest
import torch

from flex_gemm.tenstorrent.reference import build_subm_neighbor_map


def oracle(coords, kernel, dilation):
    rows = [tuple(row) for row in coords.tolist()]
    lookup = {row: i for i, row in enumerate(rows)}
    offsets = list(product(*(range(-(k // 2), k // 2 + 1) for k in kernel)))
    return torch.tensor([[lookup.get((row[0], *(c + d * o for c, d, o in
        zip(row[1:], dilation, offset))), -1) for offset in offsets] for row in rows],
        dtype=torch.int64).reshape(len(rows), len(offsets))


@pytest.mark.parametrize("kernel,dilation", [((3, 3, 3), (1, 1, 1)),
    ((1, 3, 5), (1, 2, 3)), ((3, 1, 3), (20, 1, 2))])
def test_batched_lookup_matches_coordinate_oracle(kernel, dilation):
    generator = torch.Generator().manual_seed(421)
    grid = torch.tensor(list(product(range(2), range(7), range(8), range(9))), dtype=torch.int32)
    coords = grid[torch.randperm(len(grid), generator=generator)[:457]]
    actual = build_subm_neighbor_map(coords, (2, 7, 8, 9), kernel, dilation)
    torch.testing.assert_close(actual, oracle(coords, kernel, dilation), rtol=0, atol=0)


@pytest.mark.parametrize("spatial", [(1, 2**31, 1, 1), (2, 2**32, 2**32, 2)])
def test_large_integer_coordinates_and_overflow_fallback(spatial):
    coords = torch.tensor([[0, 2**31 - 2, 0, 0], [0, 2**31 - 1, 0, 0]], dtype=torch.int64)
    torch.testing.assert_close(build_subm_neighbor_map(coords, spatial),
        oracle(coords, (3, 3, 3), (1, 1, 1)), rtol=0, atol=0)


def test_empty_and_invalid_coordinates():
    assert build_subm_neighbor_map(torch.empty(0, 4, dtype=torch.int32), (1, 2, 2, 2)).shape == (0, 27)
    with pytest.raises(ValueError, match="duplicate"):
        build_subm_neighbor_map(torch.zeros(2, 4, dtype=torch.int32), (1, 2, 2, 2))
    for row in ([0, -1, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0]):
        with pytest.raises(ValueError, match="outside"):
            build_subm_neighbor_map(torch.tensor([row]), (1, 2, 2, 2))
