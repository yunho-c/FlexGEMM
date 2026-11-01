"""Gray-order grouping and per-32-row kernel-offset masks."""

from dataclasses import dataclass
import torch


@dataclass
class MaskSchedule:
    permutation: torch.Tensor  # sorted row -> original row, including padding
    indices: torch.Tensor
    block_masks: torch.Tensor
    active_offsets: int
    dense_offsets: int


def make_mask_schedule(indices):
    """Sort by inverse Gray code, as in FlexGEMM, then OR masks per tile row."""
    m, volume = indices.shape
    if not 1 <= volume <= 32:
        raise ValueError("masked GEMM requires kernel volume in [1,32]")
    bits = ((indices >= 0).long() << torch.arange(volume)).sum(1)
    order_key = bits.clone()
    for shift in (1, 2, 4, 8, 16):
        order_key ^= order_key >> shift
    order = torch.argsort(order_key, stable=True)
    mp = (m + 31) // 32 * 32
    permutation = torch.cat((order, torch.arange(m, mp)))
    padded_bits = torch.zeros(mp, dtype=torch.int64)
    padded_bits[:m] = bits[order]
    blocks = padded_bits.reshape(-1, 32)
    masks = torch.zeros(blocks.shape[0], dtype=torch.int64)
    for row in range(32):
        masks |= blocks[:, row]
    return MaskSchedule(permutation, indices[order], masks,
                        sum(int(mask).bit_count() for mask in masks), masks.numel() * volume)
