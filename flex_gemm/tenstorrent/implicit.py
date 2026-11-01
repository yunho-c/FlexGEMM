"""Fused sparse gather/matmul via TTNN's Metal program descriptors."""

from dataclasses import dataclass
from pathlib import Path
import torch
from .reference import validate
from .ops import _upload, _matmul_fidelity


def _round32(n):
    return (n + 31) // 32 * 32


def _sum_split_partials(partials):
    """Accumulate tiled FP32 [S,M,N] partials without the axis-0 sum lowering.

    The validated SDK's ttnn.sum(dim=0) loses low mantissa bits despite FP32
    input/output. Sliced FP32 elementwise additions preserve the residuals.
    This stays entirely on device and uses S-1 additions, with bounded scratch.
    """
    import ttnn
    splits, m, n = partials.shape
    result = ttnn.reshape(ttnn.slice(partials, [0, 0, 0], [1, m, n]), [m, n])
    for split in range(1, splits):
        part = ttnn.reshape(ttnn.slice(partials, [split, 0, 0], [split + 1, m, n]), [m, n])
        result = ttnn.add(result, part)
    return result


@dataclass
class PreparedSparseConv:
    """Device operands. Call run repeatedly to exclude host preprocessing/upload."""
    features: object
    weights: object
    indices: object
    bias: object
    device: object
    m: int
    ci: int
    co: int
    volume: int
    masks: object = None
    permutation: object = None
    schedule: object = None

    def run(self, *, split_k=1):
        import ttnn
        kt = self.volume * _round32(self.ci) // 32
        if isinstance(split_k, bool) or not isinstance(split_k, int) or not 1 <= split_k <= kt:
            raise ValueError(f"split_k must be an integer in [1,{kt}]")
        result = _launch(self, split_k)
        if self.masks is not None:
            result = ttnn.to_layout(result, ttnn.TILE_LAYOUT)
        if split_k > 1:
            # Partials and reduction stay FP32 until the final BF16 conversion.
            result = _sum_split_partials(result)
            result = ttnn.typecast(result, ttnn.bfloat16)
        result = ttnn.slice(result, [0, 0], [self.m, self.co])
        if self.bias is not None:
            result = ttnn.add(result, self.bias)
        return result


def prepare_implicit(feats, weight, neighbor_map, *, device, bias=None, masked=False):
    """Upload BF16 operands with channel tails zero-padded independently per offset."""
    w, indices = validate(feats, weight, neighbor_map, bias)
    m, v = indices.shape
    co, _, ci = w.shape
    if m == 0:
        raise ValueError("device GEMM requires at least one output row")
    import ttnn
    mp, cp, op = _round32(m), _round32(ci), _round32(co)
    schedule = None
    if masked:
        from .masking import make_mask_schedule
        schedule = make_mask_schedule(indices)
        indices = schedule.indices
    # At least one source row lets all-missing maps work with empty input sets.
    xpad = torch.zeros((max(1, feats.shape[0]), cp), dtype=torch.bfloat16)
    xpad[:feats.shape[0], :ci] = feats
    wpad = torch.zeros((v, cp, op), dtype=torch.bfloat16)
    wpad[:, :ci, :co] = w.permute(1, 2, 0)
    maps = torch.full((v, mp), 0xffffffff, dtype=torch.int64)
    maps[:, :m] = torch.where(indices.T < 0, 0xffffffff, indices.T)
    xdev = ttnn.from_torch(xpad, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    wdev = _upload(wpad.reshape(v * cp, op), device)
    idev = ttnn.from_torch(maps.to(torch.uint32), dtype=ttnn.uint32,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    bdev = None if bias is None else _upload(bias.reshape(1, co), device)
    p = PreparedSparseConv(xdev, wdev, idev, bdev, device, m, ci, co, v)
    if masked:
        mask_rows = torch.zeros((mp // 32, 8), dtype=torch.uint32)
        mask_rows[:, 0] = schedule.block_masks.to(torch.uint32)
        p.masks = ttnn.from_torch(mask_rows, dtype=ttnn.uint32,
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        p.permutation = ttnn.from_torch(schedule.permutation.reshape(1, mp).to(torch.uint32),
                                       dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        p.schedule = schedule
    return p


def implicit_gemm(feats, weight, neighbor_map, *, device, bias=None):
    """One-shot fused sparse gather/matmul; returns a BF16 TTNN device tensor."""
    return prepare_implicit(feats, weight, neighbor_map, device=device, bias=bias).run()


def masked_implicit_gemm(feats, weight, neighbor_map, *, device, bias=None):
    """Gray-sort rows, skip inactive block offsets, and restore rows on device."""
    return prepare_implicit(feats, weight, neighbor_map, device=device, bias=bias, masked=True).run()


def split_k_gemm(feats, weight, neighbor_map, *, device, split_k=2, masked=True, bias=None):
    """Parallel reduction slices, FP32 device partials/reduction, then one bias add."""
    return prepare_implicit(feats, weight, neighbor_map, device=device, bias=bias,
                            masked=masked).run(split_k=split_k)


def _launch(p, split_k=1):
    import ttnn
    mp, cp, op = _round32(p.m), _round32(p.ci), _round32(p.co)
    masked = p.masks is not None
    layout = ttnn.ROW_MAJOR_LAYOUT if masked else ttnn.TILE_LAYOUT
    shape = [mp, op] if split_k == 1 else [split_k, mp, op]
    out_dtype = ttnn.bfloat16 if split_k == 1 else ttnn.float32
    output = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), out_dtype, layout, p.device)
    tasks = split_k * (mp // 32) * (op // 32)
    grid = p.device.compute_with_storage_grid_size()
    ncores = min(tasks, grid.x * grid.y)
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        start, end = tasks * i // ncores, tasks * (i + 1) // ncores
        rr[c.x][c.y] = [p.features.buffer_address(), p.weights.buffer_address(),
                         p.indices.buffer_address(), start, end - start]
        wr[c.x][c.y] = [output.buffer_address(), start, end - start]
        cr[c.x][c.y] = [end - start]
        if masked:
            rr[c.x][c.y] = list(rr[c.x][c.y]) + [p.masks.buffer_address()]
            wr[c.x][c.y] = list(wr[c.x][c.y]) + [p.permutation.buffer_address()]
    reader_ct = [p.volume, cp // 32, op // 32, mp // 32, split_k]
    for tensor in (p.features, p.weights, p.indices):
        reader_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    out_bytes = 2 if split_k == 1 else 4
    writer_ct = [op // 32, mp // 32, out_bytes] + ttnn.TensorAccessorArgs(output).get_compile_time_args()
    defines = []
    if masked:
        defines = [("MASKED", "1")]
        reader_ct.extend(ttnn.TensorAccessorArgs(p.masks).get_compile_time_args())
        writer_ct.extend(ttnn.TensorAccessorArgs(p.permutation).get_compile_time_args())
    if masked or split_k > 1:
        defines.append(("DYNAMIC_K", "1"))
    root = Path(__file__).parent / "kernels"
    kernels = [
        ttnn.KernelDescriptor(kernel_source=str(root / "reader.cpp"), core_ranges=ranges,
                              compile_time_args=reader_ct, runtime_args=rr, defines=defines,
                              config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(root / "writer.cpp"), core_ranges=ranges,
                              compile_time_args=writer_ct,
                              runtime_args=wr, defines=defines, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(root / "compute.cpp"), core_ranges=ranges,
                              compile_time_args=[p.volume * cp // 32], runtime_args=cr, defines=defines,
                              config=ttnn.ComputeConfigDescriptor(
                                  math_fidelity=_matmul_fidelity(p.device), fp32_dest_acc_en=True)),
    ]
    cbs = [ttnn.CBDescriptor(total_size=page * depth, core_ranges=ranges,
                             format_descriptors=[ttnn.CBFormatDescriptor(
                                 buffer_index=index, data_format=dtype, page_size=page)])
           for index, dtype, page, depth in [(0, ttnn.bfloat16, 2048, 2),
                                            (1, ttnn.bfloat16, 2048, 2),
                                            (2, ttnn.uint32, 128, 1),
                                            (3, ttnn.uint32, 32, 2),
                                            (4, ttnn.uint32, 128, 1),
                                            (16, out_dtype, 1024 * out_bytes, 2)]]
    operands = [p.features, p.weights, p.indices]
    if masked:
        operands += [p.masks, p.permutation]
    ttnn.generic_op(operands + [output],
                    ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return output
