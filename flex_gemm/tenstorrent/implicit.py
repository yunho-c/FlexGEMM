"""Fused sparse gather/matmul via TTNN's Metal program descriptors."""

from dataclasses import dataclass
from pathlib import Path
import torch
from .reference import validate
from .ops import _upload


def _round32(n):
    return (n + 31) // 32 * 32


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

    def run(self):
        import ttnn
        result = _launch(self)
        result = ttnn.slice(result, [0, 0], [self.m, self.co])
        if self.bias is not None:
            result = ttnn.add(result, self.bias)
        return result


def prepare_implicit(feats, weight, neighbor_map, *, device, bias=None):
    """Upload BF16 operands with channel tails zero-padded independently per offset."""
    import ttnn
    w, indices = validate(feats, weight, neighbor_map, bias)
    m, v = indices.shape
    co, _, ci = w.shape
    if m == 0:
        raise ValueError("device GEMM requires at least one output row")
    mp, cp, op = _round32(m), _round32(ci), _round32(co)
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
    return PreparedSparseConv(xdev, wdev, idev, bdev, device, m, ci, co, v)


def implicit_gemm(feats, weight, neighbor_map, *, device, bias=None):
    """One-shot fused sparse gather/matmul; returns a BF16 TTNN device tensor."""
    return prepare_implicit(feats, weight, neighbor_map, device=device, bias=bias).run()


def _launch(p):
    import ttnn
    mp, cp, op = _round32(p.m), _round32(p.ci), _round32(p.co)
    output = ttnn.allocate_tensor_on_device(ttnn.Shape([mp, op]), ttnn.bfloat16,
                                            ttnn.TILE_LAYOUT, p.device)
    tasks = (mp // 32) * (op // 32)
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
    reader_ct = [p.volume, cp // 32, op // 32]
    for tensor in (p.features, p.weights, p.indices):
        reader_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    root = Path(__file__).parent / "kernels"
    kernels = [
        ttnn.KernelDescriptor(kernel_source=str(root / "reader.cpp"), core_ranges=ranges,
                              compile_time_args=reader_ct, runtime_args=rr,
                              config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(root / "writer.cpp"), core_ranges=ranges,
                              compile_time_args=ttnn.TensorAccessorArgs(output).get_compile_time_args(),
                              runtime_args=wr, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(root / "compute.cpp"), core_ranges=ranges,
                              compile_time_args=[p.volume * cp // 32], runtime_args=cr,
                              config=ttnn.ComputeConfigDescriptor(
                                  math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)),
    ]
    cbs = [ttnn.CBDescriptor(total_size=page * depth, core_ranges=ranges,
                             format_descriptors=[ttnn.CBFormatDescriptor(
                                 buffer_index=index, data_format=dtype, page_size=page)])
           for index, dtype, page, depth in [(0, ttnn.bfloat16, 2048, 2),
                                            (1, ttnn.bfloat16, 2048, 2),
                                            (2, ttnn.uint32, 128, 1),
                                            (16, ttnn.bfloat16, 2048, 2)]]
    ttnn.generic_op([p.features, p.weights, p.indices, output],
                    ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return output
