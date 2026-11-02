"""Fresh BF16-to-integer-digit conversion with bounded per-core scratch."""

from pathlib import Path


def feature_digits(features, inverse, cp):
    """Fuse SFPU extraction and write two final row-major tables plus zero rows."""
    import ttnn
    rows, ci = features.shape
    if features.layout != ttnn.TILE_LAYOUT or features.dtype != ttnn.bfloat16:
        raise ValueError("digit extraction requires tiled BF16 features")
    if cp != (ci + 31) // 32 * 32 or rows <= 0:
        raise ValueError("invalid feature shape/padded channels")
    device = features.device()
    outputs = [ttnn.allocate_tensor_on_device(ttnn.Shape([rows + 1, cp]),
        ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG) for _ in range(2)]
    tasks = (rows + 31) // 32 * (cp // 32)
    grid = device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(min(tasks, grid.x * grid.y))]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    reader, compute, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        begin, end = tasks * i // len(cores), tasks * (i + 1) // len(cores)
        reader[c.x][c.y] = [features.buffer_address(), inverse.buffer_address(), begin, end - begin]
        compute[c.x][c.y] = [end - begin]
        writer[c.x][c.y] = [outputs[0].buffer_address(), outputs[1].buffer_address(), begin, end - begin]
    read_ct = []
    for tensor in (features, inverse):
        read_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    write_ct = [rows, ci, cp // 32]
    for tensor in outputs:
        write_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    root = Path(__file__).parent / "kernels"
    kernels = [
        ttnn.KernelDescriptor(kernel_source=str(root / "feature_digits_reader.cpp"), core_ranges=ranges,
            compile_time_args=read_ct, runtime_args=reader, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(root / "feature_digits_compute.cpp"), core_ranges=ranges,
            runtime_args=compute, config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True, math_approx_mode=False)),
        ttnn.KernelDescriptor(kernel_source=str(root / "feature_digits_writer.cpp"), core_ranges=ranges,
            compile_time_args=write_ct, runtime_args=writer, config=ttnn.WriterConfigDescriptor()),
    ]
    cbs = [ttnn.CBDescriptor(total_size=page * depth, core_ranges=ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)])
        for index, dtype, page, depth in ((0, ttnn.bfloat16, 2048, 2), (2, ttnn.uint32, 32, 1),
            (3, ttnn.uint32, 64, 1), (16, ttnn.bfloat16, 2048, 2), (17, ttnn.bfloat16, 2048, 2))]
    ttnn.generic_op([features, inverse, *outputs],
        ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return tuple(outputs)
