"""Experimental direct tiled gather for the resident explicit algorithm."""

from pathlib import Path

from .implicit import _round32


def tiled_gather(features, topology, start, end, ci):
    """Gather full-neighborhood row chunks; start must be tile aligned."""
    import ttnn
    if start % 32 or not 0 <= start < end <= len(topology.host_indices):
        raise ValueError("tiled gather requires a tile-aligned start and valid rows")
    if topology.masks is not None:
        raise ValueError("tiled gather requires original-order topology")
    volume = topology.host_indices.shape[1]
    cp, mp = _round32(ci), _round32(end - start)
    output = ttnn.allocate_tensor_on_device(ttnn.Shape([end - start, volume * cp]),
        ttnn.bfloat16, ttnn.TILE_LAYOUT, topology.device)
    tasks = mp // 32 * volume
    grid = topology.device.compute_with_storage_grid_size()
    ncores = min(tasks, grid.x * grid.y)
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    runtime = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        begin, finish = tasks * i // ncores, tasks * (i + 1) // ncores
        runtime[c.x][c.y] = [features.buffer_address(), topology.indices.buffer_address(),
            output.buffer_address(), start // 32, begin, finish - begin]
    compile_time = [volume, cp // 32]
    for tensor in (features, topology.indices, output):
        compile_time.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    kernel = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).parent / "kernels/gather.cpp"),
        core_ranges=ranges, compile_time_args=compile_time, runtime_args=runtime,
        config=ttnn.ReaderConfigDescriptor())
    cbs = [ttnn.CBDescriptor(total_size=page, core_ranges=ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)])
        for index, dtype, page in ((0, ttnn.bfloat16, 2048), (2, ttnn.uint32, 128))]
    ttnn.generic_op([features, topology.indices, output],
        ttnn.ProgramDescriptor(kernels=[kernel], cbs=cbs, semaphores=[]))
    return output
