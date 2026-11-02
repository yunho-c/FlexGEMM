// SPDX-License-Identifier: MIT
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    DeviceZoneScopedN("FG_OUTPUT_TILE_COPY");
    constexpr uint32_t PAGE_BYTES = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, get_arg_val<uint32_t>(0));
    const auto dst = TensorAccessor(dst_args, get_arg_val<uint32_t>(1));
    const uint32_t begin = get_arg_val<uint32_t>(2);
    const uint32_t end = begin + get_arg_val<uint32_t>(3);
    const uint32_t offset = get_arg_val<uint32_t>(4);
    constexpr uint32_t BATCH = 8;
    cb_reserve_back(0, BATCH);
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t tile = begin; tile < end; tile += BATCH) {
        const uint32_t count = (end - tile < BATCH) ? end - tile : BATCH;
        for (uint32_t i = 0; i < count; ++i) {
            noc_async_read_page(tile + i, src, scratch + i * PAGE_BYTES);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < count; ++i) {
            noc_async_write_page(offset + tile + i, dst, scratch + i * PAGE_BYTES);
        }
        noc_async_write_barrier();
    }
}
