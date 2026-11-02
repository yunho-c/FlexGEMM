// SPDX-License-Identifier: MIT
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

// Materialize each gathered tile once, directly in the layout consumed by
// TTNN matmul. All feature values are fetched anew on every invocation.
void kernel_main() {
    DeviceZoneScopedN("FG_EXPLICIT_GATHER");
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t map_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t row_tile_base = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);
    constexpr uint32_t V = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr auto xa = TensorAccessorArgs<2>();
    constexpr auto ma = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, x_addr);
    const auto map = TensorAccessor(ma, map_addr);
    const auto out = TensorAccessor(oa, out_addr);
    cb_reserve_back(0, 1);
    cb_reserve_back(2, 1);
    const uint32_t tile = get_write_ptr(0), scratch = get_write_ptr(2);
    for (uint32_t task = start; task < start + count; ++task) {
        const uint32_t mt = task / V, v = task % V;
        noc_async_read(get_noc_addr(v, map) + (row_tile_base + mt) * 128, scratch, 128);
        noc_async_read_barrier();
        const auto ids = reinterpret_cast<volatile uint32_t*>(scratch);
        for (uint32_t kt = 0; kt < CT; ++kt) {
            auto words = reinterpret_cast<volatile uint32_t*>(tile);
            for (uint32_t i = 0; i < 512; ++i) words[i] = 0;
            for (uint32_t r = 0; r < 32; ++r) {
                if (ids[r] != 0xffffffff) {
                    const uint64_t src = get_noc_addr(ids[r], x) + kt * 64;
                    const uint32_t dst = tile + ((r / 16) * 512 + (r % 16) * 16) * 2;
                    noc_async_read(src, dst, 32);
                    noc_async_read(src + 32, dst + 512, 32);
                }
            }
            noc_async_read_barrier();
            noc_async_write_page((mt * V + v) * CT + kt, out, tile);
            noc_async_write_barrier();
        }
    }
}
