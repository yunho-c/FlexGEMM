// SPDX-License-Identifier: MIT
#include "api/dataflow/dataflow_api.h"

// Gather directly from row-major features into the four faces of a BF16 tile.
// DRAM contains features, weights, and indices, but never an expanded im2col.
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t map_addr = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    constexpr uint32_t V = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr auto xa = TensorAccessorArgs<3>();
    constexpr auto wa = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto ma = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, x_addr);
    const auto w = TensorAccessor(wa, w_addr);
    const auto map = TensorAccessor(ma, map_addr);
    cb_reserve_back(2, 1);
    const uint32_t scratch = get_write_ptr(2);
    for (uint32_t task = start; task < start + count; ++task) {
        const uint32_t mt = task / NT, nt = task % NT;
        for (uint32_t v = 0; v < V; ++v) {
            noc_async_read(get_noc_addr(v, map) + mt * 128, scratch, 128);
            noc_async_read_barrier();
            const volatile uint32_t* ids = reinterpret_cast<volatile uint32_t*>(scratch);
            for (uint32_t kt = 0; kt < CT; ++kt) {
                cb_reserve_back(0, 1);
                cb_reserve_back(1, 1);
                const uint32_t a = get_write_ptr(0), b = get_write_ptr(1);
                volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(a);
                for (uint32_t i = 0; i < 512; ++i) words[i] = 0;
                for (uint32_t r = 0; r < 32; ++r) {
                    if (ids[r] != 0xffffffff) {
                        const uint64_t src = get_noc_addr(ids[r], x) + kt * 64;
                        const uint32_t dst = a + ((r / 16) * 512 + (r % 16) * 16) * 2;
                        noc_async_read(src, dst, 32);
                        noc_async_read(src + 32, dst + 512, 32);
                    }
                }
                noc_async_read_page((v * CT + kt) * NT + nt, w, b);
                noc_async_read_barrier();
                cb_push_back(0, 1);
                cb_push_back(1, 1);
            }
        }
    }
}
