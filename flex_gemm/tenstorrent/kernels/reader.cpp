// SPDX-License-Identifier: MIT
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

// Gather directly from row-major features into the four faces of a BF16 tile.
// DRAM contains features, weights, and indices, but never an expanded im2col.
void kernel_main() {
    DeviceZoneScopedN("FG_IMPLICIT_READER");
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t map_addr = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    constexpr uint32_t V = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr uint32_t MT = get_compile_time_arg_val(3);
    constexpr uint32_t SPLITS = get_compile_time_arg_val(4);
    constexpr auto xa = TensorAccessorArgs<5>();
    constexpr auto wa = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto ma = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, x_addr);
    const auto w = TensorAccessor(wa, w_addr);
    const auto map = TensorAccessor(ma, map_addr);
#ifdef MASKED
    constexpr auto mask_args = TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    const auto masks = TensorAccessor(mask_args, get_arg_val<uint32_t>(5));
#endif
    cb_reserve_back(2, 1);
    const uint32_t scratch = get_write_ptr(2);
    for (uint32_t task = start; task < start + count; ++task) {
        const uint32_t split = task / (MT * NT);
        const uint32_t mt = (task / NT) % MT, nt = task % NT;
        const uint32_t begin = (V * CT) * split / SPLITS;
        const uint32_t end = (V * CT) * (split + 1) / SPLITS;
#ifdef DYNAMIC_K
        cb_reserve_back(3, 1);
        const uint32_t header = get_write_ptr(3);
#ifdef MASKED
        noc_async_read_page(mt, masks, header);
        noc_async_read_barrier();
        const uint32_t mask = *reinterpret_cast<volatile uint32_t*>(header);
        uint32_t active = 0;
        for (uint32_t k = begin; k < end; ++k) active += (mask >> (k / CT)) & 1;
#else
        const uint32_t active = end - begin;
#endif
        *reinterpret_cast<volatile uint32_t*>(header) = active ? active : 1;
        cb_push_back(3, 1);
        if (active == 0) {
            // A zero product clears destination registers even on empty blocks.
            cb_reserve_back(0, 1);
            cb_reserve_back(1, 1);
            auto a = reinterpret_cast<volatile uint32_t*>(get_write_ptr(0));
            auto b = reinterpret_cast<volatile uint32_t*>(get_write_ptr(1));
            for (uint32_t i = 0; i < 512; ++i) { a[i] = 0; b[i] = 0; }
            cb_push_back(0, 1);
            cb_push_back(1, 1);
        }
#endif
        for (uint32_t v = begin / CT; v < (end + CT - 1) / CT; ++v) {
#ifdef MASKED
            if (((mask >> v) & 1) == 0) continue;
#endif
            noc_async_read(get_noc_addr(v, map) + mt * 128, scratch, 128);
            noc_async_read_barrier();
            const volatile uint32_t* ids = reinterpret_cast<volatile uint32_t*>(scratch);
            for (uint32_t kt = 0; kt < CT; ++kt) {
                if (v * CT + kt < begin || v * CT + kt >= end) continue;
                {
                DeviceZoneScopedSumN1("FG_READER_BUFFER_WAIT");
                cb_reserve_back(0, 1);
                cb_reserve_back(1, 1);
                }
                {
                DeviceZoneScopedSumN2("FG_READER_GATHER_AND_WEIGHT");
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
                }
                cb_push_back(0, 1);
                cb_push_back(1, 1);
            }
        }
    }
}
