// SPDX-License-Identifier: MIT
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t ROWS = get_compile_time_arg_val(0);
    constexpr uint32_t CI = get_compile_time_arg_val(1), CT = get_compile_time_arg_val(2);
    constexpr auto ha = TensorAccessorArgs<3>();
    constexpr auto la = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    const auto high = TensorAccessor(ha, get_arg_val<uint32_t>(0));
    const auto low = TensorAccessor(la, get_arg_val<uint32_t>(1));
    const uint32_t begin = get_arg_val<uint32_t>(2), count = get_arg_val<uint32_t>(3);
    cb_reserve_back(3, 1);
    const uint32_t zero = get_write_ptr(3);
    for (uint32_t i = 0; i < 16; ++i) reinterpret_cast<volatile uint32_t*>(zero)[i] = 0;
    for (uint32_t task = begin; task < begin + count; ++task) {
        cb_wait_front(16, 1);
        cb_wait_front(17, 1);
        const uint32_t h = get_read_ptr(16), l = get_read_ptr(17);
        const uint32_t mt = task / CT, kt = task % CT;
        for (uint32_t r = 0; r < 32 && mt * 32 + r < ROWS; ++r) {
            const uint32_t offset = ((r / 16) * 512 + (r % 16) * 16) * 2;
            if constexpr (CI % 32) {
                if (kt == CT - 1) {
                    for (uint32_t c = CI % 32; c < 32; ++c) {
                        const uint32_t pos = offset + (c / 16) * 512 + (c % 16) * 2;
                        *reinterpret_cast<volatile uint16_t*>(h + pos) = 0;
                        *reinterpret_cast<volatile uint16_t*>(l + pos) = 0;
                    }
                }
            }
            const uint64_t hd = get_noc_addr(mt * 32 + r, high) + kt * 64;
            const uint64_t ld = get_noc_addr(mt * 32 + r, low) + kt * 64;
            noc_async_write(h + offset, hd, 32);
            noc_async_write(h + offset + 512, hd + 32, 32);
            noc_async_write(l + offset, ld, 32);
            noc_async_write(l + offset + 512, ld + 32, 32);
        }
        if (mt == 0) {
            noc_async_write(zero, get_noc_addr(ROWS, high) + kt * 64, 64);
            noc_async_write(zero, get_noc_addr(ROWS, low) + kt * 64, 64);
        }
        noc_async_write_barrier();
        cb_pop_front(16, 1);
        cb_pop_front(17, 1);
    }
}
