// SPDX-License-Identifier: MIT
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t MT = get_compile_time_arg_val(1);
    constexpr uint32_t ELEM_BYTES = get_compile_time_arg_val(2);
    constexpr auto args = TensorAccessorArgs<3>();
    const auto out = TensorAccessor(args, addr);
#ifdef MASKED
    constexpr auto perm_args = TensorAccessorArgs<args.next_compile_time_args_offset()>();
    const auto perm = TensorAccessor(perm_args, get_arg_val<uint32_t>(3));
    cb_reserve_back(4, 1);
    const uint32_t scratch = get_write_ptr(4);
#endif
    for (uint32_t t = start; t < start + count; ++t) {
        cb_wait_front(16, 1);
#ifdef MASKED
        const uint32_t split = t / (MT * NT);
        const uint32_t mt = (t / NT) % MT, nt = t % NT;
        noc_async_read(get_noc_addr(0, perm) + mt * 128, scratch, 128);
        noc_async_read_barrier();
        const auto rows = reinterpret_cast<volatile uint32_t*>(scratch);
        const uint32_t tile = get_read_ptr(16);
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t src = tile + ((r / 16) * 512 + r % 16 * 16) * ELEM_BYTES;
            const uint64_t dst = get_noc_addr(split * MT * 32 + rows[r], out) + nt * 32 * ELEM_BYTES;
            noc_async_write(src, dst, 16 * ELEM_BYTES);
            noc_async_write(src + 256 * ELEM_BYTES, dst + 16 * ELEM_BYTES, 16 * ELEM_BYTES);
        }
#else
        noc_async_write_page(t, out, get_read_ptr(16));
#endif
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
