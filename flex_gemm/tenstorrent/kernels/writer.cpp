// SPDX-License-Identifier: MIT
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(args, addr);
    for (uint32_t t = start; t < start + count; ++t) {
        cb_wait_front(16, 1);
        noc_async_write_page(t, out, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
