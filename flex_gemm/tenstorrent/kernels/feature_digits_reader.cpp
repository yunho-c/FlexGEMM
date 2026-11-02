// SPDX-License-Identifier: MIT
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto xa = TensorAccessorArgs<0>();
    constexpr auto ia = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, get_arg_val<uint32_t>(0));
    const auto inverse = TensorAccessor(ia, get_arg_val<uint32_t>(1));
    const uint32_t begin = get_arg_val<uint32_t>(2), count = get_arg_val<uint32_t>(3);
    // The first FP32 element is the current invocation's inverse scale.
    cb_reserve_back(2, 1);
    noc_async_read(get_noc_addr(0, inverse), get_write_ptr(2), 32);
    noc_async_read_barrier();
    cb_push_back(2, 1);
    for (uint32_t i = begin; i < begin + count; ++i) {
        cb_reserve_back(0, 1);
        noc_async_read_page(i, x, get_write_ptr(0));
        noc_async_read_barrier();
        cb_push_back(0, 1);
    }
}
