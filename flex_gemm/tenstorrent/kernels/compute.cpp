// SPDX-License-Identifier: MIT
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    matmul_init(0, 1);
    for (uint32_t task = 0; task < count; ++task) {
        uint32_t steps = KT;
#ifdef DYNAMIC_K
        cb_wait_front(3, 1);
        steps = read_tile_value(3, 0, 0);
        cb_pop_front(3, 1);
#endif
        tile_regs_acquire();
        for (uint32_t k = 0; k < steps; ++k) {
            cb_wait_front(0, 1);
            cb_wait_front(1, 1);
            matmul_tiles(0, 1, 0, 0, 0);
            cb_pop_front(0, 1);
            cb_pop_front(1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(16, 1);
        pack_tile(0, 16);
        cb_push_back(16, 1);
        tile_regs_release();
    }
}
