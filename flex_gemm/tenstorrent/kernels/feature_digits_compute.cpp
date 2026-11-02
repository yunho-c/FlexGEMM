// SPDX-License-Identifier: MIT
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"

using namespace ckernel;

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    init_sfpu(0, 16);
    cb_wait_front(2, 1);
    UNPACK(asm volatile("fence"));
    const uint32_t inverse_bits = read_tile_value(2, 0, 0);
    for (uint32_t i = 0; i < count; ++i) {
        cb_wait_front(0, 1);
        tile_regs_acquire();
        copy_tile(0, 0, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inverse_bits);
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(0, 1);
        rounding_op_tile_init();
        round_tile(0, 0);
        sub_binary_tile_init();
        sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(1, 0, 1);
        binop_with_scalar_tile_init();
        mul_unary_tile(1, 0x43800000);  // 256.0f
        rounding_op_tile_init();
        round_tile(1, 0);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(1);
        tile_regs_commit();
        cb_reserve_back(16, 1);
        cb_reserve_back(17, 1);
        tile_regs_wait();
        pack_tile(0, 16);
        pack_tile(1, 17);
        tile_regs_release();
        cb_pop_front(0, 1);
        cb_push_back(16, 1);
        cb_push_back(17, 1);
    }
    cb_pop_front(2, 1);
}
