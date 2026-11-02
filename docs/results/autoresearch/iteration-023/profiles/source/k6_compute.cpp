// SPDX-License-Identifier: Apache-2.0
// K6 implicit-GEMM sparse conv (tt/sparse_k6.py, OPT_PLAN K6), compute.
//
// Per (chunk, live block of this core): tilize the present taps' gathered rows (CB_GATH, row-major [32, CIN_T * 32]
// per tap) into CB_TL at slot p * CIN_T (CB_TL always takes TG * CIN_T pages per block: fixed increments), then for
// each output tile n: DEST[0] (+)= sum over present taps p, input tiles c of TL[p, c] x W[p, c, n] (BF16 or FP32 weights,
// HiFi3, fp32 DEST, tap-major / channel order = the stock im2col matmul's K order). Between chunks the fp32 partial
// goes through CB_ACC (UnpackToDestFp32 reload: exact). Last chunk, the epilogue in the order of the stock programs:
//   y = acc + bias                       add_binary_tile<NearestEven> (binary_ng's fp32 SFPU add)
//   y = y + identity     (HAS_RES)       add_binary_tile<NearestEven>
//   y = relu(y)          (RELU)          relu_tile
//   CB_Y <- y            (WRITE_Y)       fp32 tile (the residual source / tap)
//   out = bf16(y)                        typecast_tile<Float32, Float16_b> (ttnn.typecast), packed untilized into
//                                        CB_OUT (32 rows x COUT_T * 32 bf16, row-major: the next table's rows)
// CT args: [0] per-core RT-arg count (P3) [1] K [2] CIN_T [3] COUT_T [4] NCH [5] TG [6] NBLK [7] NC [8] HAS_RES
//          [9] WRITE_Y [10] RELU [11] SPLIT [12] WRITE_TABLE. Per-core RT args: [me].
// WRITE_TABLE=0 packs the FP32 product directly, omitting BF16 conversion/untilization.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"

using namespace ckernel;

constexpr uint32_t K = get_compile_time_arg_val(1);
constexpr uint32_t CIN_T = get_compile_time_arg_val(2);
constexpr uint32_t COUT_T = get_compile_time_arg_val(3);
constexpr uint32_t NCH = get_compile_time_arg_val(4);
constexpr uint32_t TG = get_compile_time_arg_val(5);
constexpr uint32_t NBLK = get_compile_time_arg_val(6);
constexpr uint32_t NC = get_compile_time_arg_val(7);
constexpr uint32_t HAS_RES = get_compile_time_arg_val(8);
constexpr uint32_t WRITE_Y = get_compile_time_arg_val(9);
constexpr uint32_t RELU = get_compile_time_arg_val(10);
constexpr uint32_t SPLIT = get_compile_time_arg_val(11);
constexpr uint32_t WRITE_TABLE = get_compile_time_arg_val(12);
static_assert(WRITE_Y || WRITE_TABLE);

constexpr uint32_t CB_GATH = 0, CB_W = 1, CB_BIAS = 2, CB_IDENT = 3, CB_META = 4, CB_TL = 5, CB_ACC = 6;
constexpr uint32_t CB_GATH2 = 11, CB_OUT = 16, CB_Y = 17;
constexpr uint32_t WCH = TG * CIN_T * COUT_T;
constexpr uint32_t TLN = TG * CIN_T;

void kernel_main() {
    const uint32_t me = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_TL, CB_W, WRITE_TABLE ? CB_OUT : CB_Y);
    // BH: the DEST remap of pack-untilize is a MATH-thread setting; arm it once here, outside any DEST section (as
    // the stock matmul's untilize_out path leaves it armed for its later matmuls)
    if constexpr (WRITE_TABLE) {
        pack_untilize_dest_init<1, COUT_T>(CB_OUT);
        pack_untilize_uninit(CB_OUT);
    }
    cb_wait_front(CB_BIAS, COUT_T);

    for (uint32_t ch = 0; ch < NCH; ++ch) {
        const uint32_t t0 = ch * TG;
        const uint32_t nt = (K - t0) < TG ? (K - t0) : TG;
        const bool last = ch == NCH - 1;
        cb_wait_front(CB_W, WCH);
        for (uint32_t b = me; b < NBLK; b += NC) {
            cb_wait_front(CB_META, 1);
            UNPACK(asm volatile("fence"));
            const uint32_t mask = read_tile_value(CB_META, 0, 0);
            const uint32_t live = read_tile_value(CB_META, 0, 1);
            cb_pop_front(CB_META, 1);
            if (!live) {
                continue;
            }
            // ---- tilize the present taps
            cb_reserve_back(CB_TL, TLN);
            if (mask) {
                reconfig_data_format_srca(CB_GATH);
                pack_reconfig_data_format(CB_TL);
                tilize_init(CB_GATH, CIN_T, CB_TL);
                uint32_t k = 0;
                for (uint32_t p = 0; p < nt; ++p) {
                    if (mask & (1u << p)) {
                        const uint32_t cb = (SPLIT && (k & 1)) ? CB_GATH2 : CB_GATH;
                        ++k;
                        cb_wait_front(cb, CIN_T);
#ifndef K6_NOTILIZE
                        tilize_block(cb, CIN_T, CB_TL, 0, p * CIN_T);
#endif
                        cb_pop_front(cb, CIN_T);
                    }
                }
                tilize_uninit(CB_GATH, CB_TL);
            }
            cb_push_back(CB_TL, TLN);
            cb_wait_front(CB_TL, TLN);
            if (last) {
                if constexpr (WRITE_TABLE) {
                    cb_reserve_back(CB_OUT, COUT_T);
                }
                if constexpr (HAS_RES) {
                    cb_wait_front(CB_IDENT, COUT_T);
                }
            }
            for (uint32_t n = 0; n < COUT_T; ++n) {
                tile_regs_acquire();
                if (ch > 0) {
                    reconfig_data_format_srca(CB_ACC);
                    copy_tile_init(CB_ACC);
                    cb_wait_front(CB_ACC, 1);
                    copy_tile(CB_ACC, 0, 0);
                    cb_pop_front(CB_ACC, 1);
                } else {
                    // DEST is not guaranteed zero here (pack-untilize does not clear it): start from +0
                    fill_tile_init();
                    fill_tile(0, 0.0f);
                }
                reconfig_data_format_srca(CB_W);
                reconfig_data_format_srcb(CB_TL);
                matmul_init(CB_TL, CB_W);
                for (uint32_t p = 0; p < nt; ++p) {
                    if (mask & (1u << p)) {
#ifndef K6_NOMM
                        for (uint32_t c = 0; c < CIN_T; ++c) {
                            matmul_tiles(CB_TL, CB_W, p * CIN_T + c, (p * CIN_T + c) * COUT_T + n, 0);
                        }
#endif
                    }
                }
                if (!last) {
                    tile_regs_commit();
                    cb_reserve_back(CB_ACC, 1);
                    tile_regs_wait();
                    pack_reconfig_data_format(CB_ACC);
                    pack_tile(0, CB_ACC);
                    tile_regs_release();
                    cb_push_back(CB_ACC, 1);
                    continue;
                }
                reconfig_data_format_srca(CB_BIAS);
                copy_tile_init(CB_BIAS);
                copy_tile(CB_BIAS, n, 1);
                add_binary_tile_init();
                add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
                if constexpr (HAS_RES) {
                    reconfig_data_format_srca(CB_IDENT);
                    copy_tile_init(CB_IDENT);
                    copy_tile(CB_IDENT, n, 1);
                    add_binary_tile_init();
                    add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
                }
                if constexpr (RELU) {
                    relu_tile_init();
                    relu_tile(0);
                }
                if constexpr (WRITE_Y && WRITE_TABLE) {
                    copy_dest_values_init();
                    copy_dest_values<DataFormat::Float32>(0, 2);
                }
                if constexpr (WRITE_TABLE) {
                    typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
                    typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
                }
                tile_regs_commit();
                if constexpr (WRITE_Y) {
                    cb_reserve_back(CB_Y, 1);
                }
                tile_regs_wait();
                if constexpr (WRITE_Y) {
                    pack_reconfig_data_format(CB_Y);
                    pack_tile(WRITE_TABLE ? 2 : 0, CB_Y);
                }
                if constexpr (WRITE_TABLE) {
                    pack_reconfig_data_format(CB_OUT);
                    pack_untilize_dest_init<1, COUT_T, false, TILE_C_DIM, false, false>(CB_OUT);
                    pack_untilize_dest<1, COUT_T>(CB_OUT, 1, n);
                    pack_untilize_uninit(CB_OUT);
                }
                tile_regs_release();
                if constexpr (WRITE_Y) {
                    cb_push_back(CB_Y, 1);
                }
            }
            if (last) {
                if constexpr (WRITE_TABLE) {
                    cb_push_back(CB_OUT, COUT_T);
                }
                if constexpr (HAS_RES) {
                    cb_pop_front(CB_IDENT, COUT_T);
                }
            }
            cb_pop_front(CB_TL, TLN);
        }
        cb_pop_front(CB_W, WCH);
    }
    cb_pop_front(CB_BIAS, COUT_T);
}
