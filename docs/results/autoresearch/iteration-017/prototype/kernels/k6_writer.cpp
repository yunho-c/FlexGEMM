// SPDX-License-Identifier: Apache-2.0
// K6 implicit-GEMM sparse conv (tt/sparse_k6.py, OPT_PLAN K6), writer (RISCV_0 / BRISC).
// Core 0 first writes the zero sentinel row (page ROWS) of the output table. Then, for each block of this core (the
// reader's order), a live block's COUT_T fp32 tiles (WRITE_Y) go to y (tile b * COUT_T + n) and its 32 untilized
// bf16 rows (CB_OUT) to the output table (pages 32 b .. 32 b + 31). Dead blocks write nothing.
// Without SPLIT the live flags come from the reader (CB_WMETA, last chunk). With SPLIT this kernel also walks every
// (chunk, block), reads the index rows itself (prefetched, double-buffered), and gathers the present taps of odd rank
// into CB_GATH2 (k6_gather.hpp); a block's outputs are written one block later, after the next block's gathers, so
// the compute never waits for this RISC's writes.
// CT args: [0] per-core RT-arg count (P3) [1] COUT_T [2] NBLK [3] NC [4] ROWS [5] WRITE_Y [6] SPLIT [7] K [8] CIN_T
// [9] NCH [10] TG [11] SENT [12] DEAD_SKIP, then the TensorAccessorArgs of the output table, y, the input table and
// the index. Common RT args: [table_out, y, table_in, index]. Per-core RT args: [me].
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "k6_gather.hpp"

constexpr uint32_t COUT_T = get_compile_time_arg_val(1);
constexpr uint32_t NBLK = get_compile_time_arg_val(2);
constexpr uint32_t NC = get_compile_time_arg_val(3);
constexpr uint32_t ROWS = get_compile_time_arg_val(4);
constexpr uint32_t WRITE_Y = get_compile_time_arg_val(5);
constexpr uint32_t SPLIT = get_compile_time_arg_val(6);
constexpr uint32_t K = get_compile_time_arg_val(7);
constexpr uint32_t CIN_T = get_compile_time_arg_val(8);
constexpr uint32_t NCH = get_compile_time_arg_val(9);
constexpr uint32_t TG = get_compile_time_arg_val(10);
constexpr uint32_t SENT = get_compile_time_arg_val(11);
constexpr uint32_t DEAD_SKIP = get_compile_time_arg_val(12);
constexpr auto o_args = TensorAccessorArgs<13>();
constexpr auto y_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
constexpr auto t_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
constexpr auto i_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();

constexpr uint32_t CB_WMETA = 8, CB_ZERO = 9, CB_GATH2 = 11, CB_IDX2 = 12, CB_ZBUF2 = 13, CB_OUT = 16, CB_Y = 17;
constexpr uint32_t ROWB = COUT_T * 64;
constexpr uint32_t ROWB_IN = CIN_T * 64;
constexpr uint32_t IDXB = K * 32 * 4;
constexpr uint32_t TB32 = 4096;

template <typename OA, typename YA>
inline void write_block(uint32_t b, const OA& out, const YA& yy) {
    if constexpr (WRITE_Y) {
        for (uint32_t n = 0; n < COUT_T; ++n) {
            cb_wait_front(CB_Y, 1);
            noc_async_write(get_read_ptr(CB_Y), yy.get_noc_addr(b * COUT_T + n), TB32);
            noc_async_writes_flushed();
            cb_pop_front(CB_Y, 1);
        }
    }
    cb_wait_front(CB_OUT, COUT_T);
    uint32_t rp = get_read_ptr(CB_OUT);
    for (uint32_t v = 0; v < 32; ++v) {
        noc_async_write(rp, out.get_noc_addr(b * 32 + v), ROWB);
        rp += ROWB;
    }
    noc_async_writes_flushed();
    cb_pop_front(CB_OUT, COUT_T);
}

void kernel_main() {
    const uint32_t me = get_arg_val<uint32_t>(0);
    const auto out = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), ROWB);
    const auto yy = TensorAccessor(y_args, get_common_arg_val<uint32_t>(1), TB32);
    if (me == 0) {
        cb_reserve_back(CB_ZERO, 1);
        const uint32_t zp = get_write_ptr(CB_ZERO);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zp);
        for (uint32_t j = 0; j < ROWB / 4; ++j) {
            z[j] = 0;
        }
        noc_async_write(zp, out.get_noc_addr(ROWS), ROWB);
        noc_async_write_barrier();
        cb_push_back(CB_ZERO, 1);
    }
    if constexpr (!SPLIT) {
        for (uint32_t b = me; b < NBLK; b += NC) {
            cb_wait_front(CB_WMETA, 1);
            invalidate_l1_cache();
            const uint32_t live = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_WMETA))[0];
            cb_pop_front(CB_WMETA, 1);
            if (live) {
                write_block(b, out, yy);
            }
        }
    } else {
        const auto tab = TensorAccessor(t_args, get_common_arg_val<uint32_t>(2), ROWB_IN);
        const auto idx = TensorAccessor(i_args, get_common_arg_val<uint32_t>(3), IDXB);
        cb_reserve_back(CB_ZBUF2, 1);
        const uint32_t zero_l1 = get_write_ptr(CB_ZBUF2);
        {
            volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_l1);
            for (uint32_t j = 0; j < CIN_T * 512; ++j) {
                z[j] = 0;
            }
        }
        const uint64_t zero_noc = get_noc_addr(zero_l1);
        cb_reserve_back(CB_IDX2, 2);
        const uint32_t idx_buf[2] = {get_write_ptr(CB_IDX2), get_write_ptr(CB_IDX2) + IDXB};
        uint32_t cur = 0;
        if (me < NBLK) {
            noc_async_read(idx.get_noc_addr(me), idx_buf[0], IDXB);
        }
        uint32_t pending = NBLK;            // a live block of the last chunk whose outputs are still to be written
        for (uint32_t ch = 0; ch < NCH; ++ch) {
            const uint32_t t0 = ch * TG;
            const uint32_t nt = (K - t0) < TG ? (K - t0) : TG;
            const bool last = ch == NCH - 1;
            for (uint32_t b = me; b < NBLK; b += NC) {
                noc_async_read_barrier();
                invalidate_l1_cache();
                asm volatile("" ::: "memory");
                const uint32_t* ip = reinterpret_cast<const uint32_t*>(idx_buf[cur]);
                auto prefetch = [&]() {
                    uint32_t nb = b + NC, nch = ch;
                    if (nb >= NBLK) {
                        nb = me;
                        ++nch;
                    }
                    if (nch < NCH) {
                        noc_async_read(idx.get_noc_addr(nb), idx_buf[cur ^ 1], IDXB);
                    }
                };
                uint32_t live, mask;
                k6_scan<K, SENT, DEAD_SKIP>(ip, t0, nt, live, mask);
                if (live) {
                    k6_gather<CIN_T, ROWB_IN, SENT, 1, 2>(ip, t0, nt, mask, CB_GATH2, zero_noc, tab, prefetch);
                } else {
                    prefetch();
                }
                cur ^= 1;
                if (last && live) {
                    if (pending != NBLK) {
                        write_block(pending, out, yy);
                    }
                    pending = b;
                }
            }
        }
        if (pending != NBLK) {
            write_block(pending, out, yy);
        }
        noc_async_read_barrier();
        cb_push_back(CB_IDX2, 2);
        cb_push_back(CB_ZBUF2, 1);
    }
    noc_async_write_barrier();
}
