// SPDX-License-Identifier: Apache-2.0
// K6 implicit-GEMM sparse conv (tt/sparse_k6.py, OPT_PLAN K6), reader (RISCV_0 / NCRISC).
//
// One layer y[o] = sum_k W_k^T x[nmap[o, k]] (+ bias, + identity, ReLU) over 32-row output blocks. Block b of this
// core: b = me, me + NC, me + 2 NC, ... (strided, so the live blocks, a prefix of the level, spread over all cores).
// Taps are processed in NCH chunks of TG taps (the chunk's fp32 weights resident in L1).
//
// Per chunk:  the chunk's weight tiles -> CB_W (always TG * CIN_T * COUT_T pages; the tail chunk part filled).
// Per block:  the block's index row (K * 32 uint32, tile order: entry k * 32 + v = row of output 32 b + v at tap k)
//             -> scratch; live = any entry != SENT (rows past the level count are all-sentinel: skipped whole);
//             mask = taps of the chunk with any entry != SENT (all-sentinel taps contribute exact zeros: skipped).
//             meta page {mask, live} -> CB_META (compute); in the last chunk {live} -> CB_WMETA (writer).
//             For each tap in mask: CIN_T row-major pages = 32 table rows (CIN_T * 64 B each; sentinel rows zero)
//             -> CB_GATH (one read barrier per block).
//             Last chunk, HAS_RES: the COUT_T fp32 identity tiles of the block -> CB_IDENT.
// Once:       the COUT_T fp32 bias tiles (rows replicated) -> CB_BIAS.
//
// CT args: [0] per-core RT-arg count (P3) [1] K [2] CIN_T [3] COUT_T [4] NCH [5] TG [6] NBLK [7] NC [8] SENT
//          [9] HAS_RES [10] DEAD_SKIP (0: all-sentinel blocks are computed like the stock layer, relu(bias)
//          (+ identity), so every output row is written and deterministic) [11] MCAST [12] NRECV
//          [13] SPLIT (the writer gathers the odd-rank present taps into CB_GATH2; no CB_WMETA), then the TensorAccessorArgs of table, index, weight, bias, identity.
// Common RT args: [table, index, weight, bias, identity]. Per-core RT args: [me, is_sender, sender x, y (virtual),
// the receivers' rectangle x0, y0, x1, y1 (virtual, NOC 0 order)].
// MCAST: the chunk's weights are read from DRAM once per grid row, by the row's first core, and multicast to the
// NRECV other cores of the row (the stock 1D-matmul handshake: each receiver frees its CB_W, sets its 'valid'
// semaphore to 0 and increments the sender's 'ready' semaphore; the sender waits for NRECV, multicasts the data,
// flushes, then multicasts valid = 1). The rectangle never contains the sender.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "k6_gather.hpp"

constexpr uint32_t K = get_compile_time_arg_val(1);
constexpr uint32_t CIN_T = get_compile_time_arg_val(2);
constexpr uint32_t COUT_T = get_compile_time_arg_val(3);
constexpr uint32_t NCH = get_compile_time_arg_val(4);
constexpr uint32_t TG = get_compile_time_arg_val(5);
constexpr uint32_t NBLK = get_compile_time_arg_val(6);
constexpr uint32_t NC = get_compile_time_arg_val(7);
constexpr uint32_t SENT = get_compile_time_arg_val(8);
constexpr uint32_t HAS_RES = get_compile_time_arg_val(9);
constexpr uint32_t DEAD_SKIP = get_compile_time_arg_val(10);
constexpr uint32_t MCAST = get_compile_time_arg_val(11);
constexpr uint32_t NRECV = get_compile_time_arg_val(12);
constexpr uint32_t SPLIT = get_compile_time_arg_val(13);
constexpr uint32_t NPARTS = SPLIT ? 2 : 1;
constexpr auto t_args = TensorAccessorArgs<14>();
constexpr auto i_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
constexpr auto b_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto r_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();

constexpr uint32_t CB_GATH = 0, CB_W = 1, CB_BIAS = 2, CB_IDENT = 3, CB_META = 4, CB_IDX = 7, CB_WMETA = 8, CB_ZBUF = 10;
constexpr uint32_t ROWB = CIN_T * 64;      // table row bytes (bf16)
constexpr uint32_t IDXB = K * 32 * 4;      // index row bytes
constexpr uint32_t TB32 = 4096;
constexpr uint32_t WCH = TG * CIN_T * COUT_T;

void kernel_main() {
    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t is_sender = get_arg_val<uint32_t>(1);
    const uint32_t snx = get_arg_val<uint32_t>(2), sny = get_arg_val<uint32_t>(3);
    // NOC 1 runs its coordinates the other way: the rectangle's start is the far corner there (as ttnn's matmuls)
    const bool n1 = noc_index == 1;
    const uint32_t mx0 = get_arg_val<uint32_t>(n1 ? 6 : 4), my0 = get_arg_val<uint32_t>(n1 ? 7 : 5);
    const uint32_t mx1 = get_arg_val<uint32_t>(n1 ? 4 : 6), my1 = get_arg_val<uint32_t>(n1 ? 5 : 7);
    const uint32_t ready_addr = get_semaphore(0);
    const uint32_t valid_addr = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* ready_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    volatile tt_l1_ptr uint32_t* valid_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);
    const auto tab = TensorAccessor(t_args, get_common_arg_val<uint32_t>(0), ROWB);
    const auto idx = TensorAccessor(i_args, get_common_arg_val<uint32_t>(1), IDXB);
    const auto wt = TensorAccessor(w_args, get_common_arg_val<uint32_t>(2), TB32);
    const auto bs = TensorAccessor(b_args, get_common_arg_val<uint32_t>(3), TB32);
    const auto rs = TensorAccessor(r_args, get_common_arg_val<uint32_t>(4), TB32);

    // bias once
    cb_reserve_back(CB_BIAS, COUT_T);
    {
        uint32_t wp = get_write_ptr(CB_BIAS);
        for (uint32_t n = 0; n < COUT_T; ++n) {
            noc_async_read(bs.get_noc_addr(n), wp + n * TB32, TB32);
        }
        noc_async_read_barrier();
    }
    cb_push_back(CB_BIAS, COUT_T);

    // a zeroed region of one tap (32 rows x CIN_T * 64 B): the source of the per-tap zero fills
    cb_reserve_back(CB_ZBUF, 1);
    const uint32_t zero_l1 = get_write_ptr(CB_ZBUF);
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_l1);
        for (uint32_t j = 0; j < CIN_T * 512; ++j) {
            z[j] = 0;
        }
    }
    const uint64_t zero_noc = get_noc_addr(zero_l1);

    // two index-row buffers: the next (chunk, block)'s row is prefetched while this block gathers
    cb_reserve_back(CB_IDX, 2);
    const uint32_t idx_buf[2] = {get_write_ptr(CB_IDX), get_write_ptr(CB_IDX) + IDXB};
    uint32_t cur = 0;
    if (me < NBLK) {
        noc_async_read(idx.get_noc_addr(me), idx_buf[0], IDXB);
    }

    for (uint32_t ch = 0; ch < NCH; ++ch) {
        const uint32_t t0 = ch * TG;
        const uint32_t nt = (K - t0) < TG ? (K - t0) : TG;
        const bool last = ch == NCH - 1;
        // weights of the chunk: tiles [t0 * CIN_T * COUT_T, (t0 + nt) * CIN_T * COUT_T)
        cb_reserve_back(CB_W, WCH);
        {
            const uint32_t wp = get_write_ptr(CB_W);
            const uint32_t first = t0 * CIN_T * COUT_T;
            const uint32_t cnt = nt * CIN_T * COUT_T;
            if (!MCAST || is_sender) {
                for (uint32_t i = 0; i < cnt; ++i) {
                    noc_async_read(wt.get_noc_addr(first + i), wp + i * TB32, TB32);
                }
                noc_async_read_barrier();
            }
            if constexpr (MCAST) {
                if (is_sender) {
                    noc_semaphore_wait(ready_ptr, NRECV);
                    noc_semaphore_set(ready_ptr, 0);
                    noc_async_write_multicast(wp, get_noc_multicast_addr(mx0, my0, mx1, my1, wp), cnt * TB32, NRECV,
                                              true);
                    noc_async_writes_flushed();
                    *valid_ptr = 1;
                    noc_semaphore_set_multicast(valid_addr, get_noc_multicast_addr(mx0, my0, mx1, my1, valid_addr),
                                                NRECV);
                    noc_async_write_barrier();
                } else {
                    noc_semaphore_set(valid_ptr, 0);
                    noc_semaphore_inc(get_noc_addr(snx, sny, ready_addr), 1);
                    noc_semaphore_wait(valid_ptr, 1);
                }
            }
        }
        cb_push_back(CB_W, WCH);

        for (uint32_t b = me; b < NBLK; b += NC) {
            noc_async_read_barrier();
            invalidate_l1_cache();
            asm volatile("" ::: "memory");
            const uint32_t* ip = reinterpret_cast<const uint32_t*>(idx_buf[cur]);
            bool prefetched = false;
            auto prefetch = [&]() {
                uint32_t nb = b + NC, nch = ch;
                if (nb >= NBLK) {
                    nb = me;
                    ++nch;
                }
                if (nch < NCH) {
                    noc_async_read(idx.get_noc_addr(nb), idx_buf[cur ^ 1], IDXB);
                }
                prefetched = true;
            };
            uint32_t live, mask;
            k6_scan<K, SENT, DEAD_SKIP>(ip, t0, nt, live, mask);
            cb_reserve_back(CB_META, 1);
            {
                volatile tt_l1_ptr uint32_t* mp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_META));
                mp[0] = mask;
                mp[1] = live;
            }
            cb_push_back(CB_META, 1);
            if (last && !SPLIT) {
                cb_reserve_back(CB_WMETA, 1);
                volatile tt_l1_ptr uint32_t* mp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_WMETA));
                mp[0] = live;
                cb_push_back(CB_WMETA, 1);
            }
            if (!live) {
                prefetch();
                cur ^= 1;
                continue;
            }
            // the present taps of rank 0 mod NPARTS (all of them without SPLIT; the writer gathers the others)
            k6_gather<CIN_T, ROWB, SENT, 0, NPARTS>(ip, t0, nt, mask, CB_GATH, zero_noc, tab, prefetch);
            if (!prefetched) {
                prefetch();
            }
            if constexpr (HAS_RES) {
                if (last) {
                    // (the identity reads complete under the next block's barrier at the latest; push after ours)
                    cb_reserve_back(CB_IDENT, COUT_T);
                    uint32_t wp = get_write_ptr(CB_IDENT);
                    for (uint32_t n = 0; n < COUT_T; ++n) {
                        noc_async_read(rs.get_noc_addr(b * COUT_T + n), wp + n * TB32, TB32);
                    }
                    noc_async_read_barrier();
                    cb_push_back(CB_IDENT, COUT_T);
                }
            }
            cur ^= 1;
        }
    }
    cb_push_back(CB_IDX, 2);
    cb_push_back(CB_ZBUF, 1);
}
