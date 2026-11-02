// SPDX-License-Identifier: Apache-2.0
// K6 implicit-GEMM sparse conv (tt/sparse_k6.py): the gather steps shared by the reader (RISCV_1) and, with
// SPLIT, the writer (RISCV_0), so both data-movement RISCs issue row reads (each on its own NoC).
#pragma once
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// live (any index of the block != SENT; always 1 without DEAD_SKIP) and the mask of the chunk's taps [t0, t0 + nt)
// with any index != SENT. ip: the block's index row (K x 32, tap-major).
template <uint32_t K, uint32_t SENT, uint32_t DEAD_SKIP>
inline void k6_scan(const uint32_t* ip, uint32_t t0, uint32_t nt, uint32_t& live, uint32_t& mask) {
    live = 1;
    if constexpr (DEAD_SKIP) {
        live = 0;
        for (uint32_t i = 0; i < K * 32; ++i) {
            if (ip[i] != SENT) {
                live = 1;
                break;
            }
        }
    }
    mask = 0;
    if (live) {
        for (uint32_t p = 0; p < nt; ++p) {
            const uint32_t* q = ip + (t0 + p) * 32;
            for (uint32_t v = 0; v < 32; ++v) {
                if (q[v] != SENT) {
                    mask |= (1u << p);
                    break;
                }
            }
        }
    }
}

// The present taps of `mask` whose rank (order among the present taps) is = PART mod NPARTS: each tap's 32 rows of
// CIN_T * 64 B into one CIN_T-page region of `cb` (the CB holds 2 TG CIN_T pages and every push is CIN_T pages, so
// a region is contiguous; regions wrap at the CB end). (1) every region zeroed by one local NoC copy from `zero_noc`
// (sentinel rows stay zero), barrier, (2) the rows' reads, `before_barrier()` (the index prefetch), barrier, pushes.
template <uint32_t CIN_T, uint32_t ROWB, uint32_t SENT, uint32_t PART, uint32_t NPARTS, typename Acc, typename F>
inline void k6_gather(const uint32_t* ip, uint32_t t0, uint32_t nt, uint32_t mask, uint32_t cb, uint64_t zero_noc,
                      const Acc& tab, F&& before_barrier) {
    uint32_t mine = 0;
    {
        uint32_t k = 0;
        for (uint32_t p = 0; p < nt; ++p) {
            if (mask & (1u << p)) {
                if (k % NPARTS == PART) {
                    mine |= (1u << p);
                }
                ++k;
            }
        }
    }
    const uint32_t np = __builtin_popcount(mine);
    if (!np) {
        before_barrier();
        return;
    }
    cb_reserve_back(cb, np * CIN_T);
    uint32_t wp = get_write_ptr(cb);
    const uint32_t lim = get_local_cb_interface(cb).fifo_limit;
    const uint32_t sz = get_local_cb_interface(cb).fifo_size;
    uint32_t zp = wp;
    for (uint32_t p = 0; p < np; ++p) {
        noc_async_read(zero_noc, zp, CIN_T * 2048);
        zp += CIN_T * 2048;
        if (zp >= lim) {
            zp -= sz;
        }
    }
    noc_async_read_barrier();
    for (uint32_t p = 0; p < nt; ++p) {
        if (!(mine & (1u << p))) {
            continue;
        }
        const uint32_t* q = ip + (t0 + p) * 32;
        uint32_t rp = wp;
        for (uint32_t v = 0; v < 32; ++v) {
            const uint32_t r = q[v];
            if (r != SENT) {
#ifndef K6_NOREAD
                noc_async_read(tab.get_noc_addr(r), rp, ROWB);
#endif
            }
            rp += ROWB;
        }
        wp += CIN_T * 2048;
        if (wp >= lim) {
            wp -= sz;
        }
    }
    before_barrier();
    noc_async_read_barrier();
    for (uint32_t p = 0; p < np; ++p) {
        cb_push_back(cb, CIN_T);
    }
}
