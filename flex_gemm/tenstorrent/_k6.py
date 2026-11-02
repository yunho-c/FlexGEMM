# SPDX-License-Identifier: Apache-2.0
"""Pinned Apache-2.0 K6 layer adapted for FlexGEMM digit products.

See K6-NOTICE. The integer weights and zero bias are supplied by digit_implicit;
there is no model loader, encoder or feature cache here. Only HiFi3 is enabled.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


__all__ = ["K6Layer", "CB_BUDGET"]

_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "k6")
TB32, TB16 = 4096, 2048
N_RT = 1
N_RT_READER = 8
# fp32 weight tiles resident per chunk (112 x 4 KB = 448 KB): l1-l3 and conv_out in one chunk, down3 in 2, l4 in 4
WEIGHT_BUDGET_TILES = 112
# total circular-buffer bytes per core a layer may use (the L1 tables of K6_TABLES=l1 sit above the CB region): taps
# per chunk shrink until the layer's CBs fit
CB_BUDGET = 900 * 1024


class K6Layer:
    """One sparse conv layer of a :class:`ttaw.models.sparse_encoder.SparseConvSpec` as the K6 ``generic_op``."""

    def __init__(self, spec: Any, *, budget_tiles: int = WEIGHT_BUDGET_TILES):
        self.spec = spec
        self.fidelity = "HiFi3"
        self.name = spec.name
        self.K = spec.kernel_volume
        self.cin_t = spec.cin_pad // 32
        self.cout_t = spec.cout_pad // 32
        self.budget_tiles = int(budget_tiles)
        # weights read once per grid row and multicast along it
        self.mcast = True
        # both data-movement RISCs gather (the writer the odd-rank present taps)
        self.split = True
        self.tg, self.nch = self.plan(0)
        self._dev: Optional[Tuple[Any, Any]] = None

    def cb_bytes(self, tg: int, nch: int, nbmax: int, has_res: bool = True, write_y: bool = True,
                 write_table: bool = True, weight_tile_bytes: int = TB32) -> int:
        """Circular-buffer bytes per core of the layer (the descriptor in :meth:`__call__`)."""
        ci, co, K = self.cin_t, self.cout_t, self.K
        b = 2 * tg * ci * TB16 + tg * ci * co * weight_tile_bytes + co * TB32 + 4 * 32 + tg * ci * TB16 + 2 * K * 128 + 4 * 32 \
            + ci * TB16
        b += co * 64 + 2 * co * TB16 if write_table else 0
        b += 2 * co * TB32 if has_res else 0
        b += nbmax * co * TB32 if nch > 1 else 0
        b += 2 * co * TB32 if write_y else 0
        b += 2 * tg * ci * TB16 + 2 * K * 128 + ci * TB16 if self.split else 0
        return b

    def plan(self, nbmax: int) -> Tuple[int, int]:
        """(taps per chunk, chunks): the fewest chunks whose weights fit ``budget_tiles`` and whose CBs fit
        :data:`CB_BUDGET` with ``nbmax`` blocks per core."""
        per_tap = self.cin_t * self.cout_t
        nch = max(1, -(-self.K * per_tap // self.budget_tiles))
        while True:
            tg = -(-self.K // nch)
            n = -(-self.K // tg)
            if tg == 1 or self.cb_bytes(tg, n, nbmax) <= CB_BUDGET:
                return tg, n
            nch += 1

    def _upload(self, device: Any) -> Tuple[Any, Any]:
        if self._dev is None:
            raise ValueError("digit weights must be prepared before K6 execution")
        return self._dev

    def __call__(self, table: Any, index: Any, identity: Any = None, write_y: bool = False, dead_skip: bool = True,
                 table_memory: str = "dram", write_table: bool = True):
        """``table`` bf16 ROW_MAJOR ``[1, 1, V + 1, cin_pad]`` (row V zero), ``index`` uint32 ROW_MAJOR
        ``[rows / 32, K * 32]`` (C13 tile order), ``identity`` fp32 TILE ``[1, 1, rows, cout_pad]`` or None ->
        ``(y, table_out)``: y the fp32 TILE output (``write_y``; else None) and the next bf16 ROW_MAJOR table
        ``[1, 1, rows + 1, cout_pad]`` (zero last row). ``dead_skip``: only live blocks' rows are written (else
        all-sentinel blocks are computed as the stock layer does: every row written, deterministic).
        ``write_table=False`` omits the BF16 table and requires ``write_y=True``.
        Weights may be BF16 digits or their exact FP32 encoding. Tap planning
        deliberately retains the original FP32-weight/table-output CB bound."""
        import ttnn
        if not write_table and not write_y:
            raise ValueError("K6 requires at least one output")

        spec = self.spec
        dev = table.device()
        if int(table.shape[-1]) != spec.cin_pad:
            raise ValueError(f"{self.name}: table width {int(table.shape[-1])} != {spec.cin_pad}")
        if int(index.shape[-1]) != self.K * 32:
            raise ValueError(f"{self.name}: index has {int(index.shape[-1])} columns, expected {self.K * 32}")
        nblk = int(index.shape[0])
        rows = nblk * 32
        sent = int(table.shape[-2]) - 1
        has_res = identity is not None
        w, b = self._upload(dev)
        if w.dtype not in (ttnn.bfloat16, ttnn.float32):
            raise ValueError("K6 weights must be BF16 or FP32")
        weight_tile_bytes = TB16 if w.dtype == ttnn.bfloat16 else TB32
        tmc = ttnn.L1_MEMORY_CONFIG if table_memory == "l1" else ttnn.DRAM_MEMORY_CONFIG
        out = None
        if write_table:
            out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows + 1, spec.cout_pad]), ttnn.bfloat16,
                                                 ttnn.ROW_MAJOR_LAYOUT, dev, tmc)
        y = None
        if write_y:
            y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, spec.cout_pad]), ttnn.float32,
                                               ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        g = dev.compute_with_storage_grid_size()
        nc = int(g.x) * int(g.y)
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        mcast = self.mcast and int(g.x) > 1
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i in range(nc):
            cx, cy = i % g.x, i // g.x
            s_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(0, cy))
            a_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(1 if g.x > 1 else 0, cy))
            b_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(g.x - 1, cy))
            rd[cx][cy] = [i, int(cx == 0), s_.x, s_.y, a_.x, a_.y, b_.x, b_.y]
            wr[cx][cy] = [i]
            cp[cx][cy] = [i]
        nbmax = -(-nblk // nc)
        tg, nch = self.plan(nbmax)
        self.tg, self.nch = tg, nch
        K, cin_t, cout_t = self.K, self.cin_t, self.cout_t

        def cb(idx, pages, dtype, size):
            return ttnn.CBDescriptor(total_size=pages * size, core_ranges=crs, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=size)])

        cbs = [cb(0, 2 * tg * cin_t, ttnn.bfloat16, TB16),               # gathered tap rows (row-major)
               cb(1, tg * cin_t * cout_t, w.dtype, weight_tile_bytes),   # the chunk's weights
               cb(2, cout_t, ttnn.float32, TB32),                        # bias tiles
               cb(4, 4, ttnn.float32, 32),                               # meta (compute)
               cb(5, tg * cin_t, ttnn.bfloat16, TB16),                   # tilized taps
               cb(7, 2, ttnn.float32, K * 128),                          # reader's index rows (double buffer)
               cb(8, 4, ttnn.float32, 32),                               # meta (writer)
               cb(10, 1, ttnn.bfloat16, cin_t * TB16)]                   # reader's zero tap region
        if write_table:
            cbs += [cb(9, 1, ttnn.bfloat16, cout_t * 64),                # the zero output row
                    cb(16, 2 * cout_t, ttnn.bfloat16, TB16)]             # untilized output rows
        if has_res:
            cbs.append(cb(3, 2 * cout_t, ttnn.float32, TB32))
        if nch > 1:
            cbs.append(cb(6, nbmax * cout_t, ttnn.float32, TB32))       # fp32 partials between chunks
        if write_y:
            cbs.append(cb(17, 2 * cout_t, ttnn.float32, TB32))
        split = self.split
        if split:
            cbs += [cb(11, 2 * tg * cin_t, ttnn.bfloat16, TB16),            # the writer's gathered tap rows
                    cb(12, 2, ttnn.float32, K * 128),                      # the writer's index rows
                    cb(13, 1, ttnn.bfloat16, cin_t * TB16)]                # the writer's zero tap region
        um = [ttnn.UnpackToDestMode.Default] * 64
        for i in (2, 3, 6):
            um[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=getattr(ttnn.MathFidelity, self.fidelity), fp32_dest_acc_en=True,
                                            math_approx_mode=False)
        ccfg.unpack_to_dest_mode = um
        SRC = ttnn.KernelDescriptor.SourceType.FILE_PATH

        def acc(t):
            return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


        def kd(name, ct, rt, common, config):
            return ttnn.KernelDescriptor(kernel_source=os.path.join(_KDIR, name), source_type=SRC, core_ranges=crs,
                                         compile_time_args=ct, defines=[], runtime_args=rt,
                                         common_runtime_args=common, config=config)

        ident = identity if has_res else w
        yt = y if write_y else out
        ot = out if write_table else y  # unused accessor when the table writer is compiled out
        rd_ct = [N_RT_READER, K, cin_t, cout_t, nch, tg, nblk, nc, sent, int(has_res), int(bool(dead_skip)),
                 int(mcast), int(g.x) - 1, int(split), weight_tile_bytes] + acc(table) + acc(index) + acc(w) \
            + acc(b) + acc(ident)
        wr_ct = [N_RT, cout_t, nblk, nc, rows, int(bool(write_y)), int(split), K, cin_t, nch, tg, sent,
                 int(bool(dead_skip)), int(bool(write_table))] + acc(ot) + acc(yt) + acc(table) + acc(index)
        cp_ct = [N_RT, K, cin_t, cout_t, nch, tg, nblk, nc, int(has_res), int(bool(write_y)), int(bool(spec.relu)),
                 int(split), int(bool(write_table))]
        ks = [kd("k6_reader.cpp", rd_ct, rd, [table.buffer_address(), index.buffer_address(), w.buffer_address(),
                                              b.buffer_address(), ident.buffer_address()],
                 ttnn.ReaderConfigDescriptor()),
              kd("k6_writer.cpp", wr_ct, wr, [ot.buffer_address(), yt.buffer_address(), table.buffer_address(),
                                              index.buffer_address()],
                 ttnn.WriterConfigDescriptor()),
              kd("k6_compute.cpp", cp_ct, cp, [], ccfg)]
        ios: List[Any] = [table, index, w, b]
        if has_res:
            ios.append(identity)
        if write_y:
            ios.append(y)
        if write_table:
            ios.append(out)
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0),
                ttnn.SemaphoreDescriptor(id=1, core_ranges=crs, initial_value=0)]
        ttnn.generic_op(ios, ttnn.ProgramDescriptor(kernels=ks, semaphores=sems, cbs=cbs))
        return y, out

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "kernel": "k6", "K": self.K, "cin_t": self.cin_t, "cout_t": self.cout_t,
                "taps_per_chunk": self.tg, "chunks": self.nch}
