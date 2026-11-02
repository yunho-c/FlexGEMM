# SPDX-License-Identifier: Apache-2.0
"""K6: the implicit-GEMM sparse conv as one ``generic_op`` per layer (OPT_PLAN K6, ``BEVFUSION_SPARSE_KERNEL=k6``).

A stock C28 layer (``ttaw.models.sparse_encoder.TtSparseConv``) is 6-7 programs: a TILE embedding writes the whole
``[rows, K * Cin_pad]`` bf16 im2col operand to DRAM, a matmul reads it back, then fp32 bias / residual adds, a bf16
cast, an untilize and a pad that copies the table to append the zero sentinel row. K6 does the layer in one program
(``kernels/k6_{reader,compute,writer}.cpp``):

- the reader gathers each output block's tap rows (32 rows of ``Cin_pad`` bf16 per tap) straight from the ROW_MAJOR
  table into L1; taps whose 32 indices are all the sentinel are skipped (they add exact zeros), and blocks whose
  indices are all the sentinel (the rows past the level count) are skipped whole;
- the compute tilizes the tap rows and accumulates the tap matmuls (fp32 weights, HiFi4, fp32 DEST) in the stock K
  order, adds the fp32 bias (and the fp32 identity) on the SFPU, applies the ReLU, writes the fp32 output when a
  later layer needs it as a residual (or it is tapped), casts to bf16 and packs the rows untilized;
- the writer writes the rows into the next table (``[1, 1, rows + 1, Cout_pad]`` bf16 ROW_MAJOR) whose sentinel row
  core 0 zeroes, so no pad copy is needed.

Rows of dead blocks are left unwritten in the outputs: no index points at them (every index is an active row or the
sentinel), and residual / tap consumers only read live blocks. Compare outputs on the active rows only.

Layers whose fp32 weights exceed the L1 budget (level 4: 1.7 MB, down3: 0.86 MB) run in tap chunks with the fp32
partials kept in L1 (exact reload). Blocks are assigned to cores strided (block ``b`` on core ``b % NC``) so the live
prefix of a level spreads over the whole grid. Parameters: compile-time = the layer geometry (K, Cin / Cout tiles,
chunks, blocks, cores, sentinel, residual / output flags); common RT args = the buffer addresses; one per-core RT
arg (the core's index; count in CT arg 0, probe P3). The grid comes from the device.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

__all__ = ["K6Layer", "K6Encoder", "supported", "WEIGHT_BUDGET_TILES"]

_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
TB32, TB16 = 4096, 2048
N_RT = 1
N_RT_READER = 8
# fp32 weight tiles resident per chunk (112 x 4 KB = 448 KB): l1-l3 and conv_out in one chunk, down3 in 2, l4 in 4
WEIGHT_BUDGET_TILES = int(os.environ.get("BEVFUSION_K6_WEIGHT_TILES", "112"))
# total circular-buffer bytes per core a layer may use (the L1 tables of K6_TABLES=l1 sit above the CB region): taps
# per chunk shrink until the layer's CBs fit
CB_BUDGET = int(os.environ.get("BEVFUSION_K6_CB_BYTES", str(900 * 1024)))


def supported(spec: Any, weight_terms: int = 1) -> bool:
    """K6 covers one-term (bf16) output tables, fp32 weights, K <= 32 taps."""
    return int(weight_terms) == 1 and spec.out_terms == 1 and spec.kernel_volume <= 32 \
        and spec.cin_pad % 32 == 0 and spec.cout_pad % 32 == 0


class K6Layer:
    """One sparse conv layer of a :class:`ttaw.models.sparse_encoder.SparseConvSpec` as the K6 ``generic_op``."""

    def __init__(self, spec: Any, *, budget_tiles: int = WEIGHT_BUDGET_TILES):
        self.spec = spec
        self.name = spec.name
        self.K = spec.kernel_volume
        self.cin_t = spec.cin_pad // 32
        self.cout_t = spec.cout_pad // 32
        self.budget_tiles = int(budget_tiles)
        # weights read once per grid row and multicast along it (BEVFUSION_K6_MCAST=0: every core reads DRAM)
        self.mcast = os.environ.get("BEVFUSION_K6_MCAST", "1") == "1"
        # both data-movement RISCs gather (the writer the odd-rank present taps; BEVFUSION_K6_SPLIT=0: the reader only)
        self.split = os.environ.get("BEVFUSION_K6_SPLIT", "1") == "1"
        self.tg, self.nch = self.plan(0)
        self._dev: Optional[Tuple[Any, Any]] = None

    def cb_bytes(self, tg: int, nch: int, nbmax: int, has_res: bool = True, write_y: bool = True) -> int:
        """Circular-buffer bytes per core of the layer (the descriptor in :meth:`__call__`)."""
        ci, co, K = self.cin_t, self.cout_t, self.K
        b = 2 * tg * ci * TB16 + tg * ci * co * TB32 + co * TB32 + 4 * 32 + tg * ci * TB16 + 2 * K * 128 + 4 * 32 \
            + co * 64 + ci * TB16 + 2 * co * TB16
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
        import torch
        import ttnn

        if self._dev is None:
            def up(a):
                return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a, np.float32)), dtype=ttnn.float32,
                                       layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            w = up(self.spec.matrix())                                     # [K * cin_pad, cout_pad]
            b = up(np.repeat(self.spec.bias_row(), 32, axis=0))            # [32, cout_pad], rows replicated
            self._dev = (w, b)
        return self._dev

    def release(self) -> None:
        if self._dev is not None:
            from ..ttaw.models.second import free_maps

            free_maps(*self._dev)
            self._dev = None

    def __call__(self, table: Any, index: Any, identity: Any = None, write_y: bool = False, dead_skip: bool = True,
                 table_memory: str = "dram"):
        """``table`` bf16 ROW_MAJOR ``[1, 1, V + 1, cin_pad]`` (row V zero), ``index`` uint32 ROW_MAJOR
        ``[rows / 32, K * 32]`` (C13 tile order), ``identity`` fp32 TILE ``[1, 1, rows, cout_pad]`` or None ->
        ``(y, table_out)``: y the fp32 TILE output (``write_y``; else None) and the next bf16 ROW_MAJOR table
        ``[1, 1, rows + 1, cout_pad]`` (zero last row). ``dead_skip``: only live blocks' rows are written (else
        all-sentinel blocks are computed as the stock layer does: every row written, deterministic)."""
        import ttnn

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
        tmc = ttnn.L1_MEMORY_CONFIG if table_memory == "l1" else ttnn.DRAM_MEMORY_CONFIG
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
               cb(1, tg * cin_t * cout_t, ttnn.float32, TB32),           # the chunk's weights
               cb(2, cout_t, ttnn.float32, TB32),                        # bias tiles
               cb(4, 4, ttnn.float32, 32),                               # meta (compute)
               cb(5, tg * cin_t, ttnn.bfloat16, TB16),                   # tilized taps
               cb(7, 2, ttnn.float32, K * 128),                          # reader's index rows (double buffer)
               cb(8, 4, ttnn.float32, 32),                               # meta (writer)
               cb(9, 1, ttnn.bfloat16, cout_t * 64),                     # the zero row (core 0)
               cb(10, 1, ttnn.bfloat16, cin_t * TB16),                   # reader's zero tap region
               cb(16, 2 * cout_t, ttnn.bfloat16, TB16)]                  # untilized output rows
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
        ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                            math_approx_mode=False)
        ccfg.unpack_to_dest_mode = um
        SRC = ttnn.KernelDescriptor.SourceType.FILE_PATH

        def acc(t):
            return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

        dbg = [(f"K6_{d.strip().upper()}", "1") for d in os.environ.get("BEVFUSION_K6_DEBUG", "").split(",")
               if d.strip()]                                           # timing experiments only (wrong outputs)

        def kd(name, ct, rt, common, config):
            return ttnn.KernelDescriptor(kernel_source=os.path.join(_KDIR, name), source_type=SRC, core_ranges=crs,
                                         compile_time_args=ct, defines=list(dbg), runtime_args=rt,
                                         common_runtime_args=common, config=config)

        ident = identity if has_res else w
        yt = y if write_y else out
        rd_ct = [N_RT_READER, K, cin_t, cout_t, nch, tg, nblk, nc, sent, int(has_res), int(bool(dead_skip)),
                 int(mcast), int(g.x) - 1, int(split)] + acc(table) + acc(index) + acc(w) \
            + acc(b) + acc(ident)
        wr_ct = [N_RT, cout_t, nblk, nc, rows, int(bool(write_y)), int(split), K, cin_t, nch, tg, sent,
                 int(bool(dead_skip))] + acc(out) + acc(yt) + acc(table) + acc(index)
        cp_ct = [N_RT, K, cin_t, cout_t, nch, tg, nblk, nc, int(has_res), int(bool(write_y)), int(bool(spec.relu)),
                 int(split)]
        ks = [kd("k6_reader.cpp", rd_ct, rd, [table.buffer_address(), index.buffer_address(), w.buffer_address(),
                                              b.buffer_address(), ident.buffer_address()],
                 ttnn.ReaderConfigDescriptor()),
              kd("k6_writer.cpp", wr_ct, wr, [out.buffer_address(), yt.buffer_address(), table.buffer_address(),
                                              index.buffer_address()],
                 ttnn.WriterConfigDescriptor()),
              kd("k6_compute.cpp", cp_ct, cp, [], ccfg)]
        ios: List[Any] = [table, index, w, b]
        if has_res:
            ios.append(identity)
        if write_y:
            ios.append(y)
        ios.append(out)
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0),
                ttnn.SemaphoreDescriptor(id=1, core_ranges=crs, initial_value=0)]
        ttnn.generic_op(ios, ttnn.ProgramDescriptor(kernels=ks, semaphores=sems, cbs=cbs))
        return y, out

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "kernel": "k6", "K": self.K, "cin_t": self.cin_t, "cout_t": self.cout_t,
                "taps_per_chunk": self.tg, "chunks": self.nch}


class K6Encoder:
    """Drop-in for :class:`ttaw.models.sparse_encoder.SparseEncoder` (same call / describe / release) that runs every
    supported layer as a :class:`K6Layer` and the others with the stock layer. Holds the stock encoder (its specs,
    and its layers as the fallback)."""

    def __init__(self, stock: Any, *, weight_terms: Any = 1, table_memory: str = "dram", mk_mode: str = "none"):
        self.stock = stock
        self.table_memory = table_memory
        self.l1_input = os.environ.get("BEVFUSION_K6_L1_INPUT", "1") == "1"
        self.name = stock.name
        self.specs = stock.specs
        wt = weight_terms
        self.k6: List[Optional[K6Layer]] = []
        for s in self.specs:
            t = int(wt.get(s.name, 1)) if isinstance(wt, Mapping) else int(wt)
            self.k6.append(K6Layer(s) if supported(s, t) else None)
        # layer outputs read later as residual identities
        self._res_src = {s.residual for s in self.specs if s.residual is not None}
        self._last_use: Dict[str, int] = {}
        for i, s in enumerate(self.specs):
            if s.residual is not None:
                self._last_use[s.residual] = i
        # MK-S (a), OPT_PLAN 3.3 (BEVFUSION_SPARSE_MK=level): each level's sub-manifold layers as one megakernel
        self.mk_mode = str(mk_mode)
        self.mks: Dict[int, Tuple[int, Any]] = {}
        if self.mk_mode == "level" and self.table_memory == "l1" and all(k.split for k in self.k6 if k is not None):
            from .sparse_mks import MKSLevel, level_groups

            for a, b in level_groups(self.specs, self.k6):
                self.mks[a] = (b, MKSLevel([self.k6[j] for j in range(a, b)]))

    @staticmethod
    def _l1(t: Any) -> bool:
        """MK-S gathers every table of a level through one L1-interleaved accessor layout."""
        import ttnn

        return t.memory_config().buffer_type == ttnn.BufferType.L1

    @property
    def index_names(self) -> List[str]:
        return self.stock.index_names

    @property
    def input_width(self) -> int:
        return self.stock.input_width

    @property
    def input_terms(self) -> int:
        return self.stock.input_terms

    @property
    def layers(self):
        return self.stock.layers

    def __call__(self, table: Any, indices: Mapping[str, Any], taps: Optional[Dict[str, Any]] = None,
                 keep_table: bool = True, free_input: bool = False):
        """``free_input`` (P1: the caller's L1 VFE table): free ``table`` after the first layer, as the L1 copy made
        here is (the later layers' circular buffers need that L1)."""
        from ..ttaw.models.second import free_maps

        missing = [n for n in self.index_names if n not in indices]
        if missing:
            raise KeyError(f"{self.name}: missing gather indices {missing}")
        cur = table
        if self.table_memory == "l1" and self.k6[0] is not None and self.l1_input:
            import ttnn

            if table.memory_config().buffer_type != ttnn.BufferType.L1:    # P1 hands over an L1 table already
                cur = ttnn.to_memory_config(table, ttnn.L1_MEMORY_CONFIG)  # the VFE input table: gathered from L1
        outs: Dict[str, Any] = {}
        y = None
        n = len(self.specs)
        group_end, group_ys, group_out = -1, {}, None
        for i, (spec, k6, stock) in enumerate(zip(self.specs, self.k6, self.stock.layers)):
            ident = outs[spec.residual] if spec.residual else None
            last = i == n - 1
            need_y = taps is not None or spec.name in self._res_src or (last and not keep_table)
            if i in self.mks and self._l1(cur):
                # the level megakernel runs layers [i, end); the loop below only does their bookkeeping
                end, mk = self.mks[i]
                names = [self.specs[j].name for j in range(i, end)]
                ys_needed, res_from = [], []
                ident0 = None
                for j in range(i, end):
                    sj = self.specs[j]
                    ys_needed.append(taps is not None or sj.name in self._res_src or (j == n - 1 and not keep_table))
                    if sj.residual is None:
                        res_from.append(None)
                    elif sj.residual in names:
                        res_from.append(names.index(sj.residual))
                    else:
                        if ident0 is not None and outs[sj.residual] is not ident0:
                            raise ValueError(f"{mk.name}: two outside identities")
                        ident0 = outs[sj.residual]
                        res_from.append(-1)
                own = cur is not table or free_input
                ys, group_out = mk(cur, indices[spec.index], ident0, ys_needed, dead_skip=taps is None,
                                   reuse_input=own, res_from=res_from)
                group_ys = {names[j]: ys[j] for j in range(len(names))}
                group_end = end
                if own and group_out is not cur:
                    free_maps(cur)
                cur = None
            if i < group_end:
                y = group_ys[spec.name]
                nxt = group_out if i == group_end - 1 else None
            elif k6 is not None:
                y, nxt = k6(cur, indices[spec.index], ident, write_y=need_y, dead_skip=taps is None,
                                 table_memory=self.table_memory)
            else:
                y = stock(cur, indices[spec.index], ident)
                nxt = stock.to_table(y) if (not last or keep_table) else None
            if cur is not None and (cur is not table or free_input):
                free_maps(cur)
            for name, lu in list(self._last_use.items()):
                if lu == i and name in outs and taps is None:
                    free_maps(outs.pop(name))
            if y is not None:
                outs[spec.name] = y
                if taps is not None:
                    taps[spec.name] = y
            if i < group_end and nxt is None:
                continue
            cur = nxt
            if not last and taps is None and spec.name not in self._last_use and spec.name in outs:
                free_maps(outs.pop(spec.name))
        if not keep_table and cur is not None:
            free_maps(cur)
            cur = None
        return y, cur

    def release(self) -> None:
        for k in self.k6:
            if k is not None:
                k.release()
        self.stock.release()

    def describe(self) -> Dict[str, Any]:
        d = self.stock.describe()
        d["kernel"] = "k6"
        d["k6"] = [k.describe() if k is not None else None for k in self.k6]
        if self.mks:
            d["mks"] = [mk.describe() for _, mk in self.mks.values()]
        return d
