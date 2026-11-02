"""Unverified Wormhole comparator adapter; no published performance claim.

K6 source is pinned in provenance.json. Its kernels are unchanged. Conversion
and bounded output assembly below belong inside each timed forward. Static
weights and integer indices may be prepared once. No feature cache is kept.
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch


def round32(value):
    return (value + 31) // 32 * 32


class Spec:
    def __init__(self, weight, bias):
        self.name = 'matched_sparse_convolution'
        self.cout, self.cin = weight.shape[0], weight.shape[-1]
        self.cin_pad, self.cout_pad = round32(self.cin), round32(self.cout)
        w = weight.bfloat16().float().reshape(self.cout, -1, self.cin)
        self.kernel_volume = w.shape[1]
        self.relu, self.out_terms = False, 1
        self._weight = torch.zeros(self.kernel_volume, self.cin_pad, self.cout_pad)
        self._weight[:, :self.cin, :self.cout] = w.permute(1, 2, 0)
        self._bias = torch.zeros(1, self.cout_pad)
        if bias is not None:
            self._bias[0, :self.cout] = bias.bfloat16().float()

    def matrix(self):
        return self._weight.reshape(-1, self.cout_pad).numpy()

    def bias_row(self):
        return self._bias.numpy()


def index_rows(mapping, input_rows):
    """K6 page order: one output-row tile, then tap, then row within tile."""
    rows, volume = mapping.shape
    mapping = mapping.long().clone()
    mapping[mapping == 0xffffffff] = -1
    if bool(((mapping < -1) | (mapping >= input_rows)).any()):
        raise ValueError('invalid neighbor index')
    padded = torch.full((round32(rows), volume), input_rows, dtype=torch.int64)
    padded[:rows] = torch.where(mapping < 0, input_rows, mapping)
    return padded.reshape(-1, 32, volume).permute(0, 2, 1).reshape(-1, volume * 32).to(torch.uint32)


class Adapter:
    def __init__(self, weight, bias, mapping, input_rows, *, device, fidelity, row_chunk=8192):
        import ttnn
        if fidelity not in ('HiFi3', 'HiFi4') or row_chunk <= 0 or row_chunk % 32:
            raise ValueError('explicit fidelity and positive tile-aligned row chunk required')
        self.spec = Spec(weight, bias)
        if self.spec.kernel_volume > 32 or self.spec.kernel_volume != mapping.shape[1]:
            raise ValueError('K6 supports at most 32 taps and requires matching topology')
        loader = importlib.util.spec_from_file_location('pinned_k6_comparator', Path(__file__).with_name('sparse_k6.py'))
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        self.layer = module.K6Layer(self.spec, fidelity=fidelity)
        self.device, self.input_rows, self.rows = device, input_rows, len(mapping)
        self.fidelity, self.row_chunk = fidelity, row_chunk
        self.chunks = []
        grid = device.compute_with_storage_grid_size()
        for start in range(0, self.rows, row_chunk):
            stop = min(start + row_chunk, self.rows)
            indices = index_rows(mapping[start:stop], input_rows)
            nbmax = (indices.shape[0] + grid.x * grid.y - 1) // (grid.x * grid.y)
            tg, nch = self.layer.plan(nbmax)
            cb_bytes = self.layer.cb_bytes(tg, nch, nbmax, has_res=False, write_y=False)
            if cb_bytes > module.CB_BUDGET:
                raise ValueError('K6 per-core CB plan exceeds budget; channel partitioning is required')
            index = ttnn.from_torch(indices, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.chunks.append((start, stop, index, {'tap_chunk':tg, 'tap_chunks':nch, 'cb_bytes':cb_bytes}))
        self.layer._upload(device)

    def __call__(self, features):
        import ttnn
        # Use the incumbent's exact output-copy helper; include its source hash
        # in a future comparator report. It changes neither arithmetic nor rows.
        from flex_gemm.tenstorrent.digits import _copy_output_rows
        if tuple(features.shape) != (self.input_rows, self.spec.cin):
            raise ValueError('feature shape mismatch')
        table = ttnn.to_layout(features, ttnn.ROW_MAJOR_LAYOUT)
        table = ttnn.pad(table, [(0, 1), (0, self.spec.cin_pad - self.spec.cin)], 0)
        output = ttnn.allocate_tensor_on_device(ttnn.Shape([self.rows, self.spec.cout]),
            ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG)
        for start, stop, index, plan in self.chunks:
            _, rows = self.layer(table, index, identity=None, write_y=False,
                                 dead_skip=False, table_memory='dram')
            rows = ttnn.slice(rows, [0, 0, 0, 0], [1, 1, stop - start, self.spec.cout_pad])
            tiled = ttnn.to_layout(ttnn.reshape(rows, [stop - start, self.spec.cout_pad]), ttnn.TILE_LAYOUT)
            tiled = ttnn.slice(tiled, [0, 0], [stop - start, self.spec.cout])
            _copy_output_rows(tiled, output, start)
            del rows, tiled
        return output
