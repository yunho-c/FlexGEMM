"""Four-product K6 prototype; fresh feature work and all output work are timed."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def round32(value):
    return (value + 31) // 32 * 32


def index_rows(mapping, input_rows):
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
        from flex_gemm.tenstorrent.digits import prepare_sparse_digit_weights
        if fidelity != 'HiFi3' or row_chunk <= 0 or row_chunk % 32:
            raise ValueError('fixed HiFi3 and positive tile-aligned chunk required')
        self.weights = prepare_sparse_digit_weights(weight, device=device, bias=bias)
        w = self.weights
        self.spec = SimpleNamespace(name='four_product_sparse_convolution', cin=w.ci, cout=w.co,
            cin_pad=round32(w.ci), cout_pad=round32(w.co), kernel_volume=w.volume, relu=False, out_terms=1)
        if w.volume > 32 or w.volume != mapping.shape[1]:
            raise ValueError('K6 requires matching topology and at most 32 taps')
        loader = importlib.util.spec_from_file_location('pinned_digit_k6', Path(__file__).with_name('sparse_k6.py'))
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        self.layers = [module.K6Layer(self.spec, fidelity=fidelity) for _ in range(2)]
        zero_bias = ttnn.zeros([32, self.spec.cout_pad], dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for layer, digit in zip(self.layers, w.digits):
            layer._dev = (ttnn.typecast(digit.weights, ttnn.float32), zero_bias)
        self.device, self.input_rows, self.rows = device, input_rows, len(mapping)
        self.fidelity, self.row_chunk = fidelity, row_chunk
        self.chunks = []
        grid = device.compute_with_storage_grid_size()
        physical = [[device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
                     for x in range(grid.x)] for y in range(grid.y)]
        contiguous = all(all(row[x].x == row[0].x + x and row[x].y == row[0].y
                            for x in range(grid.x)) for row in physical)
        for layer in self.layers:
            layer.mcast = layer.mcast and contiguous
        self.port_settings = {'physical_core_rows': [[[c.x,c.y] for c in row] for row in physical],
            'multicast':self.layers[0].mcast,'split_gather':self.layers[0].split,
            'product_calls':4,'fp32_product_output':True,'unused_bf16_output_included':True,
            'integer_weight_storage':'FP32 encoding of incumbent BF16 integer digits'}
        for start in range(0, self.rows, row_chunk):
            stop = min(start + row_chunk, self.rows)
            indices = index_rows(mapping[start:stop], input_rows)
            nbmax = (indices.shape[0] + grid.x * grid.y - 1) // (grid.x * grid.y)
            tg, nch = self.layers[0].plan(nbmax)
            cb_bytes = self.layers[0].cb_bytes(tg, nch, nbmax, has_res=False, write_y=True)
            if cb_bytes > module.CB_BUDGET:
                raise ValueError('K6 per-core CB plan exceeds budget; fallback or partitioning required')
            index = ttnn.from_torch(indices, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.chunks.append((start,stop,index,{'tap_chunk':tg,'tap_chunks':nch,'cb_bytes':cb_bytes}))

    def product(self, table, index, digit):
        import ttnn
        y, unused = self.layers[digit](table,index,identity=None,write_y=True,
            dead_skip=False,table_memory='dram')
        del unused
        return ttnn.reshape(y,[index.shape[0]*32,self.spec.cout_pad])

    def __call__(self, features):
        import ttnn
        from flex_gemm.tenstorrent.digits import _feature_digits, _scale_and_bias, _copy_output_rows
        if tuple(features.shape) != (self.input_rows,self.spec.cin):
            raise ValueError('feature shape mismatch')
        high, low, fs = _feature_digits(features,self.spec.cin_pad)
        scale = ttnn.multiply(ttnn.repeat(fs,[1,self.weights.co]),self.weights.scale)
        scale_tile = ttnn.multiply(self.weights.scale_tile,fs)
        output = ttnn.allocate_tensor_on_device(ttnn.Shape([self.rows,self.spec.cout]),
            ttnn.bfloat16,ttnn.TILE_LAYOUT,self.device,ttnn.DRAM_MEMORY_CONFIG)
        for start,stop,index,plan in self.chunks:
            top = self.product(high,index,0)
            cross_a, cross_b = self.product(high,index,1), self.product(low,index,0)
            cross = ttnn.add(cross_a,cross_b)
            del cross_a,cross_b
            bottom = self.product(low,index,1)
            result = ttnn.add(ttnn.add(top,ttnn.multiply(cross,1/256)),ttnn.multiply(bottom,1/65536))
            del top,cross,bottom
            result = ttnn.slice(result,[0,0],[stop-start,self.spec.cout])
            result = _scale_and_bias(result,self.weights,scale,scale_tile)
            result = ttnn.typecast(result,ttnn.bfloat16)
            _copy_output_rows(result,output,start)
            del result
        return output
