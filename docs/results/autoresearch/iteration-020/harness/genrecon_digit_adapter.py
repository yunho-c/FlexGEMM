"""Isolated BF16 GenRecon adapter using validated FlexGEMM digit convolutions.

Only sparse convolutions change. Dense linear, normalization, residual and
subdivision logic remain inherited. This is an integration experiment, not a
claim of FP32 decoder parity. No original GenRecon source is modified.
"""
import time
import os
from genrecon.backends.tenstorrent.sparse_decoder import TenstorrentSparseDecoder

class DigitSparseDecoder(TenstorrentSparseDecoder):
    def run(self, *args, **kwargs):
        from stage_topology_cache import StageTopologyCache
        cache=self.topology_cache
        if not isinstance(cache,StageTopologyCache):return super().run(*args,**kwargs)
        cache.begin()
        try:return super().run(*args,**kwargs)
        finally:self.stats['stage_cache']=cache.finish()

    def c2s(self, *args, **kwargs):
        from stage_topology_cache import StageTopologyCache
        if isinstance(self.topology_cache,StageTopologyCache):
            assert self.topology_cache.large is None,'large metadata retained into C2S'
        return super().c2s(*args,**kwargs)

    def __init__(self, model, device, **kwargs):
        import ttnn
        from genrecon.modules.sparse.conv.conv import SparseConv3d
        from flex_gemm.tenstorrent import prepare_sparse_digit_weights
        if kwargs.get("precision", "bf16") != "bf16":
            raise ValueError("this adapter implements the BF16 application track only")
        kwargs["precision"] = "bf16"
        kwargs["algorithm"] = "explicit_gemm"
        start = time.perf_counter()
        super().__init__(model, device, **kwargs)
        self.digit_layers = []
        for name, module in model.named_modules():
            if isinstance(module, SparseConv3d):
                del self.weights[name]
                self.weights[name] = prepare_sparse_digit_weights(module.weight.detach(), device=device,
                    bias=None if module.bias is None else module.bias.detach())
                self.digit_layers.append(name)
        self.algorithm = os.environ["FLEXGEMM_DIGIT_ALGORITHM"]
        if self.algorithm not in ("digit_explicit_gemm", "digit_implicit_gemm"):
            raise ValueError("unsupported integration algorithm")
        ttnn.synchronize_device(device)
        self.prepare_seconds = time.perf_counter() - start

    def conv(self, name, module, x, coords, batch, spatial, stage):
        from flex_gemm.tenstorrent import sparse_conv_device
        start = time.perf_counter()
        top = self.topology_cache.get(coords, batch, spatial, module.kernel_size,
            module.dilation, stage, self.device, False)
        self.stats["topology_seconds"] += time.perf_counter() - start
        cp = (module.in_channels + 31) // 32 * 32
        row_chunk = max(32, 1048576 // cp // 32 * 32)
        result = sparse_conv_device(x, self.weights[name], top,
            algorithm=self.algorithm, row_chunk=row_chunk)
        path, reason = self.algorithm, None
        if self.algorithm == "digit_implicit_gemm":
            weight = self.weights[name]
            plan = top.digit_plans[(weight.ci, weight.co, weight.volume, row_chunk)]
            path, reason = plan.info["path"], plan.info["reason"]
        self.stats.setdefault("digit_calls", []).append({"name": name,
            "rows": len(coords), "ci": module.in_channels, "co": module.out_channels,
            "row_chunk": row_chunk, "path": path, "fallback_reason": reason})
        from stage_topology_cache import StageTopologyCache
        if isinstance(self.topology_cache,StageTopologyCache):self.topology_cache.after_conv(name)
        return result

def enable_digit_decoder(model, device, **kwargs):
    model._tenstorrent_backend = DigitSparseDecoder(model, device, **kwargs)
    return model
