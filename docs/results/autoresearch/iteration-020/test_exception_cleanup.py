"""Check an exception after metadata retention, not just input validation."""
import sys
sys.path.insert(0,'/home/user/GitHub/GenRecon')
sys.path.insert(0,'/home/user/GitHub/flexgemm-autoresearch')
sys.path.insert(0,'/home/user/tt-scratchpad/flexgemm-autoresearch/integration-stage-cache-v1')
import pytest
import torch
from genrecon_digit_adapter import DigitSparseDecoder,TenstorrentSparseDecoder
from stage_topology_cache import StageTopologyCache


def test_exception_after_retention_releases_metadata(monkeypatch):
    backend=DigitSparseDecoder.__new__(DigitSparseDecoder)
    backend.topology_cache=StageTopologyCache(max_cached_rows=1)
    backend.stats={}
    def fail_after_retention(self,*args,**kwargs):
        self.topology_cache.large=((),torch.zeros(3,4,dtype=torch.int32),object())
        raise RuntimeError('injected failure after topology retention')
    monkeypatch.setattr(TenstorrentSparseDecoder,'run',fail_after_retention)
    with pytest.raises(RuntimeError,match='injected failure'):
        backend.run(None)
    assert backend.topology_cache.large is None and not backend.topology_cache.active
    assert backend.stats['stage_cache']['large_entries_after_return']==0
    assert backend.stats['stage_cache']['events']==[{'event':'release','reason':'return_or_error','rows':3}]
