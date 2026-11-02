import os
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,'/home/user/GitHub/GenRecon')
sys.path.insert(0,'/home/user/GitHub/flexgemm-autoresearch')
from genrecon.modules.sparse import config, SparseTensor
config.set_conv_backend('reference')
from stage_topology_cache import StageTopologyCache,metadata_usage
from genrecon.backends.tenstorrent.sparse_decoder import TopologyCache


def test_host_budget_and_actual_coordinate_identity(monkeypatch):
    import flex_gemm.tenstorrent as fg
    from types import SimpleNamespace
    calls=[]
    def build(coords,*args):
        calls.append(coords.clone());return torch.arange(len(coords)).reshape(-1,1)
    monkeypatch.setattr(fg,'build_subm_neighbor_map',build)
    monkeypatch.setattr(fg,'prepare_sparse_topology',lambda m,n,device,masked:
        SimpleNamespace(host_indices=m.clone(),device=device))
    class Device:
        def id(self):return 0
    dev=Device();c=StageTopologyCache(max_cached_rows=1);coords=torch.zeros(3,4,dtype=torch.int32)
    def get(x=coords,stage=0,device=dev):return c.get(x,1,(8,8,8),(1,1,1),(1,1,1),stage,device,False)
    c.begin();a=get();assert get(coords.clone()) is a and c.large_hits==1
    changed=coords.clone();changed[1,1]=1
    assert get(changed) is not a
    assert get(coords,1) is not a
    assert get(coords,1,Device()) is not a
    c.finish();assert c.large is None
    c.begin();assert get() is not a
    c.finish()
    c.host_limit=0;c.begin();get();assert c.large is None;c.finish()
    assert len(calls)==6


@pytest.fixture(scope='module')
def device():
    import ttnn
    torch.set_num_threads(8)
    d=ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


def test_device_metadata_bound_and_new_features(device):
    import ttnn
    from flex_gemm.tenstorrent import prepare_sparse_digit_weights,sparse_conv_device
    gen=torch.Generator().manual_seed(20262201)
    coords=torch.stack([torch.zeros(65),torch.arange(65),torch.zeros(65),torch.zeros(65)],1).int()
    weight=prepare_sparse_digit_weights(torch.randn(35,3,3,3,17,generator=gen)*.02,device=device)
    cache=StageTopologyCache(max_cached_rows=1);cache.begin()
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    top=cache.get(coords,1,(65,1,1),(3,3,3),(1,1,1),0,device,False)
    for i in range(2):
        x=torch.randn(65,17,generator=gen).bfloat16()
        xd=ttnn.from_torch(x,device=device,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT)
        actual=sparse_conv_device(xd,weight,top,algorithm='digit_implicit_gemm',row_chunk=32)
        expected=sparse_conv_device(xd,weight,top,algorithm='digit_explicit_gemm',row_chunk=32)
        torch.testing.assert_close(ttnn.to_torch(actual),ttnn.to_torch(expected),atol=.025,rtol=.025)
        cache.after_conv('blocks.0.0.conv')
        assert cache.get(coords.clone(),1,(65,1,1),(3,3,3),(1,1,1),0,device,False) is top
    trace=ttnn.graph.end_graph_capture()
    usage=metadata_usage(top,cache.large[1],cache.banks)
    import json
    if isinstance(trace,str):trace=json.loads(trace)
    allocations={int(n['params']['address']):int(n['params']['max_size_per_bank'])*cache.banks
        for n in trace if n.get('node_type')=='buffer_allocate' and n.get('params',{}).get('type')=='DRAM'}
    for b in usage['buffers']:assert b['bank_bound_bytes']==allocations[b['address']]
    cache.device_limit=0;cache.after_conv('blocks.0.0.conv');assert cache.large is None
    cache.finish()


@pytest.mark.parametrize('changed',[False,True])
def test_decoder_stage_lifetime_and_feature_refresh(device,changed,monkeypatch):
    from experiments.tenstorrent.test_sparse_decoder import problem
    from genrecon_digit_adapter import enable_digit_decoder
    monkeypatch.setenv('FLEXGEMM_DIGIT_ALGORITHM','digit_implicit_gemm')
    model,x=problem()
    with torch.no_grad():_,guides=model.forward_features(x,return_subs=True)
    if changed:x=x.replace(x.feats*.5+.03125)
    enable_digit_decoder(model,device,topology_cache=TopologyCache(max_cached_rows=1),precision='bf16')
    baseline=model.forward_features(x,guide_subs=guides)
    model.disable_tenstorrent()
    cache=StageTopologyCache(max_cached_rows=1)
    enable_digit_decoder(model,device,topology_cache=cache,precision='bf16')
    actual=model.forward_features(x,guide_subs=guides)
    torch.testing.assert_close(actual.coords,baseline.coords,atol=0,rtol=0)
    torch.testing.assert_close(actual.feats,baseline.feats,atol=0,rtol=0)
    assert cache.large is None and not cache.active
    stats=model._tenstorrent_backend.stats['stage_cache']
    assert stats['large_hits']>0 and any(e.get('reason')=='before_c2s' for e in stats['events'])
    with pytest.raises(ValueError,match='finite'):
        model.forward_features(x.replace(torch.full_like(x.feats,float('nan'))))
    assert cache.large is None and not cache.active
    model.disable_tenstorrent()
