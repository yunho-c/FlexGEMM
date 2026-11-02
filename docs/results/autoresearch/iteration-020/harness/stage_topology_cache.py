"""One bounded integer topology retained only within a decoder forward."""
import torch
from genrecon.backends.tenstorrent.sparse_decoder import TopologyCache


def metadata_usage(top, coords, banks):
    """Aligned DRAM bank bound from real page metadata; no feature buffers."""
    import ttnn
    tensors=[top.indices,*top.gathers.values()]
    for plan in top.digit_plans.values():
        tensors.extend(chunk[2] for chunk in plan.chunks)
    buffers={}
    for t in tensors:
        assert t.dtype==ttnn.uint32 and t.memory_config()==ttnn.DRAM_MEMORY_CONFIG
        address=int(t.buffer_address())
        pages=int(t.buffer_num_pages());page=int(t.buffer_aligned_page_size())
        buffers[address]={'address':address,'pages':pages,'aligned_page_bytes':page,
            'bank_bound_bytes':((pages+banks-1)//banks)*page*banks}
    return {'host_bytes':top.host_indices.untyped_storage().nbytes()+coords.untyped_storage().nbytes(),
        'device_bytes':sum(b['bank_bound_bytes'] for b in buffers.values()),
        'banks':banks,'buffers':list(buffers.values())}


class StageTopologyCache(TopologyCache):
    def __init__(self, max_cached_rows=262144, *, host_limit=256<<20, device_limit=256<<20):
        super().__init__(max_cached_rows)
        if host_limit<0 or device_limit<0:raise ValueError('cache limits must be nonnegative')
        self.host_limit,self.device_limit=host_limit,device_limit
        self.large=None;self.active=False;self.events=[];self.banks=None
        self.large_hits=0;self.large_builds=0

    def release_large(self, reason):
        if self.large is not None:
            self.events.append({'event':'release','reason':reason,'rows':len(self.large[1])})
        self.large=None

    def begin(self):
        self.release_large('begin');self.active=True;self.events=[]
        self.large_hits=0;self.large_builds=0
        self.initial_counts=(self.hits,self.builds)

    def finish(self):
        self.release_large('return_or_error');self.active=False
        return {'host_limit_bytes':self.host_limit,'device_limit_bytes':self.device_limit,
            'large_hits':self.large_hits,'large_builds':self.large_builds,
            'all_hits':self.hits-self.initial_counts[0],
            'all_builds':self.builds-self.initial_counts[1],
            'large_entries_after_return':int(self.large is not None),'events':list(self.events)}

    def clear(self):
        self.release_large('clear');self.banks=None
        super().clear()

    def get(self, coords, batch, spatial, kernel, dilation, stage, device, masked):
        if masked:
            self.release_large('masked_layout')
            return super().get(coords,batch,spatial,kernel,dilation,stage,device,masked)
        if self.device is not device:
            self.release_large('device_change');self.banks=None
        key=(batch,tuple(spatial),tuple(kernel),tuple(dilation),stage,masked)
        if self.large is not None:
            old_key,old_coords,top=self.large
            if key==old_key and top.device is device and torch.equal(old_coords,coords):
                self.hits+=1;self.large_hits+=1
                return top
            self.release_large('layout_change')
        top=super().get(coords,batch,spatial,kernel,dilation,stage,device,masked)
        if self.active and len(coords)>self.max_cached_rows:
            self.large_builds+=1
            host_bytes=top.host_indices.untyped_storage().nbytes()+coords.numel()*coords.element_size()
            if host_bytes<=self.host_limit:
                self.large=(key,coords.clone(),top)
            else:
                self.events.append({'event':'reject','reason':'host_budget','rows':len(coords),
                    'host_bytes':host_bytes})
        return top

    def after_conv(self, name):
        if self.large is None:return
        # The wide conv's own map is needed until dispatch, but no stage map is
        # retained into C2S feature expansion. No explicit deallocation races.
        if name.endswith('.conv1'):
            self.release_large('before_c2s');return
        import ttnn
        _,coords,top=self.large
        if self.banks is None:self.banks=ttnn.get_memory_view(top.device,ttnn.BufferType.DRAM).num_banks
        usage=metadata_usage(top,coords,self.banks)
        event=dict(usage,event='retain',rows=len(coords),layer=name)
        if usage['host_bytes']>self.host_limit or usage['device_bytes']>self.device_limit:
            event.update(event='reject',reason='metadata_budget')
            self.events.append(event);self.release_large('metadata_budget')
        else:self.events.append(event)
