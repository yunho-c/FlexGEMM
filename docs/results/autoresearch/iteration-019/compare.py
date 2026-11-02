"""Compare complete decoder outputs after all timing workers have exited."""
import hashlib
import json
from pathlib import Path
import statistics
import torch

ROOT = Path('/home/user/tt-scratchpad/flexgemm-autoresearch/campaign-019')
torch.set_num_threads(16)

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def metrics(a, e):
    a, e = a.double().reshape(-1), e.double().reshape(-1)
    assert a.shape == e.shape
    if not len(a):
        return dict(finite=True, relative_l2=0., pcc=1., max_abs=0.)
    ac, ec = a-a.mean(), e-e.mean()
    return dict(finite=bool(torch.isfinite(a).all() and torch.isfinite(e).all()),
        relative_l2=float((a-e).norm()/e.norm().clamp_min(1e-30)),
        pcc=float((ac*ec).sum()/(ac.norm()*ec.norm()).clamp_min(1e-30)),
        max_abs=float((a-e).abs().max()))

def compare(a, e, geometry=False, signs=False):
    same = torch.equal(a['coords'],e['coords'])
    if same:
        av, ev = a['feats'],e['feats']
    else:
        bounds = torch.maximum(a['coords'].amax(0),e['coords'].amax(0)).long()+1
        def keys(c):
            k=c[:,0].long()
            for axis in range(1,4): k=k*bounds[axis]+c[:,axis]
            return k
        ak=keys(a['coords']); ek,order=keys(e['coords']).sort()
        pos=torch.searchsorted(ek,ak)
        found=(pos<len(ek)) & (ek[pos.clamp_max(len(ek)-1)]==ak)
        av=a['feats'][found];ev=e['feats'][order[pos[found]]]
    m=metrics(av,ev)
    r=dict(coordinates_equal=same, actual_voxels=len(a['coords']),
        reference_voxels=len(e['coords']), common_voxels=len(av),
        missing_voxels=len(e['coords'])-len(av), extra_voxels=len(a['coords'])-len(av),
        common_features=m, common_features_bitwise_equal=torch.equal(av,ev),
        strict_continuous_gate_passed=same and m['finite'] and m['relative_l2']<=1e-4 and m['pcc']>=.999999)
    if geometry:r['common_edge_sign_mismatches']=int(((av[:,3:6]>0)!=(ev[:,3:6]>0)).sum())
    if signs:r['common_sign_mismatches']=int(((av>0)!=(ev>0)).sum())
    return r

def main():
    names=('baseline-a','candidate-b','baseline-c')
    runs={n:read(ROOT/n/'result.json') for n in names}
    for n,r in runs.items():
        supervisor=read(ROOT/(n+'.runner.json'))
        assert supervisor['returncode'] in (0,1) and supervisor['timeout'] is None
        assert not supervisor['device_recovery_required']
        assert r['phase']=='complete' and r['sources_unchanged']
        assert r['gate']==runs['baseline-a']['gate']
        assert r['reference_files_sha256']==runs['baseline-a']['reference_files_sha256']
        assert r['checkpoint_sha256']==runs['baseline-a']['checkpoint_sha256']
        assert r['integration_source_sha256']==runs['baseline-a']['integration_source_sha256']
    result={'scope':'Additional matched BF16 application comparison; does not replace strict saved CPU gates',
        'cpu_statuses':{n:r['status'] for n,r in runs.items()},'comparisons':{},'timing':{},'artifacts':[]}
    files=('shape_tt_fixed.pt','shape_tt_predicted.pt','texture_tt_fixed.pt','texture_tt_predicted.pt')
    guides=('fixed_tt_guides.pt','predicted_tt_guides.pt')
    for actual,expected in [('candidate-b','baseline-a'),('candidate-b','baseline-c'),('baseline-c','baseline-a')]:
        item={}
        for f in files:
            a=torch.load(ROOT/actual/f,map_location='cpu',weights_only=True)
            e=torch.load(ROOT/expected/f,map_location='cpu',weights_only=True)
            item[f]=compare(a,e,geometry=f.startswith('shape'))
            del a,e
        for f in guides:
            a=torch.load(ROOT/actual/f,map_location='cpu',weights_only=True)
            e=torch.load(ROOT/expected/f,map_location='cpu',weights_only=True)
            assert len(a)==len(e)==4
            item[f]=[compare(aa,ee,signs=True) for aa,ee in zip(a,e)]
            del a,e
        result['comparisons'][actual+'_vs_'+expected]=item
    for label in ('shape','texture'):
        times={n:r['warm'][label]['seconds'] for n,r in runs.items()}
        med={n:statistics.median(v) for n,v in times.items()}
        anchor=(med['baseline-a']+med['baseline-c'])/2
        result['timing'][label]={'seconds':times,'median_seconds':med,
            'bracket_ratio':anchor/med['candidate-b'],
            'baseline_drift_fraction':abs(med['baseline-a']-med['baseline-c'])/anchor,
            'scope':'Two warm samples per worker, one A/B/A bracket; diagnostic only'}
    for n in names:
        for p in sorted((ROOT/n).glob('*.pt')):
            result['artifacts'].append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'cpu_statuses':result['cpu_statuses'],'timing':result['timing']},indent=2))

if __name__=='__main__':main()
