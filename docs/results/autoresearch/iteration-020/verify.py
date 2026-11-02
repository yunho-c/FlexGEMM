"""Audit the stage-cache experiment without opening the device."""
import hashlib,json,statistics
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
archive=read(ROOT/'archive.json')
for e in archive['files']:
    p=ROOT/e['path'];assert sha(p)==e['sha256'] and p.stat().st_size==e['bytes'],e['path']
names=('baseline-a','candidate-b','baseline-c')
runs={n:read(ROOT/n/'result.json') for n in names}
plan=read(ROOT/'plan.json');comparison=read(ROOT/'comparison.json');decision=read(ROOT/'decision.json')
for n,r in runs.items():
    runner=read(ROOT/(n+'.runner.json'))
    assert runner['returncode']==1 and runner['timeout'] is None and not runner['device_recovery_required']
    assert runner['status']=='complete' and r['phase']=='complete'
    assert r['status']=='numerical_failure' and r['sources_unchanged']
    assert r['flex_gemm']['commit']==plan['kernel_commit']
    assert r['algorithm']=='digit_implicit_gemm'
    assert r['topology_policy']==('stage' if n=='candidate-b' else 'baseline')
    assert r['gate']=={'relative_l2':.0001,'pcc':.999999,'coordinates':'exact','subdivision_signs':'exact',
        'geometry_edge_signs':'exact','scope':'decoder numerical parity; real-scene reconstruction quality is separate'}
    for key in ('gate','reference_files_sha256','checkpoint_sha256','integration_source_sha256','flex_gemm'):
        assert r[key]==runs['baseline-a'][key]
    assert r['topology_cache_max_rows']==262144
    assert not r['shape_predicted']['coordinates_equal'] and not r['fixed_shape']['captured_wide']['passed']
    assert r['fixed_shape']['shape_fixed']['features']['relative_l2']>r['gate']['relative_l2']
    for label,w in r['warm'].items():
        assert len(w['seconds'])==len(w['stats'])==3
        assert w['repeat_outputs_equal'] and w['repeat_guides_equal']
        assert statistics.median(w['seconds'])==w['median_seconds']
    for path,digest in r['integration_source_sha256'].items():assert sha(ROOT/'harness'/Path(path).name)==digest
for name,digest in plan['integrity'].items():assert sha(ROOT/'harness'/name)==digest
count=0
for group in comparison['comparisons'].values():
    for value in group.values():
        for v in value if isinstance(value,list) else [value]:
            count+=1
            assert v['coordinates_equal'] and v['common_features_bitwise_equal'] and v['strict_continuous_gate_passed']
            assert v['common_features']['max_abs']==v['common_features']['relative_l2']==0
            assert v.get('common_edge_sign_mismatches',0)==v.get('common_sign_mismatches',0)==0
assert count==36
for label,t in comparison['timing'].items():
    med={n:statistics.median(runs[n]['warm'][label]['seconds']) for n in names}
    assert med==t['median_seconds'];anchor=(med['baseline-a']+med['baseline-c'])/2
    assert t['bracket_ratio']==anchor/med['candidate-b']
    assert t['baseline_drift_fraction']==abs(med['baseline-a']-med['baseline-c'])/anchor
assert decision['timing']==comparison['timing'] and len(comparison['artifacts'])==18
r=runs['candidate-b']
stats=[r['fixed_shape']['stats'],r['shape_stats'],r['texture_fixed_stats'],r['texture_predicted_stats']]
stats += [s for w in r['warm'].values() for s in w['stats']]
assert len(stats)==10
host_max=device_max=0
for s in stats:
    c=s['stage_cache'];assert c['large_entries_after_return']==0 and c['large_hits']==5 and c['large_builds']==2
    assert c['host_limit_bytes']==c['device_limit_bytes']==256<<20
    assert sum(e.get('reason')=='before_c2s' for e in c['events'])==1
    assert sum(e.get('reason')=='host_budget' for e in c['events'])==1
    retained=[e for e in c['events'] if e['event']=='retain'];assert len(retained)==5
    for e in retained:
        assert e['host_bytes']<=c['host_limit_bytes'] and e['device_bytes']<=c['device_limit_bytes']
        assert len({b['address'] for b in e['buffers']})==len(e['buffers'])
        assert e['device_bytes']==sum(b['bank_bound_bytes'] for b in e['buffers'])
        for b in e['buffers']:
            assert b['bank_bound_bytes']==((b['pages']+e['banks']-1)//e['banks'])*b['aligned_page_bytes']*e['banks']
        host_max=max(host_max,e['host_bytes']);device_max=max(device_max,e['device_bytes'])
assert host_max==decision['maximum_retained_host_tensor_bytes']
assert device_max==decision['maximum_retained_device_metadata_bytes']
for name,count in [('watcher-tests.xml',3),('cpu-tests.xml',1),('exception-tests.xml',1)]:
    suite=ET.parse(ROOT/name).find('testsuite');assert int(suite.attrib['tests'])==count
    assert all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped'))
watcher=read(ROOT/'watcher-tests.runner.json')
assert watcher['returncode']==0 and watcher['timeout'] is None and not watcher['device_recovery_required']
print(f'Verified {len(archive["files"])} files, 36 bitwise-equal matched records, 10 bounded cache lifetimes, 3 Watcher/2 CPU tests, 4 normal workers; retained host/device maxima {host_max}/{device_max} bytes; original CPU gates remain failed')
