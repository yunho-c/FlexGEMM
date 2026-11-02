"""Verify archived complete-decoder evidence without opening a device."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

archive=read(ROOT/'archive.json')
for e in archive['files']:
    p=ROOT/e['path'];assert sha(p)==e['sha256'] and p.stat().st_size==e['bytes'],e['path']
names=('baseline-a','candidate-b','baseline-c')
runs={n:read(ROOT/n/'result.json') for n in names}
plan=read(ROOT/'plan.json');comparison=read(ROOT/'comparison.json');decision=read(ROOT/'decision.json')
assert runs['candidate-b']['flex_gemm']['commit']==plan['candidate_commit']==decision['candidate_commit']
assert runs['baseline-a']['flex_gemm']['commit']==runs['baseline-c']['flex_gemm']['commit']==plan['baseline_commit']
assert decision['saved_cpu_gates_unchanged']
for n,r in runs.items():
    runner=read(ROOT/(n+'.runner.json'))
    assert runner['returncode']==1 and runner['timeout'] is None and not runner['device_recovery_required']
    assert runner['status']=='complete' and r['phase']=='complete'
    assert r['status']=='numerical_failure' and r['sources_unchanged']
    assert r['gate']=={'relative_l2':.0001,'pcc':.999999,'coordinates':'exact','subdivision_signs':'exact',
        'geometry_edge_signs':'exact','scope':'decoder numerical parity; real-scene reconstruction quality is separate'}
    for key in ('gate','reference_files_sha256','checkpoint_sha256','integration_source_sha256'):
        assert r[key]==runs['baseline-a'][key]
    assert r['topology_cache_max_rows']==262144
    assert not r['shape_predicted']['coordinates_equal']
    assert not r['fixed_shape']['captured_wide']['passed']
    assert r['fixed_shape']['shape_fixed']['features']['relative_l2']>r['gate']['relative_l2']
    for label,w in r['warm'].items():
        assert len(w['seconds'])==len(w['stats'])==2
        assert w['repeat_outputs_equal'] and w['repeat_guides_equal']
        assert statistics.median(w['seconds'])==w['median_seconds']
        if n=='candidate-b':
            for stats in w['stats']:
                calls=stats['digit_calls'];assert len(calls)==40
                assert sum(c['path']=='digit_implicit_gemm' for c in calls)==16
                assert sum(c['path']=='digit_explicit_gemm' for c in calls)==24
    for path,digest in r['integration_source_sha256'].items():
        assert sha(ROOT/'harness'/Path(path).name)==digest
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
    assert med==t['median_seconds']
    anchor=(med['baseline-a']+med['baseline-c'])/2
    assert t['bracket_ratio']==anchor/med['candidate-b']
    assert t['baseline_drift_fraction']==abs(med['baseline-a']-med['baseline-c'])/anchor
assert decision['timing']==comparison['timing']
assert len(comparison['artifacts'])==18
print(f'Verified {len(archive["files"])} archived files, 3 normally completed workers, unchanged failed CPU gates, 36 bitwise-equal matched output/guide records, and diagnostic A/B/A timings')
