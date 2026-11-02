"""Verify iteration 017 archived evidence without opening a device."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
sys.path.insert(0,str(REPO))
from benchmarks.tenstorrent.memory import workspace_bound
from benchmarks.tenstorrent.sweep import comparison,summarize_comparisons

def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
archive=read(ROOT/'archive.json')
assert digest(REPO/'benchmarks/tenstorrent/memory.py')==archive['memory_source_sha256']
for e in archive['files']+[archive['allocation_event_bundle']]:assert digest(ROOT/e['path'])==e['sha256'],e['path']
summary=read(ROOT/'campaign-summary.json')
assert summary['status']=='complete' and len(summary['trials'])==24 and len(summary['unsupported'])==12
for t in summary['trials']:
 r=read(ROOT/'prototype-cases'/f"{t['case']}.json")
 assert t['status']==r['status']=='passed_probe_gates'
 assert all(c['passed'] for c in r['checks']) and r['fidelity']=='HiFi3'
 for key in ('input_unchanged','source_unchanged','fixture_unchanged','contract_unchanged','probe_unchanged','comparator_unchanged'):assert r[key]

bundle=json.loads(gzip.decompress((ROOT/'allocation-events.json.gz').read_bytes()))
assert len(bundle)==33
maximum=0
for name,events in bundle.items():
 r=read(ROOT/name);assert r['status']=='passed_probe_gates'
 w=r['workspace'];o=w['excluded_final_output']
 b=workspace_bound(events,dram_banks=w['dram_banks'],l1_capacity_bytes=w['all_l1_capacity_bytes'],limit_bytes=w['workspace_limit_bytes'],output_address=o['address'],expected_output_bytes=o['expected_padded_bytes'])
 assert b['trace_valid'] and b['within_limit']
 for key in ('new_dram_peak_bytes','dram_allocations','inclusive_upper_bound_bytes','temporary_dram_peak_bytes','workspace_upper_bound_bytes'):assert b[key]==w[key],(name,key)
 assert events[b['excluded_final_output']['event_index']]['counter']==o['event_index']
 maximum=max(maximum,b['workspace_upper_bound_bytes'])

scripts={digest(p) for p in ROOT.rglob('*.py')}
runners=list(ROOT.rglob('*.runner.json'));assert len(runners)==50
codes=[]
for p in runners:
 runner=read(p);r=read(p.with_name(p.name.replace('.runner.json','.json')))
 failed=p.name in ('product-oracle.runner.json','product-oracle-bf16weights.runner.json')
 assert runner['returncode']==int(failed) and runner['timeout'] is None and not runner['device_recovery_required']
 codes.append(runner['returncode'])
 if failed:assert r['status']=='failure' and 'exact_integer_match' in r['error']
 else:assert r['status'] in ('passed','passed_probe_gates')
 if 'probe_sha256' in r:assert r['probe_sha256'] in scripts
 if 'comparator_sha256' in r:
  assert any(all(digest(ROOT/folder/name)==sha for name,sha in r['comparator_sha256'].items()) for folder in ('prototype','prototype-v0','bf16-weight-control'))
assert codes.count(0)==48 and codes.count(1)==2
oracle=read(ROOT/'product-baseline-comparison.json');checks=[c for row in oracle['cases'] for c in row['checks']]
assert oracle['status']=='passed' and len(checks)==48
assert all(c['k6_matches_incumbent'] and c['k6_vs_incumbent_max_abs_error']==0 for c in checks)
assert sum(c['exact_integer_match'] for c in checks)==32
assert all(c['exact_integer_match']==c['incumbent_product_exact_integer_match'] for c in checks)

for directory in ('confirm-narrow','confirm-largest','confirm-narrow-extra'):
 s=read(ROOT/directory/'summary.json');assert s['status']=='complete'
 for row in s['comparisons']:
  offset=row['round']*3
  reports=[read(ROOT/directory/t['report']) for t in s['trials'][offset:offset+3]]
  reports[1]=dict(reports[1],status='passed')
  assert reports[0]['case']['sha256']==reports[1]['fixture_sha256']==reports[2]['case']['sha256']
  assert comparison(*reports)==row['comparison']
original=read(ROOT/'confirm-narrow/summary.json');extra=read(ROOT/'confirm-narrow-extra/summary.json')
clean=original['comparisons'][1:]+extra['comparisons']
for i,row in enumerate(clean):row['round']=i
clean_report=read(ROOT/'clean-narrow-confirmation.json')
assert clean_report['comparisons']==clean
assert clean_report['results']==summarize_comparisons(clean,[original['case']],3)
assert read(ROOT/'timing-interference-note.json')['affected_round']=='confirm-narrow round 0'
for row in read(ROOT/'unpaired-comparison.json')['cases']:
 bpath=ROOT.parent/row['baseline_report'];assert digest(bpath)==row['baseline_report_sha256']
 b=read(bpath);r=read(ROOT/'prototype-cases'/f"{row['case']}.json")
 assert b['case']['sha256']==r['fixture_sha256']
 assert row['baseline_ms']==b['timing']['median_ms'] and row['prototype_ms']==r['timing']['median_ms']
 assert row['unpaired_ratio']==row['baseline_ms']/row['prototype_ms']
print(f"Verified {len(archive['files'])} files, 33 memory replays, 50 normal exits "
      f"(48 zero, 2 retained oracle failures), 48 incumbent-equivalent products; "
      f"maximum workspace {maximum} bytes")
