"""Verify iteration023 operator evidence without opening a device."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
sys.path.insert(0,str(REPO))
from benchmarks.tenstorrent.memory import workspace_bound
from benchmarks.tenstorrent.sweep import comparison,summarize_comparisons

def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=read(ROOT/'archive.json')
assert digest(REPO/'benchmarks/tenstorrent/memory.py')==a['memory_source_sha256']
for e in a['files']+[a['allocation_event_bundle']]:assert digest(ROOT/e['path'])==e['sha256'],e['path']
full=read(ROOT/'full-validation/summary.json');assert full['status']=='complete' and len(full['trials'])==36
paths=[];hashes=[]
for t in full['trials']:
 r=read(ROOT/'full-validation'/f"{t['case']}.json")
 assert t['status']==r['status']=='passed'
 assert all(c['passed'] for c in r['checks'])
 for k in ('input_unchanged','protected_files_unchanged','fixture_unchanged','candidate_source_unchanged'):assert r[k]
 p=r['digit_implicit_policy'];paths.append(p['path'])
 assert r['candidate']['sha256']['flex_gemm/tenstorrent/kernels/k6/k6_gather.hpp']
 hashes.append(r['candidate']['sha256'])
 assert r['work']['model']==('digit_implicit_k6_fp32_output_v1_bf16_weights' if p['path']=='digit_implicit_gemm' else 'digit_explicit_v4')
assert paths.count('digit_implicit_gemm')==24 and paths.count('digit_explicit_gemm')==12
assert all(h==hashes[0] for h in hashes)
regression=read(ROOT/'full-validation/shape-norm-silu-513-known-failure.json')
assert regression['status']=='passed' and regression['digit_implicit_policy']['reason']=='per_core_cb_budget'

bundle=json.loads(gzip.decompress((ROOT/'allocation-events.json.gz').read_bytes()));assert len(bundle)==36
maximum=0
for name,events in bundle.items():
 r=read(ROOT/name);w=r['workspace'];o=w['excluded_final_output']
 b=workspace_bound(events,dram_banks=w['dram_banks'],l1_capacity_bytes=w['all_l1_capacity_bytes'],limit_bytes=w['workspace_limit_bytes'],output_address=o['address'],expected_output_bytes=o['expected_padded_bytes'])
 assert b['trace_valid'] and b['within_limit']
 for k in ('new_dram_peak_bytes','dram_allocations','inclusive_upper_bound_bytes','temporary_dram_peak_bytes','workspace_upper_bound_bytes'):assert b[k]==w[k],(name,k)
 assert events[b['excluded_final_output']['event_index']]['counter']==o['event_index']
 maximum=max(maximum,b['workspace_upper_bound_bytes'])
runners=list(ROOT.rglob('*.runner.json'));assert len(runners)==79
for p in runners:
 r=read(p)
 expected_code=1 if p.name=='application.runner.json' else 0
 assert r['returncode']==expected_code and r['timeout'] is None and not r['device_recovery_required']
for name,count in [('cpu-tests.xml',15),('watcher-tests.xml',22)]:
 suite=ET.parse(ROOT/name).find('testsuite');assert int(suite.attrib['tests'])==count
 assert all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped'))
for name in ('narrow','largest','wide-feasible','fallback'):
 directory=ROOT/f'confirm-{name}';s=read(directory/'summary.json');assert s['status']=='complete' and len(s['trials'])==9
 for row in s['comparisons']:
  start=(row['round']-1)*3 if min(t['round'] for t in s['trials'])==1 else row['round']*3
  reports=[read(directory/t['path']) for t in s['trials'][start:start+3]]
  assert all(r['status']=='passed' for r in reports)
  assert reports[0]['case']['sha256']==reports[1]['case']['sha256']==reports[2]['case']['sha256']
  assert reports[1]['candidate']['sha256']==hashes[0]
  assert comparison(*reports)==row['comparison']
 assert summarize_comparisons(s['comparisons'],s['cases'],3)==s['results']
for row in read(ROOT/'unpaired-comparison.json')['cases']:
 p=ROOT.parent/row['baseline_report'];assert digest(p)==row['baseline_report_sha256']
 baseline=read(p);r=read(ROOT/'full-validation'/f"{row['case']}.json")
 assert baseline['case']['sha256']==r['case']['sha256']
 assert row['baseline_ms']==baseline['timing']['median_ms'] and row['candidate_ms']==r['timing']['median_ms']
 assert row['unpaired_ratio']==row['baseline_ms']/row['candidate_ms']
 assert baseline['digit_implicit_policy']['path']==r['digit_implicit_policy']['path']
 assert [(c['start'],c['end'],c['tap_chunk'],c['tap_chunks']) for c in baseline['digit_implicit_policy']['chunks']]==[(c['start'],c['end'],c['tap_chunk'],c['tap_chunks']) for c in r['digit_implicit_policy']['chunks']]
 old_work,new_work=dict(baseline['work']),dict(r['work'])
 if r['digit_implicit_policy']['path']=='digit_implicit_gemm':
  assert not r['digit_implicit_policy']['unused_bf16_output_included']
  assert old_work.pop('model')=='digit_implicit_k6_fp32_output_v1' and new_work.pop('model')=='digit_implicit_k6_fp32_output_v1_bf16_weights'
  assert old_work.pop('requested_weight_bytes')==2*new_work.pop('requested_weight_bytes')
  assert r['digit_implicit_policy']['weight_tile_bytes']==2048
  assert r['digit_implicit_policy']['cb_eligibility']=='conservative_fp32_weight_bound'
  for old_chunk,new_chunk in zip(baseline['digit_implicit_policy']['chunks'],r['digit_implicit_policy']['chunks']):
   ct,nt=(r['case']['ci']+31)//32,(r['case']['co']+31)//32
   assert old_chunk['cb_bytes']-new_chunk['cb_bytes']==new_chunk['tap_chunk']*ct*nt*2048
 assert old_work==new_work
print(f"Verified {len(a['files'])} files, 36 passing cases (24 implicit/12 fallback), "
      f"36 allocator replays, 79 normal workers, 22 Watcher/15 CPU tests; "
      f"maximum workspace {maximum} bytes")

a=read(ROOT/'application/result.json');assert a['phase']=='complete' and a['status']=='numerical_failure' and a['sources_unchanged']
assert a['flex_gemm']['sha256']==hashes[0]
assert all(w['repeat_outputs_equal'] and w['repeat_guides_equal'] for w in a['warm'].values())
c=read(ROOT/'application-comparison.json');count=0
for value in c['comparisons'].values():
 for e in value if isinstance(value,list) else [value]:
  count+=1;assert e['coordinates_equal'] and e['common_features_bitwise_equal'] and e['common_features']['max_abs']==0
assert count==12
print('Also verified 12 bitwise-identical full-decoder output/guide records; original CPU gates remain failed')

# Replay the post-promotion diagnostics using their archived parser and source.
import copy,importlib.util,tarfile,tempfile
profile_root=ROOT/'profiles'
def archived_module(name, path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module
parser=archived_module('benchmarks.tenstorrent.csv_parser023',profile_root/'csv_parser.py').parse_device_csv
instrument=archived_module('benchmarks.tenstorrent.k6_profile023',profile_root/'worker.py').instrument
analyzer=archived_module('analysis023',profile_root/'analysis_driver.py')
archive=read(ROOT/'archive.json');expected=read(ROOT/'profile-analysis.json')
assert digest(REPO/'docs/results/autoresearch/iteration-022/analysis.json')==expected['baseline_sha256']
expected_by_path={x['report']:x for x in expected['candidate']}
assert len(archive['profile_reports'])==len(expected_by_path)==5
count=0
with tempfile.TemporaryDirectory() as temporary:
 temporary=Path(temporary)
 with tarfile.open(profile_root/'selected-csv.tar.xz','r:xz') as tar:
  members=tar.getmembers();assert len(members)==74
  for member in members:
   assert member.isfile() and Path(member.name).name==member.name and member.name.endswith('.csv')
   (temporary/member.name).write_bytes(tar.extractfile(member).read())
 for entry in archive['profile_reports']:
  raw=gzip.decompress((ROOT/entry['path']).read_bytes());assert hashlib.sha256(raw).hexdigest()==entry['original_sha256']
  report=json.loads(raw);assert report['status']=='passed' and report['phase']=='complete'
  assert report['candidate']['sha256']==hashes[0] and not report['scoring_eligible']
  assert all(report[k] for k in ('source_unchanged','input_unchanged','fixture_unchanged','evaluator_unchanged'))
  assert all(x['passed'] for x in report['checks'])
  for name,filename in [('k6_profile.py','worker.py'),('profile.py','csv_parser.py')]:
   key=next(k for k in report['evaluator_sha256'] if k.endswith('/'+name))
   assert digest(profile_root/filename)==report['evaluator_sha256'][key]
  for f in (profile_root/'source').iterdir():assert digest(f)==hashes[0]['flex_gemm/tenstorrent/kernels/k6/'+f.name]
  generated=temporary/(Path(entry['original_report']).stem+'.kernels')
  assert instrument(profile_root/'source',generated,report['mode'])==report['instrumented_source_sha256']
  for record in report['profiles']:
   path=temporary/Path(record['csv']).name;assert digest(path)==record['csv_sha256']
   saved=copy.deepcopy(record['parsed']);saved['source']=str(path)
   parsed=parser(path);assert parsed==saved
   assert parsed['unmatched_starts']==parsed['unmatched_ends']==0
   if report['scope']=='chunks':assert len(parsed['runs'])==1 and parsed['runs'][0]['cores']==64
   record['csv']=str(path);record['parsed']=parsed;count+=1
  recomputed=analyzer.analyze_report(report,Path(entry['original_report']))
  assert recomputed==expected_by_path[entry['original_report']]
  if report['scope']=='operator':
   assert all(x['programs']==1924 and x['k6_programs']==584 and x['phase_overlap_us']==0 for x in recomputed['samples'])
assert count==74
print('Also replayed 74 diagnostic CSV segments, including both complete largest-operator captures (1924 programs / 584 K6 products each).')
