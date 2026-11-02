"""Replay archived profiler observations without opening a TT device."""
import copy,gzip,hashlib,importlib.util,json,sys,tempfile,tarfile
from pathlib import Path
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
def archived_module(name, filename):
 spec=importlib.util.spec_from_file_location('benchmarks.tenstorrent.'+name,HERE/filename)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module
parse_device_csv=archived_module('csv_parser_022','csv-parser.py').parse_device_csv
instrument=archived_module('k6_profile_022','harness-v1.py').instrument
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('profile_analysis',HERE/'analyze.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
a=json.loads((HERE/'archive.json').read_text())
for e in a['files']:assert sha(HERE/e['path'])==e['sha256'],e['path']
assert analysis.union_cycles([(0,5),(3,10),(20,21)])==11
expected=json.loads((HERE/'analysis.json').read_text())
expected={Path(x['report']).name:x for x in expected}
assert len(expected)==16
cpu=ET.parse(HERE/'cpu-tests.xml').getroot()
assert sum(int(x.get('tests',0)) for x in cpu.iter('testsuite'))==15
assert all(int(x.get('failures',0))+int(x.get('errors',0))==0 for x in cpu.iter('testsuite'))
reference_source=None;count=checks=0
with tempfile.TemporaryDirectory() as tmp:
 tmp=Path(tmp)
 with tarfile.open(HERE/'selected-profiles.tar.xz', 'r:xz') as tar:
  members=tar.getmembers()
  assert len(members)==272
  for member in members:
   assert member.isfile() and Path(member.name).name==member.name and member.name.endswith('.csv')
   (tmp/member.name).write_bytes(tar.extractfile(member).read())
 for e in a['original_reports']:
  b=gzip.decompress((HERE/e['archive']).read_bytes())
  assert hashlib.sha256(b).hexdigest()==e['sha256']
  report=json.loads(b);name=Path(e['path']).name
  runner=json.loads((HERE/'runners'/Path(e['path']).with_suffix('.runner.json').name).read_text())
  assert runner['timeout'] is None and not runner['device_recovery_required']
  if name=='narrow-compute.json':
   assert runner['returncode']==1 and report['status']=='execution_failure' and report['phase']=='complete'
   assert 'assert csv_path.exists()' in report['traceback'] and not report['profiles']
   continue
  assert runner['returncode']==0 and report['status']=='passed'
  assert report['phase']=='complete' and not report['scoring_eligible']
  assert all(report[k] for k in ('input_unchanged','source_unchanged','fixture_unchanged','evaluator_unchanged'))
  assert all(c['passed'] for c in report['checks']);checks+=len(report['checks'])
  assert report['candidate']['commit']=='ec0fe26038769aad1e2bae88c159fe2298e35588'
  source=report['candidate']['sha256']
  if reference_source is None:reference_source=source
  assert source==reference_source
  # The committed diagnostic worker is exactly the worker that ran.
  key=next(k for k in report['evaluator_sha256'] if k.endswith('/k6_profile.py'))
  assert sha(HERE/'harness-v1.py')==report['evaluator_sha256'][key]
  key=next(k for k in report['evaluator_sha256'] if k.endswith('/profile.py'))
  assert sha(HERE/'csv-parser.py')==report['evaluator_sha256'][key]
  for f in (HERE/'baseline-kernels').iterdir():
   assert sha(f)==source['flex_gemm/tenstorrent/kernels/k6/'+f.name]
  generated=tmp/(Path(name).stem+'.kernels')
  assert instrument(HERE/'baseline-kernels',generated,report['mode'])==report['instrumented_source_sha256']
  for p in report['profiles']:
   path=tmp/Path(p['csv']).name
   assert sha(path)==p['csv_sha256']
   parsed=parse_device_csv(path)
   saved=copy.deepcopy(p['parsed']);saved['source']=str(path)
   assert parsed==saved,p['label']
   assert parsed['clock_mhz']==1000 and parsed['unmatched_starts']==parsed['unmatched_ends']==0
   if report['scope']=='chunks':
    assert len(parsed['runs'])==1 and parsed['runs'][0]['cores']==64
    unpack={x['zone'] for x in parsed['runs'][0]['accumulated_zones'] if x['risc']=='TRISC_0'}
    allowed={'CB-COMPUTE-WAIT-FRONT'}
    if report['mode']=='compute':allowed.add('FG_K6_WAIT_GATHER')
    elif report['mode']=='weight':allowed.add('FG_K6_WAIT_WEIGHT')
    assert unpack==allowed,(name,unpack)
   p['csv']=str(path);p['parsed']=parsed;count+=1
  replay=analysis.analyze(report,Path(e['path']))
  assert replay==expected[name],name
  if report['scope']=='operator':
   assert len(replay['samples'])==2
   assert all(x['programs']==650 and x['k6_programs']==192 for x in replay['samples'])
assert count==272 and checks==456
print('Verified 16 passing workers, 456 checks, 272 replayed CSV segments, 15 CPU tests; initial failed probe retained.')
print('Full operator: both captures have all 650 programs, including 192 K6 products; all statistics reproduced.')
