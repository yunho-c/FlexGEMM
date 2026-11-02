"""A/B/A prototype confirmation after all eligible fixture probes pass."""
import json,os,pathlib,subprocess,sys
ROOT=pathlib.Path('/home/user/GitHub/flexgemm-evaluator');S=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch');OUT=S/'iteration-017'
sys.path.insert(0,str(ROOT))
from benchmarks.tenstorrent.common import write_json,sha256
from benchmarks.tenstorrent.sweep import comparison,summarize_comparisons
c=json.loads((OUT/'campaign-summary.json').read_text());assert c['status']=='complete' and len(c['trials'])==24 and all(t['status']=='passed_probe_gates' for t in c['trials'])
assert not any(os.getenv(k,'0') not in ('','0') for k in ('TT_METAL_WATCHER','TT_METAL_DEVICE_PROFILER','TT_METAL_DPRINT_CORES'))
for name,corpus,case,chunk in [('narrow','poster-shape-corpus-v1','n388659-128x128-k3x3x3',8192),('largest','full-sampler-shape-corpus-v1','n2389000-64x64-k3x3x3',16384)]:
 directory=OUT/('confirm-'+name);directory.mkdir(exist_ok=False)
 summary={'status':'running','scope':'matched completed-operator A/B/A; candidate uses prototype harness with same external numerical/residency/workspace gate; no production integration','repeat':11,'rounds':3,'case':case,'trials':[],'comparisons':[],'baseline_root':str(ROOT),'prototype_probe_sha256':sha256('/tmp/flexgemm_k6_digit_probe.py')}
 def trial(kind,round_index):
  path=directory/f"trial-{len(summary['trials']):04d}.json"
  if kind=='baseline':
   command=[sys.executable,'-m','benchmarks.tenstorrent.run','--corpus',str(S/corpus),'--case',case,'--candidate-root',str(ROOT),'--config',json.dumps({'algorithm':'digit_explicit_gemm','row_chunk':chunk}),'--output',str(path),'--repeat','11']
  else:command=[sys.executable,'/tmp/flexgemm_k6_digit_probe.py','--corpus',str(S/corpus),'--case',case,'--fidelity','HiFi3','--row-chunk',str(chunk),'--output',str(path),'--repeat','11']
  code=subprocess.run(command,cwd=ROOT).returncode;r=json.loads(path.read_text());runner=json.loads(path.with_suffix('.runner.json').read_text())
  summary['trials'].append({'kind':kind,'round':round_index,'report':path.name,'status':r['status'],'returncode':code,'timing':r.get('timing')});write_json(directory/'summary.json',summary)
  assert not runner['device_recovery_required'] and runner['timeout'] is None
  if code:raise RuntimeError('stop confirmation after failed probe')
  if kind=='prototype':
   assert r['status']=='passed_probe_gates' and r['workspace']['within_limit'] and all(x['passed'] for x in r['checks'])
   return dict(r,status='passed')
  assert r['status']=='passed';return r
 for round_index in range(3):
  before=trial('baseline',round_index);candidate=trial('prototype',round_index);after=trial('baseline',round_index)
  assert before['case']['sha256']==candidate['fixture_sha256']==after['case']['sha256']
  summary['comparisons'].append({'candidate':'four_product_k6','case':case,'round':round_index,'comparison':comparison(before,candidate,after)})
  summary['results']=summarize_comparisons(summary['comparisons'],[case],3);write_json(directory/'summary.json',summary)
  print(name,round_index,summary['comparisons'][-1]['comparison'],flush=True)
 summary['status']='complete';write_json(directory/'summary.json',summary)
