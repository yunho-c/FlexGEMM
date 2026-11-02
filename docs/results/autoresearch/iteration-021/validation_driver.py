import json,pathlib,subprocess,sys
repo=pathlib.Path('/home/user/GitHub/flexgemm-autoresearch');scratch=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch');out=scratch/'iteration-021/full-validation'
sys.path.insert(0,str(repo))
from benchmarks.tenstorrent.common import write_json
summary={'status':'running','trials':[],'scope':'36 unchanged BF16 operator fixtures, full residency/numerical/integrity/workspace gates'}
for corpus in ('corpus-v2','poster-shape-corpus-v1','full-sampler-shape-corpus-v1','autoware-lidar-corpus-v1'):
 manifest=json.loads((scratch/corpus/'manifest.json').read_text())
 for entry in manifest['cases']:
  name=entry['name'];cp=(entry['ci']+31)//32*32;chunk=max(32,1048576//cp//32*32);config={'algorithm':'digit_implicit_gemm','row_chunk':chunk};path=out/(name+'.json')
  command=[sys.executable,'-m','benchmarks.tenstorrent.run','--corpus',str(scratch/corpus),'--case',name,'--config',json.dumps(config),'--output',str(path),'--mode','memory','--repeat','3','--compile-timeout','480']
  code=subprocess.run(command,cwd=repo).returncode;runner=json.loads(path.with_suffix('.runner.json').read_text());r=json.loads(path.read_text())
  row={'corpus':corpus,'case':name,'config':config,'status':r['status'],'returncode':code,'median_ms':r.get('timing',{}).get('median_ms'),'workspace':r.get('workspace'),'dispatch':r.get('digit_implicit_policy')}
  summary['trials'].append(row);write_json(out/'summary.json',summary);print(json.dumps({k:v for k,v in row.items() if k!='workspace'}),flush=True)
  if runner['device_recovery_required'] or (code and r['status']!='numerical_or_integrity_failure'):raise RuntimeError('campaign stopped after execution failure')
summary['status']='complete';write_json(out/'summary.json',summary)
