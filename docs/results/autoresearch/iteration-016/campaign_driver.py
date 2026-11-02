import json,pathlib,subprocess,sys
root=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch');out=root/'iteration-016';cases=json.loads((out/'preflight.json').read_text())['cases'];summary={'status':'running','scope':'unpaired matched-operand HiFi3 K6 port study, no global promotion','trials':[],'unsupported':[]}
for c in cases:
 if not c['eligible']:
  summary['unsupported'].append({'case':c['case'],'reason':'per-core CB feasibility check fails','plans':c['plans']});continue
 path=out/'hifi3-cases'/(c['case']+'.json')
 code=subprocess.run([sys.executable,'/tmp/flexgemm_k6_probe.py','--corpus',str(root/c['corpus']),'--case',c['case'],'--fidelity','HiFi3','--row-chunk',str(c['row_chunk']),'--output',str(path)]).returncode
 r=json.loads(path.read_text());runner=json.loads(path.with_suffix('.runner.json').read_text());row={'case':c['case'],'corpus':c['corpus'],'row_chunk':c['row_chunk'],'status':r['status'],'returncode':code,'median_ms':r.get('timing',{}).get('median_ms'),'workspace_bytes':r.get('workspace',{}).get('workspace_upper_bound_bytes')};summary['trials'].append(row);(out/'campaign-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(row),flush=True)
 if runner['device_recovery_required'] or r['status']=='execution_failure':raise RuntimeError('stop after execution failure; no automatic retry/reset')
summary['status']='complete';(out/'campaign-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
