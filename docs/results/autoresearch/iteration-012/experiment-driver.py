import gzip, hashlib, json, pathlib, subprocess, sys, time
repo=pathlib.Path('/home/user/GitHub/flexgemm-autoresearch')
scratch=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch')
out=scratch/'iteration-012'; old=repo/'docs/results/autoresearch/iteration-011'
sys.path.insert(0,str(repo))
from benchmarks.tenstorrent.common import write_json
old_events=json.loads(gzip.decompress((old/'allocation-events.json.gz').read_bytes()))

def events(trace):
 return [n for n in trace if n.get('node_type') in ('buffer_allocate','buffer_deallocate') and n.get('params',{}).get('type')=='DRAM']
def normalized(trace):
 active={}; rows=[]; next_id=0; resident={}
 for n in events(trace):
  p=n['params'];key=(p['device_id'],p['address']);kind=n['node_type']
  if kind=='buffer_allocate':
   assert key not in active
   active[key]=next_id;next_id+=1;ordinal=active[key]
  else:
   ordinal=active.pop(key,None)
   if ordinal is None:
    if key not in resident:resident[key]=-len(resident)-1
    ordinal=resident[key]
  rows.append([kind,ordinal,{k:v for k,v in p.items() if k not in ('device_id','address')}])
 return rows

def run(case,corpus,config,path,root=repo,mode='memory'):
 command=[sys.executable,'-m','benchmarks.tenstorrent.run','--corpus',str(scratch/corpus),'--case',case,'--candidate-root',str(root),'--config',json.dumps(config),'--output',str(path),'--repeat','3','--mode',mode,'--compile-timeout','480']
 proc=subprocess.run(command,cwd=repo)
 r=json.loads(path.read_text());runner=json.loads(path.with_suffix('.runner.json').read_text())
 assert runner['returncode']==proc.returncode and not runner['device_recovery_required'],runner
 return proc.returncode,r,runner

for kind in ('l1','host'):
 path=out/('reject-'+kind+'.json')
 code,r,runner=run('tune-scattered-35-17x45','corpus-v2',{'algorithm':'digit_explicit_gemm','row_chunk':32768},path,pathlib.Path('/home/user/GitHub/flexgemm-residency-'+kind),mode='timing')
 assert code==1 and r['status']=='numerical_or_integrity_failure',r['status']
 assert all(c['elementwise_failures']==0 and c['relative_l2']<=r['contract']['max_relative_l2'] and c['dtype_valid'] and c['layout_valid'] and not c['tensor_contract_valid'] for c in r['checks'])
 assert all(c['device_valid'] and not c['dram_valid'] for c in r['checks']) if kind=='l1' else all(not c['device_valid'] for c in r['checks'])
 print('Expected residency rejection:',kind,flush=True)

trials=[('full-validation/tune-scattered-35-17x45.json','corpus-v2'),('full-validation/n4490-1024x4096-k3x3x3.json','poster-shape-corpus-v1'),('full-validation/n388659-128x128-k3x3x3.json','poster-shape-corpus-v1'),('external-validation/autoware-test-pcd-21359-256x256.json','autoware-lidar-corpus-v1'),('full-validation/n2389000-64x64-k3x3x3.json','full-sampler-shape-corpus-v1')]
summary={'status':'running','scope':'evaluator instrumentation and residency gates; candidate kernels unchanged','comparisons':[]}
for key,corpus in trials:
 original=json.loads((old/key).read_text());case=original['case']['name'];path=out/'memory-checks'/(case+'.json')
 code,r,runner=run(case,corpus,original['config'],path)
 assert code==0 and r['status']=='passed',r['status']
 assert r['candidate']['sha256']==original['candidate']['sha256']
 assert not r['memory_capture']['python_io_recording']
 raw=path.with_suffix('.allocations.json.gz').read_bytes();trace=json.loads(gzip.decompress(raw))
 before,after=normalized(old_events[key]),normalized(trace)
 if before!=after:
  first=next((i for i,(a,b) in enumerate(zip(before,after)) if a!=b),min(len(before),len(after)))
  write_json(out/(case+'-trace-difference.json'),{'first_index':first,'old_events':len(before),'new_events':len(after),'old':before[max(0,first-2):first+3],'new':after[max(0,first-2):first+3]})
  raise AssertionError('Allocation sequence changed: '+case)
 w,ow=r['workspace'],original['workspace']
 fields=('new_dram_peak_bytes','dram_allocations','temporary_dram_peak_bytes','workspace_upper_bound_bytes','all_l1_capacity_bytes')
 assert all(w[k]==ow[k] for k in fields)
 fields_out=('aligned_bytes','expected_padded_bytes','excluded_bytes')
 assert all(w['excluded_final_output'][k]==ow['excluded_final_output'][k] for k in fields_out)
 old_runner=json.loads((old/key).with_suffix('.runner.json').read_text())
 item={'case':case,'status':'passed','report':str(path.relative_to(out)),'previous_report':str(old/key),'candidate_sources_identical':True,'normalized_dram_events_identical':True,'normalized_event_sha256':hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest(),'dram_events':len(after),'workspace_bounds_identical':True,'final_output_exclusion_identical':True,'workspace_upper_bound_bytes':w['workspace_upper_bound_bytes'],'previous_runner_seconds':old_runner['elapsed_seconds'],'runner_seconds':runner['elapsed_seconds'],'capture_ms':r['memory_capture']['elapsed_ms']}
 summary['comparisons'].append(item);write_json(out/'memory-comparison.json',summary);print(json.dumps(item),flush=True)
summary['status']='complete';write_json(out/'memory-comparison.json',summary)
