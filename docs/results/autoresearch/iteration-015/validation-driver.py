import fcntl,json,os,pathlib,subprocess,sys
root=pathlib.Path('/home/user/GitHub/flexgemm-autoresearch');out=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch/iteration-015');scratch=out.parent
sys.path.insert(0,str(root))
from benchmarks.tenstorrent.run import monitor
assert json.loads((out/'screen-narrow.runner.json').read_text())['status']=='passed'
with open('/tmp/genrecon-tt-pjrt.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (out/'watcher-tests.log').open('w') as log:
  r=monitor([sys.executable,'-m','pytest','tests/tenstorrent/test_resident.py','-q','-k','digit and not fused_feature_digits','--junitxml='+str(out/'watcher-tests.xml')],log=log,report_path=out/'watcher-tests.json',compile_timeout=300,execution_timeout=180,env=dict(os.environ,TT_METAL_WATCHER='1'))
 (out/'watcher-tests.runner.json').write_text(json.dumps(r,indent=2)+'\n');print('Watcher',r,flush=True)
 assert r['returncode']==0 and not r['device_recovery_required']
subprocess.run([sys.executable,'/tmp/flexgemm_full015.py'],cwd=root,check=True)
s=json.loads((out/'full-validation/summary.json').read_text());assert s['status']=='complete' and len(s['trials'])==36 and all(r['status']=='passed' for r in s['trials'])
for name,corpus,case,chunk in [('narrow','poster-shape-corpus-v1','n388659-128x128-k3x3x3',8192),('largest','full-sampler-shape-corpus-v1','n2389000-64x64-k3x3x3',16384),('wide','poster-shape-corpus-v1','n4490-1024x4096-k3x3x3',1024)]:
 subprocess.run([sys.executable,'-m','benchmarks.tenstorrent.sweep','--corpus',str(scratch/corpus),'--cases',case,'--baseline-root','/home/user/GitHub/flexgemm-evaluator','--baseline',json.dumps({'algorithm':'digit_explicit_gemm','row_chunk':chunk}),'--candidates',str(out/(name+'-configs.json')),'--output',str(out/('confirm-'+name)),'--rounds','3','--repeat','11'],cwd=root,check=True)
subprocess.run([sys.executable,'-m','benchmarks.tenstorrent.run','--corpus',str(scratch/'poster-shape-corpus-v1'),'--case','n388659-128x128-k3x3x3','--config',json.dumps({'algorithm':'digit_explicit_gemm','row_chunk':8192}),'--output',str(out/'narrow-components.json'),'--workload','components','--repeat','2'],cwd=root,check=True)
