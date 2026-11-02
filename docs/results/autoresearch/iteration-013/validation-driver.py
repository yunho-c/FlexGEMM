import fcntl,json,os,pathlib,subprocess,sys
root=pathlib.Path('/home/user/GitHub/flexgemm-autoresearch');out=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch/iteration-013')
sys.path.insert(0,str(root))
from benchmarks.tenstorrent.run import monitor
with open('/tmp/genrecon-tt-pjrt.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (out/'watcher-tests.log').open('w') as log:
  r=monitor([sys.executable,'-m','pytest','tests/tenstorrent/test_resident.py','-q','-k','digit','--junitxml='+str(out/'watcher-tests.xml')],log=log,report_path=out/'watcher-tests.json',compile_timeout=240,execution_timeout=180,env=dict(os.environ,TT_METAL_WATCHER='1'))
 (out/'watcher-tests.runner.json').write_text(json.dumps(r,indent=2)+'\n');print('Watcher',r,flush=True)
 assert r['returncode']==0 and not r['device_recovery_required']
subprocess.run([sys.executable,'/tmp/flexgemm_full013.py'],cwd=root,check=True)
s=json.loads((out/'full-validation/summary.json').read_text())
assert s['status']=='complete' and len(s['trials'])==36 and all(r['status']=='passed' for r in s['trials'])
subprocess.run([sys.executable,'-m','benchmarks.tenstorrent.sweep','--corpus','/home/user/tt-scratchpad/flexgemm-autoresearch/poster-shape-corpus-v1','--cases','n388659-128x128-k3x3x3','--baseline-root','/home/user/GitHub/flexgemm-evaluator','--baseline',json.dumps({'algorithm':'digit_explicit_gemm','row_chunk':8192}),'--candidates',str(out/'confirm-configs.json'),'--output',str(out/'confirm-narrow'),'--rounds','3','--repeat','11'],cwd=root,check=True)
subprocess.run([sys.executable,'-m','benchmarks.tenstorrent.run','--corpus','/home/user/tt-scratchpad/flexgemm-autoresearch/poster-shape-corpus-v1','--case','n388659-128x128-k3x3x3','--config',json.dumps({'algorithm':'digit_explicit_gemm','row_chunk':8192}),'--output',str(out/'narrow-components.json'),'--workload','components','--repeat','2'],cwd=root,check=True)
