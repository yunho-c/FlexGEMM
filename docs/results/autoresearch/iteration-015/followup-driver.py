import fcntl,json,os,pathlib,subprocess,sys,time
root=pathlib.Path('/home/user/GitHub/flexgemm-autoresearch');out=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch/iteration-015');identity=json.loads((out/'followup-parent.json').read_text());stat=pathlib.Path(f"/proc/{identity['pid']}/stat")
while True:
 try:
  fields=stat.read_text().split()
  if fields[21]!=identity['start_time'] or fields[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(2)
for name in ('full-validation','confirm-narrow','confirm-largest','confirm-wide'):
 d=json.loads((out/name/'summary.json').read_text());assert d['status']=='complete' and all(t['status']=='passed' for t in d['trials'])
assert json.loads((out/'narrow-components.runner.json').read_text())['status']=='passed'
# Strengthen the helper oracle to compare raw BF16/FP32 bit patterns, including signed zero.
p=root/'tests/tenstorrent/test_resident.py';s=p.read_text();old='            torch.testing.assert_close(ttnn.to_torch(a), ttnn.to_torch(b), atol=0, rtol=0)'
assert s.count(old)==1
s=s.replace(old,'''            ah, bh = ttnn.to_torch(a), ttnn.to_torch(b)
            bits = torch.int16 if ah.dtype == torch.bfloat16 else torch.int32
            torch.testing.assert_close(ah.view(bits), bh.view(bits), atol=0, rtol=0)''');p.write_text(s)
sys.path.insert(0,str(root))
from benchmarks.tenstorrent.run import monitor
with open('/tmp/genrecon-tt-pjrt.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (out/'feature-bit-tests.log').open('w') as log:
  r=monitor([sys.executable,'-m','pytest','tests/tenstorrent/test_resident.py','-q','-k','fused_feature_digits','--junitxml='+str(out/'feature-bit-tests.xml')],log=log,report_path=out/'feature-bit-tests.json',compile_timeout=300,execution_timeout=180,env=dict(os.environ,TT_METAL_WATCHER='1'))
 (out/'feature-bit-tests.runner.json').write_text(json.dumps(r,indent=2)+'\n');print('Feature raw-bit oracle',r,flush=True);assert r['returncode']==0 and not r['device_recovery_required']
subprocess.run([sys.executable,'-m','benchmarks.tenstorrent.sweep','--corpus',str(out.parent/'corpus-v2'),'--cases','tune-scattered-35-17x45','--baseline-root','/home/user/GitHub/flexgemm-evaluator','--baseline',json.dumps({'algorithm':'digit_explicit_gemm','row_chunk':32768}),'--candidates',str(out/'small-configs.json'),'--output',str(out/'confirm-small'),'--rounds','3','--repeat','51'],cwd=root,check=True)
