import fcntl,json,os,pathlib,sys,time
root=pathlib.Path('/home/user/GitHub/flexgemm-autoresearch')
out=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch/iteration-022')
sys.path.insert(0,str(root))
from benchmarks.tenstorrent.run import monitor
label,mode,case,corpus,chunk,scope=sys.argv[1:]
path=out/(label+'.json')
assert not path.exists()
env=dict(os.environ,FLEX_GEMM_EVALUATOR_LOCKED='1',TT_METAL_DEVICE_PROFILER='1',TT_METAL_PROFILER_SUM='1',TT_METAL_PROFILER_MID_RUN_DUMP='1',TT_METAL_PROFILER_DIR=str(out/(label+'.profile')))
for k in ('TT_METAL_WATCHER','TT_METAL_PROFILER_ACCUMULATE','TT_METAL_DPRINT_CORES'):env.pop(k,None)
command=[sys.executable,'-m','benchmarks.tenstorrent.k6_profile','--corpus',str(out.parent/corpus),'--case',case,'--row-chunk',chunk,'--mode',mode,'--scope',scope,'--output',str(path)]
with open('/tmp/genrecon-tt-pjrt.lock','a') as lock:
 start=time.monotonic()
 while True:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:
   if time.monotonic()-start>30:raise SystemExit('device busy')
   time.sleep(.2)
 with path.with_suffix('.log').open('w') as log:
  r=monitor(command,log=log,report_path=path,compile_timeout=900,execution_timeout=180,env=env)
 r.update(command=command,lock='/tmp/genrecon-tt-pjrt.lock',lock_wait_seconds=time.monotonic()-start-r['elapsed_seconds'])
 path.with_suffix('.runner.json').write_text(json.dumps(r,indent=2)+'\n')
 print(r,flush=True)
 raise SystemExit(r['returncode'])
