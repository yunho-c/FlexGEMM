import fcntl,json,os,pathlib,sys
root=pathlib.Path('/home/user/GitHub/flexgemm-autoresearch');out=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch/iteration-023')
sys.path.insert(0,str(root))
from benchmarks.tenstorrent.run import monitor
with open('/tmp/genrecon-tt-pjrt.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (out/'watcher-tests.log').open('w') as log:
  r=monitor([sys.executable,'-m','pytest','tests/tenstorrent/test_resident.py','-q','-k','digit_implicit','--junitxml='+str(out/'watcher-tests.xml')],log=log,report_path=out/'watcher-tests.json',compile_timeout=300,execution_timeout=180,env=dict(os.environ,TT_METAL_WATCHER='1'))
 (out/'watcher-tests.runner.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
 raise SystemExit(r['returncode'])
