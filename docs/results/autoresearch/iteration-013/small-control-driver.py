import pathlib,json,subprocess,sys,time
root=pathlib.Path('/home/user/GitHub/flexgemm-autoresearch');p=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch/iteration-013');identity=json.loads((p/'followup-parent.json').read_text());stat=pathlib.Path(f"/proc/{identity['pid']}/stat")
while True:
 try:
  fields=stat.read_text().split()
  if fields[21]!=identity['start_time'] or fields[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(2)
for name in ('full-validation/summary.json','confirm-narrow/summary.json'):
 d=json.loads((p/name).read_text());assert d['status']=='complete' and all(r['status']=='passed' for r in d['trials'])
assert json.loads((p/'narrow-components.runner.json').read_text())['status']=='passed'
subprocess.run([sys.executable,'-m','benchmarks.tenstorrent.sweep','--corpus','/home/user/tt-scratchpad/flexgemm-autoresearch/corpus-v2','--cases','tune-surface-256-64x64','--baseline-root','/home/user/GitHub/flexgemm-evaluator','--baseline',json.dumps({'algorithm':'digit_explicit_gemm','row_chunk':16384}),'--candidates',str(p/'small-confirm-configs.json'),'--output',str(p/'confirm-small'),'--rounds','3','--repeat','11'],cwd=root,check=True)
