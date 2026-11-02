import json,pathlib,subprocess,sys
root=pathlib.Path('/home/user/GitHub/flexgemm-autoresearch');s=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch');out=s/'iteration-018'
d=json.loads((out/'full-validation/summary.json').read_text());assert d['status']=='complete' and len(d['trials'])==36 and all(t['status']=='passed' for t in d['trials'])
for name,corpus,case,chunk in [('narrow','poster-shape-corpus-v1','n388659-128x128-k3x3x3',8192),('largest','full-sampler-shape-corpus-v1','n2389000-64x64-k3x3x3',16384),('fallback','poster-shape-corpus-v1','n4490-1024x4096-k3x3x3',1024)]:
 config=out/(name+'-configs.json');config.write_text(json.dumps({'integrated_digit_implicit':{'algorithm':'digit_implicit_gemm','row_chunk':chunk}},indent=2)+'\n')
 subprocess.run([sys.executable,'-m','benchmarks.tenstorrent.sweep','--corpus',str(s/corpus),'--cases',case,'--baseline-root','/home/user/GitHub/flexgemm-evaluator','--baseline',json.dumps({'algorithm':'digit_explicit_gemm','row_chunk':chunk}),'--candidates',str(config),'--output',str(out/('confirm-'+name)),'--rounds','3','--repeat','11'],cwd=root,check=True)
