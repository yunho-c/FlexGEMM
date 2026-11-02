import json,pathlib,subprocess,sys
root=pathlib.Path('/home/user/GitHub/flexgemm-autoresearch')
out=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch/iteration-023')
for name in ('narrow','largest','wide-feasible','fallback'):
 a=json.loads((out/f'confirm-{name}/summary.json').read_text());assert a['status']=='complete' and all(x['status']=='passed' for x in a['trials'])
# The application's original numerical-failure status is expected and retained.
command=[sys.executable,str(out.parent/'integration-stage-cache-v1/run_scene.py'),'--candidate-root',str(root),'--reference-dir','/home/user/tt-scratchpad/genrecon/sparse-decoder-poster-bounded','--output-dir',str(out/'application'),'--warm-repeats','1','--algorithm','digit_implicit_gemm','--topology-policy','stage']
r=subprocess.run(command,cwd=root)
a=json.loads((out/'application.runner.json').read_text());report=json.loads((out/'application/result.json').read_text())
assert r.returncode==1 and a['timeout'] is None and not a['device_recovery_required']
assert report['phase']=='complete' and report['status']=='numerical_failure' and report['sources_unchanged']
for label,mode,case,corpus,chunk in [
 ('probe-narrow-weight-v1','weight','n388659-128x128-k3x3x3','poster-shape-corpus-v1','8192'),
 ('probe-largest-weight-v1','weight','n2389000-64x64-k3x3x3','full-sampler-shape-corpus-v1','16384'),
 ('probe-largest-compute-v1','compute','n2389000-64x64-k3x3x3','full-sampler-shape-corpus-v1','16384'),
 ('probe-wide-weight-v1','weight','n388659-128x512-k3x3x3','poster-shape-corpus-v1','8192')]:
 subprocess.run([sys.executable,'/tmp/flexgemm_profile023.py',label,mode,case,corpus,chunk,'chunks'],cwd=root,check=True)
subprocess.run([sys.executable,'/tmp/flexgemm_profile023.py','probe-largest-operator-v1','base','n2389000-64x64-k3x3x3','full-sampler-shape-corpus-v1','16384','operator'],cwd=root,check=True)
print('Application and all diagnostic workers terminated normally.',flush=True)
