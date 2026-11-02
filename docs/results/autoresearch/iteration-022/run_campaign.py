import json,os,pathlib,subprocess,sys
root=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch/iteration-022')
cases=[('narrow','n388659-128x128-k3x3x3','poster-shape-corpus-v1','8192'),('wide','n388659-128x512-k3x3x3','poster-shape-corpus-v1','8192'),('largest','n2389000-64x64-k3x3x3','full-sampler-shape-corpus-v1','16384')]
plan=[]
for group,case,corpus,chunk in cases:
 for mode in ('base','weight','compute','gather','multicast'):
  if group=='narrow' and mode=='compute':continue
  plan.append([group+'-'+mode+'-v1',mode,case,corpus,chunk,'chunks'])
plan.append(['narrow-operator-v1','base',cases[0][1],cases[0][2],cases[0][3],'operator'])
(root/'campaign-plan.json').write_text(json.dumps({'baseline':'ec0fe26038769aad1e2bae88c159fe2298e35588','diagnostic_only':True,'planned':plan,'completed_first':'narrow-compute-v1','discarded_first_probe':'narrow-compute: missing mid-run CSV dump; custom slot 1 conflicted with SDK UNPACK timer'},indent=2)+'\n')
for args in plan:
 print('START',args[0],flush=True)
 r=subprocess.run([sys.executable,'/tmp/flexgemm_profile022.py',*args])
 print('DONE',args[0],r.returncode,flush=True)
 if r.returncode:sys.exit(r.returncode)
