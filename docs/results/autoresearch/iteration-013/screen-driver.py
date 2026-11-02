import pathlib,json,re,subprocess,sys
root=pathlib.Path('/home/user/GitHub/flexgemm-autoresearch');out=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch/iteration-013');source=root/'flex_gemm/tenstorrent/digits.py'
sys.path.insert(0,str(root))
from benchmarks.tenstorrent.common import write_json
summary={'status':'running','scope':'unpaired narrow-layer screening; numerical gates unchanged','subblock_policy':'SDK order, FP32 area <=4','cb_estimate_limit_bytes':1048576,'trials':[]}
for block in (2,6,9,12,18,27):
 text=source.read_text();text=re.sub(r'^MATMUL_K_BLOCK = \d+$','MATMUL_K_BLOCK = '+str(block),text,flags=re.M);source.write_text(text)
 (out/f'k{block}-sdk-subblocks.patch').write_bytes(subprocess.check_output(['git','diff','--','flex_gemm/tenstorrent/digits.py','benchmarks/tenstorrent/evaluate.py','tests/tenstorrent/test_resident.py'],cwd=root))
 path=out/'kblock-screen'/f'k{block}.json'
 command=[sys.executable,'-m','benchmarks.tenstorrent.run','--corpus','/home/user/tt-scratchpad/flexgemm-autoresearch/poster-shape-corpus-v1','--case','n388659-128x128-k3x3x3','--config',json.dumps({'algorithm':'digit_explicit_gemm','row_chunk':8192}),'--repeat','5','--output',str(path)]
 code=subprocess.run(command,cwd=root).returncode;r=json.loads(path.read_text());runner=json.loads(path.with_suffix('.runner.json').read_text())
 summary['trials'].append({'k_block':block,'path':str(path.relative_to(out)),'status':r['status'],'returncode':code,'median_ms':r.get('timing',{}).get('median_ms'),'plan':r.get('matmul_policy')})
 write_json(out/'kblock-screen-summary.json',summary);print(json.dumps(summary['trials'][-1]),flush=True)
 if runner['device_recovery_required'] or (code and r['status']!='numerical_or_integrity_failure'):raise RuntimeError('screen stopped after execution failure')
summary['status']='complete';write_json(out/'kblock-screen-summary.json',summary)
