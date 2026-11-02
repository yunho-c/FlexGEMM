"""Watcher oracle for all four K6 integer products; serialized and supervised."""
import fcntl,hashlib,json,os,pathlib,subprocess,sys,time,traceback
ROOT=pathlib.Path('/home/user/GitHub/flexgemm-evaluator');K6=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch/comparators/k6-four-product-bf16-probe');OUT=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch/iteration-017/product-oracle-bf16weights.json')
sys.path[:0]=[str(ROOT),str(K6)]
from benchmarks.tenstorrent.common import sha256,write_json
from benchmarks.tenstorrent.run import monitor
from benchmarks.tenstorrent.evaluate import source_state

def worker():
 import torch,ttnn
 from adapter import Adapter
 from flex_gemm.tenstorrent.digits import _feature_digits
 assert os.getenv('FLEXGEMM_PRODUCT_ORACLE_LOCKED')=='1'
 torch.set_num_threads(8)
 sources={str(p.relative_to(K6)):sha256(p) for p in sorted(K6.rglob('*')) if p.suffix in ('.py','.cpp','.hpp')}
 report={'status':'running','phase':'prepare','watcher':os.getenv('TT_METAL_WATCHER'),'probe_sha256':sha256(__file__),'comparator_sha256':sources,'source':source_state(ROOT),'cases':[]};write_json(OUT,report);device=None
 try:
  device=ttnn.open_device(device_id=0)
  for seed,m,n,ci,co,k,chunk in [(1701,1,3,1,1,1,32),(1702,65,83,17,45,27,32),(1703,257,321,128,128,27,256)]:
   g=torch.Generator().manual_seed(seed);weight=(torch.randn(co,k,ci,generator=g)*.02).bfloat16();bias=torch.randn(co,generator=g).bfloat16();mapping=torch.randint(-1,n,(m,k),generator=g)
   if m>=65:mapping[:32]=-1;mapping[32]=n-1;mapping[-1]=-1
   a=Adapter(weight,bias,mapping,n,device=device,fidelity='HiFi3',row_chunk=chunk)
   row={'seed':seed,'rows':m,'input_rows':n,'ci':ci,'co':co,'volume':k,'row_chunk':chunk,'checks':[]}
   wdigits=[ttnn.to_torch(d.weights).to(torch.int64) for d in a.weights.digits]
   x=torch.randn(n,ci,generator=g).bfloat16()
   for variant,values in enumerate((x,(x.float()*.5+.03125).bfloat16())):
    features=ttnn.from_torch(values,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=device)
    high,low,scale=_feature_digits(features,a.spec.cin_pad);tables=[high,low]
    host=[ttnn.to_torch(t).to(torch.int64) for t in tables]
    for start,stop,index,plan in a.chunks:
     ids=mapping[start:stop].clone();ids[ids<0]=n
     for fi,wi in ((0,0),(0,1),(1,0),(1,1)):
      gathered=host[fi][ids].reshape(stop-start,k*a.spec.cin_pad)
      exact=gathered@wdigits[wi]
      absolute_bound=(gathered.abs()@wdigits[wi].abs()).max().item()
      assert absolute_bound<2**24,('not guaranteed FP32-exact',absolute_bound)
      product=a.product(tables[fi],index,wi);actual=ttnn.to_torch(product).float()[:stop-start]
      passed=torch.equal(actual,exact.float())
      row['checks'].append({'variant':variant,'start':start,'stop':stop,'feature_digit':fi,'weight_digit':wi,'absolute_sum_bound':absolute_bound,'exact_integer_match':passed,'max_abs_error':float((actual-exact.float()).abs().max()),'tap_chunks':plan['tap_chunks']})
      if not passed:
       bad=torch.nonzero(actual != exact.float())[:8];row['checks'][-1]['first_differences']=[{'row':int(i),'column':int(j),'actual':float(actual[i,j]),'expected':int(exact[i,j])} for i,j in bad];report['cases'].append(row);write_json(OUT,report)
      assert passed,row['checks'][-1]
      del product,gathered,exact,actual
    assert torch.equal(ttnn.to_torch(features),values)
    del high,low,scale,tables,features,host
   report['cases'].append(row);write_json(OUT,report);del a,wdigits
  report['source_unchanged']=source_state(ROOT)['sha256']==report['source']['sha256'];report['probe_unchanged']=sha256(__file__)==report['probe_sha256'];report['comparator_unchanged']=sources=={str(p.relative_to(K6)):sha256(p) for p in sorted(K6.rglob('*')) if p.suffix in ('.py','.cpp','.hpp')}
  assert report['source_unchanged'] and report['probe_unchanged'] and report['comparator_unchanged']
  report['status']='passed'
 except Exception as e:report.update(status='failure',error=str(e),traceback=traceback.format_exc())
 finally:
  if device is not None:ttnn.close_device(device)
  report['phase']='complete';write_json(OUT,report)
 return 0 if report['status']=='passed' else 1

if __name__=='__main__':
 if '--worker' in sys.argv:raise SystemExit(worker())
 assert not OUT.exists()
 env=dict(os.environ,TT_METAL_WATCHER='1',FLEXGEMM_PRODUCT_ORACLE_LOCKED='1',BEVFUSION_K6_MCAST='1',BEVFUSION_K6_SPLIT='1',BEVFUSION_K6_DEBUG='',BEVFUSION_K6_WEIGHT_TILES='112',BEVFUSION_K6_CB_BYTES=str(900*1024))
 with open('/tmp/genrecon-tt-pjrt.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with OUT.with_suffix('.log').open('w') as log:r=monitor([sys.executable,__file__,'--worker'],log=log,report_path=OUT,compile_timeout=300,execution_timeout=180,env=env)
  write_json(OUT.with_suffix('.runner.json'),r);print(r)
  raise SystemExit(r['returncode'])
