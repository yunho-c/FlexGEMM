"""Unscored higher-fidelity study with the original external numerical gates.

This does not alter or qualify under the fixed HiFi3 autoresearch contract.
"""
import argparse,fcntl,json,math,os,pathlib,subprocess,sys,time,traceback
import torch

ROOT=pathlib.Path('/home/user/GitHub/flexgemm-evaluator')
sys.path.insert(0,str(ROOT))
from benchmarks.tenstorrent.common import correctness,sha256,summarize,write_json
from benchmarks.tenstorrent.evaluate import source_state,tensor_contract
from benchmarks.tenstorrent.run import monitor


def worker(args):
 import ttnn
 from flex_gemm.tenstorrent import prepare_sparse_weights,prepare_sparse_topology
 from flex_gemm.tenstorrent.gather import tiled_gather
 from flex_gemm.tenstorrent.digits import _copy_output_rows
 if not os.getenv('FLEXGEMM_FIDELITY_PROBE_LOCKED'):raise RuntimeError('outer supervisor required')
 torch.set_num_threads(8)
 manifest=json.loads((args.corpus/'manifest.json').read_text());entry=next(c for c in manifest['cases'] if c['name']==args.case)
 fixture_path=args.corpus/entry['file'];assert sha256(fixture_path)==entry['sha256']
 contract_path=args.corpus/'contract.json';assert sha256(contract_path)==manifest['contract_sha256']
 contract=json.loads(contract_path.read_text());fixture=torch.load(fixture_path,weights_only=True,map_location='cpu')
 report={'status':'running','scope':'unscored fidelity study; outside fixed HiFi3 contract; same BF16 operand goldens and numerical thresholds',
  'fidelity':args.fidelity,'case':entry,'fixture_sha256':entry['sha256'],'original_contract_sha256':manifest['contract_sha256'],
  'numerical_gate':{k:contract[k] for k in ('atol','rtol','max_relative_l2')},'row_chunk':args.row_chunk,'checks':[],'samples':[],
  'source':source_state(ROOT),'probe_sha256':sha256(__file__),'promotion':'not_evaluated',
  'tt_metal_commit':subprocess.check_output(['git','-C',os.environ['TT_METAL_HOME'],'rev-parse','HEAD']).decode().strip()}
 def phase(name):report['phase']=name;write_json(args.output,report);print(name,flush=True)
 device=None
 try:
  phase('prepare');device=ttnn.open_device(device_id=0)
  assert device.arch()==ttnn.Arch.WORMHOLE_B0
  cfg=ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity,args.fidelity),math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=False)
  weights=prepare_sparse_weights(fixture['weight'],device=device)
  topology=prepare_sparse_topology(fixture['mapping'],len(fixture['features'][0]),device=device)
  features=[ttnn.from_torch(x,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=device) for x in fixture['features']]
  report['input_contract']=[tensor_contract(f,device) for f in features]
  assert all(c['tensor_contract_valid'] for c in report['input_contract'])
  bias=fixture['bias'];co=entry['co'];ci=entry['ci'];cp=(ci+31)//32*32;rows=len(fixture['mapping'])
  bias_row=None if bias is None else ttnn.from_torch(bias.bfloat16().float()[None],dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,device=device)
  bias_tile=None if bias is None else ttnn.from_torch(bias.bfloat16().float()[None].expand(32,-1).contiguous(),dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,device=device)
  def forward(index):
   table=ttnn.to_layout(features[index],ttnn.ROW_MAJOR_LAYOUT)
   if cp!=ci:table=ttnn.pad(table,[(0,0),(0,cp-ci)],0)
   output=ttnn.allocate_tensor_on_device(ttnn.Shape([rows,co]),ttnn.bfloat16,ttnn.TILE_LAYOUT,device,ttnn.DRAM_MEMORY_CONFIG)
   for start in range(0,rows,args.row_chunk):
    end=min(start+args.row_chunk,rows);count=end-start
    gathered=tiled_gather(table,topology,start,end,ci)
    result=ttnn.matmul(gathered,weights.weights,dtype=ttnn.float32,compute_kernel_config=cfg)
    del gathered
    result=ttnn.slice(result,[0,0],[count,co])
    if bias is not None:
     if count%32:result=ttnn.add(result,ttnn.repeat(bias_row,[count,1]),dtype=ttnn.float32)
     else:result=ttnn.reshape(ttnn.add(ttnn.reshape(result,[count//32,32,co]),ttnn.reshape(bias_tile,[1,32,co])),[count,co])
    result=ttnn.typecast(result,ttnn.bfloat16);_copy_output_rows(result,output,start)
    del result
   ttnn.synchronize_device(device);return output
  def check(result,index,label):
   actual=ttnn.to_torch(result).float();c=correctness(actual,fixture['expected'][index],contract)
   c.update(tensor_contract(result,device));c.update(variant=index,label=label);c['passed'] &= c['tensor_contract_valid'];report['checks'].append(c)
   return actual
  phase('warmup');first=[]
  for i in range(2):
   result=forward(i);first.append(check(result,i,'warmup'));del result
  phase('measure')
  for j in range(3):
   i=j%2;start=time.perf_counter();result=forward(i);report['samples'].append((time.perf_counter()-start)*1000)
   actual=check(result,i,'timed');report['checks'][-1]['repeat_bitwise_equal']=torch.equal(actual,first[i]);report['checks'][-1]['passed'] &= report['checks'][-1]['repeat_bitwise_equal'];del result
  report['timing']=summarize(report['samples']);report['input_unchanged']=all(torch.equal(ttnn.to_torch(f),x) for f,x in zip(features,fixture['features']))
  report['source_unchanged']=source_state(ROOT)['sha256']==report['source']['sha256'];report['fixture_unchanged']=sha256(fixture_path)==entry['sha256']
  report['contract_unchanged']=sha256(contract_path)==manifest['contract_sha256'];report['probe_unchanged']=sha256(__file__)==report['probe_sha256']
  report['status']='numerical_pass' if all(c['passed'] for c in report['checks']) and report['input_unchanged'] and report['source_unchanged'] and report['fixture_unchanged'] and report['contract_unchanged'] and report['probe_unchanged'] else 'numerical_or_integrity_failure'
 except Exception as error:report.update(status='execution_failure',error=str(error),traceback=traceback.format_exc())
 finally:
  phase('close_device')
  if device is not None:ttnn.close_device(device)
  phase('complete')
 return 0 if report['status']=='numerical_pass' else 1


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus',type=pathlib.Path,required=True);p.add_argument('--case',required=True);p.add_argument('--fidelity',choices=('HiFi3','HiFi4'),required=True);p.add_argument('--row-chunk',type=int,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--worker',action='store_true');args=p.parse_args()
 if args.row_chunk<=0 or args.row_chunk%32:p.error('positive tile-aligned row chunk required')
 if args.output.exists():p.error('fresh output path required')
 args.output.parent.mkdir(parents=True,exist_ok=True)
 if args.worker:return worker(args)
 with open('/tmp/genrecon-tt-pjrt.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with args.output.with_suffix('.log').open('w') as log:
   result=monitor([sys.executable,__file__,*sys.argv[1:],'--worker'],log=log,report_path=args.output,compile_timeout=300,execution_timeout=180,env=dict(os.environ,FLEXGEMM_FIDELITY_PROBE_LOCKED='1'))
  write_json(args.output.with_suffix('.runner.json'),result);print(result)
  return result['returncode']

if __name__=='__main__':raise SystemExit(main())
