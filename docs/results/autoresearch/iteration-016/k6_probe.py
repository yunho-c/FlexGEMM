"""Unscored matched-operand K6 portability/accuracy probe; no workspace promotion."""
import argparse,fcntl,gzip,json,math,os,pathlib,subprocess,sys,time,traceback
import torch

ROOT=pathlib.Path('/home/user/GitHub/flexgemm-evaluator')
K6=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch/comparators/k6-wormhole-probe')
def comparator_hashes():
 return {str(p.relative_to(K6)):sha256(p) for p in sorted(K6.rglob('*')) if p.suffix in ('.py','.cpp','.hpp')}

sys.path.insert(0,str(ROOT))
from benchmarks.tenstorrent.common import correctness,sha256,summarize,write_json
from benchmarks.tenstorrent.evaluate import source_state,tensor_contract
from benchmarks.tenstorrent.run import monitor


def worker(args):
 import ttnn
 sys.path.insert(0,str(K6))
 from adapter import Adapter
 from flex_gemm.tenstorrent import prepare_sparse_weights,prepare_sparse_topology
 from flex_gemm.tenstorrent.gather import tiled_gather
 from flex_gemm.tenstorrent.digits import _copy_output_rows
 if not os.getenv('FLEXGEMM_FIDELITY_PROBE_LOCKED'):raise RuntimeError('outer supervisor required')
 torch.set_num_threads(8)
 manifest=json.loads((args.corpus/'manifest.json').read_text());entry=next(c for c in manifest['cases'] if c['name']==args.case)
 fixture_path=args.corpus/entry['file'];assert sha256(fixture_path)==entry['sha256']
 contract_path=args.corpus/'contract.json';assert sha256(contract_path)==manifest['contract_sha256']
 contract=json.loads(contract_path.read_text());fixture=torch.load(fixture_path,weights_only=True,map_location='cpu')
 report={'status':'running','scope':'unscored K6 portability study with same BF16-rounded operand values, numerical/residency/workspace gates; native HiFi4 is separate from HiFi3 contract; no paired timing or promotion' , 'comparator_sha256':comparator_hashes(), 'upstream':json.loads((K6/'provenance.json').read_text()),
  'fidelity':args.fidelity,'case':entry,'fixture_sha256':entry['sha256'],'original_contract_sha256':manifest['contract_sha256'],
  'numerical_gate':{k:contract[k] for k in ('atol','rtol','max_relative_l2')},'row_chunk':args.row_chunk,'checks':[],'samples':[],
  'source':source_state(ROOT),'probe_sha256':sha256(__file__),'promotion':'not_evaluated',
  'tt_metal_commit':subprocess.check_output(['git','-C',os.environ['TT_METAL_HOME'],'rev-parse','HEAD']).decode().strip()}
 def phase(name):report['phase']=name;write_json(args.output,report);print(name,flush=True)
 device=None
 try:
  phase('prepare');device=ttnn.open_device(device_id=0)
  assert device.arch()==ttnn.Arch.WORMHOLE_B0
  layer=Adapter(fixture['weight'],fixture['bias'],fixture['mapping'],len(fixture['features'][0]),device=device,fidelity=args.fidelity,row_chunk=args.row_chunk)
  report['port_settings']=layer.port_settings
  report['plans']=[{'start':a,'end':b,**plan} for a,b,_,plan in layer.chunks]
  features=[ttnn.from_torch(x,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=device) for x in fixture['features']]
  report['input_contract']=[tensor_contract(f,device) for f in features]
  assert all(c['tensor_contract_valid'] for c in report['input_contract'])
  def forward(index):
   result=layer(features[index]);ttnn.synchronize_device(device);return result
  def check(result,index,label):
   actual=ttnn.to_torch(result).float();c=correctness(actual,fixture['expected'][index],contract)
   c.update(tensor_contract(result,device));c.update(variant=index,label=label);c['passed'] &= c['tensor_contract_valid'];report['checks'].append(c)
   return actual
  phase('warmup');first=[]
  for i in range(2):
   result=forward(i);first.append(check(result,i,'warmup'));del result
  for i in range(2):
   result=forward(i);del result
  from benchmarks.tenstorrent.memory import workspace_bound
  phase('capture_memory');ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
  try:
   ttnn.graph.disable_python_io_recording();result=forward(0)
  finally:trace=ttnn.graph.end_graph_capture()
  trace_path=args.output.with_suffix('.allocations.json.gz')
  with gzip.open(trace_path,'wt') as stream:json.dump(trace,stream)
  dram=ttnn.get_memory_view(device,ttnn.BufferType.DRAM);l1=ttnn.get_memory_view(device,ttnn.BufferType.L1)
  report['workspace']=workspace_bound(trace,dram_banks=dram.num_banks,l1_capacity_bytes=l1.num_banks*l1.total_bytes_per_bank,limit_bytes=contract['workspace_limit_bytes'],output_address=result.buffer_address(),expected_output_bytes=math.prod(result.padded_shape)*2)
  report['workspace'].update(trace_path=str(trace_path),trace_sha256=sha256(trace_path))
  check(result,0,'memory_capture');del result
  phase('measure')
  for j in range(3):
   i=j%2;start=time.perf_counter();result=forward(i);report['samples'].append((time.perf_counter()-start)*1000)
   actual=check(result,i,'timed');report['checks'][-1]['repeat_bitwise_equal']=torch.equal(actual,first[i]);report['checks'][-1]['passed'] &= report['checks'][-1]['repeat_bitwise_equal'];del result
  report['timing']=summarize(report['samples']);report['input_unchanged']=all(torch.equal(ttnn.to_torch(f),x) for f,x in zip(features,fixture['features']))
  report['source_unchanged']=source_state(ROOT)['sha256']==report['source']['sha256'];report['fixture_unchanged']=sha256(fixture_path)==entry['sha256']
  report['contract_unchanged']=sha256(contract_path)==manifest['contract_sha256'];report['probe_unchanged']=sha256(__file__)==report['probe_sha256']
  report['comparator_unchanged']=comparator_hashes()==report['comparator_sha256']
  report['status']='passed_probe_gates' if all(c['passed'] for c in report['checks']) and report['input_unchanged'] and report['source_unchanged'] and report['fixture_unchanged'] and report['contract_unchanged'] and report['probe_unchanged'] and report['comparator_unchanged'] and report['workspace']['within_limit'] else 'numerical_or_integrity_failure'
 except Exception as error:report.update(status='execution_failure',error=str(error),traceback=traceback.format_exc())
 finally:
  phase('close_device')
  if device is not None:ttnn.close_device(device)
  phase('complete')
 return 0 if report['status']=='passed_probe_gates' else 1


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus',type=pathlib.Path,required=True);p.add_argument('--case',required=True);p.add_argument('--fidelity',choices=('HiFi3','HiFi4'),required=True);p.add_argument('--row-chunk',type=int,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--worker',action='store_true');args=p.parse_args()
 if args.row_chunk<=0 or args.row_chunk%32:p.error('positive tile-aligned row chunk required')
 if args.output.exists():p.error('fresh output path required')
 args.output.parent.mkdir(parents=True,exist_ok=True)
 if args.worker:return worker(args)
 for k,v in {'BEVFUSION_K6_DEBUG':'','BEVFUSION_K6_MCAST':'1','BEVFUSION_K6_SPLIT':'1','BEVFUSION_K6_WEIGHT_TILES':'112','BEVFUSION_K6_CB_BYTES':str(900*1024)}.items():os.environ[k]=v
 with open('/tmp/genrecon-tt-pjrt.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with args.output.with_suffix('.log').open('w') as log:
   result=monitor([sys.executable,__file__,*sys.argv[1:],'--worker'],log=log,report_path=args.output,compile_timeout=300,execution_timeout=180,env=dict(os.environ,FLEXGEMM_FIDELITY_PROBE_LOCKED='1'))
  write_json(args.output.with_suffix('.runner.json'),result);print(result)
  return result['returncode']

if __name__=='__main__':raise SystemExit(main())
