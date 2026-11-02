"""Structural work model for the four-call K6 prototype, not traffic counters."""
import json,pathlib,sys
import torch
ROOT=pathlib.Path('/home/user/GitHub/flexgemm-evaluator');S=pathlib.Path('/home/user/tt-scratchpad/flexgemm-autoresearch');OUT=S/'iteration-017'
sys.path.insert(0,str(ROOT))
from benchmarks.tenstorrent.common import sha256,write_json
summary=json.loads((OUT/'campaign-summary.json').read_text());assert summary['status']=='complete'
rows=[]
for t in summary['trials']:
 report_path=OUT/'prototype-cases'/(t['case']+'.json');r=json.loads(report_path.read_text());fixture_path=S/t['corpus']/r['case']['file'];assert sha256(fixture_path)==r['fixture_sha256'];f=torch.load(fixture_path,map_location='cpu',weights_only=True);mapping=f['mapping'].long();mapping[mapping==0xffffffff]=-1
 m,v=mapping.shape;ci,co=r['case']['ci'],r['case']['co'];cp=(ci+31)//32*32;op=(co+31)//32*32;edges=int((mapping>=0).sum());padded=torch.full(((m+31)//32*32,v),-1,dtype=torch.int64);padded[:m]=mapping
 active=(padded.reshape(-1,32,v)>=0).any(1);active_taps=int(active.sum());tile_products=4*active_taps*(cp//32)*(op//32);issued=tile_products*2*32**3
 gy=len(r['port_settings']['physical_core_rows']);gx=len(r['port_settings']['physical_core_rows'][0]);senders=gy if r['port_settings']['multicast'] else gx*gy
 blocks=(m+31)//32;chunks=len(r['plans']);weight_payload=4*chunks*senders*v*cp*op*4
 index_payload=sum(4*2*((p['end']-p['start']+31)//32)*v*128*p['tap_chunks'] for p in r['plans'])
 unused=sum(4*(((p['end']-p['start']+31)//32)*32+1)*op*2 for p in r['plans'])
 rows.append({'case':t['case'],'report_sha256':sha256(report_path),'fixture_sha256':r['fixture_sha256'],'valid_interactions':edges,'active_block_taps':active_taps,'dense_block_taps':blocks*v,'useful_flops':2*edges*ci*co,'issued_matmul_flops':issued,'tile_products':tile_products,'explicit_im2col_dram_bytes':0,'requested_feature_read_bytes':4*edges*cp*2,'requested_weight_read_bytes':weight_payload,'requested_index_read_bytes':index_payload,'fp32_product_write_bytes':4*blocks*32*op*4,'unused_bf16_table_write_bytes':unused,'final_output_copy_bytes':blocks*32*op*4,'median_ms':r['timing']['median_ms'],'issued_tflops':issued/r['timing']['median_ms']/1e9,'workspace_upper_bound_bytes':r['workspace']['workspace_upper_bound_bytes']})
write_json(OUT/'structural-work.json',{'scope':'Logical custom-kernel matmul calls and requested payloads, derived from frozen operands and actual plans; not measured DRAM/NoC bus traffic or a complete I/O lower bound. Excludes SFPU preparation/reconstruction traffic and on-chip movement. FP32 weights and four separately gathered products.','source_commit':'8c1c896b19892e9abcd340de9f8f8942f13c0728','cases':rows})
print('modeled',len(rows),'cases')
