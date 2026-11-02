import hashlib,importlib.util,json,statistics
from pathlib import Path
ROOT=Path('/home/user/tt-scratchpad/flexgemm-autoresearch/iteration-023')
REPO=Path('/home/user/GitHub/flexgemm-autoresearch')
BASE=REPO/'docs/results/autoresearch/iteration-022/analysis.json'
helper_path=Path(__file__).with_name('analyze.py')
if not helper_path.exists():helper_path=REPO/'docs/results/autoresearch/iteration-022/analyze.py'
spec=importlib.util.spec_from_file_location('analysis022',helper_path)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
read=lambda p:json.loads(p.read_text())
def analyze_report(report,path):
 result=helper.analyze(report,path)
 if report['scope']=='operator':
  for raw,sample in zip(report['profiles'],result['samples']):
   spans=helper.intervals(raw['csv']);mhz=raw['parsed']['clock_mhz']
   k6={r['run_host_id'] for r in raw['parsed']['runs'] if any(x['zone']=='FG_K6_COMPUTE' for x in r['custom_zones'])}
   first=min(map(int,k6))
   categories={key:[] for key in ('feature_preparation_and_scale','k6_products','loop_reconstruction_epilogue_assembly')}
   for span in spans:
    key=('k6_products' if span['run_host_id'] in k6 else 'feature_preparation_and_scale' if int(span['run_host_id'])<first else 'loop_reconstruction_epilogue_assembly')
    categories[key].append((span['start'],span['end']))
   sample['phase_unions']={k:{'programs':len(v),'microseconds':helper.union_cycles(v)/mhz} for k,v in categories.items()}
   sample['phase_overlap_us']=sum(x['microseconds'] for x in sample['phase_unions'].values())-sample['kernel_union_us']
 return result


def main():
 baselines=read(BASE);records=[];comparisons=[]
 for p in sorted(ROOT.glob('probe-*-v1.json')):
  r=read(p);assert r['status']=='passed'
  a=analyze_report(r,p);records.append(a)
  old=next((x for x in baselines if (x['case'],x['mode'],x['scope'])==(a['case'],a['mode'],a['scope'])),None)
  item=dict(case=a['case'],mode=a['mode'],scope=a['scope'],paired=False,
            candidate_median_kernel_union_us=statistics.median(x['kernel_union_us'] for x in a['samples']))
  if old is not None:
   assert [x['label'] for x in a['samples']]==[x['label'] for x in old['samples']]
   item['baseline_median_kernel_union_us']=statistics.median(x['kernel_union_us'] for x in old['samples'])
   item['selected_zones']={}
   for zone in ('FG_K6_WAIT_WEIGHT@TRISC_0','FG_K6_WAIT_GATHER@TRISC_0','FG_K6_MATMUL_REGION@TRISC_1'):
    if zone not in a['samples'][0]['core_totals_us']:continue
    item['selected_zones'][zone]={}
    for label,source in (('baseline',old),('candidate',a)):
     item['selected_zones'][zone][label]={'median_core_us':statistics.median(x['core_totals_us'][zone]['median'] for x in source['samples']),
         'median_core_fraction':statistics.median(x['core_fractions'][zone]['median'] for x in source['samples'])}
  comparisons.append(item)
 assert len(records)==5
 result=dict(scope='Unpaired diagnostic observations, not promotion scores or bus counters',baseline_analysis=str(BASE),baseline_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),candidate=records,comparison=comparisons)
 (ROOT/'profile-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
