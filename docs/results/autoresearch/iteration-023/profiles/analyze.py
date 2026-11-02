import collections,csv,json,statistics
from pathlib import Path
ROOT=Path('/home/user/tt-scratchpad/flexgemm-autoresearch/iteration-022')

def union_cycles(intervals):
 merged=[]
 for start,end in sorted(intervals):
  assert end>=start
  if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
  else:merged.append([start,end])
 return sum(b-a for a,b in merged)

def intervals(path):
 starts=collections.defaultdict(list);runs=collections.defaultdict(list)
 with open(path) as f:
  f.readline()
  for raw in csv.DictReader(f):
   r={k.strip():(v or '').strip() for k,v in raw.items()}
   if not r['zone name'].endswith('-KERNEL'):continue
   key=tuple(r[k] for k in ('PCIe slot','run host ID','core_x','core_y','RISC processor type','timer_id'))
   if r['type']=='ZONE_START':starts[key].append(int(r['time[cycles since reset]']))
   elif r['type']=='ZONE_END':
    assert starts[key],key
    runs[key[:2]].append((starts[key].pop(),int(r['time[cycles since reset]'])))
 assert not any(starts.values())
 return [{'device':k[0],'run_host_id':k[1],'start':min(x[0] for x in v),'end':max(x[1] for x in v)} for k,v in runs.items()]

def describe(values):
 return dict(min=min(values),median=statistics.median(values),max=max(values))

def analyze(report,path):
 assert report['status']=='passed'
 summaries=[]
 for p in report['profiles']:
  runs=p['parsed']['runs'];freq=p['parsed']['clock_mhz']
  assert p['parsed']['unmatched_starts']==p['parsed']['unmatched_ends']==0
  ints=intervals(p['csv']);assert len(runs)==len(ints)
  span=(max(x['end'] for x in ints)-min(x['start'] for x in ints))/freq
  occupied=sum(union_cycles([(x['start'],x['end']) for x in ints if x['device']==d]) for d in {x['device'] for x in ints})/freq
  assert len({x['device'] for x in ints})==1
  scopes=collections.defaultdict(list);fractions=collections.defaultdict(list)
  for run in runs:
   core_durations={(tuple(x['core']),x['risc']):x['duration_us'] for x in run['custom_zones'] if x['zone'] in ('FG_K6_COMPUTE','FG_K6_READER','FG_K6_WRITER')}
   for x in run['accumulated_zones']:
    key=x['zone']+'@'+x['risc']
    scopes[key].append(x['total_us'])
    duration=core_durations.get((tuple(x['core']),x['risc']))
    if duration is not None:fractions[key].append(x['total_us']/duration)
  summaries.append(dict(label=p['label'],wall_ms=p['wall_ms'],programs=len(runs),device_span_us=span,
   kernel_union_us=occupied,inter_program_gap_us=span-occupied,
   k6_programs=sum(any(x['zone']=='FG_K6_COMPUTE' for x in r['custom_zones']) for r in runs),
   k6_kernel_spans_us=[r['kernel_span_us'] for r in runs if any(x['zone']=='FG_K6_COMPUTE' for x in r['custom_zones'])],
   core_totals_us={k:describe(v) for k,v in scopes.items()},core_fractions={k:describe(v) for k,v in fractions.items()}))
 return dict(report=str(path),case=report['case']['name'],mode=report['mode'],scope=report['scope'],samples=summaries,
  checks=len(report['checks']),all_checks_pass=all(x['passed'] for x in report['checks']))

def main():
 result=[]
 for p in sorted(ROOT.glob('*-v1.json')):
  a=json.loads(p.read_text())
  if a.get('status')=='passed':result.append(analyze(a,p))
 (ROOT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 for item in result:
  s=item['samples'];print(item['case'],item['mode'],item['scope'],'samples',len(s),'kernel us',round(statistics.median(x['kernel_union_us'] for x in s),2),'wall ms',round(statistics.median(x['wall_ms'] for x in s),3))
  keys=sorted({k for x in s for k in x['core_fractions']})
  for k in keys:
   if k.startswith('FG_') or 'WAIT-FRONT' in k:print(' ',k,round(statistics.median(x['core_fractions'][k]['median'] for x in s if k in x['core_fractions'])*100,2),'%')
if __name__=='__main__':main()
