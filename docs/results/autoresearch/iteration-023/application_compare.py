"""Compare new BF16-weight kernel application outputs with iteration021."""
import importlib.util,json,hashlib
from pathlib import Path
import torch
ROOT=Path('/home/user/tt-scratchpad/flexgemm-autoresearch/iteration-023')
BASE=ROOT.parent/'iteration-021'/'application'
ACTUAL=ROOT/'application'
helper_path=Path(__file__).with_name('application_compare_helpers.py')
if not helper_path.exists():helper_path=Path('/tmp/flexgemm_compare020.py')
spec=importlib.util.spec_from_file_location('compare020',helper_path)
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
read=helpers.read;sha=helpers.sha
a,b=read(ACTUAL/'result.json'),read(BASE/'result.json')
assert a['phase']=='complete' and a['sources_unchanged'] and a['status']==b['status']=='numerical_failure'
for key in ('gate','reference_files_sha256','checkpoint_sha256','integration_source_sha256','topology_policy'):
    assert a[key]==b[key],key
expected=read(ROOT.parent/'iteration-021'/'application-comparison.json')['artifacts']
for row in expected:
    p=Path(row['path'])
    if p.parent==BASE:assert sha(p)==row['sha256']
result={'scope':'New kernel versus iteration021 stage-cache application; identical adapter, inputs and checkpoints; no paired latency claim',
    'cpu_status':'numerical_failure','baseline_report':str(BASE/'result.json'),'baseline_report_sha256':sha(BASE/'result.json'),
    'comparisons':{},'artifacts':[]}
for name in ('shape_tt_fixed.pt','shape_tt_predicted.pt','texture_tt_fixed.pt','texture_tt_predicted.pt','fixed_tt_guides.pt','predicted_tt_guides.pt'):
    x=torch.load(ACTUAL/name,map_location='cpu',weights_only=True)
    y=torch.load(BASE/name,map_location='cpu',weights_only=True)
    if isinstance(x,list):
        assert len(x)==len(y)==4
        item=[helpers.compare(xx,yy,signs=True) for xx,yy in zip(x,y)]
    else:item=helpers.compare(x,y,geometry=name.startswith('shape'))
    result['comparisons'][name]=item
    for entry in item if isinstance(item,list) else [item]:
        assert entry['coordinates_equal'] and entry['common_features_bitwise_equal']
    del x,y
    for p in (ACTUAL/name,BASE/name):result['artifacts'].append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
(ROOT/'application-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print('All 12 complete output/guide records match iteration021 bitwise; original strict CPU gates remain failed')
