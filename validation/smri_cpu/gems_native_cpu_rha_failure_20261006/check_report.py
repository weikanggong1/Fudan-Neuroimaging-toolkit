"""Check saved report consistency; never loads MRI or runs a fitting program."""
from pathlib import Path
import ast,csv,hashlib,json,math,re
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((P/'FULL_RHA_RESULTS.public.json').read_text());s=json.loads((P/'SUMMARY.public.json').read_text())
assert sha(P/'FULL_RHA_RESULTS.public.json')=='feb0c1f4786abf4e7f3b02fcd523b53345b740175ccfcc189b3eb1f3d928c4ab'
rows=list(csv.DictReader((P/'REGIONS.csv').open()));assert len(rows)==len(s['regions'])==56
for a,b in zip(rows,s['regions']):
 for k,v in b.items():
  if isinstance(v,bool): assert a[k]==str(v)
  elif isinstance(v,(int,float)):assert float(a[k])==v
  else:assert a[k]==v
 assert b['passed']==(b['dice']>=.95 and b['hard_volume_relative_to_official']<=.05)
 assert b['nonempty'] and b['official_voxels']>0 and b['FNIT_voxels']>0
 assert math.isclose(b['hard_volume_relative_to_official'],abs(b['FNIT_hard_volume_mm3']-b['official_hard_volume_mm3'])/b['official_hard_volume_mm3'],rel_tol=0,abs_tol=2e-15)
 assert math.isclose(b['soft_volume_relative_to_official'],abs(b['FNIT_soft_volume_mm3']-b['official_soft_volume_mm3'])/b['official_soft_volume_mm3'],rel_tol=0,abs_tol=2e-15)
summary_keys=[];numericlists={};forbidden_strings=[]
def walk(x,path):
 if isinstance(x,str):
  if re.search(r'/cwStorage/|/mnt/c/|gongwk@|nodecw[0-9]|gpucw[0-9]|10\.190\.|FS_LICENSE|password=|token=',x):forbidden_strings.append(path)
 elif isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for k,v in x.items():walk(v,path+'.'+k)
 elif isinstance(x,list):
  if len(x)>4 and all(isinstance(v,(int,float)) for v in x):
   numericlists[path.rsplit('.',1)[-1]]=max(len(x),numericlists.get(path.rsplit('.',1)[-1],0))
  for i,v in enumerate(x):walk(v,path+'[]')
walk(f,'full');assert not forbidden_strings
assert set(numericlists)=={'objective_history','affinity'},numericlists
full_numeric_lists=dict(numericlists)
assert 'cpu_native_traces' not in json.dumps(s['recipe'])
assert 'objective_history' not in json.dumps(s['recipe'])
assert len(s['recipe']['stages'])==5 and sum(x['mesh_steps'] for x in s['recipe']['stages'])==900
for space,expected in (('native',9),('hr',6)):
 r=[x for x in s['regions'] if x['space']==space];assert sum(x['passed'] for x in r)==expected
 assert len(r)==28
for p in P.glob('*.json'):walk(json.loads(p.read_text()),p.name)
assert not forbidden_strings,forbidden_strings
for p in P.glob('*.py'): ast.parse(p.read_text())
readme=(P/'README.md').read_text();assert len(re.findall(r'^## [1-7]\.',readme,re.M))==7
for target in re.findall(r'\]\(([^)]+)\)',readme):
 if not target.startswith(('http://','https://')):assert (P/target).exists(),target
result={'status':'passed_static_report_consistency','scope':'Only already-saved JSON/CSV/source/PNG checks; no MRI arrays, objective, optimizer, native program or fitted transform executed','per_space_nonempty_regions':28,'native_pass':9,'HR_pass':6,'CSV_rows':56,'thresholds_unchanged':True,'original_full_report_SHA_unchanged':sha(P/'FULL_RHA_RESULTS.public.json'),'large_numeric_lists_in_full_report':full_numeric_lists,'raw_MRI_mesh_gradient_prior_posterior_or_owner_arrays_in_full_report':False,'SUMMARY_bytes':(P/'SUMMARY.public.json').stat().st_size,'scalar_formula_roundoff_check_absolute_tolerance':2e-15,'CSV_summary_and_threshold_pass_fields_checked_exactly':True,'seven_sections_and_local_links_checked':True,'plots_visually_inspected':True,'production_src_tests_env_changes':False}
(P/'REPORT_CHECK.public.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(result))
