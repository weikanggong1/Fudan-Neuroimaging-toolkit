"""Read cached actual reports; return only whitelisted public dataset metrics.
No pipeline/driver export, MRI/license reads, remote mutations or benchmark runs.
"""
import argparse,datetime,hashlib,json,math,re
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fnit-root', type=Path, required=True, help='已有固定索引和冻结运行记录的 FNIT 根目录')
parser.add_argument('--excluded-program-root', type=Path, action='append', default=[], help='仅检查元数据目录引用；不能证明运行时隔离')
args=parser.parse_args()
ROOT=args.fnit_root.resolve()
QR=ROOT/'runs/recon_accuracy_20261003/precision_candidate_3a_ten_case_queue_v1'
IDS=('ds000114_sub-06','ds000114_sub-07','ds000114_sub-04','ds000114_sub-05','ds000114_sub-08','ds000030_sub-10159','ds000030_sub-10171','ds000030_sub-10189','ds000030_sub-10193')
RUNS=ROOT/'runs/recon_accuracy_20261003'
API_FIRST=RUNS/'precision_api_evaluation_recovered_ed16ceeb_v1'
API_REST=RUNS/'precision_api_evaluation_recovered_ed16ceeb_remaining_api_v1'
cache={}
def read(path,records,key,required=True):
 p=Path(path);s=str(p)
 if s not in cache:
  try:
   before=p.stat();content=p.read_bytes();after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
   cache[s]=(json.loads(content),{'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
  except FileNotFoundError:
   if required:raise
   cache[s]=({}, {'status':'unknown'})
 data,receipt=cache[s];records[key]=receipt;return data

PUBLIC_ENUMS={'unknown','complete','measured','not_assessed','not_assessed_vertex_correspondence','not_assessed_invalid_space','disabled','cli','initialized_cuda_api','preserved_preinitialized_unknown','passed','failed','mm','mm2','mm3'}
def public_tree(value):
 if value is None or isinstance(value,(bool,int)):return value
 if isinstance(value,float):return value if math.isfinite(value) else 'unknown'
 if isinstance(value,str):
  return value if value in PUBLIC_ENUMS or re.fullmatch(r'[0-9a-f]{64}',value) else 'unknown'
 if isinstance(value,list):
  # Coordinates, vertex identifiers and individual intersection pairs are private.
  return []
 if isinstance(value,dict):
  return {k:public_tree(v) for k,v in value.items()
          if re.fullmatch(r'[A-Za-z0-9_.+/-]+',k) and not k.startswith('/')
          and not any(x in k.lower() for x in ('first_100','first100','coordinate','pairs_detail','pair_details','source_path'))
          and k not in ('subject','host','path','command','anomaly_vertex_ids_first_100')}
 return 'unknown'

def plan_post(value,case):
 if isinstance(value,dict):
  if value.get('id',value.get('case'))==case:
   for key in ('post_evaluation','post_eval','evaluation'):
    if isinstance(value.get(key),dict) and 'config' in value[key]:return value[key]
   if 'config' in value:return value
  for child in value.values():
   result=plan_post(child,case)
   if result:return result
 elif isinstance(value,list):
  for child in value:
   result=plan_post(child,case)
   if result:return result
 return None

def api_post(case,records):
 root=API_FIRST if case.endswith('10159') else API_REST
 names=('single-case-evaluation-only.plan.template.json','plan.json') if case.endswith('10159') else ('remaining-api-evaluation-only.plan.json',)
 for name in names:
  if (root/name).is_file():
   post=plan_post(read(root/name,records,'api_plan'),case)
   if post:return post
 for candidate in (root/(case+'.evaluation.json'),root/'evaluations'/(case+'.evaluation.json'),root/'evalroot'/(case+'.evaluation.json')):
  if candidate.is_file():return {'config':str(candidate)}
 raise ValueError('API evaluation configuration not found for '+case)

def private_path_provenance(*values):
 strings=[]
 def walk(value):
  if isinstance(value,str):strings.append(value)
  elif isinstance(value,dict):
   for key,item in value.items():walk(key);walk(item)
  elif isinstance(value,list):
   for item in value:walk(item)
 for value in values:walk(value)
 prefixes=[str(p.resolve()).rstrip('/') for p in args.excluded_program_root]
 return {'check_performed':bool(prefixes),
         'excluded_root_referenced':any(prefix in value for prefix in prefixes for value in strings) if prefixes else 'not_assessed',
         'scope':'metadata reference check only; not process, library or isolation verification'}

def selected(data,keys):
 result={k:public_tree(data[k]) if k in data else 'unknown' for k in keys}
 # 名称仅来自公开 atlas/LUT 的具名字段，不开放任意来源字符串。
 if 'name' in keys and isinstance(data.get('name'),str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,100}',data['name']):
  result['name']=data['name']
 return result

indexes={}
# Actual canonical content read, never returned.
for name in ('README.md','INDEX.md','INDEX.json'):
 p=ROOT/name
 if p.is_file():
  content=p.read_bytes();indexes[name]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
assert 'README.md' in indexes and any(k in indexes for k in ('INDEX.md','INDEX.json'))
global_records={};plan=read(QR/'plan.json',global_records,'original_queue_plan');queue=read(QR/'queue.json',global_records,'original_queue_status')
for name,root in (('api_initial',API_FIRST),('api_remaining',API_REST)):
 read(root/'queue.json',global_records,name+'_queue')
rows=[]
for case in IDS:
 record={}
 if case.startswith('ds000030_'):post=api_post(case,record)
 elif plan.get('initial_evaluation',{}).get('id')==case:post=plan['initial_evaluation']
 else:post=next(r['post_evaluation'] for r in plan['cases'] if r['id']==case)
 config=read(post['config'],record,'evaluation_config')
 actual=read(config['evaluated_config'],record,'actual_raw_config')
 official=read(config['official_config'],record,'official_raw_config')
 out=Path(config['output']);checkpoint=read(out/'checkpoint.json',record,'checkpoint')
 assert checkpoint['status']=='complete' and len(checkpoint['phases'])==18 and all(p['status']=='complete' for p in checkpoint['phases'].values())
 assert checkpoint['config_sha256']==record['evaluation_config']['sha256']
 assert checkpoint['case']==case and config['case']==case and checkpoint['evaluated_role']==config['evaluated_role']=='precision_candidate'
 assert checkpoint['schema']=='fnit-recon-pair-evaluation-v1'
 assert re.fullmatch(r'[0-9a-f]{64}',checkpoint['script_sha256']) and re.fullmatch(r'[0-9a-f]{64}',checkpoint['role_helper_sha256'])
 if post.get('config_sha256'):assert post['config_sha256']==record['evaluation_config']['sha256']
 if post.get('script_sha256'):assert post['script_sha256']==checkpoint['script_sha256']
 raw_diag=Path(actual['diagnostic_root']);official_diag=Path(official['diagnostic_root'])
 completion=read(raw_diag/'completion.json',record,'fnit_raw_completion');launch=read(raw_diag/'launch.json',record,'fnit_raw_launch')
 official_completion=read(official_diag/'completion.json',record,'official_raw_completion');official_launch=read(official_diag/'launch.json',record,'official_raw_launch')
 assert actual['code_commit']=='3a0c9aba6321b4981fd8174b4b191515459aa38b' and actual['input_sha256']==official['input_sha256']
 assert completion.get('code_commit')==actual['code_commit']
 assert completion.get('source_archive_sha256')==actual['source_archive_sha256']
 assert actual['source_archive_sha256']=='03cc806fb449a8620c8caa78b1af86dfaff6b64ab9e9e7c5210cf12184aa7f31'
 assert completion['execution_status']=='complete' and completion['pipeline_status']=='complete' and completion['exit_code']==completion['child_exit_code']==0
 assert official_completion['execution_status']=='complete' and official_completion['exit_code']==official_completion['child_exit_code']==0
 monitor=read(raw_diag.parent/'monitor/monitor.json',record,'fnit_raw_monitor',False)
 strict=read(out/'strict_precision_candidate_vs_official.json',record,'strict_138')
 assert type(strict['checked']) is int and strict['checked']==138 and type(strict['passed']) is int and 0<=strict['passed']<=138
 assert checkpoint['strict_138']=={k:strict[k] for k in ('checked','passed')}
 all_regions=read(out/'all_regions_precision_candidate_vs_official.json',record,'all_regions',False)
 no_th3=read(out/'no_th3_precision_candidate_vs_official.json',record,'no_th3',False)
 geometry=read(out/'geometry_precision_candidate_vs_official.json',record,'geometry')
 binding=read(out/'execution_binding.json',record,'execution_binding')
 region=read(out/'region_precision_candidate_vs_official.json',record,'region_statistics')
 dice=read(out/'dice_precision_candidate_vs_official.json',record,'label_dice')
 local=read(out/'local_precision_candidate_vs_official.json',record,'local_anomalies')
 quality={}
 for role in ('precision_candidate','official'):
  report=read(out/('quality_'+role)/'report.json',record,'quality_'+role)
  quality[role]={'status':report.get('status','unknown'),'hemispheres':{h:{k:public_tree(v) for k,v in data.items() if k in ('sphere_orientation','topology','vertex_links','white_pial_crossings','ordered_faces_and_vertex_counts_preserved')} for h,data in report.get('hemispheres',{}).items()}}
 surfaces={}
 for stage in ('orig.nofix','orig.premesh','orig','white.preaparc','white','pial','sphere','sphere.reg'):
  report=read(out/('surface_'+stage+'_precision_candidate_vs_official.json'),record,'surface_'+stage)
  surfaces[stage]={h:public_tree(data.get(stage,{})) for h,data in report.get('stages',{}).items()}
 aparc={}
 for metric in ('ThickAvg','SurfArea','GrayVol','MeanCurv'):
  data=region.get('aparc_68',{}).get(metric,all_regions.get('aparc',{}).get(metric,{}))
  aparc[metric]=selected(data,('matched_regions','mae','maximum_absolute_error','median_absolute_relative_error_percent','p90_absolute_relative_error_percent','pearson_r'))
  values=data.get('per_region',{})
  aparc[metric]['worst_absolute_region']=max(values,key=lambda k:values[k].get('absolute_error',-1)) if values else 'unknown'
  aparc[metric]['per_region']=public_tree(values)
 assert set(dice.get('files',{})).issubset({'aseg.mgz','aparc+aseg.mgz','aparc.DKTatlas+aseg.mgz','aparc.a2009s+aseg.mgz','wmparc.mgz','ribbon.mgz','filled.mgz'})
 dice_summary={filename:{**selected(data,('minimum_dice','p05_dice','median_dice','different_voxels','worst_labels')),'per_label':{label:selected(value,('name','dice','reference_voxels','candidate_voxels')) for label,value in data.get('per_label',{}).items()}} for filename,data in dice.get('files',{}).items()}
 timings={'fnit_raw_entry_command_seconds':completion.get('command_seconds','unknown'),'fnit_internal_api_total_seconds':completion.get('pipeline_total_seconds','unknown'),'official_raw_entry_command_seconds':official_completion.get('command_wall_seconds','unknown'),'monitor_wall_seconds':monitor.get('command_wall_seconds','unknown'),'scope':'original raw entry end-to-end and existing official run; no new repeats; admission wait and later comparison excluded'}
 a,b=timings['fnit_raw_entry_command_seconds'],timings['official_raw_entry_command_seconds'];timings['observed_official_to_fnit_entry_ratio']=b/a if isinstance(a,(int,float)) and isinstance(b,(int,float)) and a>0 else 'unknown'
 timings['comparison_phase_seconds']={k:v.get('seconds','unknown') for k,v in checkpoint['phases'].items()}
 timings['comparison_lock_wait_seconds']={k:v.get('lock_wait_seconds','unknown') for k,v in checkpoint['phases'].items()}
 protocol={'same_raw_t1_sha256':actual['input_sha256']==official['input_sha256'],'same_launch_host':launch.get('host')==official_launch.get('host') if launch.get('host') and official_launch.get('host') else 'unknown','same_declared_gpu':actual.get('gpu_uuid')==official.get('gpu_uuid'),'fnit_threads':actual.get('threads','unknown'),'official_threads':official.get('threads','unknown'),'official_version':official_completion.get('code_version','unknown'),'fnit_invocation':actual.get('invocation','unknown'),'new_repeated_benchmark':False}
 rows.append({'case':case,'dataset':'OpenNeuro '+case.split('_')[0],'production_commit':actual['code_commit'],'raw_t1_sha256':actual['input_sha256'],'source_archive_sha256':actual['source_archive_sha256'],'evaluation_status':'18_phases_complete','overall_metric_equivalence':'not_assessed','protocol':protocol,'timing_seconds':timings,'memory':{**selected(monitor,('peak_sampled_process_bytes','sampling_interval_requested_seconds','maximum_sampling_gap_seconds','failed_app_queries','samples','continuous_peak_verified')),'scope':'same-time sampled monitored parent/children; not continuous peak or whole-card peak'},'strict_138':selected(strict,('checked','passed','all_pass')),'label_dice':dice_summary,'aparc_68':aparc,'all_regions':public_tree(all_regions),'no_th3':public_tree(no_th3.get('atlases',{})),'local_maps':public_tree(local.get('maps',{})),'surface_bidirectional_vertex_to_triangle':surfaces,'quality':quality,'geometry':public_tree({k:v for k,v in geometry.items() if k in ('surfaces','volumes')}),'original_comparison_tool_binding':{'evaluator_sha256':checkpoint['script_sha256'],'role_helper_sha256':checkpoint['role_helper_sha256']},'private_resource_path_check':private_path_provenance(config,actual,official,binding),'source_reports':record})
result={'source_production_commit':'3a0c9aba6321b4981fd8174b4b191515459aa38b','current_validation_tools_commit':'ed16ceebd796fa910e9b58e07e965da4917a6cf7','overall_metric_equivalence':'not_assessed','schema':'fnit-public-existing-nine-comparisons-20261007-v1','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'privacy':'public metrics and opaque original-file hashes only; no private paths, network identifiers or license fields','canonical_index_receipts':indexes,'queue_receipts':global_records,'scope':'read once per original path; no pipeline/driver export, imaging execution or new timing repeats','cases':rows}
encoded=json.dumps(result,indent=2,allow_nan=False)
assert not re.search(r'/cwStorage|/public/software|/home/|/mnt/|\b(?:\d{1,3}\.){3}\d{1,3}\b|FS_LICENSE|license\.txt',encoded)
assert len(rows)==9 and len({r['case'] for r in rows})==9
print(encoded)
