"""仅CPU准备十例精度候选顺序计划；绝不调用guard/driver的main或启动queue。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
COMMIT = '3a0c9aba6321b4981fd8174b4b191515459aa38b'
ARCHIVE = '03cc806fb449a8620c8caa78b1af86dfaff6b64ab9e9e7c5210cf12184aa7f31'
PLAN_SHA = '3b6b76bb0b816899d3e07f8fe339c2ea9f1b6b3247d4a70501de2ef63451dd9e'
TEMPLATE_SHA = 'b7c5bd990ecc90b542c87323cfa1872cae1bf3246b8ea887a8f478e229ca0942'
TOOL_SHAS = {
 'after_startup_stage_whole.py':'273aac30511db8b74adf285ab1c8039c21c1f4e8e32033b22512b1a172c2f244',
 'resource_admission.py':'77449d11c47bd7d8bf1e289012a507d5436f94fb74abc153da2e7b35365de62c',
 'execute_whole_case.py':'6690d0e37682a024ef2daaa06d9e3c366ac2d41922905f1c0d39b3603ca93249',
 'evaluate_pair.py':'4e6a96e57009a5be4bc3c085809042aae093a7039c864a380b80c3574e114418',
 'run_monitored.py':'d0760007a9733f63c5aa57de42ec49adf95ccae06d64addc737349e7d57cedb2',
}
QUEUE_SHA='681201b6606e31e5d008c0edb7f2611bb479a3059520b281dcee1d1b9ada10f3'


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def record(path):return {'path':str(path),'sha256':digest(path),'bytes':Path(path).stat().st_size}
def frozen(path, expected):
 if digest(path)!=expected:raise ValueError('frozen SHA mismatch: '+str(path))
 return read(path)

def write(path,value):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,default=ROOT/'runs/recon_accuracy_20261003/precision_candidate_3a_ten_case_queue_v1')
 args=parser.parse_args();started=time.perf_counter();base=ROOT/'runs/recon_accuracy_20261003'
 tools=ROOT/'workspaces/recon_accuracy_20261003/precision_whole_tools_v1'
 queue=ROOT/'workspaces/recon_accuracy_20261003/precision_queue_tools_v1/run_resource_replay_queue.py'
 plan_path=base/'precision_candidate_3a_remaining_plan_v2/plan.json'
 template_path=base/'precision_candidate_3a_whole_sub06_v1/evaluation_precision_vs_official_v1.config.json'
 plan=frozen(plan_path,PLAN_SHA);template=frozen(template_path,TEMPLATE_SHA)
 for name,expected in TOOL_SHAS.items():
  if digest(tools/name)!=expected:raise ValueError('frozen tool changed: '+name)
 if digest(queue)!=QUEUE_SHA:raise ValueError('queue changed')
 if plan['source_commit']!=COMMIT or plan['source_archive_sha256']!=ARCHIVE:raise ValueError('candidate differs')
 if template['evaluated_role']!='precision_candidate' or template['evaluated_commit']!=COMMIT:raise ValueError('template role/source differs')
 if Path(template['output']).exists():raise FileExistsError('initial evaluation output already exists')
 initial_config=Path(template['evaluated_config']);actual=read(initial_config)
 completion_path=Path(actual['diagnostic_root'])/'completion.json';completion=read(completion_path)
 pipeline_path=Path(actual['output'])/'fnit-native-free-run.json';pipeline=read(pipeline_path)
 if (completion.get('execution_status')!='complete' or completion.get('pipeline_status')!='complete'
     or type(completion.get('exit_code')) is not int or completion['exit_code']!=0
     or type(completion.get('child_exit_code')) is not int or completion['child_exit_code']!=0
     or completion.get('code_commit')!=COMMIT or completion.get('source_archive_sha256')!=ARCHIVE
     or pipeline.get('status')!='complete' or pipeline.get('mesh_validation',{}).get('status')!='passed'
     or pipeline.get('output_validation',{}).get('present')!=138
     or pipeline.get('output_validation',{}).get('expected')!=138
     or pipeline.get('input')!=actual['input'] or pipeline.get('subject_dir')!=actual['output']):
  raise ValueError('initial raw completion/mesh/output/path gate failed')
 # Read-only byte checks: no raw/weights/license content copied into reports.
 for item in plan['file_bindings']:
  if digest(item['path'])!=item['sha256']:raise ValueError('original plan binding drift: '+item['path'])
 jobs=plan['jobs']
 if len(jobs)!=9 or len({j['case_id'] for j in jobs})!=9:raise ValueError('nine distinct jobs required')
 loaded=[]
 for job in jobs:
  whole=frozen(job['whole_config']['path'],job['whole_config']['sha256'])
  guard=frozen(job['guard_config']['path'],job['guard_config']['sha256'])
  if whole['invocation']!=job['expected_invocation'] or whole['threads']!=4 or whole['code_commit']!=COMMIT or whole['source_archive_sha256']!=ARCHIVE:raise ValueError('job protocol differs')
  for path in [whole['output'],whole['diagnostic_root'],guard['retry_root'],guard['admission_report']]:
   if Path(path).exists():raise FileExistsError('existing nominal/actual execution destination: '+path)
  original_official=Path(template['official_config']).parent/('official_'+job['case_id']+'.json')
  official=read(original_official)
  if official['input_sha256']!=whole['input_sha256'] or official['input']!=whole['input']:raise ValueError('official raw binding differs')
  if guard['retry_root']!=job['retry_root'] or guard['whole_case_config']!=job['whole_config']['path']:
   raise ValueError('guard/job paths differ')
  loaded.append((job,whole,guard,original_official))
 args.output.mkdir(parents=True,exist_ok=False)
 manifest={'schema':'fnit-precision-ten-case-plan-preparation-v1','status':'preparing','algorithm_entered':False,
 'started_utc':now(),'builder':record(Path(__file__)),'input_plan':record(plan_path),'initial_template':record(template_path),
 'source_commit':COMMIT,'source_archive_sha256':ARCHIVE,'file_bindings_rechecked':len(plan['file_bindings']),
 'queue_tool':record(queue),'tools':[record(tools/n) for n in TOOL_SHAS],
 'initial_actual_receipts':[record(initial_config),record(completion_path),record(pipeline_path)],
 'prepare_only_driver_command':'NOT EXECUTED: frozen execute_whole_case.py has no --prepare-only option',
 'prepare_scope':'CPU call after_startup_stage_whole.prepare_launch only; guard main and algorithm never entered',
 'preparations':[],'generated_files':[]}
 # Importing these standard-library validation modules does not import torch or create CUDA contexts.
 sys.path.insert(0,str(tools))
 spec=importlib.util.spec_from_file_location('frozen_prepare',tools/'after_startup_stage_whole.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 queue_plan={'execution_role':'precision_candidate','python':actual['python'],
 'admission_script':str(tools/'resource_admission.py'),'admission_script_sha256':TOOL_SHAS['resource_admission.py'],
 'shared_lock':template['lock'],'initial_evaluation':{'id':template['case'],'actual_config':str(initial_config),
 'script':str(tools/'evaluate_pair.py'),'script_sha256':TOOL_SHAS['evaluate_pair.py'],
 'config':str(template_path),'config_sha256':TEMPLATE_SHA},'cases':[]}
 try:
  for job,whole,guard,official_path in loaded:
   mark=time.perf_counter();row={'case':job['case_id'],'function':'after_startup_stage_whole.prepare_launch',
     'module_sha256':TOOL_SHAS['after_startup_stage_whole.py'],'config':record(Path(job['whole_config']['path'])),
     'guard':record(Path(job['guard_config']['path'])),'status':'preparing','algorithm_entered':False}
   manifest['preparations'].append(row)
   try:
    row['prepared_launch']=mod.prepare_launch(job['whole_config']['path'],guard)
    row.update(exit_code=0,status='prepared_not_executed')
   except Exception as error:
    row.update(exit_code=1,status='preparation_failed',error=repr(error));raise
   finally:row['wall_seconds']=time.perf_counter()-mark
   config=dict(template);config.update(case=job['case_id'],official_config=str(official_path),
      evaluated_config=str(Path(job['retry_root'])/'retry_config.json'),evaluated_resources=guard['admission_report'],
      output=str(args.output/'evaluations'/job['case_id']))
   config_path=args.output/'evaluation_configs'/(job['case_id']+'.json');write(config_path,config)
   manifest['generated_files'].append(record(config_path))
   queue_plan['cases'].append({'id':job['case_id'],'original_config':job['whole_config']['path'],
     'retry_root':job['retry_root'],'admission_report':guard['admission_report'],
     'post_evaluation':{'script':str(tools/'evaluate_pair.py'),'script_sha256':TOOL_SHAS['evaluate_pair.py'],
       'config':str(config_path),'config_sha256':digest(config_path)}})
  if sum(w['invocation']=='initialized_cuda_api' for _,w,_,_ in loaded)!=5:raise ValueError('five API cases required')
  if sum(w['invocation']=='cli' for _,w,_,_ in loaded)!=3 or actual['invocation']!='cli':raise ValueError('four CLI cases including initial required')
  # CLI: initial completed sub06 + three remaining ds000114 cases = four.
  write(args.output/'plan.json',queue_plan);manifest['generated_files'].append(record(args.output/'plan.json'))
  manifest.update(status='prepared_not_launched',queue_command=[actual['python'],str(queue),'--plan',str(args.output/'plan.json'),'--report',str(args.output/'queue.json')],queue_started=False)
 finally:
  manifest.update(finished_utc=now(),preparation_total_wall_seconds=time.perf_counter()-started)
  write(args.output/'preparation_manifest.json',manifest)
 print(json.dumps({'status':manifest['status'],'manifest':record(args.output/'preparation_manifest.json'),'plan':record(args.output/'plan.json')}))


if __name__=='__main__':main()
