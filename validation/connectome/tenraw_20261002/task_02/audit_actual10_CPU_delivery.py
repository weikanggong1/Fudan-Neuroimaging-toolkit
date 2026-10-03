"""Read and hash existing actual ten-case CPU references; no MRI/GPU solver is launched."""
import base64,hashlib,json
from pathlib import Path

REMOTE_PROGRAM=r"""import base64,hashlib,json,socket,platform
from pathlib import Path
from datetime import datetime,timezone
E=json.loads(__EXPECTED_JSON__)
R=Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002/task_02')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
checks=[];unique_outputs={};source_bindings={};diags={};cases={}
def check(label,value):
 if not value:raise AssertionError(label)
 checks.append(label)
def verify(record,label):
 p=Path(record['path']);actual=sha(p);check(label+':SHA',actual==record['sha256']);return p
report=load(verify(E['summary']['report']['remote'],'final_evaluation'))
case_map_path=verify(E['summary']['actual_completed_case_map'],'actual_case_map');case_map=load(case_map_path)
check('actual_10_models_and_diagnostics',report['completed_case_count']==report['same_input_diagnostic_completed_case_count']==10 and case_map['models_completed']==10 and len(case_map['actual_completed_case_map'])==10)
for s,row in E['summary']['cases'].items():
 cp=verify(row['consumer_contract'],s+':consumer');rp=verify(row['modeling_report'],s+':report');fp=verify(row['figure'],s+':brainfigure');dp=verify(row['same_input_CPU_tensor_diagnostic'],s+':sameinputdiag')
 c=load(cp);r=load(rp);diagnostic=load(dp);check(s+':case_and_complete',c['case_id']=='sub-'+s and c['execution_completed'] and r['execution_completed'] and c['state']==r['state']=='completed')
 check(s+':model_SHA_and_CPU',c['harness_sha256']==r['harness_sha256']=='616b3f01197447da8255c20592165f066f03bc20b48765b9612ea6ada29d11c1' and c['CPU_threads']==r['threads']==8)
 check(s+':consumer_original_report',c['modeling_report_sha256']==sha(rp) and c['modeling_report']==str(rp))
 check(s+':sameinput_report_dictionary',diagnostic==row['same_input_CPU_tensor_diagnostic']['report'])
 check(s+':sameinput_FA',diagnostic['fa']==report['cases'][s]['same_input_CPU_tensor_diagnostic']['report']['fa'])
 check(s+':map_uses_actual_namespace',case_map['actual_completed_case_map'][s]['consumer_contract']==row['consumer_contract'])
 check(s+':actualcommands13',len(r['commands'])==13 and all(x['returncode']==0 for x in r['commands']))
 check(s+':command_time_per_label',{x['label']:x['wall_s'] for x in r['commands']}==row['official_stage_wall_s'])
 for key,item in c['files'].items():
  p=Path(item['path'])
  if str(p) not in unique_outputs:
   actual=sha(p);check(s+':output:'+key,actual==item['sha256']);unique_outputs[str(p)]={'SHA256':actual,'bytes':p.stat().st_size}
  else:check(s+':alias:'+key,unique_outputs[str(p)]['SHA256']==item['sha256'])
 check(s+':sameinput_3keys',[diagnostic['input_sha256'][key] for key in ('dwi','gradient','brain_mask')]==[c['files'][key]['sha256'] for key in ('official_corrected_dwi','official_gradient_mrtrix','brain_mask')])
 diags[s]={'path':str(dp),'sha256':sha(dp),'bytes_base64':base64.b64encode(dp.read_bytes()).decode()}
 cases[s]={'consumer_SHA256':sha(cp),'report_SHA256':sha(rp),'same_input_diagnostic_SHA256':sha(dp),'brain_SHA256':sha(fp),'original_modeling_namespace':str(cp.parent.parent),'command_count':len(r['commands'])}
source=E['source_contract']
source_payload={}
for name,row in source.items():
 if isinstance(row,dict) and 'path' in row and 'sha256' in row:
  p=verify(row,'CON11:'+name);source_bindings[name]={'path':str(p),'sha256':sha(p)}
  if name in ['actual_status','actual_dispatcher_freeze','actual_launch','actual_model_freeze','actual_configuration','actual_alignment_binding']:source_payload[name]=load(p)
for name,row in source['actual_raw_control_source_bindings'].items():
 if isinstance(row,dict):verify(row,'CON11control:'+name)
worker=R/'official_modeling_cohort_cpu_v3.py';wrapper=R/'orchestrate_CON11_CPU_subset_v2.py';check('science616b_metadata_bf62',sha(worker)=='616b3f01197447da8255c20592165f066f03bc20b48765b9612ea6ada29d11c1' and sha(wrapper)=='bf62ce93000e79215317514fcd61ba0a15a06487ea1d775fb5342bd85f1206d4')
dispatcher=source_payload['actual_dispatcher_freeze'];model=source_payload['actual_model_freeze'];check('precursor_distinct_original_freezes',source['actual_dispatcher_freeze']['sha256']=='12066e3cc3269cdd5d28e85115b4bb4ed505ab22c984690ac66df733e073a372' and source['actual_model_freeze']['sha256']=='d27ed9a016db4c5938de69e5023a7de6448b731a7a95d23b83dcb09fb9f4e8b4' and dispatcher['subset_created'] is False and dispatcher['output_root_planned']==str(Path(source['actual_model_freeze']['path']).parent) and dispatcher['worker']==model['worker'] and dispatcher['wrapper']==model['wrapper'] and dispatcher['base_config']==model['base_config'])
check('completed_vs_precursor',source_payload['actual_status']['subset_completed'] is True and source_payload['actual_status']['consumer_contract']==source['actual_consumer'])
check('sameinput_historicalCON07_preserved',report['cases']['CON07']['same_input_CPU_tensor_diagnostic']['report']['direction_antipodal_degrees']['max']==E['summary']['cases']['CON07']['same_input_CPU_tensor_diagnostic']['report']['direction_antipodal_degrees']['max'])
print(json.dumps({'observed_UTC':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),'python':platform.python_version(),'CPU_hash_and_metadata_only':True,'MRI_or_GPU_solver_started':False,'check_count':len(checks),'checks':checks,'cases':cases,'unique_actual_model_output_files':unique_outputs,'source_bindings':source_bindings,'CON11_distinct_actual_metadata':source_payload,'frozen_worker_SHA256':sha(worker),'actual_wrapper_SHA256':sha(wrapper),'same_input_diagnostic_original_bytes':diags},indent=2,allow_nan=False))
"""

def main():
    import argparse,subprocess
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-root',type=Path,required=True)
    parser.add_argument('--control-path',type=Path,required=True)
    parser.add_argument('--host',default='gongwk@10.190.248.228')
    parser.add_argument('--port',type=int,default=39516)
    parser.add_argument('--output',type=Path,required=True,help='New local proof; refuses replacement')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    expected={'summary':load(args.local_root/'actual_CPU_budget_modeling_10of10_summary.json'),'source_contract':load(args.local_root/'actual_CON11_source_contract.json')}
    code=REMOTE_PROGRAM.replace('__EXPECTED_JSON__',repr(json.dumps(expected,allow_nan=False)))
    result=subprocess.run(['ssh','-S',str(args.control_path),'-o','ControlMaster=no','-o','BatchMode=yes','-p',str(args.port),args.host,'/usr/bin/python3 -'],input=code,text=True,capture_output=True,check=True)
    proof=json.loads(result.stdout)
    # Retain complete original bytes for missing diagnostic JSON, with exact original SHA.
    original=proof.pop('same_input_diagnostic_original_bytes')
    for subject,item in original.items():
        data=base64.b64decode(item['bytes_base64']);assert hashlib.sha256(data).hexdigest()==item['sha256']
        destination=args.local_root/'actual_official_CPU_models'/subject/'same_input_CPU_tensor_diagnostic.json'
        if destination.exists():assert destination.read_bytes()==data
        else:
            with destination.open('xb') as file:file.write(data)
        item.pop('bytes_base64')
    proof['same_input_diagnostic_original_byte_receipts']=original
    proof['local_audit_tool_SHA256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    proof['local_summary_SHA256']=hashlib.sha256((args.local_root/'actual_CPU_budget_modeling_10of10_summary.json').read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as file:file.write(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'checks':proof['check_count'],'cases':len(proof['cases']),'unique_actual_outputs_hashed':len(proof['unique_actual_model_output_files']),'proof_SHA256':hashlib.sha256(args.output.read_bytes()).hexdigest()}))

def load(path):return json.loads(Path(path).read_text())

if __name__=='__main__':main()
