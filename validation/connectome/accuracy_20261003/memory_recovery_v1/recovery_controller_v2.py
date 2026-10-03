"""Retain v1 validation failure; verify its completed CON09 and dispatch only CON10.

Only namespace-generated prepared/dwi and prepared/bvecs paths may differ.
Their actual bytes, all external input paths/bytes, science, budgets and frozen
worker remain strictly identical. Never reexecute the completed CON09.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path('/cwStorage/home/gongwk/Notebook_code/FNIT/runs/connectome-accuracy-memory-recovery-20261003-v1')
spec=importlib.util.spec_from_file_location('immutable_recovery_v1',ROOT/'recovery_controller.py')
v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)
v1.bound(ROOT/'recovery_controller.py','333dc44f5bc8adbec2299f3e8fbf945f410a60bbb1cf19082bd3ab7a5c02a331')

def compare_input_bindings(expected,observed,original_job,recovery_job,original_outputs,recovery_outputs):
    v1.require(set(observed)==set(expected),'actual input coverage differs')
    changed={};derived={'prepared/dwi':'data.nii.gz','prepared/bvecs':'data.eddy_rotated_bvecs'}
    v1.require(set(derived)<=set(observed),'actual produced DWI/bvec input records missing')
    for key in observed:
        if key not in derived:
            v1.require(observed[key]==expected[key],f'external raw/FS/resource differs: {key}')
        else:
            original_path=Path(original_job)/'connectome/preproc/eddy'/derived[key]
            recovery_path=Path(recovery_job)/'connectome/preproc/eddy'/derived[key]
            v1.require(expected[key]['path']==str(original_path) and observed[key]['path']==str(recovery_path),'prepared output outside exact respective new namespace')
            v1.require(observed[key]['sha256']==expected[key]['sha256'] and observed[key]['size_bytes']==expected[key]['size_bytes'],'prepared output bytes differ; no tolerance')
            for path,record,outputs in ((original_path,expected[key],original_outputs),(recovery_path,observed[key],recovery_outputs)):
                v1.require(outputs.get(str(path))=={'sha256':record['sha256'],'size_bytes':record['size_bytes']},'prepared input disagrees with actual worker producer output ledger')
            changed[key]={'original':expected[key],'recovery':observed[key],'byte_SHA_equal':True,
                'actual_producer_outputs_verified':True,'scope':'each run generated this corrected output from raw; path difference required by fresh namespace'}
    return changed

def verify_actual_case(config,clone,driver,cases,bindings,case_id):
    old=Path(config['run_root'])/'candidate'/case_id;new=ROOT/'candidate'/case_id
    report=json.loads((new/'gpu_report.json').read_bytes());wall=json.loads((new/'raw_bids_wall.json').read_bytes())
    old_gpu=json.loads((old/'gpu_report.json').read_bytes());old_wall=json.loads((old/'raw_bids_wall.json').read_bytes())
    v1.require(report['status']==wall['status']=='completed' and report['exit_code']==wall['exit_code']==0,'recovery science did not complete')
    v1.require(report['case_id']==case_id and report['version']=='candidate','wrong completed case/version')
    driver.cohort.check_wall_report(wall,clone,cases[case_id]);driver.cohort.check_selected_inputs(wall,cases[case_id])
    subject=bindings['cases'][case_id]['anatomy']['directory']
    v1.require(wall['cli_arguments']==driver.cohort.cli_command(clone,cases[case_id],new,anatomy_subject=subject),'actual science CLI differs')
    v1.require(report['source_before']['source_fingerprint']==report['source_after']['source_fingerprint']==v1.SCIENCE,'recovery source differs')
    v1.require(report['anatomy']==report['anatomy_after']==bindings['cases'][case_id]['anatomy']['files'],'recovery FS differs')
    v1.require(report['raw_dwi_cli_total_runtime_seconds']==wall['total_runtime_seconds'],'actual timer binding differs')
    v1.require(Path(report['wall_report'])==new/'raw_bids_wall.json','actual wall report path differs')
    budget=driver.cohort.memory_budget(wall);v1.require(budget==report['memory_budget'],'GPU/wall memory ledger differs');v1.strict_budget(budget)
    monitor=wall['gpu_process_memory'];v1.require(monitor['backend']=='pynvml' and budget['status']=='observed_below_budget' and not budget['monitor_issues'],'strict direct NVML gate failed')
    before=json.loads((ROOT/f'{case_id}_original_before.json').read_bytes());after=json.loads((ROOT/f'{case_id}_original_after.json').read_bytes())
    v1.require(before==after,'original preserved bytes differ')
    observed=v1.verify_input_inventory(wall);expected=before['inputs']
    changed=compare_input_bindings(expected,observed,old,new,old_gpu['outputs']['files'],report['outputs']['files'])
    command=list(report['command']);original_command=old_gpu['command']
    v1.require(command[0]==v1.MONITOR_PYTHON and original_command[0]==v1.ORIGINAL_PYTHON,'interpreter command differs')
    command[0]=original_command[0]
    for flag in ('--report','--result-export-dir','--output-dir'):
        v1.require(command.count(flag)==original_command.count(flag)==1,'ambiguous output path flag')
        command[command.index(flag)+1]=original_command[original_command.index(flag)+1]
    v1.require(command==original_command,'science argv differs beyond interpreter and three output paths')
    return {'status':'completed','gpu_report':v1.bound(new/'gpu_report.json'),'wall_report':v1.bound(new/'raw_bids_wall.json'),
            'memory':budget,'independent_CLI_seconds':report['raw_dwi_cli_total_runtime_seconds'],
            'scientific_array_comparison':'not_assessed_CPU_followup','original_bytes_preserved':True,
            'generated_prepared_output_binding':changed,'external_inputs_byte_identical':True,
            'changed_command_fields':['interpreter','--report','--result-export-dir','--output-dir']}

def run():
    v1.require(not (ROOT/'validation_recovery_v2.json').exists(),'refuse duplicate validation recovery')
    error=json.loads((ROOT/'controller_failure.json').read_bytes())
    v1.require(error['error']=='ValueError: recovery science used different input/resource bytes','only the actual v1 prepared-output path guard may be recovered')
    config,driver,cases,bindings=v1.frozen();done,terminal=v1.formal_completed(config);v1.require(done,'original formal phase not complete')
    clone=json.loads((ROOT/'recovery_configuration.json').read_bytes());expected=copy.deepcopy(config)
    expected.update(run_root=str(ROOT),gpu_python=v1.MONITOR_PYTHON,frozen_sources=copy.deepcopy(config['declared_source_manifests']))
    v1.require(clone==expected,'immutable recovery clone changed')
    plan=json.loads((ROOT/'frozen_recovery_plan.json').read_bytes());v1.require([r['case_id'] for r in plan['explicit_selection']]==['sub-CON09','sub-CON10'],'original explicit selection differs')
    v1.load_runtime();finished=verify_actual_case(config,clone,driver,cases,bindings,'sub-CON09')
    v1.require(not (ROOT/'candidate/sub-CON10').exists(),'CON10 must remain strictly undispatched')
    for source,dest in (('status.json','controller_v1_terminal_snapshot.json'),('CPU_status.json','CPU_status_v1.json')):
        v1.require(not (ROOT/dest).exists(),'prior snapshot path exists');(ROOT/dest).write_bytes((ROOT/source).read_bytes())
    validation={'scope':'private verifier path-contract repair only; v1 failure retained; completed CON09 not rerun',
                'old_failure':v1.bound(ROOT/'controller_failure.json'),'v1_controller':v1.bound(ROOT/'recovery_controller.py'),
                'v2_controller':v1.bound(__file__),'original_terminal':terminal,'immutable_plan':v1.bound(ROOT/'frozen_recovery_plan.json'),
                'completed_CON09':finished,'pending_GPU_only':['sub-CON10'],'science_source_parameters_budgets_unchanged':True,'time_utc':v1.utc()}
    v1.atomic(ROOT/'validation_recovery_v2.json',validation)
    status={'status':'running_selected_monitor_recovery','start_utc':v1.utc(),'validation_recovery':v1.bound(ROOT/'validation_recovery_v2.json'),
            'original_configuration':v1.bound(v1.CONFIG),'cases':{'sub-CON09':finished},'scope':'same independent memory recovery; retained v1 verifier failure; original science/time never replaced'}
    v1.atomic(ROOT/'status.json',status)
    case_id='sub-CON10';original=Path(config['run_root'])/'candidate'/case_id
    wall=json.loads((original/'raw_bids_wall.json').read_bytes());subject=bindings['cases'][case_id]['anatomy']['directory']
    def snapshot():
        return {'config':v1.bound(v1.CONFIG),'GPU':v1.bound(original/'gpu_report.json'),'wall':v1.bound(original/'raw_bids_wall.json'),
                'inputs':v1.verify_input_inventory(wall),'scientific_output_inventory':v1.inventory(original),'runtime':v1.load_runtime(),
                'raw':driver.cohort.verify_inputs(cases[case_id]),'FS':driver.cohort.check_anatomy(subject,config['atlases'])}
    before=snapshot();v1.require(before['FS']==bindings['cases'][case_id]['anatomy']['files'],'actual supplied FS changed')
    v1.atomic(ROOT/f'{case_id}_original_before.json',before)
    def anatomy(actual_config,case,version,job):
        record=bindings['cases'][case['case_id']];prior=driver.bound(record['prior_gpu_report'])
        v1.require(prior['status']=='completed' and prior['case_id']==case['case_id'] and prior['anatomy']==record['anatomy']['files'] and prior['raw_input_provenance']==case['input_files'],'same FS origin differs')
        return {'anatomy':record['anatomy']['files']}
    def subject_path(actual_config,case,job):return subject
    status['cases'][case_id]={'status':'running','start_utc':v1.utc()};v1.atomic(ROOT/'status.json',status)
    extra=('--result-export-dir',str(ROOT/'candidate'/case_id/'returned_result'))
    actual=driver.cohort.worker({'action':'gpu','config':clone,'case':cases[case_id],'version':'candidate'},
        anatomy_loader=anatomy,anatomy_subject=subject_path,extra_wall_arguments=extra)
    after=snapshot();v1.require(after==before,'original/input/FS/runtime bytes changed during CON10');v1.frozen()
    v1.atomic(ROOT/f'{case_id}_original_after.json',after)
    v1.require(actual['status']=='completed','actual CON10 science failed')
    status['cases'][case_id]=verify_actual_case(config,clone,driver,cases,bindings,case_id)
    status.update(status='GPU_execution_completed_CPU_comparison_pending',end_utc=v1.utc());v1.atomic(ROOT/'status.json',status)

if __name__=='__main__':
    try:run()
    except Exception as error:
        v1.atomic(ROOT/'controller_failure_v2.json',{'status':'failed','time_utc':v1.utc(),'error':f'{type(error).__name__}: {error}'});raise
