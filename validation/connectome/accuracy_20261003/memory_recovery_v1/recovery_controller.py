"""Private monitor-only recovery coordinator; frozen worker and science stay read-only.

Default only waits/checks. --execute-after-formal permits explicit root-selected
CON09/CON10/(conditionally CON11) after all twelve original runs completed.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HUB = Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
NEW = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_accuracy_20261003_v1')
ROOT = HUB/'runs/connectome-accuracy-memory-recovery-20261003-v1'
CONFIG = NEW/'formal_frozen_v1/accuracy_configuration.json'
CONFIG_SHA = 'f8eba1cbf3eab2baa549172b32be2c9d3b1014ad478f6e3702d7987378f52f8c'
MONITOR_PYTHON = '/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002/nvml_monitor_runtime_v1/venv/bin/python'
ORIGINAL_PYTHON = '/cwStorage/home/gongwk/Notebook_code/fnit_conda_env_956b1a9/bin/python'
SCIENCE = 'a27fe1ad0aca34c23b62017dc0bacb6b7a4c44d3423bf855a840509ffc1b6e82'
PREFLIGHT_SHA = '627b26ef0a9abc47d0d5bcf626487b5e239cee783c22a14f9529d973437506bc'
LIMIT = 20_000_000_000

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20),b''): digest.update(block)
    return digest.hexdigest()

def require(condition,message):
    if not condition: raise ValueError(message)

def bound(path,expected=None,allow_symlink=False):
    path=Path(path)
    require(path.is_file() and (allow_symlink or not path.is_symlink()),f'actual regular file required: {path}')
    actual=sha(path)
    require(expected is None or actual==expected,f'changed bound file: {path}')
    record={'path':str(path),'sha256':actual,'size_bytes':path.stat().st_size}
    if path.is_symlink(): record.update(is_symlink=True,resolved_target=str(path.resolve()))
    return record

def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def atomic(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temp.replace(path)

def register_index():
    # Same lock already used by the other active FNIT task index writer.
    lock=HUB/'admin/fmri-index-update.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        index_path=HUB/'INDEX.json'; md_path=HUB/'INDEX.md'
        prior_index=index_path.read_bytes(); prior_md=md_path.read_bytes()
        index=json.loads(prior_index); key=ROOT.name
        require(key not in index.get('active_tasks',{}),'index namespace already occupied')
        require(not any(row.get('path')==str(ROOT) for row in index['entries']),'indexed run already occupied')
        entry={'name':key,'group':'runs','kind':'directory','path':str(ROOT),'old_path':None,
               'action':'new_independent_monitor_recovery','pin_reasons':['immutable_original_formal_runs']}
        index['entries'].append(entry)
        index.setdefault('active_tasks',{})[key]={'runs':str(ROOT.relative_to(HUB)),
            'frozen_scientific_revision':'1fe86ab8347b29d9c47be8109627736576222912',
            'original_configuration':str(CONFIG),'original_configuration_sha256':CONFIG_SHA,
            'scope':'Only memory measurement eligibility; original raw12 science, time and monitor issues preserved'}
        index['updated_utc']=utc()
        section=f'\n## 独立 connectome 显存补测（2026-10-03）\n\n- [`{key}`](runs/{key})：原正式 12 次全部完成后，复用冻结候选源码和输入进行直接 NVML 补测；原科学结果、时间及监控失败记录保留。\n'
        # Never regenerate the central index or replace another task's section.
        require(index_path.read_bytes()==prior_index and md_path.read_bytes()==prior_md,'index changed while locked')
        atomic(index_path,index)
        temp=md_path.with_name('.INDEX.md.memory-recovery.tmp'); temp.write_bytes(prior_md+section.encode()); temp.replace(md_path)
        atomic(ROOT/'index_registration.json',{'time_utc':utc(),'lock':str(lock),
            'before':{'INDEX.json':hashlib.sha256(prior_index).hexdigest(),'INDEX.md':hashlib.sha256(prior_md).hexdigest()},
            'after':{'INDEX.json':sha(index_path),'INDEX.md':sha(md_path)},'appended_entry':entry})

def load_runtime():
    pre=ROOT/'runtime_identity_preflight.json'
    proof=json.loads(pre.read_bytes()); bound(pre,PREFLIGHT_SHA)
    require(proof['scientific_runtime_equal'] is True,'preflight does not prove identical runtime')
    for key,python in (('original_runtime',ORIGINAL_PYTHON),('monitor_runtime',MONITOR_PYTHON)):
        record=proof[key]
        require(Path(python).resolve()==Path(record['binary']),f'Python binary path changed: {key}')
        bound(record['binary'],record['binary_sha256'])
        for item in record['modules'].values(): bound(item['path'],item['sha256'])
        for path,item in record['shared_libraries'].items(): bound(path,item['sha256'])
    nvml=proof['monitor_runtime']['nvml']; bound(nvml['path'],nvml['sha256'])
    return {'runtime_preflight':bound(pre),'runtime_paths_and_bytes_checked':True}

def frozen():
    bound(CONFIG,CONFIG_SHA); config=json.loads(CONFIG.read_bytes())
    path=Path(config['worker_script']).parent
    sys.path.insert(0,str(path))
    driver=importlib.import_module('benchmark_connectome_accuracy_cohort')
    require(Path(driver.__file__).resolve()==path/'benchmark_connectome_accuracy_cohort.py','wrong frozen driver import')
    driver.verify_tools(config)
    manifest=driver.bound(config['raw_manifest']); bindings=driver.bound(config['input_bindings'])
    cases=driver.validate_plan(config,manifest,bindings)
    observed={name:driver.cohort.source_manifest(source) for name,source in config['sources'].items()}
    require(observed==config['declared_source_manifests'],'actual frozen source inventory differs')
    require(observed['candidate']['source_fingerprint']==SCIENCE,'candidate scientific source changed')
    return config,driver,cases,bindings

def formal_completed(config):
    path=Path(config['run_root'])/'status.json'; state=json.loads(path.read_bytes())
    expected={f"{row['version']}/{row['case_id']}" for row in config['execution_order']}
    require(len(expected)==12,'original phase declaration is not twelve runs')
    if state['status']!='execution_completed':
        require(state['status']=='running','original phase failed; refuse automated recovery')
        return False,{'status':state['status'],'actual_rows':{k:v['status'] for k,v in state['cases'].items()},'scientific_GPU_dispatched':False}
    require(state.get('end_utc') and set(state['cases'])==expected,'original terminal row inventory differs')
    require(all(row['status']=='completed' for row in state['cases'].values()),'original twelve rows not all completed')
    require(state['configuration']['sha256']==CONFIG_SHA and state['configuration']['path']==str(CONFIG),'original phase config binding differs')
    return True,{'original_terminal_status':bound(path),'rows':{k:v['status'] for k,v in state['cases'].items()},'end_utc':state['end_utc']}

def strict_budget(budget):
    values=budget.get('measurements',{})
    require(set(values)=={'process_tree','allocated_bytes','reserved_bytes'},'all original three memory peaks required')
    require(all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and 0<x<LIMIT for x in values.values()),
            'original measured peak missing/invalid or exceeds strict 20e9; cannot recover by monitoring')

def inventory(job):
    # Raw BIDS staging uses the existing pipeline's symlinks. Preserve and
    # bind their real targets; they are never a reused corrected-DWI output.
    return {str(path.relative_to(job)):bound(path,allow_symlink=True) for directory in ('connectome','returned_result')
            for path in sorted((job/directory).rglob('*')) if path.is_file()}

def select(config,driver,cases,bindings):
    rows=[]; inspected={}
    for case_id in ('sub-CON09','sub-CON10','sub-CON11'):
        job=Path(config['run_root'])/'candidate'/case_id
        gpu=json.loads((job/'gpu_report.json').read_bytes()); wall=json.loads((job/'raw_bids_wall.json').read_bytes())
        require(gpu['status']==wall['status']=='completed' and gpu['exit_code']==wall['exit_code']==0,'original actual scientific execution failed')
        require(gpu['case_id']==case_id and gpu['version']=='candidate','original report arm/case mismatch')
        require(gpu['source_before']['source_fingerprint']==gpu['source_after']['source_fingerprint']==SCIENCE,'original scientific source mismatch')
        driver.cohort.check_wall_report(wall,config,cases[case_id]); driver.cohort.check_selected_inputs(wall,cases[case_id])
        subject=bindings['cases'][case_id]['anatomy']['directory']
        require(wall['cli_arguments']==driver.cohort.cli_command(config,cases[case_id],job,anatomy_subject=subject),'original argv differs')
        require(gpu['anatomy']==gpu['anatomy_after']==bindings['cases'][case_id]['anatomy']['files'],'original FS binding differs')
        require(Path(gpu['wall_report'])==job/'raw_bids_wall.json','original wall path differs')
        budget=driver.cohort.memory_budget(wall)
        require(budget==gpu['memory_budget'],'original worker/wall memory ledger differs')
        strict_budget(budget)
        inspected[case_id]={'original_budget':budget,'selected':budget['status']=='not_fully_measured','original_gpu':bound(job/'gpu_report.json'),'original_wall':bound(job/'raw_bids_wall.json')}
        if case_id in ('sub-CON09','sub-CON10'): require(budget['status']=='not_fully_measured' and budget['monitor_issues'],'required original monitor issue missing')
        if budget['status']=='not_fully_measured':
            require(budget['monitor_issues'],'monitor-incomplete lacks actual issues')
            rows.append({'version':'candidate','case_id':case_id,'reason':'original_actual_monitor_incomplete','original':inspected[case_id]})
        else: require(budget['status']=='observed_below_budget','original CON11 has an unexplained gate failure')
    return rows,inspected

def verify_input_inventory(wall):
    return {name:bound(item['path'],item['sha256']) for name,item in wall['inputs'].items() if item.get('exists')}

def run(args):
    if args.prepare:
        require(not ROOT.exists() and not ROOT.is_symlink(),'refuse preexisting recovery namespace')
        ROOT.mkdir(parents=True)
        register_index()
        atomic(ROOT/'status.json',{'status':'prepared_CPU_only','time_utc':utc(),'scientific_GPU_dispatched':False})
        return 0
    require(ROOT.is_dir() and (ROOT/'index_registration.json').is_file(),'run must be freshly prepared and indexed')
    config,driver,cases,bindings=frozen()
    status={'status':'waiting_original_formal','start_utc':utc(),'identity':{'hostname':socket.gethostname(),'python':sys.executable,'pid':os.getpid()},
            'original_configuration':bound(CONFIG),'controller':bound(__file__),'scope':'independent memory eligibility; original science/timing not replaced','cases':{}}
    while True:
        done,observation=formal_completed(config)
        status['original_observation']=observation; status['last_check_utc']=utc()
        atomic(ROOT/'status.json',status)
        if done: break
        if not args.execute_after_formal: return 0
        time.sleep(45)  # CPU waiting only; no GPU worker, lock acquisition or CUDA context.
    rows,inspected=select(config,driver,cases,bindings)
    plan={'original_configuration':bound(CONFIG),'original_terminal':observation,'explicit_selection':rows,'inspected':inspected,
          'scientific_source_fingerprint':SCIENCE,'science_protocol_unchanged':True,'runtime':load_runtime(),'time_utc':utc()}
    require(not (ROOT/'frozen_recovery_plan.json').exists(),'refuse existing recovery plan/dispatch')
    atomic(ROOT/'frozen_recovery_plan.json',plan)
    if not args.execute_after_formal: return 0
    clone=copy.deepcopy(config); clone.update(run_root=str(ROOT),gpu_python=MONITOR_PYTHON)
    clone['frozen_sources']=copy.deepcopy(config['declared_source_manifests'])
    atomic(ROOT/'recovery_configuration.json',clone)
    status.update(status='running_selected_monitor_recovery',plan=bound(ROOT/'frozen_recovery_plan.json'),configuration=bound(ROOT/'recovery_configuration.json'))
    atomic(ROOT/'status.json',status)
    def anatomy(actual_config,case,version,job):
        record=bindings['cases'][case['case_id']]; prior=driver.bound(record['prior_gpu_report'])
        require(prior['status']=='completed' and prior['case_id']==case['case_id'] and prior['anatomy']==record['anatomy']['files'] and prior['raw_input_provenance']==case['input_files'],'same completed FS origin differs')
        return {'anatomy':record['anatomy']['files']}
    def subject(actual_config,case,job): return bindings['cases'][case['case_id']]['anatomy']['directory']
    for selected in rows:
        case_id=selected['case_id']; original=Path(config['run_root'])/'candidate'/case_id
        wall=json.loads((original/'raw_bids_wall.json').read_bytes())
        before={'config':bound(CONFIG),'GPU':bound(original/'gpu_report.json'),'wall':bound(original/'raw_bids_wall.json'),
                'inputs':verify_input_inventory(wall),'scientific_output_inventory':inventory(original),'runtime':load_runtime(),
                'raw':driver.cohort.verify_inputs(cases[case_id]),'FS':driver.cohort.check_anatomy(subject(clone,cases[case_id],None),config['atlases'])}
        require(before['FS']==bindings['cases'][case_id]['anatomy']['files'],'same supplied FS changed before recovery')
        atomic(ROOT/f'{case_id}_original_before.json',before)
        status['cases'][case_id]={'status':'running','start_utc':utc()}; atomic(ROOT/'status.json',status)
        extra=('--result-export-dir',str(ROOT/'candidate'/case_id/'returned_result'))
        report=driver.cohort.worker({'action':'gpu','config':clone,'case':cases[case_id],'version':'candidate'},anatomy_loader=anatomy,anatomy_subject=subject,extra_wall_arguments=extra)
        after={'config':bound(CONFIG),'GPU':bound(original/'gpu_report.json'),'wall':bound(original/'raw_bids_wall.json'),
               'inputs':verify_input_inventory(wall),'scientific_output_inventory':inventory(original),'runtime':load_runtime(),
               'raw':driver.cohort.verify_inputs(cases[case_id]),'FS':driver.cohort.check_anatomy(subject(clone,cases[case_id],None),config['atlases'])}
        require(after==before,'original/input/FS/runtime bytes changed during recovery')
        frozen(); atomic(ROOT/f'{case_id}_original_after.json',after)
        require(report['status']=='completed','recovery raw science failed')
        require(report['source_before']['source_fingerprint']==report['source_after']['source_fingerprint']==SCIENCE,'recovery scientific source changed')
        memory=report['memory_budget']; strict_budget(memory)
        wall_new=json.loads((ROOT/'candidate'/case_id/'raw_bids_wall.json').read_bytes())
        require(wall_new['gpu_process_memory']['backend']=='pynvml','recovery did not use direct NVML')
        require(memory['status']=='observed_below_budget' and not memory['monitor_issues'],'strict recovery memory gate failed')
        require(verify_input_inventory(wall_new)==before['inputs'],'recovery science used different input/resource bytes')
        command=list(report['command']); old_command=json.loads((original/'gpu_report.json').read_bytes())['command']
        require(command[0]==MONITOR_PYTHON and old_command[0]==ORIGINAL_PYTHON,'unexpected interpreter command')
        command[0]=old_command[0]
        for flag in ('--report','--result-export-dir','--output-dir'):
            require(command.count(flag)==old_command.count(flag)==1,'ambiguous changed output flag')
            command[command.index(flag)+1]=old_command[old_command.index(flag)+1]
        require(command==old_command,'scientific worker argv changed beyond interpreter/output paths')
        status['cases'][case_id]={'status':'completed','gpu_report':bound(ROOT/'candidate'/case_id/'gpu_report.json'),
            'wall_report':bound(ROOT/'candidate'/case_id/'raw_bids_wall.json'),'memory':memory,
            'independent_CLI_seconds':report['raw_dwi_cli_total_runtime_seconds'],'scientific_array_comparison':'not_assessed_CPU_followup',
            'original_bytes_preserved':True,'changed_command_fields':['interpreter','--report','--result-export-dir','--output-dir']}
        atomic(ROOT/'status.json',status)
    status.update(status='GPU_execution_completed_CPU_comparison_pending',end_utc=utc())
    atomic(ROOT/'status.json',status); return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--execute-after-formal',action='store_true')
    args=parser.parse_args()
    try: raise SystemExit(run(args))
    except Exception as error:
        if ROOT.is_dir(): atomic(ROOT/'controller_failure.json',{'status':'failed','time_utc':utc(),'error':f'{type(error).__name__}: {error}'})
        raise
