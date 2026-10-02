"""Explicit selected-case monitor recovery; immutable original guards, fresh raw DWI.

Private benchmark orchestration only. Default is CPU preflight. GPU execution
requires --execute and an explicit, nonempty (arm, case) selection from root.
"""
from __future__ import annotations
import argparse
import inspect
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True  # Original frozen helper/environment directories stay read-only.

MODE = 'selected_same_round_anatomy_fresh_raw_dwi_monitor_recovery'
GPU = 'GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba'
LOCK = '/tmp/fnit-recon-five-20261002-gongwk.gpu.lock'
ATLASES = ['fs-aparc', 'aparc+tian-s1', 'aparc.a2009s+tian-s1', 'glasser+tian-s1',
           'glasser+tian-s4', 'schaefer200+tian-s1', 'schaefer500+tian-s4', 'schaefer1000+tian-s4']
ARM_SOURCE_FINGERPRINTS = {'candidate':'9fd44cbc49c9cdfc16c9a8cff2971fec3239b450dce41222e6ef059861eb0dc7',
                           'baseline':'deefeb6908c3c14a9aa7b4cf154c8df941a56abffd4a4e893045a1c4d1ddfd4a'}
HELPERS = ('benchmark_connectome_raw_cohort', 'benchmark_connectome_raw_recovery',
           'benchmark_connectome_raw_rerun')
OLD_REPORTS = ('gpu_report.json', 'raw_bids_wall.json', 'staged_eligibility.json',
               'anatomy_origin.json', 'staged_anatomy_binding.json')
PROBE = r'''
import hashlib, importlib, importlib.metadata as md, json, pathlib, sys, torch
names = ('torch','torch._C','numpy','numpy.core._multiarray_umath','nibabel')
modules = {}
for name in names:
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    modules[name] = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'version':getattr(module,'__version__',None)}
path = pathlib.Path(sys.executable).resolve()
result = {'gpu_python':sys.executable,'python_binary':str(path),
          'python_binary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'python_version':sys.version,'modules':modules,'CUDA_initialized':torch.cuda.is_initialized()}
try:
    import pynvml
    path = pathlib.Path(pynvml.__file__).resolve()
    result['NVML'] = {'package_version':md.version('nvidia-ml-py'),'module_path':str(path),
                      'module_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
except ImportError: result['NVML'] = None
print(json.dumps(result))
'''



def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def read_bound(binding):
    if set(binding) != {'path', 'sha256'} or not Path(binding['path']).is_absolute():
        raise ValueError('binding requires an absolute path and exact SHA-256')
    path = Path(binding['path'])
    if path.is_symlink() or sha(path) != binding['sha256']:
        raise ValueError(f'bound bytes changed: {path}')
    return json.loads(path.read_bytes())


def binding(path):
    return {'path':str(Path(path).absolute()), 'sha256':sha(path)}


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def overlaps(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    return a.is_relative_to(b) or b.is_relative_to(a)


def validate_selection(declaration):
    rows = declaration.get('selections')
    if not isinstance(rows, list) or not rows:
        raise ValueError('root must supply a nonempty explicit selection; no inferred cases')
    seen = set()
    for row in rows:
        if set(row) != {'arm','case_id','reason','origin_config','origin_driver','source_fingerprint','helpers'}:
            raise ValueError('selected row must declare exact origin config/driver/source/helpers')
        key = (row['arm'], row['case_id'])
        if row['arm'] not in ('baseline','candidate') or row['case_id'] not in ['sub-CON01', *[f'sub-CON{i:02d}' for i in range(3,12)]] or key in seen:
            raise ValueError('duplicate, invalid arm/case or excluded CON02')
        if row['reason'] not in ('monitor_incomplete','original_not_dispatched','original_queue_stopped_before_compute'):
            raise ValueError('explicit recovery reason required')
        seen.add(key)
    return rows


def load_helpers(row):
    paths = row['helpers']
    needed = {*HELPERS, *(['benchmark_connectome_staged_gpu'] if row['arm']=='candidate' else [])}
    if set(paths) != needed: raise ValueError('explicit exact original helper ledger required')
    directories = set()
    for name, bound in paths.items():
        read = Path(bound['path'])
        if read.name != name+'.py' or read.is_symlink() or sha(read) != bound['sha256']:
            raise ValueError('original helper SHA or filename changed')
        directories.add(read.parent.resolve())
    if len(directories) != 1: raise ValueError('original helper imports must share their frozen directory')
    directory = directories.pop()
    sys.path.insert(0, str(directory))
    modules = {}
    for name in needed:
        loaded = importlib.import_module(name)
        if Path(loaded.__file__).resolve() != Path(paths[name]['path']).resolve() or sha(loaded.__file__) != paths[name]['sha256']:
            raise ValueError('imported helper differs from declared original helper')
        modules[name] = loaded
    return (modules['benchmark_connectome_raw_cohort'], modules['benchmark_connectome_raw_rerun'],
            modules.get('benchmark_connectome_staged_gpu'))


def check_protocol(config, row):
    arm = row['arm']
    if set(config['sources']) != {arm} or set(config['frozen_sources']) != {arm}:
        raise ValueError('original source arm changed')
    expected = {'pilot':False, 'n_seeds':100000, 'seed':0, 'eddy_gp_seed':12345,
                'cuda_alloc_conf':'expandable_segments:True', 'gpu_uuid':GPU,
                'cuda_visible_devices':'1', 'gpu_lock':LOCK, 'device':'cuda:0',
                'cpu_threads':8, 'gpu_cpu_threads':8, 'atlases':ATLASES}
    for key, value in expected.items():
        if isinstance(value,int) and not isinstance(value,bool) and isinstance(config.get(key),bool):
            raise ValueError('numeric protocol parameter cannot be boolean')
        if config.get(key) != value: raise ValueError(f'original scientific protocol differs: {key}')
    if row['source_fingerprint']!=ARM_SOURCE_FINGERPRINTS[arm]:
        raise ValueError('recovery must retain actual 641 candidate / da427 common baseline source')
    if config['frozen_sources'][arm]['source_fingerprint'] != row['source_fingerprint']:
        raise ValueError('selected source fingerprint differs from original frozen arm')
    forbidden = {'preprocessed_root','corrected_dwi','rotated_bvecs','resume','skip_topup','skip_eddy'}
    if forbidden.intersection(config): raise ValueError('preprocessed input or skipped stage forbidden')


def origin(row, cohort, rerun, staged):
    config = read_bound(row['origin_config'])
    state = read_bound(row['origin_driver'])
    if state.get('config') != config or not state.get('end_utc') or state.get('status') in ('running','preflighting','claiming'):
        raise ValueError('original driver must actually be terminal and identify original config')
    check_protocol(config, row)
    rerun.verify_resources(config)
    if sha(cohort.__file__) != config['worker_script_sha256']:
        raise ValueError('original mature cohort worker differs')
    if row['arm']=='candidate':
        staged.verify_source(config)
        manifest = read_bound(config['input_manifest'])
    else:
        rerun.verify_common_source(config)
        path = Path(config['run_root'])/'input_manifest.json'
        manifest = json.loads(path.read_bytes())
        old = rerun.origin_state(config)
        if sha(path) != config['rerun_origin']['input_manifest_sha256']:
            raise ValueError('rerun raw manifest differs from original cohort')
    cases = [c for c in cohort.validate_manifest(manifest) if c['case_id']==row['case_id']]
    if len(cases)!=1: raise ValueError('case absent or ambiguous in original raw manifest')
    if cohort.source_manifest(config['sources'][row['arm']])['source_fingerprint'] != row['source_fingerprint']:
        raise ValueError('actual original frozen numerical source changed')
    return config, state, cases[0]


def anatomy(row, config, case, cohort, rerun, staged):
    cohort.verify_inputs(case)
    if row['arm']=='candidate':
        checked = staged.verify_preparation(config, case)
        record, observation = staged.preparation_driver_case(config, case)
        if record['preparation_report'] != checked['preparation_report']:
            raise ValueError('actual preparation differs from same-round driver case')
        recon = copy.deepcopy(checked['preparation_report']['reconstruction_result'])
        subject = checked['anatomy_subject_dir']
        proof = {'original_preparation':checked, 'preparation_driver_case':record,
                 'driver_observation':observation}
    else:
        recon, report_path, subject = rerun.validate_original(config, case)
        recon = copy.deepcopy(recon)
        actual = cohort.check_anatomy(subject, config['atlases'])
        if actual != recon['anatomy']: raise ValueError('original FS anatomy bytes changed')
        geometry = cohort.validate_anatomy_child(subject, actual, config['anatomy_validation_python'])
        if geometry.get('status')!='actual_images_surfaces_annotations_read':
            raise ValueError('same-round actual FS geometry read failed')
        recon.update(status='completed', anatomy_geometry=geometry)
        proof = {'original_recon_report':binding(report_path), 'anatomy_geometry':geometry,
                 'original_reconstruction_failure':recon.get('error')}
    if cohort.check_anatomy(subject, config['atlases']) != recon['anatomy']:
        raise ValueError('same-round FS anatomy differs from its original ledger')
    cohort.verify_inputs(case)
    recon.pop('rerun',None); recon.pop('recovery',None)
    recon['staged_anatomy'] = {'mode':MODE,'same_round_official_FS':True,
        'prepared_anatomy_subject_dir':str(subject),'raw_dwi_preprocessing_resumed':False,
        'source_fingerprint':row['source_fingerprint']}
    return recon, str(subject), proof


def probe(python):
    env = os.environ.copy(); env.pop('PYTHONPATH',None); env['CUDA_VISIBLE_DEVICES']=''; env['PYTHONDONTWRITEBYTECODE']='1'
    result = subprocess.run([python,'-c',PROBE],env=env,capture_output=True,text=True,check=True)
    return json.loads(result.stdout)


def runtime_check(declaration, original_python):
    if (declaration.get('scope')!='optional_NVML_monitor_environment'
            or declaration.get('science_modules_unchanged') is not True
            or declaration.get('CUDA_initialized') is not False
            or declaration.get('production_source_and_wall_bytes_changed') is not False
            or declaration.get('scientific_GPU_work_dispatched') is not False):
        raise ValueError('root actual isolated monitor preflight required')
    if declaration['original_runtime']['gpu_python']!=original_python or declaration['gpu_python']==original_python:
        raise ValueError('root runtime must bind original Python and isolated new Python')
    before, after = probe(original_python), probe(declaration['gpu_python'])
    for observed, name in ((before,'original_runtime'),(after,'new_runtime')):
        expected=declaration[name]
        for field in ('gpu_python','python_binary','python_binary_sha256','python_version','modules','CUDA_initialized'):
            if observed[field]!=expected[field]: raise ValueError(f'actual runtime changed since root preflight: {name}/{field}')
        if observed['CUDA_initialized'] is not False: raise ValueError('CPU runtime probe initialized CUDA')
    for field in ('python_binary','python_binary_sha256','python_version','modules'):
        if before[field]!=after[field]: raise ValueError(f'isolated runtime changed scientific code: {field}')
    nvml=declaration['NVML']
    if nvml.get('package')!='nvidia-ml-py' or nvml.get('package_version')!='13.580.82':
        raise ValueError('official optional NVML distribution version not verified')
    if after['NVML']!={k:nvml[k] for k in ('package_version','module_path','module_sha256')}:
        raise ValueError('actual NVML module differs from root preflight')
    monitor=nvml.get('real_monitor_probe',{})
    if (monitor.get('backend')!='pynvml' or monitor.get('gpu_uuid')!=GPU
            or monitor.get('failed_samples')!=0 or monitor.get('unresolved_device_samples')!=0
            or monitor.get('errors') or monitor.get('samples',0)<10
            or monitor.get('max_observed_interval_seconds',float('inf'))>5
            or nvml.get('CUDA_initialized_before') is not False or nvml.get('CUDA_initialized_after') is not False):
        raise ValueError('root monitor preflight is incomplete; not a case memory budget result')
    return {'original':before,'isolated':after,'scientific_runtime_equal':True,'CUDA_initialized':False}


def eligible(budget):
    # Exact staged gate; errors/gaps are computed by mature cohort.memory_budget.
    measurements = budget.get('measurements',{})
    groups = (('process_tree',),('allocated_bytes','peak_allocated_bytes','max_memory_allocated_bytes'),
              ('reserved_bytes','peak_reserved_bytes','max_memory_reserved_bytes'))
    def valid(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and 0 <= value < 20_000_000_000
    return (budget.get('status')=='observed_below_budget' and not budget.get('monitor_issues')
            and all(valid(v) for v in measurements.values() if v is not None)
            and all(any(valid(measurements.get(key)) for key in names) for names in groups))


def fresh_config(original, run_root, python, stop_path):
    config = copy.deepcopy(original)
    config.update(run_root=str(run_root),gpu_python=python,stop_dispatch_path=str(stop_path))
    config.pop('resources_manifest',None)
    return config


def validate_reason(row, old_job, cohort):
    gpu, wall, output = (Path(old_job)/name for name in ('gpu_report.json','raw_bids_wall.json','connectome'))
    if row['reason']=='original_not_dispatched':
        for path in (gpu,wall,output,Path(old_job)/'raw_bids_wall.log'):
            if path.exists() or path.is_symlink():
                raise ValueError('original_not_dispatched requires strictly absent old GPU/wall/connectome')
        return {'reason':row['reason'],'original_GPU_and_wall_absent':True,'old_failure_report_fabricated':False}
    if row['reason']=='original_queue_stopped_before_compute':
        if not gpu.is_file() or gpu.is_symlink(): raise ValueError('queued-stop requires the actual old failed GPU report')
        for path in (wall,output,Path(old_job)/'raw_bids_wall.log'):
            if path.exists() or path.is_symlink(): raise ValueError('queued-stop cannot have started raw-DWI outputs')
        stopped=json.loads(gpu.read_bytes())
        expected={'status':'failed','action':'gpu','case_id':row['case_id'],'version':row['arm']}
        if any(stopped.get(k)!=v for k,v in expected.items()): raise ValueError('queued-stop report case/action/status differs')
        if stopped.get('error')!={'type':'RuntimeError','message':'STOP_DISPATCH prevents starting the queued raw-DWI computation'}:
            raise ValueError('only the exact mature queued STOP failure is recoverable by this reason')
        if stopped.get('source_before',{}).get('source_fingerprint')!=row['source_fingerprint']:
            raise ValueError('actual source observed after original lock acquisition differs')
        if any(key in stopped for key in ('command','exit_code','gpu_command_wall_seconds','raw_dwi_cli_total_runtime_seconds','wall_report','outputs','memory_budget','source_after','gpu_memory')):
            raise ValueError('queued-stop report shows a scientific subprocess or wall result')
        return {'reason':row['reason'],'original_science_started':False,'original_failed_report_preserved':binding(gpu),
                'original_wall_and_connectome_absent':True,'original_error':stopped['error']}
    if not gpu.is_file() or not wall.is_file() or not output.is_dir() or gpu.is_symlink() or wall.is_symlink() or output.is_symlink():
        raise ValueError('monitor_incomplete requires actual original GPU/wall/output')
    original_gpu=json.loads(gpu.read_bytes()); original_wall=json.loads(wall.read_bytes())
    if any(x.get('status')!='completed' or x.get('exit_code')!=0 for x in (original_gpu,original_wall)):
        raise ValueError('monitor recovery cannot relabel an actual scientific failure')
    budget=cohort.memory_budget(original_wall)
    if budget.get('status')!='not_fully_measured':
        raise ValueError('monitor_incomplete requires an actually incomplete original memory ledger')
    return {'reason':row['reason'],'original_execution_completed':True,'original_memory_budget':budget,
            'original_GPU_report':binding(gpu),'original_wall_report':binding(wall)}


def build_resources_compat(config, arm, rerun):
    """Only the two actual frozen signatures are supported."""
    parameters=inspect.signature(rerun.build_resources).parameters
    if arm=='baseline' and tuple(parameters)==('config',):
        return rerun.build_resources(config)
    if (arm=='candidate' and tuple(parameters)==('config','source_version')
            and parameters['source_version'].kind==inspect.Parameter.KEYWORD_ONLY):
        return rerun.build_resources(config,source_version=arm)
    raise ValueError('unsupported frozen resource helper API; retain guards and fail before science')


def matured_worker_compat(cohort, payload, loader, subject_loader, extra_arguments):
    """Use new callbacks or scoped old anatomy hooks; worker bytes/math unchanged."""
    parameters=inspect.signature(cohort.worker).parameters
    callbacks={'anatomy_loader','anatomy_subject','extra_cli_arguments'}
    if (tuple(parameters)==('payload','anatomy_loader','anatomy_subject','extra_cli_arguments')
            and all(parameters[name].kind==inspect.Parameter.KEYWORD_ONLY for name in callbacks)):
        return cohort.worker(payload,anatomy_loader=loader,anatomy_subject=subject_loader,extra_cli_arguments=extra_arguments)
    if tuple(parameters)!=('payload',) or extra_arguments:
        raise ValueError('unsupported frozen worker API; cannot drop scientific scheduling arguments')
    old_loader=cohort.load_recon_for_gpu; old_subject=cohort.gpu_anatomy_subject
    # Each _worker is a clean single-case process. Only these orchestration
    # globals are temporarily rebound; original source files stay untouched.
    try:
        cohort.load_recon_for_gpu=loader; cohort.gpu_anatomy_subject=subject_loader
        return cohort.worker(payload)
    finally:
        cohort.load_recon_for_gpu=old_loader; cohort.gpu_anatomy_subject=old_subject


def worker(payload):
    row, new = payload['selection'], payload['config']
    if sha(__file__) != payload['driver_sha256']: raise ValueError('recovery driver bytes changed')
    cohort, rerun, staged = load_helpers(row)
    original, old_state, case = origin(row,cohort,rerun,staged)
    declared_runtime = read_bound(payload['runtime_declaration'])
    if new != fresh_config(original,new['run_root'],declared_runtime['gpu_python'],new['stop_dispatch_path']):
        raise ValueError('new config changed more than namespace/runtime/dispatch metadata')
    for protected in (original['run_root'], *original['sources'].values(), Path(row['origin_driver']['path']).parent):
        if overlaps(new['run_root'],protected): raise ValueError('recovery namespace overlaps original data/source/reports')
    runtime = runtime_check(declared_runtime,original['gpu_python'])
    recon, subject, proof = anatomy(row,original,case,cohort,rerun,staged)
    for protected in (case['bids_root'],subject):
        if overlaps(new['run_root'],protected): raise ValueError('new raw-DWI output overlaps raw inputs or same-round FS')
    prior_resources=read_bound(original['resources_manifest'])
    resources = build_resources_compat(new,row['arm'],rerun)
    prior={item['role']+':'+item['path']:item for item in prior_resources['files'] if item['role']!='gpu_python'}
    current={item['role']+':'+item['path']:item for item in resources['files'] if item['role']!='gpu_python'}
    if prior!=current: raise ValueError('non-monitor scientific resource identity changed')
    if sha(original['wall_script'])!=declared_runtime['NVML']['original_wall_helper']['sha256']:
        raise ValueError('wall helper bytes differ from root NVML validation')
    old_job = Path(original['run_root'])/row['arm']/case['case_id']
    reason=validate_reason(row,old_job,cohort)
    snapshots = {}
    for name in OLD_REPORTS:
        path = old_job/name
        if path.is_symlink(): raise ValueError('old case report cannot be a symlink')
        if path.exists():
            value = json.loads(path.read_bytes())
            if name=='gpu_report.json' and value.get('status')=='running':
                raise ValueError('original selected case computation still running')
            snapshots[name] = {'binding':binding(path),'value':value}
        else: snapshots[name] = {'absent_at_bind':True}
    helper_API={'build_resources':str(inspect.signature(rerun.build_resources)),
                'worker':str(inspect.signature(cohort.worker)),
                'validate_anatomy_child':str(inspect.signature(cohort.validate_anatomy_child)) if hasattr(cohort,'validate_anatomy_child') else 'fixture_only',
                'candidate_verify_preparation':str(inspect.signature(staged.verify_preparation)) if staged and hasattr(staged,'verify_preparation') else None}
    bound = {'helper_API':helper_API,'mode':MODE,'arm':row['arm'],'case_id':case['case_id'],'case':case,'reason':reason,
        'origin_case_state':old_state.get('cases',{}).get(row['arm']+'/'+case['case_id']),
        'original_config':row['origin_config'],'original_driver':row['origin_driver'],
        'old_case_reports':snapshots,'same_round_anatomy':proof,'anatomy':recon['anatomy'],
        'anatomy_subject_dir':subject,'runtime':runtime,'resources':resources,
        'FS_recomputed':False,'raw_DWI_execution_policy':'complete_from_raw','old_DWI_outputs_used':False}
    if payload['action']=='preflight': return bound
    if payload['action']!='gpu': raise ValueError('recovery never runs recon or preprocessing resume')
    job = Path(new['run_root'])/row['arm']/case['case_id']
    if Path(new['stop_dispatch_path']).exists(): return {'status':'dispatch_paused','gpu_started':False}
    cohort.require_fresh(job)
    cohort.atomic_json(job/'recovery_binding.json',bound)
    resource_path=job/'recovery_resources.json'; cohort.atomic_json(resource_path,resources)
    active={**new,'resources_manifest':binding(resource_path)}
    cohort.atomic_json(job/'recovery_config.json',active)
    def revalidate():
        origin(row,cohort,rerun,staged); read_bound(payload['runtime_declaration'])
        if validate_reason(row,old_job,cohort)!=reason: raise ValueError('original recovery reason changed')
        if runtime_check(declared_runtime,original['gpu_python'])!=runtime: raise ValueError('runtime changed since binding')
        fresh_recon,fresh_subject,fresh_proof=anatomy(row,original,case,cohort,rerun,staged)
        if fresh_subject!=subject or fresh_recon!=recon or fresh_proof.get('original_preparation',fresh_proof)!=proof.get('original_preparation',proof):
            raise ValueError('same-round anatomy provenance changed since binding')
        for name,value in snapshots.items():
            if 'binding' in value: read_bound(value['binding'])
            elif (old_job/name).exists() or (old_job/name).is_symlink():
                raise ValueError('originally absent old case report appeared during recovery')
        rerun.verify_resources(active)
    def loader(config,c,arm,j):
        revalidate(); return copy.deepcopy(recon)
    def subject_loader(config,c,j):
        if Path(new['stop_dispatch_path']).exists(): raise RuntimeError('STOP_DISPATCH prevents science after GPU lock wait')
        revalidate(); return subject
    result=matured_worker_compat(cohort,{'action':'gpu','config':active,'case':case,'version':row['arm']},
        loader,subject_loader,staged.extra_arguments(original.get('candidate_cli_arguments',[])) if staged else ())
    result['execution_scope']='selected monitor recovery: same-round fresh official FS reused; new full raw-DWI stage; not continuous cold pipeline'
    ok=result.get('status')=='completed' and result.get('exit_code')==0 and eligible(result.get('memory_budget',{}))
    validation_error=None
    try: revalidate()
    except Exception as error: ok=False; validation_error={'type':type(error).__name__,'message':str(error)}
    eligibility={'status':'execution_complete_memory_observed_below_budget' if ok else 'not_eligible',
        'execution_status':result.get('status'),'memory_budget':result.get('memory_budget'),
        'validation_error':validation_error,'scope':MODE,'full_ten_complete':False}
    cohort.atomic_json(job/'gpu_report.json',result); cohort.atomic_json(job/'recovery_eligibility.json',eligibility)
    return {'gpu_result':result,'eligibility':eligibility,'binding':bound,
            'configuration':binding(job/'recovery_config.json'),'new_config':active,
            'science_worker':binding(cohort.__file__),'recovery_worker':binding(__file__),
            'resources_manifest':binding(resource_path),'job_root':str(job),
            'GPU_report':binding(job/'gpu_report.json'),
            'wall_report':binding(job/'raw_bids_wall.json') if (job/'raw_bids_wall.json').is_file() else None}


def make_selection(pairs, origins):
    """Generate byte bindings only from root's explicit pair list and stopped origins."""
    if not isinstance(pairs,list) or not pairs: raise ValueError('explicit nonempty pairs required')
    seen=set(); rows=[]
    for pair in pairs:
        if set(pair)!={'arm','case_id','reason'}: raise ValueError('explicit pairs require arm, case_id and reason')
        arm=pair['arm']
        if arm not in ('baseline','candidate') or (arm,pair['case_id']) in seen or pair['reason'] not in ('monitor_incomplete','original_not_dispatched','original_queue_stopped_before_compute'):
            raise ValueError('invalid or duplicate explicit recovery pair/reason')
        seen.add((arm,pair['case_id'])); paths=origins[arm]
        config_binding=binding(paths['configuration']); driver_binding=binding(paths['driver_status'])
        config=read_bound(config_binding); state=read_bound(driver_binding)
        if state.get('config')!=config or not state.get('end_utc') or state.get('status') in ('running','preflighting','claiming'):
            raise ValueError('bind only actually terminal original driver configs')
        directory=Path(config['worker_script']).parent
        names=(*HELPERS,*(['benchmark_connectome_staged_gpu'] if arm=='candidate' else []))
        row={**pair,'origin_config':config_binding,'origin_driver':driver_binding,
             'source_fingerprint':config['frozen_sources'][arm]['source_fingerprint'],
             'helpers':{name:binding(directory/(name+'.py')) for name in names}}
        check_protocol(config,row); rows.append(row)
    declaration={'schema_version':1,'scope':MODE,'selections':rows}
    validate_selection(declaration)
    return declaration


def run(options):
    rows=validate_selection(json.loads(options.selection.read_bytes()))
    runtime_binding=binding(options.runtime_declaration); runtime=read_bound(runtime_binding)
    report=Path(options.report_dir); root=Path(options.run_root)
    if not report.is_absolute() or not root.is_absolute() or overlaps(report,root): raise ValueError('fresh absolute disjoint namespaces required')
    if report.parent.is_symlink() or root.parent.is_symlink(): raise ValueError('output/report parents cannot be symbolic links')
    if report.exists() or root.exists() or report.is_symlink() or root.is_symlink(): raise FileExistsError('never reuse output/report namespace')
    for row in rows:
        original=read_bound(row['origin_config']); check_protocol(original,row)
        previous=read_bound(row['origin_driver'])
        if previous.get('config')!=original or not previous.get('end_utc') or previous.get('status') in ('running','preflighting','claiming'):
            raise ValueError('old driver must actually be terminal before any recovery plan')
        for protected in (original['run_root'], *original['sources'].values(), Path(row['origin_driver']['path']).parent):
            if overlaps(report,protected) or overlaps(root,protected): raise ValueError('new namespaces overlap original artifacts')
    report.mkdir(parents=True)
    state={'mode':MODE,'status':'preflighting','selected_pairs':[r['arm']+'/'+r['case_id'] for r in rows],
           'full_ten_complete':False,'config_changes':['run_root','gpu_python','stop_dispatch_path','resources_manifest'],
           'selection_binding':binding(options.selection),'runtime_declaration':runtime_binding,'cases':{}}
    atomic(report/'status.json',state)
    for index,row in enumerate(rows):
        for kind,key in (('config','origin_config'),('driver','origin_driver')):
            source=Path(row[key]['path']); read_bound(row[key])
            (report/f'{index}-original_{kind}.bytes.json').write_bytes(source.read_bytes())
    if options.plan_only:
        state.update(status='plan_only_NO_remote_worker_NO_GPU',science_GPU_started=False)
        atomic(report/'status.json',state); return 0
    if options.execute: root.mkdir(parents=True)
    for index,row in enumerate(rows):
        original=read_bound(row['origin_config']); new=fresh_config(original,root,runtime['gpu_python'],report/'STOP_DISPATCH')
        payload={'action':'gpu' if options.execute else 'preflight','selection':row,'config':new,
                 'runtime_declaration':runtime_binding,'driver_sha256':sha(__file__)}
        transport={**new,'worker_script':str(Path(__file__).resolve())}
        # Transport uses the original cohort SSH implementation, not production.
        # Load only the stdlib cohort transport under a unique module name.
        # Baseline and candidate original helpers can have different frozen bytes;
        # their validation imports run in separate remote worker processes.
        helper=row['helpers']['benchmark_connectome_raw_cohort']
        if sha(helper['path'])!=helper['sha256']: raise ValueError('transport helper changed')
        spec=importlib.util.spec_from_file_location('recovery_transport_'+str(index),helper['path'])
        cohort=importlib.util.module_from_spec(spec); spec.loader.exec_module(cohort)
        if (report/'STOP_DISPATCH').exists(): state['status']='dispatch_paused'; break
        started=time.perf_counter()
        atomic(report/f'{index}-new_config.json',new)
        try:
            response=cohort.remote(transport,'gpu',payload,report/f'{index}-stderr.log')
        except Exception as error:
            state.update(status='failed_recovery_preflight_or_execution',error={'type':type(error).__name__,'message':str(error)})
            atomic(report/'status.json',state); return 1
        pair=row['arm']+'/'+row['case_id']
        case_status=('completed' if response.get('eligibility',{}).get('status')=='execution_complete_memory_observed_below_budget'
                     else 'preflight_completed' if not options.execute else 'failed_or_ineligible')
        state['cases'][pair]={'case_id':row['case_id'],'version':row['arm'],'status':case_status,
                             'response':response,'recovery_head_wall_seconds':time.perf_counter()-started}
        atomic(report/f'{index}-response.json',response)
        atomic(report/f'{index}-new_config.json',response.get('new_config',new))
        if response.get('status')=='dispatch_paused':
            state['cases'][pair]['status']='dispatch_paused'; state['status']='dispatch_paused'; break
        if options.execute:
            origins=json.loads((report/'recovery_origins.json').read_bytes()) if (report/'recovery_origins.json').exists() else {'schema_version':1,'scope':'explicit_actual_GPU_monitor_recovery','bindings':[]}
            old_job=Path(original['run_root'])/row['arm']/row['case_id']
            origin_reports=response['binding']['old_case_reports']
            origins['bindings'].append({'arm':row['arm'],'case_id':row['case_id'],'reason':row['reason'],
                'original':{'GPU_report':origin_reports['gpu_report.json'].get('binding'),
                    'wall_report':origin_reports['raw_bids_wall.json'].get('binding'),
                    'driver_snapshot':binding(report/f'{index}-original_driver.bytes.json'),'configuration':row['origin_config']},
                'replacement':{'root':response['job_root'],'driver_status':binding(report/'status.json'),
                    'configuration':response['configuration'],'runtime_preflight':runtime_binding},
                'science_worker':response['science_worker'],'recovery_worker':response['recovery_worker'],
                'resources_manifest':response['resources_manifest']})
            atomic(report/'recovery_origins.json',origins)
        if options.execute and response.get('eligibility',{}).get('status')!='execution_complete_memory_observed_below_budget':
            state['status']='selected_recovery_failed_or_ineligible'; break
        atomic(report/'status.json',state)
    else: state['status']='completed_selected_subset' if options.execute else 'CPU_preflight_completed_GPU_not_started'
    state['selected_attempted']=len(state['cases'])
    state['selected_completed']=sum(v['response'].get('eligibility',{}).get('status')=='execution_complete_memory_observed_below_budget' for v in state['cases'].values()) if options.execute else 0
    state['full_ten_complete']=False
    atomic(report/'status.json',state)
    origin_path=report/'recovery_origins.json'
    if origin_path.exists():
        origins=json.loads(origin_path.read_bytes())
        for row in origins['bindings']: row['replacement']['driver_status']=binding(report/'status.json')
        atomic(origin_path,origins)
    return 0 if state['status'] in ('completed_selected_subset','CPU_preflight_completed_GPU_not_started') else 1


def main():
    if len(sys.argv)>1 and sys.argv[1]=='_worker':
        print(json.dumps(worker(json.load(sys.stdin)),allow_nan=False)); return 0
    if len(sys.argv)>1 and sys.argv[1]=='bind-selection':
        p=argparse.ArgumentParser(description='Bind only root explicit pairs to actually terminal original configs')
        for name in ('pairs','origins','output'): p.add_argument('--'+name,type=Path,required=True)
        o=p.parse_args(sys.argv[2:])
        if o.output.exists(): raise FileExistsError('selection declaration must be new')
        value=make_selection(json.loads(o.pairs.read_bytes()),json.loads(o.origins.read_bytes()))
        o.output.write_text(json.dumps(value,indent=2)+'\n'); return 0
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('selection','runtime-declaration','run-root','report-dir'): p.add_argument('--'+name,type=Path,required=True)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--plan-only',action='store_true',help='Only validate explicit declaration/config shapes and create snapshots; no remote worker or GPU')
    group.add_argument('--execute',action='store_true',help='Root deployment only: execute selected raw-DWI cases; default CPU preflight')
    return run(p.parse_args())

if __name__=='__main__': raise SystemExit(main())
