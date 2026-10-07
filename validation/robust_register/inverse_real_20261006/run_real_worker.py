"""One bounded CPU stage or saved-output score; no official command function.

This worker is prepared only. Separate rigid/affine processes are deliberate;
the affine API reads the actual rigid MGH written by the first process.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def atomic_report(path, report):
    payload = (json.dumps(report, indent=2, ensure_ascii=False,
                          allow_nan=False)+'\n').encode()
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name+'.writing.'+str(os.getpid()))
    try:
        fd = os.open(temporary, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def check_bindings(plan):
    actual = {}
    for key, item in plan['bindings'].items():
        path = Path(item['path'])
        value = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        if value != {k: item[k] for k in ('bytes', 'sha256')}:
            raise ValueError('frozen source/input/reference changed: '+key)
        actual[key] = value
    return actual


def configure(plan):
    cores = plan['physical_cores']
    if len(set(cores)) != 8 or len(cores) != 8:
        raise ValueError('eight distinct physical CPU cores required')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        if os.environ.get(key) != '8':
            raise ValueError('CPU eight-thread setting differs: '+key)
    for key in ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','OPENBLAS_CORETYPE',
                'FS_SetVoxToRasXform_Change_VoxSize'):
        if key in os.environ:
            raise ValueError('environment injection is not allowed: '+key)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA must remain hidden')
    if os.environ.get('PYTHONDONTWRITEBYTECODE') != '1':
        raise ValueError('frozen sources must not create caches')
    os.sched_setaffinity(0, cores)
    resource.setrlimit(resource.RLIMIT_AS, (20000000000, 20000000000))
    import torch
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    return actual_flags()


def actual_flags():
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('a CPU-only phase initialized CUDA')
    if 'fnit.robust_register' in sys.modules:
        raise RuntimeError('experiment loaded the production robust namespace')
    if any(n == 'fnit.gems' or n.startswith('fnit.gems.') for n in sys.modules):
        raise RuntimeError('experiment unexpectedly loaded GEMS')
    return {'CPU_affinity': sorted(os.sched_getaffinity(0)),
            'Torch_threads': torch.get_num_threads(),
            'Torch_interop_threads': torch.get_num_interop_threads(),
            'Torch_version': str(torch.__version__),
            'CUDA_initialized': False, 'Torch_GPU_allocated_bytes': 0,
            'Torch_GPU_reserved_bytes': 0,
            'TF32_CPU_unused': bool(torch.backends.cuda.matmul.allow_tf32),
            'source_namespace_production_loaded': False,
            'native_GEMS_loaded': False,
            'address_space_cap_bytes': resource.getrlimit(resource.RLIMIT_AS)[0],
            'system_load': list(os.getloadavg())}


def run_stage(plan, phase):
    from importlib import import_module
    import nibabel as nib
    import numpy as np
    from load_experiment import load_package
    robust_register = load_package(plan, candidate=True).robust_register
    stage = Path(plan['run_directory'])/'stages'
    if phase == 'rigid':
        stage.mkdir(mode=0o700, exist_ok=False)
        source = plan['moving']
        boundary = {'original_moving_sha256': digest(source)}
    else:
        rigid_path = stage/'rigid.header.mgz'
        prior = json.loads((Path(plan['run_directory'])/'rigid/report.private.json').read_text())
        if prior['status'] != 'completed' or prior['scientific_returncode'] != 0:
            raise ValueError('affine requires the successful own rigid receipt')
        if digest(rigid_path) != prior['result']['mapped_MGZ']['sha256']:
            raise ValueError('saved rigid MGH changed before affine reload')
        reload_started = time.monotonic()
        source = nib.load(str(rigid_path))
        # Materialize the stored image in this fresh process before the API.
        values = np.array(source.dataobj, copy=True)
        boundary = {'rigid_MGH_sha256': digest(rigid_path),
                    'actual_reloaded_shape': list(source.shape),
                    'actual_reloaded_dtype': str(source.header.get_data_dtype()),
                    'actual_reloaded_data_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
                    'reload_and_read_seconds': time.monotonic()-reload_started}
        del values
    started = time.monotonic()
    result = robust_register(source, plan['fixed'], mode=phase, **plan['parameters'])
    api_wall = time.monotonic()-started
    levels = result.report['stages']
    if len(levels) > 2 or sum(len(x['steps']) for x in levels) > 10:
        raise ValueError('declared two-level/five-update-per-level scope exceeded')
    destination = stage/(phase+'.header.mgz')
    lta_path = stage/(phase+'.lta')
    io_started = time.monotonic()
    nib.save(result.header_image, str(destination))
    mgh_wall = time.monotonic()-io_started
    io_started = time.monotonic()
    result.transform.save(lta_path)
    lta_wall = time.monotonic()-io_started
    sampling = import_module('fnit._robust_inverse_validation_20261006._sampling')
    if sampling.LOCAL_INVERSE_CALLS <= 0:
        raise RuntimeError('the actual registration did not call the local inverse')
    return {'stage_report': result.report, 'API_call_seconds': api_wall,
            'MGH_save_seconds': mgh_wall, 'LTA_save_seconds': lta_wall,
            'stage_input_boundary': boundary,
            'mapped_MGZ': {'bytes': destination.stat().st_size, 'sha256': digest(destination)},
            'LTA': {'bytes': lta_path.stat().st_size, 'sha256': digest(lta_path)},
            'official_commands_executed': 0, 'native_GEMS_calls': 0,
            'actual_local_inverse_calls': sampling.LOCAL_INVERSE_CALLS,
            'actual_local_inverse_function_module': sampling.native_inverse.__module__,
            'actual_local_sampling_source_SHA256': digest(sampling.__file__),
            'actual_level_count': len(levels),
            'actual_parameter_updates': sum(len(x['steps']) for x in levels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approved-plan-sha', required=True)
    parser.add_argument('--phase', choices=('rigid','affine','score'), required=True)
    options = parser.parse_args()
    if digest(options.plan) != options.approved_plan_sha:
        raise ValueError('explicit approved PLAN SHA differs')
    plan = json.loads(Path(options.plan).read_text())
    sys.path.insert(0, str(Path(plan['source_directory'])/'src'))
    os.umask(0o077)
    directory = Path(plan['run_directory'])/options.phase
    directory.mkdir(mode=0o700, exist_ok=False)
    report = {'phase': options.phase, 'status': 'starting', 'PID': os.getpid(),
              'plan_sha256': options.approved_plan_sha,
              'scope': 'one isolated CPU API or saved-output score; no official/GEMS/GPU'}
    code = 1; started = time.monotonic()
    try:
        report['bindings_before'] = check_bindings(plan)
        report['actual_flags_before'] = configure(plan)
        if options.phase == 'score':
            from score_saved_outputs import score
            report['result'] = score(plan, directory)
            code = 0 if report['result']['all_declared_gates_pass'] else 2
            report['status'] = 'completed' if code == 0 else 'completed_gates_failed'
        else:
            report['result'] = run_stage(plan, options.phase)
            code = 0; report['status'] = 'completed'
    except BaseException as error:
        report['status'] = 'failed'
        report['exception'] = {'type': type(error).__name__, 'message': str(error)}
        (directory/'exception.private.txt').write_text(traceback.format_exc())
    finally:
        report['wall_seconds'] = time.monotonic()-started
        report['maxrss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        try:
            report['bindings_after'] = check_bindings(plan)
            report['actual_flags_after'] = actual_flags()
            before = {k:v for k,v in report['actual_flags_before'].items()
                      if k != 'system_load'}
            after = {k:v for k,v in report['actual_flags_after'].items()
                     if k != 'system_load'}
            if before != after:
                raise RuntimeError('CPU precision/resource flags changed during phase')
            report['precision_resource_flags_before_after_exact'] = True
            from load_experiment import actual_module_bindings
            report['actual_experiment_modules'] = actual_module_bindings()
        except BaseException as error:
            report['postcheck_exception'] = {'type': type(error).__name__, 'message': str(error)}
            report['status'] = 'failed_postcheck'; code = 1
        report['scientific_returncode'] = code
        atomic_report(directory/'report.private.json', report)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
