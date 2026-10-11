"""Private read-only cProfile of the already validated frozen FNIT sequence."""
from pathlib import Path
import cProfile
import faulthandler
import hashlib
import importlib.metadata
import json
import os
import pstats
import resource
import runpy
import sys
import time

faulthandler.enable()
start = time.perf_counter()
root = Path(os.environ['FNIT_CACHE_ROOT'])
old = root / 'workspaces/topology_ga_finalize_20261010_v2'
base = root / 'workspaces/topology_candidate_cycle_20261010_v2'
workspace = root / 'workspaces/topology_candidate_cycle_profile_20261010_v2'
run = root / 'runs/topology_candidate_cycle_profile_20261010_v2'
if run.exists():
    raise FileExistsError('new diagnostic run required')
run.mkdir(mode=0o700)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
binding = json.loads((old / 'PATCH_SOURCE.json').read_text())
for name, value in binding['payload'].items():
    data = (old / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != value['sha256'] or len(data) != value['size']:
        raise RuntimeError('existing frozen patch changed: ' + name)
sys.path[:0] = [str(old / 'tools'), str(base / 'tools'), str(base / 'src')]
import fnit.recon_all
fnit.recon_all.__path__.insert(0, str(old / 'src/fnit/recon_all'))
program = old / 'tools/benchmark_recon_topology_ga_finalize.py'
namespace = runpy.run_path(str(program), run_name='fnit_readonly_profile_target')
subject = root / 'workspaces/recon_topology_source_mri_edges_20261010_v2/input/subject'
reference = root / 'runs/topology_source_fitness_observer_20261010_v2/repeat0'
arguments = {
    'before_retess': root / 'runs/topology_source_retessellation_observer_20261010_v3/repeat0/retess.before.bin',
    'initial_attributes': root / 'runs/topology_source_match_observer_20261010_v2/repeat0/match.before.bin',
    'initial_global_priors': reference / 'statistics.bin',
    'input_surface': subject / 'surf/lh.orig.nofix',
    'input_qsphere': subject / 'surf/lh.qsphere.nofix',
    'brain': subject / 'mri/brain.mgz',
    'after_statistics_reference': reference / 'patch_statistics.bin',
    'after_volume_reference': reference / 'volume.bin',
    'fitness_reference': reference / 'fitness.json',
    'output': run / 'candidate',
    'threads': 4,
}
inputs = {key: {'sha256': sha(value), 'size_bytes': value.stat().st_size}
          for key, value in arguments.items() if isinstance(value, Path) and key != 'output'}
receipts = {name: sha(old / name) for name in ('PATCH_SOURCE.json', 'RUNTIME_SOURCE.private.json')}
receipts['existing_base_runtime'] = sha(base / 'RUNTIME_SOURCE.private.json')
profiler = cProfile.Profile()
result = None
failure = None
profile_start = time.perf_counter()
try:
    result = profiler.runcall(namespace['run'], **arguments)
except BaseException as error:
    failure = {'type': type(error).__name__, 'message': str(error)}
finally:
    profile_wall = time.perf_counter() - profile_start
    profiler.dump_stats(str(run / 'complete_sequence.pstats'))
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, name), (primitive, calls, exclusive, cumulative, callers) in stats.stats.items():
        rows.append({'file': filename, 'line': line, 'function': name,
                     'primitive_calls': primitive, 'calls': calls,
                     'exclusive_seconds': exclusive, 'cumulative_seconds': cumulative,
                     'caller_count': len(callers)})
    by_exclusive = sorted(rows, key=lambda item: item['exclusive_seconds'], reverse=True)
    by_cumulative = sorted(rows, key=lambda item: item['cumulative_seconds'], reverse=True)
    fnit = [item for item in rows if '/fnit/recon_all/' in item['file']]
    loaded = {name: {'file': module.__file__, 'sha256': sha(module.__file__)}
              for name, module in sorted(sys.modules.items())
              if name.startswith('fnit.') and getattr(module, '__file__', None)
              and str(module.__file__).endswith('.py')}
    payload = {'status': 'profile_completed' if failure is None else 'profile_failed',
               'failure': failure,
               'scope': 'one read-only cProfile invocation of the two validated frozen fixed-order sequences; not full GA or raw T1 continuous pipeline',
               'hardware_timing': 'cProfile adds Python instrumentation overhead; not an optimization speed comparison',
               'before_attributes_and_global_priors': 'explicit frozen same-input stage inputs',
               'reference_after_used_to_generate': False,
               'threads_requested': 4,
               'runtime_versions': {'python': sys.version, 'packages': {
                   name: importlib.metadata.version(name) for name in ('numpy', 'numba', 'torch', 'nibabel')}},
               'host': os.uname().nodename,
               'cpu_affinity': sorted(os.sched_getaffinity(0)),
               'thread_environment': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'NUMBA_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
               'target_cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'gpu_memory_bytes': None,
               'maximum_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'profile_wall_seconds': profile_wall,
               'complete_driver_seconds_including_imports_before_report': time.perf_counter() - start,
               'total_profile_exclusive_seconds': stats.total_tt,
               'profiled_total_calls': stats.total_calls,
               'existing_control_sha256': sha(old / 'control.py'),
               'existing_target_program_sha256': sha(program),
               'profiler_driver_sha256': sha(__file__),
               'frozen_receipt_sha256': receipts,
               'input_sha256': inputs,
               'actual_fnit_loaded_sources': loaded,
               'reference_initial_gate': None if result is None else result.get('initial_candidate_strict_source_gate'),
               'full_final_numeric_repeat': None if result is None else result.get('status'),
               'unprofiled_original_version': 'topology_ga_finalize_20261010_v2 retained unchanged; cross-time timing not treated as speed ratio',
               'sequence_records': None if result is None else result.get('records'),
               'raw_calibration_seconds': None if result is None else result.get('raw_calibration_seconds'),
               'top_exclusive': by_exclusive[:70],
               'top_cumulative': by_cumulative[:70],
               'fnit_functions': sorted(fnit, key=lambda item: item['exclusive_seconds'], reverse=True),
               'complete_profile_functions': by_exclusive}
    (run / 'profile.private.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    (run / 'EXIT.json').write_text(json.dumps({'exit_code': 0 if failure is None else 1}) + '\n')
    print(json.dumps({'status': payload['status'], 'profile_wall_seconds': profile_wall,
                      'initial_gate': payload['reference_initial_gate'],
                      'final_repeat': payload['full_final_numeric_repeat'],
                      'top_fnit': payload['fnit_functions'][:20]}, indent=2))
if failure is not None:
    raise RuntimeError(failure)
