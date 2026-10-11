"""完整冻结候选序列的只读分调用cProfile；冷/热调用分别保存。"""
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
root=Path(os.environ['FNIT_CACHE_ROOT'])
workspace=root/'workspaces/topology_candidate_geometry_profile_20261010_v1'
run=root/'runs/topology_candidate_geometry_profile_20261010_v1'
frozen=root/'workspaces/topology_ga_finalize_20261010_v2'
base=root/'workspaces/topology_candidate_cycle_20261010_v2'
baseline=root/'workspaces/topology_candidate_hotspots_20261010_v1'
if run.exists():raise FileExistsError('fresh independent diagnostic required')
run.mkdir(mode=0o700)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
for name,entry in json.loads((frozen/'PATCH_SOURCE.json').read_text())['payload'].items():
    if sha(frozen/name)!=entry['sha256'] or (frozen/name).stat().st_size!=entry['size']:
        raise RuntimeError('frozen target changed: '+name)
baseline_manifest=json.loads((baseline/'SOURCE_MANIFEST.json').read_text())
for name,entry in baseline_manifest['payload'].items():
    if sha(baseline/name)!=entry['sha256'] or (baseline/name).stat().st_size!=entry['size_bytes']:
        raise RuntimeError('tested ce96 baseline changed: '+name)
sys.path[:0]=[str(frozen/'tools'),str(base/'tools'),str(base/'src')]
import fnit.recon_all
fnit.recon_all.__path__.insert(0,str(baseline/'src/fnit/recon_all'))
fnit.recon_all.__path__.insert(0,str(frozen/'src/fnit/recon_all'))
profiles=[]
def observe(name,function):
    def measured(*args,**kwargs):
        ordinal=sum(entry['function']==name for entry in profiles)
        profiler=cProfile.Profile()
        tick=time.perf_counter()
        try:return profiler.runcall(function,*args,**kwargs)
        finally:
            wall=time.perf_counter()-tick
            profiler.dump_stats(str(run/f'{name}.{ordinal}.pstats'))
            stats=pstats.Stats(profiler)
            functions=[]
            for (file,line,fn),(primitive,calls,exclusive,cumulative,callers) in stats.stats.items():
                functions.append(dict(file=file,line=line,function=fn,primitive_calls=primitive,
                    calls=calls,exclusive_seconds=exclusive,cumulative_seconds=cumulative))
            profiles.append(dict(function=name,ordinal=ordinal,wall_seconds=wall,
                total_profile_calls=stats.total_calls,
                functions=sorted(functions,key=lambda row:row['exclusive_seconds'],reverse=True)))
    return measured
from fnit.recon_all import topology_candidate_geometry as geometry
geometry.candidate_geometry=observe('candidate_geometry',geometry.candidate_geometry)
program=frozen/'tools/benchmark_recon_topology_ga_finalize.py'
namespace=runpy.run_path(str(program),run_name='fnit_ce96_geometry_readonly')
from fnit.recon_all import topology_candidate_cycle as cycle
cycle._candidate_state=observe('_candidate_state',cycle._candidate_state)
subject=root/'workspaces/recon_topology_source_mri_edges_20261010_v2/input/subject'
reference=root/'runs/topology_source_fitness_observer_20261010_v2/repeat0'
args=dict(before_retess=root/'runs/topology_source_retessellation_observer_20261010_v3/repeat0/retess.before.bin',
    initial_attributes=root/'runs/topology_source_match_observer_20261010_v2/repeat0/match.before.bin',
    initial_global_priors=reference/'statistics.bin',input_surface=subject/'surf/lh.orig.nofix',
    input_qsphere=subject/'surf/lh.qsphere.nofix',brain=subject/'mri/brain.mgz',
    after_statistics_reference=reference/'patch_statistics.bin',after_volume_reference=reference/'volume.bin',
    fitness_reference=reference/'fitness.json',output=run/'candidate',threads=4)
start=time.perf_counter()
try:
    result=namespace['run'](**args)
    loaded={name:sha(module.__file__) for name,module in sorted(sys.modules.items())
        if name.startswith('fnit.') and getattr(module,'__file__',None)
        and str(module.__file__).endswith('.py')}
    for name in ['smooth_surface_python','topology_candidate_cycle']:
        if loaded['fnit.recon_all.'+name]!=sha(baseline/f'src/fnit/recon_all/{name}.py'):
            raise RuntimeError('wrong actual ce96 source '+name)
    report=dict(status='complete',scope='one actual complete frozen two-sequence API; per-call cProfile distinguishes cold and warm calls; not full GA or raw T1',
        baseline_publication_commit='ce96e724c26a09ac5f543953ace2208650fc66d3',
        original_target_program_sha256=sha(program),baseline_overlay_manifest_sha256=sha(baseline/'SOURCE_MANIFEST.json'),
        driver_sha256=sha(__file__),input_sha256={name:sha(value) for name,value in args.items() if isinstance(value,Path) and name!='output'},
        profiles=profiles,complete_api_wall_seconds=time.perf_counter()-start,
        records=result['records'],calibration_seconds=result['raw_calibration_seconds'],
        source_first_gate=result['initial_candidate_strict_source_gate'],final_repeat=result['status'],
        actual_fnit_loaded_source_sha256=loaded,
        runtime_versions={'python':sys.version,**{name:importlib.metadata.version(name) for name in ['numpy','numba','torch','nibabel']}},
        cpu_affinity=sorted(os.sched_getaffinity(0)),thread_environment={name:os.environ.get(name) for name in ['OMP_NUM_THREADS','NUMBA_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
        gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_memory_bytes=None,
        maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        caveat='cProfile changes Python overhead; not an optimization performance comparison',reference_after_used_to_generate=False)
    (run/'report.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (run/'COMPLETE.json').write_text(json.dumps(dict(status='profile_complete'))+'\n')
    (run/'EXIT.json').write_text(json.dumps(dict(exit_code=0))+'\n')
    print(json.dumps(dict(status='complete',calls=len(profiles),wall=report['complete_api_wall_seconds']),indent=2))
except BaseException as error:
    (run/'FAILED.json').write_text(json.dumps(dict(type=type(error).__name__,message=str(error)))+'\n')
    (run/'EXIT.json').write_text(json.dumps(dict(exit_code=1))+'\n')
    raise
