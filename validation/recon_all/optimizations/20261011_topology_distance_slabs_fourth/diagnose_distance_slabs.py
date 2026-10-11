"""同一第四候选冻结输入的距离后端诊断；不进入生产，不启动原版程序。"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROLES=('original_serial','serial','parallel2','parallel4','root_cycle_serial')
RUNS=('cold','warm','audit')
DISTANCE_FIELDS=('distance_mm','buffer_bounds','buffer_coordinates','other_edge_faces',
                 'query_vertices','query_pseudonormals')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_source(directory,expected):
    source=Path(directory);manifest_path=source/'SOURCE_MANIFEST.private.json'
    if sha(manifest_path)!=expected:
        raise ValueError('immutable source manifest mismatch')
    manifest=json.loads(manifest_path.read_text())
    for name,record in manifest['payload'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('relative source payload required')
        path=source/relative
        if path.stat().st_size!=record['size'] or sha(path)!=record['sha256']:
            raise ValueError('immutable source payload mismatch')
    return manifest


def worker(protocol_path,role):
    started=time.perf_counter();protocol=json.loads(Path(protocol_path).read_text())
    if role not in ROLES or sha(__file__)!=protocol['diagnostic_driver_sha256']:
        raise ValueError('fixed diagnostic role and driver required')
    if protocol['threads']!=4 or protocol['cpu_affinity']!=[80,81,82,83]:
        raise ValueError('fixed total four-thread diagnostic budget required')
    os.sched_setaffinity(0,protocol['cpu_affinity'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CUDA must be hidden for this CPU stage')
    source=Path(protocol['source_directory'])
    manifest=validate_source(source,protocol['source_manifest_sha256'])
    variant=('original' if role=='original_serial' else
             'root_cycle' if role=='root_cycle_serial' else 'candidate')
    role_source=source/variant
    folder=Path(protocol['output_directory'])/role
    folder.mkdir(parents=True,mode=0o700,exist_ok=False)
    sys.path[:0]=[str(role_source/'src'),str(role_source/'tools')]
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(role_source/'src/fnit/recon_all'))
    import numpy as np
    import numba
    import torch
    import fnit.recon_all.topology_patch_fitness as fitness_module
    distance_module=importlib.import_module('fnit.recon_all.topology_defect_signed_distance')
    if sha(distance_module.__file__)!=protocol['distance_source_sha256'][variant]:
        raise ValueError('actual distance source mismatch')
    cycle_module=importlib.import_module('fnit.recon_all.topology_candidate_cycle')
    if sha(cycle_module.__file__)!=protocol['candidate_cycle_source_sha256'][variant]:
        raise ValueError('same frozen candidate cycle required')
    from benchmark_recon_topology_fourth_candidate_profile import run
    benchmark_path=role_source/'tools/benchmark_recon_topology_fourth_candidate_profile.py'
    if sha(benchmark_path)!=protocol['fourth_tool_sha256']:
        raise ValueError('actual existing fourth benchmark source mismatch')
    original=fitness_module.defect_signed_distance
    original_mask=numba.get_num_threads()
    cuda_before=torch.cuda.is_initialized()
    backend='serial' if role in ('original_serial','serial','root_cycle_serial') else 'parallel'
    target_mask=2 if role=='parallel2' else 4
    observations=[];captured=[]

    def selected_distance(**kwargs):
        entry_mask=numba.get_num_threads();wrapped_start=time.perf_counter()
        try:
            if entry_mask!=target_mask:
                numba.set_num_threads(target_mask)
            actual_mask=numba.get_num_threads()
            if actual_mask!=target_mask:
                raise RuntimeError('diagnostic distance thread mask not applied')
            result=(original(**kwargs) if role in ('original_serial','root_cycle_serial')
                    else original(**kwargs,backend=backend))
            # Hold own returned arrays; saving and comparing happen after the
            # complete benchmark returns, outside the measured candidate API.
            captured.append({name:result[name] for name in DISTANCE_FIELDS})
        finally:
            numba.set_num_threads(entry_mask)
            restored=numba.get_num_threads()==entry_mask
        observations.append(dict(backend=backend,actual_numba_mask=actual_mask,
            entry_numba_mask=entry_mask,thread_mask_restored=restored,
            wrapped_distance_api_seconds=time.perf_counter()-wrapped_start))
        if not restored:
            raise RuntimeError('diagnostic distance thread mask not restored')
        return result

    fitness_module.defect_signed_distance=selected_distance
    try:
        arguments={key:Path(value)for key,value in protocol['inputs'].items()
                   if key in ('input_surface','input_qsphere','brain','state_checkpoint','producer_progress')}
        report=run(**arguments,frozen_source_manifest=source/'SOURCE_MANIFEST.private.json',
            expected_source_manifest_sha256=protocol['source_manifest_sha256'],
            expected_checkpoint_sha256=protocol['expected_checkpoint_sha256'],
            output=folder/'candidate',defect_number=3,threads=4,
            blocker_scan_backend='numba',audit_sequence=True,skip_profile=True)
    finally:
        fitness_module.defect_signed_distance=original
        numba.set_num_threads(original_mask)
    if len(captured)!=3 or len(observations)!=3 or [r['run']for r in report['records']]!=list(RUNS):
        raise RuntimeError('complete cold/warm/audit distance observations required')
    if any(not r['thread_mask_restored']for r in observations):
        raise RuntimeError('distance observation mask restoration failed')
    artifacts={}
    for name,data in zip(RUNS,captured):
        path=folder/('distance.'+name+'.private.npz')
        np.savez_compressed(path,**data);artifacts[name]=sha(path)
    validate_source(source,protocol['source_manifest_sha256'])
    for module_name,digest in report['actual_module_sha256'].items():
        relative=variant+'/src/'+module_name.replace('.','/')
        record=manifest['payload'].get(relative+'.py') or manifest['payload'].get(relative+'/__init__.py')
        if record is None or record['sha256']!=digest:
            raise ValueError('actually loaded helper does not match immutable source')
    audit=report['records'][-1]
    if (audit['blocker_predicate_calls']!=protocol['expected_predicate_calls']
            or audit['blocker_predicate_sequence_sha256']!=protocol['expected_predicate_sequence_sha256']
            or report['actual_rng_draws']!=0 or not report['all_repeated_fields_byte_equal']
            or not report['all_repeated_controls_equal']):
        raise RuntimeError('complete frozen first-candidate strict gate failed')
    observations=[dict(run=name,**row)for name,row in zip(RUNS,observations)]
    receipt=dict(status='complete_fixed_fourth_distance_backend_diagnostic',role=role,
        source_manifest_sha256=protocol['source_manifest_sha256'],
        distance_source_sha256=protocol['distance_source_sha256'][variant],
        candidate_cycle_source_sha256=protocol['candidate_cycle_source_sha256'][variant],
        diagnostic_driver_sha256=sha(__file__),protocol_sha256=sha(protocol_path),
        global_function_replacement=True,production_default_changed=False,
        wrapper_restored=fitness_module.defect_signed_distance is original,
        final_thread_mask_restored=numba.get_num_threads()==original_mask,
        cuda_initialized_before=cuda_before,cuda_initialized_after=torch.cuda.is_initialized(),
        distance_observations=observations,distance_artifact_sha256=artifacts,
        inherited_report_sha256=sha(folder/'candidate/report.private.json'),
        complete_worker_seconds_before_report=time.perf_counter()-started,
        self_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        threads=4,cpu_affinity=sorted(os.sched_getaffinity(0)),
        actual_numba_version=numba.__version__,actual_numpy_version=np.__version__,
        actual_torch_version=torch.__version__,full_ga=False,whole_recon_all=False)
    if not receipt['wrapper_restored'] or not receipt['final_thread_mask_restored']:
        raise RuntimeError('diagnostic wrapper restoration failed')
    write(folder/'worker.private.json',receipt)
    write(folder/'COMPLETE.json',dict(status=receipt['status'],report_sha256=sha(folder/'worker.private.json')))
    return receipt


def arrays(folder,name):
    import numpy as np
    output={}
    for suffix,prefix in [('.private.npz',''),('.state.private.npz','state_')]:
        with np.load(folder/'candidate'/(name+suffix),allow_pickle=False) as archive:
            output.update({prefix+key:archive[key].copy()for key in archive.files})
    with np.load(folder/('distance.'+name+'.private.npz'),allow_pickle=False) as archive:
        output.update({'distance_'+key:archive[key].copy()for key in archive.files})
    return output


def parent(protocol_path):
    started=time.perf_counter();protocol=json.loads(Path(protocol_path).read_text())
    if sha(__file__)!=protocol['diagnostic_driver_sha256']:
        raise ValueError('actual diagnostic driver mismatch')
    source=Path(protocol['source_directory'])
    manifest=validate_source(source,protocol['source_manifest_sha256'])
    root=Path(protocol['output_directory']);root.mkdir(parents=True,mode=0o700,exist_ok=False)
    processes=[]
    for role in ROLES:
        env=dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                   MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',NUMBA_NUM_THREADS='4',
                   ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='4',
                   NUMBA_CACHE_DIR=str(root/('numba-'+role)))
        command=[sys.executable,str(Path(__file__).resolve()),'--protocol',str(protocol_path),
                 '--worker',role]
        tick=time.perf_counter()
        with (root/(role+'.stdout.private.log')).open('wb') as stdout:
            with (root/(role+'.stderr.private.log')).open('wb') as stderr:
                process=subprocess.run(command,env=env,stdout=stdout,stderr=stderr)
        receipt=dict(role=role,exit_code=process.returncode,
                     fresh_cli_seconds=time.perf_counter()-tick)
        write(root/(role+'.process.private.json'),receipt);processes.append(receipt)
        if process.returncode!=0:
            write(root/'FAILURE.private.json',dict(failed_role=role,processes=processes))
            raise RuntimeError('fixed fourth distance worker failed; private artifacts preserved')
    validate_source(source,protocol['source_manifest_sha256'])
    import numpy as np
    baseline={name:arrays(root/'original_serial',name)for name in RUNS}
    comparisons=[];workers=[]
    input_hashes=None
    for role in ROLES:
        folder=root/role
        worker_report=json.loads((folder/'worker.private.json').read_text())
        complete=json.loads((folder/'COMPLETE.json').read_text())
        if complete['report_sha256']!=sha(folder/'worker.private.json'):
            raise ValueError('complete worker report binding changed')
        inherited=json.loads((folder/'candidate/report.private.json').read_text())
        if worker_report['inherited_report_sha256']!=sha(folder/'candidate/report.private.json'):
            raise ValueError('complete inherited benchmark report binding changed')
        if input_hashes is None:input_hashes=inherited['input_sha256']
        elif input_hashes!=inherited['input_sha256']:
            raise ValueError('same immutable fourth input hashes required')
        workers.append(dict(worker=worker_report,inherited=inherited,process=processes[ROLES.index(role)]))
        for name in RUNS:
            reference=baseline[name];current=arrays(folder,name)
            differences=[]
            if set(reference)!=set(current):
                raise ValueError('complete matching candidate/distance state fields required')
            for key in reference:
                a,b=reference[key],current[key]
                if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
                    differences.append(key)
            expected_count=protocol['expected_cycle_field_count']+len(DISTANCE_FIELDS)
            control_a=json.loads((root/'original_serial/candidate'/(name+'.control.private.json')).read_text())
            control_b=json.loads((folder/'candidate'/(name+'.control.private.json')).read_text())
            passed=not differences and len(current)==expected_count and control_a==control_b
            comparisons.append(dict(role=role,run=name,field_count=len(current),
                cycle_field_count=len(current)-len(DISTANCE_FIELDS),distance_field_count=len(DISTANCE_FIELDS),
                changed_field_count=len(differences),all_shape_dtype_bytes_equal=not differences,
                complete_controls_equal=control_a==control_b,passed=passed,
                private_changed_fields=differences))
    passed=all(row['passed']for row in comparisons)
    report=dict(status='complete_fixed_fourth_distance_backend_pair'if passed else'failed_strict_stage',
        source_manifest_sha256=protocol['source_manifest_sha256'],protocol_sha256=sha(protocol_path),
        diagnostic_driver_sha256=sha(__file__),distance_source_sha256=protocol['distance_source_sha256'],
        candidate_cycle_source_sha256=protocol['candidate_cycle_source_sha256'],
        protected_retessellation_source_sha256=manifest['payload'][
            'candidate/src/fnit/recon_all/topology_retessellation_python.py']['sha256'],
        workers=workers,comparisons=comparisons,input_sha256=input_hashes,
        all_complete_fields_and_controls_exact=passed,global_function_replacement=True,
        production_default_changed=False,candidate_reference_consumed_before_outputs=False,
        source_rng_draws=0,whole_search=False,whole_recon_all=False,
        complete_pair_cli_seconds_before_report=time.perf_counter()-started)
    write(root/'report.private.json',report)
    if not passed:raise RuntimeError('complete candidate/distance field comparison failed')
    write(root/'COMPLETE.json',dict(status=report['status'],report_sha256=sha(root/'report.private.json')))
    write(root/'EXIT.json',dict(exit_code=0,strict_pair_passed=True))
    print(json.dumps(dict(status=report['status'],strict_pair_passed=True,
        comparison_count=len(comparisons),field_count=protocol['expected_cycle_field_count']+6)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--worker',choices=ROLES)
    arguments=parser.parse_args()
    if arguments.worker:worker(arguments.protocol,arguments.worker)
    else:parent(arguments.protocol)


if __name__=='__main__':main()
