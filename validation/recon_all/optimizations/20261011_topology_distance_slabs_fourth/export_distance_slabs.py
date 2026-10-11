"""第四候选距离诊断的硬类型数值/哈希导出；不复制私有报告。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from diagnose_distance_slabs import ROLES,RUNS,sha,validate_source,write


def integer(value):
    if type(value) is not int or value<0:raise ValueError('nonnegative exact integer required')
    return value


def number(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError('finite nonnegative numeric timing required')
    return value


def signed_number(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('finite exact numeric fitness required')
    return value


def boolean(value):
    if type(value) is not bool:raise ValueError('exact bool required')
    return value


def digest(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('SHA-256 hex required')
    return value


def version(value):
    pattern=r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}(?:a[0-9]+|b[0-9]+|rc[0-9]+|\.dev[0-9]+)?(?:\+[A-Za-z][A-Za-z0-9_-]{0,31})?'
    if type(value) is not str or re.fullmatch(pattern,value) is None:
        raise ValueError('machine version token required')
    return value


def export(*,input_run,frozen_source,output_directory):
    run,source,out=map(Path,(input_run,frozen_source,output_directory))
    if out.exists():raise FileExistsError('fresh public output required')
    report_path=run/'report.private.json';report=json.loads(report_path.read_text())
    complete=json.loads((run/'COMPLETE.json').read_text())
    exit_record=json.loads((run/'EXIT.json').read_text())
    if (report['status']!='complete_fixed_fourth_distance_backend_pair'
            or complete['status']!=report['status'] or complete['report_sha256']!=sha(report_path)
            or exit_record!={'exit_code':0,'strict_pair_passed':True}
            or not boolean(report['all_complete_fields_and_controls_exact'])
            or not boolean(report['global_function_replacement'])
            or boolean(report['production_default_changed'])
            or boolean(report['candidate_reference_consumed_before_outputs'])
            or integer(report['source_rng_draws'])!=0
            or boolean(report['whole_search']) or boolean(report['whole_recon_all'])):
        raise ValueError('same complete strict diagnostic pair required')
    manifest=validate_source(source,digest(report['source_manifest_sha256']))
    distance=report['distance_source_sha256']
    variants=('original','candidate','root_cycle')
    if set(distance)!=set(variants):raise ValueError('fixed source variants required')
    cycles=report['candidate_cycle_source_sha256']
    if set(cycles)!=set(variants):raise ValueError('fixed cycle variants required')
    for variant in variants:
        if digest(distance[variant])!=manifest['payload'][variant+
                '/src/fnit/recon_all/topology_defect_signed_distance.py']['sha256']:
            raise ValueError('actual distance source binding mismatch')
        if digest(cycles[variant])!=manifest['payload'][variant+
                '/src/fnit/recon_all/topology_candidate_cycle.py']['sha256']:
            raise ValueError('actual cycle source binding mismatch')
    for name,item in manifest['payload'].items():
        if name.startswith('original/'):
            relative=name.removeprefix('original/')
            for variant,allowed in [('candidate','src/fnit/recon_all/topology_defect_signed_distance.py'),
                                    ('root_cycle','src/fnit/recon_all/topology_candidate_cycle.py')]:
                if relative!=allowed and manifest['payload'].get(variant+'/'+relative)!=item:
                    raise ValueError('only the declared independent module may differ')
    public=dict(status='complete_fixed_fourth_distance_backend_pair',
        scope='same own third-defect checkpoint; fixed fourth first candidate; no complete GA or raw-T1 whole test',
        source_manifest_sha256=digest(report['source_manifest_sha256']),
        protocol_sha256=digest(report['protocol_sha256']),
        diagnostic_driver_sha256=digest(report['diagnostic_driver_sha256']),
        distance_source_sha256={k:digest(distance[k])for k in variants},
        candidate_cycle_source_sha256={k:digest(cycles[k])for k in variants},
        protected_retessellation_source_sha256=digest(report['protected_retessellation_source_sha256']),
        source_file_count=integer(len(manifest['payload'])),
        all_complete_fields_and_controls_exact=True,global_function_replacement=True,
        production_default_changed=False,candidate_reference_consumed_before_outputs=False,
        source_rng_draws=0,whole_search=False,whole_recon_all=False,
        overall_metric_equivalence='not_assessed',gpu_allocated_bytes=None,gpu_reserved_bytes=None,
        complete_pair_cli_seconds_before_report=number(report['complete_pair_cli_seconds_before_report']),
        input_sha256={},workers=[],comparisons=[])
    input_names=('nofix','qsphere','brain','checkpoint','producer_progress_snapshot')
    if set(report['input_sha256'])!=set(input_names):raise ValueError('fixed input SHA keys required')
    public['input_sha256']={k:digest(report['input_sha256'][k])for k in input_names}
    if [r['worker']['role']for r in report['workers']]!=list(ROLES):
        raise ValueError('fixed completed worker order required')
    for row,role in zip(report['workers'],ROLES):
        worker,inherited,process=row['worker'],row['inherited'],row['process']
        folder=run/role
        if (sha(folder/'worker.private.json')!=json.loads((folder/'COMPLETE.json').read_text())['report_sha256']
                or worker['inherited_report_sha256']!=sha(folder/'candidate/report.private.json')
                or inherited!=json.loads((folder/'candidate/report.private.json').read_text())
                or process!=json.loads((run/(role+'.process.private.json')).read_text())
                or process['role']!=role or integer(process['exit_code'])!=0
                or not boolean(worker['global_function_replacement'])
                or not boolean(worker['wrapper_restored'])
                or not boolean(worker['final_thread_mask_restored'])
                or boolean(worker['cuda_initialized_before']) or boolean(worker['cuda_initialized_after'])
                or integer(worker['threads'])!=4 or worker['cpu_affinity']!=[80,81,82,83]
                or integer(inherited['actual_rng_draws'])!=0
                or not boolean(inherited['all_repeated_fields_byte_equal'])
                or not boolean(inherited['all_repeated_controls_equal'])):
            raise ValueError('complete actual worker and diagnostic restoration gate required')
        for module_name,module_sha in inherited['actual_module_sha256'].items():
            if type(module_name) is not str or re.fullmatch(r'fnit(?:\.[A-Za-z_][A-Za-z_0-9]*)+',module_name) is None:
                raise ValueError('declared FNIT helper names only')
            variant=('original' if role=='original_serial' else
                     'root_cycle' if role=='root_cycle_serial' else 'candidate')
            relative=variant+'/src/'+module_name.replace('.','/')
            item=manifest['payload'].get(relative+'.py')or manifest['payload'].get(relative+'/__init__.py')
            if item is None or digest(module_sha)!=item['sha256']:
                raise ValueError('actual loaded helper hash differs from source freeze')
        observations=worker['distance_observations'];records=inherited['records']
        if [r['run']for r in observations]!=list(RUNS) or [r['run']for r in records]!=list(RUNS):
            raise ValueError('fixed cold/warm/audit order required')
        selected=dict(role=role,fresh_cli_seconds=number(process['fresh_cli_seconds']),
            worker_seconds_before_report=number(worker['complete_worker_seconds_before_report']),
            self_peak_rss_kib=integer(worker['self_peak_rss_kib']),
            self_peak_rss_bytes=integer(worker['self_peak_rss_kib'])*1024,
            total_threads=4,cpu_affinity=[80,81,82,83],cuda_hidden=True,
            global_function_replacement=True,wrapper_restored=True,thread_mask_restored=True,
            cuda_initialized_before=False,cuda_initialized_after=False,
            versions={name:version(worker['actual_'+name+'_version'])for name in ('numpy','numba','torch')},
            actual_loaded_helper_count=integer(len(inherited['actual_module_sha256'])),
            worker_report_sha256=sha(folder/'worker.private.json'),records=[])
        for observed,record,name in zip(observations,records,RUNS):
            mask=2 if role=='parallel2' else 4
            backend='serial' if role in ('original_serial','serial','root_cycle_serial') else 'parallel'
            if (observed['backend']!=backend or integer(observed['actual_numba_mask'])!=mask
                    or integer(observed['entry_numba_mask'])!=4
                    or not boolean(observed['thread_mask_restored'])
                    or not boolean(record['all_repeated_bytes_equal'])
                    or not boolean(record['all_repeated_controls_equal'])
                    or integer(record['repeat_field_count'])!=72
                    or digest(worker['distance_artifact_sha256'][name])!=sha(folder/('distance.'+name+'.private.npz'))):
                raise ValueError('complete mask, field and artifact gate required')
            for suffix,key in [('.private.npz','candidate_sha256'),
                               ('.state.private.npz','state_sha256'),
                               ('.control.private.json','complete_control_sha256')]:
                if digest(record[key])!=sha(folder/'candidate'/(name+suffix)):
                    raise ValueError('actually saved complete candidate artifact changed')
            selected['records'].append(dict(run=name,backend=backend,distance_numba_mask=mask,
                serial_kernel_active_threads=1 if backend=='serial'else None,
                thread_mask_restored=True,fitness=signed_number(record['fitness']),
                complete_cycle_seconds=number(record['complete_cycle_seconds']),
                candidate_geometry_seconds=number(record['candidate_seconds']['geometry_complete_api']),
                candidate_sff_seconds=number(record['candidate_seconds']['sff_complete_api']),
                complete_prepared_five_seconds=number(record['candidate_seconds']['complete_prepared_five_api']),
                signed_distance_complete_seconds=number(record['five_seconds']['signed_distance_complete_api']),
                wrapped_distance_api_seconds=number(observed['wrapped_distance_api_seconds']),
                retessellation_seconds=number(record['retessellation_seconds']),
                face_detection_seconds=number(record['face_detection_seconds']),
                complete_candidate_fields=72,distance_fields=6,
                all_repeated_fields_bytes_equal=True,all_repeated_controls_equal=True,
                candidate_sha256=digest(record['candidate_sha256']),state_sha256=digest(record['state_sha256']),
                controls_sha256=digest(record['complete_control_sha256']),
                distance_arrays_sha256=digest(worker['distance_artifact_sha256'][name])))
        audit=records[-1]
        if integer(audit['blocker_predicate_calls'])!=2656862:
            raise ValueError('fixed predicate-call count required')
        selected['audit_predicate_calls']=2656862
        selected['audit_predicate_sequence_sha256']=digest(audit['blocker_predicate_sequence_sha256'])
        public['workers'].append(selected)
    expected=[(role,name)for role in ROLES for name in RUNS]
    if [(r['role'],r['run'])for r in report['comparisons']]!=expected:
        raise ValueError('complete fixed comparison matrix required')
    for row in report['comparisons']:
        if (integer(row['field_count'])!=78 or integer(row['cycle_field_count'])!=72
                or integer(row['distance_field_count'])!=6 or integer(row['changed_field_count'])!=0
                or not boolean(row['all_shape_dtype_bytes_equal'])
                or not boolean(row['complete_controls_equal']) or not boolean(row['passed'])):
            raise ValueError('complete strict 72+6-field and control gate required')
        public['comparisons'].append({key:row[key]for key in ('role','run','field_count','cycle_field_count',
            'distance_field_count','changed_field_count','all_shape_dtype_bytes_equal','complete_controls_equal','passed')})
    out.mkdir(parents=True,mode=0o700,exist_ok=False)
    metrics=out/'metrics.json';write(metrics,public)
    write(out/'EXPORT_RECEIPT.json',dict(exporter_sha256=sha(__file__),
        original_private_report_sha256=sha(report_path),original_private_report_size_bytes=report_path.stat().st_size,
        source_manifest_sha256=public['source_manifest_sha256'],public_metrics_sha256=sha(metrics),
        public_metrics_size_bytes=metrics.stat().st_size,arbitrary_text_selected=False,
        paths_commands_logs_mri_selected=False))
    print(json.dumps(dict(status=public['status'],metrics_sha256=sha(metrics),comparison_count=len(expected))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input-run','frozen-source','output-directory'):parser.add_argument('--'+name,type=Path,required=True)
    export(**vars(parser.parse_args()))


if __name__=='__main__':main()
