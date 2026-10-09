"""只读GCSA缓存修复及完整有序Numba ABBA，输出严格同输入合同和计时。

reports_directory为本页公开JSON树，output_directory须已存在且无摘要。
仅汇总固定两例自产输入，不调用MRI算法、参考程序或更改数值门槛。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def summarize_results(*, reports_directory: Path, output_directory: Path) -> dict:
    """读取两例双侧原始收据，返回JSON摘要，另写全部图谱/显存CSV。

    输入均为显式目录；表面surface RAS/mm、annot同序RGB标签int32；
    已完成、原文件和标签SHA、颜色表/名称、每轮计数及输入SHA必须相同。
    逐轮标签数组没有实测，不把计数一致写成逐轮标签一致。失败抛异常。
    """
    if not output_directory.is_dir():
        raise NotADirectoryError(output_directory)
    names = ('SUMMARY.json', 'atlas_times.csv', 'memory_samples_summary.csv')
    if any((output_directory / n).exists() for n in names):
        raise FileExistsError('summary output already exists')
    hashes, atlas_rows, memory_rows, cases, guard_checks = {}, [], [], {}, []
    def read(path):
        content = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
        return json.loads(content)
    def validate(report):
        if report['status'] != 'complete' or not report['inputs_unchanged']:
            raise ValueError('unfinished or changed input')
        if report['threads'] != 4 or report['cpu_affinity'] != [76, 77, 78, 79] or report['device'] != 'cuda:1':
            raise ValueError('thread/device budget changed')
        if report['cuda_allocator']['effective'] != 'disabled':
            raise ValueError('allocator changed')
        if not report['precision']['matmul_tf32'] or not report['precision']['cudnn_tf32'] or report['precision']['cuda_autocast']['enabled']:
            raise ValueError('precision changed')
    def same_outputs(reference, candidate, scope):
        if candidate['input_sha256'] != reference['input_sha256']:
            raise ValueError('input SHA changed: ' + scope)
        if set(candidate['atlases']) != set(reference['atlases']):
            raise ValueError('atlas set changed: ' + scope)
        for atlas, value in candidate['atlases'].items():
            original = reference['atlases'][atlas]
            for key in ('output_sha256', 'labels_sha256', 'color_table_sha256', 'color_names', 'vertices'):
                if value[key] != original[key]:
                    raise ValueError('annot contract changed: ' + scope + ':' + atlas + ':' + key)
            for key in ('gibbs_history', 'islands_history'):
                if value['result'][key] != original['result'][key]:
                    raise ValueError('per-round count changed: ' + scope + ':' + atlas + ':' + key)
    guard = reports_directory / 'recon_gcsa_guard_profile_20261009_v1'
    numba = reports_directory / 'recon_gcsa_numba_profile_20261009_v2'
    controllers = {}
    for name, folder, expected in [('guard', guard, 8), ('numba', numba, 16)]:
        controller = read(folder / 'controller.json')
        if controller['status'] != 'complete' or controller['unit_exit_code'] != 0 or len(controller['children']) != expected or any(c['exit_code'] != 0 for c in controller['children']):
            raise ValueError('controller/units did not complete: ' + name)
        controllers[name] = controller
        read(folder / 'UNIT_TEST_RECEIPT.json')
        read(folder / 'PUBLIC_EXPORT_MANIFEST.json')
    source_bindings = {}
    for case in ('06', '07'):
        for hemi in ('lh', 'rh'):
            key = f'sub{case}-{hemi}'
            old = read(guard / f'{key}-control/benchmark.json')
            changed = read(guard / f'{key}-guard/benchmark.json')
            validate(old); validate(changed); same_outputs(old, changed, key + ':guard')
            guard_checks.append({'case':case, 'hemi':hemi, 'atlas_count':3,
                'annotation_file_sha_exact':True, 'labels_color_table_names_exact':True,
                'gibbs_and_islands_round_counts_exact':True})
            reports = {role:read(numba / f'{key}-{role}/benchmark.json') for role in ('A1','B1','B2','A2')}
            reference = reports['A1']
            api, cli, gibbs = {}, {}, {}
            for role, report in reports.items():
                validate(report); same_outputs(reference, report, key + ':' + role)
                expected = 'python' if role.startswith('A') else 'numba'
                if report['gibbs_backend'] != expected:
                    raise ValueError('wrong backend')
                current_binding = {'gcsa_label_python.py':report['candidate_module_sha256'], **{Path(p).name:s for p,s in report['overlay_sha256'].items()}}
                if source_bindings and current_binding != source_bindings:
                    raise ValueError('candidate source changed during ABBA')
                source_bindings = current_binding
                api[role] = report['shared_geometry_seconds_including_sync'] + sum(a['full_api_wall_seconds_including_sync'] for a in report['atlases'].values())
                child = [c for c in controllers['numba']['children'] if c['case']==case and c['hemi']==hemi and c['role']==role]
                if len(child) != 1:raise ValueError('ambiguous child')
                cli[role] = child[0]['full_process_wall_seconds']
                gibbs[role] = sum(a['reclassification_seconds'] for a in report['atlases'].values())
                for atlas, values in report['atlases'].items():
                    row = {'case':case, 'hemi':hemi, 'role':role, 'backend':expected, 'atlas':atlas,
                        'shared_geometry_once_seconds':report['shared_geometry_seconds_including_sync'],
                        'model_init_seconds':values['model_init_seconds'],
                        'reclassification_seconds_including_pack_jit':values['reclassification_seconds'],
                        'numba_pack_seconds':values.get('numba_pack_seconds'),
                        'numba_sweep_seconds_including_jit_cache_load':values.get('numba_sweep_seconds_including_jit_cache_load'),
                        'full_api_wall_seconds_including_sync':values['full_api_wall_seconds_including_sync'],
                        'output_sha256':values['output_sha256'], 'labels_sha256':values['labels_sha256']}
                    for phase, seconds in values['result']['seconds'].items():row['api_' + phase + '_seconds']=seconds
                    atlas_rows.append(row)
                memory = report['process_memory']
                memory_rows.append({'case':case, 'hemi':hemi, 'role':role,
                    **{k:memory.get(k) for k in ('status','target_gpu_uuid','sampling_interval_seconds','max_observed_interval_seconds','peak_tree_total_bytes','observed_resolved_tree_peak_bytes','peak_target_compute_process_sum_bytes','peak_target_device_used_bytes')},
                    'failed_sample_count':len(memory['failed_samples'])})
            medians = {}
            for scope, times in [('three_atlas_api_including_shared_geometry',api), ('full_process_including_start_import_hash_all_writes',cli), ('reclassification_including_pack_jit',gibbs)]:
                before = statistics.median([times['A1'],times['A2']]);after=statistics.median([times['B1'],times['B2']])
                medians[scope]={'roles_seconds':times,'python_median_seconds':before,'numba_median_seconds':after,
                    'wall_reduction_percent':100*(1-after/before),'speedup':before/after}
            cases[key]={'input_sha256':reference['input_sha256'],'candidate_source_sha256':source_bindings,
                'three_atlas_file_labels_colors_names_exact_all_roles':True,
                'gibbs_and_islands_per_round_counts_exact_all_roles':True,
                'real_per_round_label_array_sha':'not_collected; synthetic contract snapshots exact',
                'timing':medians,'geometry_sha256_by_role':{k:v['geometry_sha256'] for k,v in reports.items()}}
    result={'scope':'two real FNIT-produced T1 subjects, four hemispheres, complete three-atlas frozen-input annotation ABBA',
        'baseline_whole_commit':'589e2749','candidate_at_test':'explicit uncommitted three-module overlay bound by SHA and uploaded bundle',
        'candidate_source_sha256':source_bindings,'guard_only':guard_checks,'cases':cases,
        'execution':'all 24 complete three-atlas processes exited 0; all guard and Numba contracts passed',
        'precision':'existing GPU feature TF32 float32; no autocast; ordered CPU probabilities float64 and fastmath=False',
        'performance_scope':'GPU1, CPU76-79 total4; one hemisphere at a time with shared node load; not production dual-hemisphere group or new raw-T1 end-to-end timing',
        'memory_scope':'ownership unresolved; parent/child bytes unknown; consecutive whole-card and compute-process queries include shared load; disabled-cache Torch counters not zero memory',
        'default_backend':'python unchanged; numba explicit opt-in','strict_operator_regression':'passed exact final annot and original per-round integer histories; no tolerance lowered',
        'overall_metric_equivalence':'not_assessed_no_confirmed_prospective_thresholds',
        'official_frozen_input_stage':'not_rerun_this_experiment; actual whole 589/765 official diagnostics linked in README',
        'installation':'existing relocated Conda runtime; dependencies already declared; fresh isolated Conda not validated',
        'input_report_sha256':hashes,'summarizer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output_directory/names[0]).write_text(json.dumps(result,indent=2)+'\n')
    for name, rows in [(names[1],atlas_rows),(names[2],memory_rows)]:
        with (output_directory/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-directory',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    report=summarize_results(reports_directory=args.reports_directory,output_directory=args.output_directory)
    print(json.dumps({k:v['timing']['three_atlas_api_including_shared_geometry'] for k,v in report['cases'].items()}))
