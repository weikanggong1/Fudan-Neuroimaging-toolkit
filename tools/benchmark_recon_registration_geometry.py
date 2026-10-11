"""同输入完整球面配准ABBA；复用已验证的阶段驱动，不续跑原始T1整例。

baseline/candidate-source是独立冻结根目录，含src及STAGE_SOURCE_BINDING.json。
subject含同序sphere/smoothwm/sulc，atlas为声明的半球TIFF；surface RAS/mm。
stage-driver是既有benchmark_registration_v2.py；threads默认4，device默认
cuda:0，hemi默认lh，output-dir须不存在。输出4份完整阶段报告、独立表面、
轨迹和精确坐标/面/文件比较JSON；不写输入目录。默认TF32、float32、仅
fresh child开启缓存，显式CUDA平均。非CUDA、非法线程或已有输出抛错误；
任何子阶段失败保留其原报告并报错，没有原版程序回退。无独立官方命令。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trajectory(report):
    registration = report['registration']
    result = {}
    for phase in ('sulc_pass', 'smoothwm_pass'):
        data = registration[phase]
        result[phase] = [{k: v for k, v in row.items() if k != 'seconds'}
                         for row in data['updates']]
    data = registration['sulc_pass']
    result.update({k: data[k] for k in ('rigid_angles', 'rigid_score', 'rigid_evaluations')})
    result['negative_counts'] = registration['smoothwm_pass']['negative_counts']
    result['sulc_seed_sha256'] = registration['sulc_seed_sha256']
    return result


def evaluate_completed_reports(summary, output_dir):
    """只读完整四份真实阶段输出；严格质量失败与替换退化分别记录。

    summary含四份已完成child记录及输入SHA，output_dir为原ABBA目录。
    更新summary的逐元素/轨迹/审计/QC与中位数，不写任何原文件。
    原有严格QC门保持不变；返回False表示有质量失败，所有行仍保留。
    坐标/面/输入/轨迹/启动审计不符合时抛错误。
    """
    if (tuple(row['policy'] for row in summary['runs']) !=
            ('baseline', 'candidate', 'candidate', 'baseline') or any(
                row['returncode'] != 0 or row['status'] != 'complete'
                for row in summary['runs'])):
        raise ValueError('four successfully completed ABBA children are required')
    import numpy as np
    from nibabel.freesurfer.io import read_geometry
    first = output_dir/'01-baseline'
    positions, faces = read_geometry(str(first/'sphere.reg'))
    first_report = json.loads((first/'report.json').read_text())
    first_trajectory = trajectory(first_report)
    for index, row in enumerate(summary['runs'], 1):
        output = output_dir/f'{index:02d}-{row["policy"]}'
        current, current_faces = read_geometry(str(output/'sphere.reg'))
        correspondence = current.shape == positions.shape and np.array_equal(faces, current_faces)
        if not correspondence:
            raise RuntimeError('ordered mesh correspondence failed; indexed comparison is invalid')
        difference = np.linalg.norm(current.astype(np.float64)-positions.astype(np.float64), axis=1)
        report = json.loads((output/'report.json').read_text())
        audit_rows = [json.loads(line) for path in (output/'execution-audit').glob('*.jsonl')
                      for line in path.read_text().splitlines()]
        row['execution_audit'] = {'recorded_events': len(audit_rows),
            'blocked_events': sum(bool(item.get('blocked')) for item in audit_rows),
            'scope': 'Python launch audit; not OS library/file isolation'}
        if not audit_rows or row['execution_audit']['blocked_events']:
            raise RuntimeError('Python execution audit incomplete or an original stage program attempted')
        row['regression'] = {'ordered_faces_same': True,
            'different_coordinate_elements': int(np.count_nonzero(current != positions)),
            'mean_distance_mm': float(difference.mean()), 'p99_distance_mm': float(np.percentile(difference,99)),
            'max_distance_mm': float(difference.max(initial=0)),
            'all_update_decisions_and_seed_same': trajectory(report) == first_trajectory,
            'output_bytes_same': sha(output/'sphere.reg') == sha(first/'sphere.reg'),
            'inputs_unchanged': report['input_sha256_before'] == report['input_sha256_after']
                               == summary['inputs_sha256'], 'qc': report['qc']}
        quality = report['qc']
        row['regression']['mesh_quality_passed'] = (
            quality['finite_coordinates'] and quality['euler'] == 2
            and all(quality[key] == 0 for key in
                    ('boundary_edges', 'nonmanifold_edges',
                     'zero_area_faces_double', 'negative_radial_faces_double')))
        if row['regression']['different_coordinate_elements'] or not all(
                row['regression'][key] for key in ('all_update_decisions_and_seed_same',
                                                   'output_bytes_same', 'inputs_unchanged')):
            raise RuntimeError('registration geometry reuse introduced a numeric regression')
    for field in ('stage_api_seconds', 'fresh_process_wall_seconds'):
        summary[field+'_median'] = {policy: statistics.median(
            row[field] for row in summary['runs'] if row['policy'] == policy)
            for policy in ('baseline', 'candidate')}
    summary['numeric_replacement_regression_passed'] = True
    summary['mesh_quality_failed_runs'] = [i for i, row in enumerate(summary['runs'], 1)
        if not row['regression']['mesh_quality_passed']]
    summary['all_mesh_quality_passed'] = not summary['mesh_quality_failed_runs']
    return summary['all_mesh_quality_passed']


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-source', 'candidate-source', 'subject', 'atlas',
                 'stage-driver', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--hemi', choices=('lh', 'rh'), default='lh')
    args = parser.parse_args(argv)
    if args.threads < 1 or not args.device.startswith('cuda:'):
        raise ValueError('positive threads and an explicit CUDA device are required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit = Path(__file__).parent/'recon_python_execution_audit'
    if not (audit/'sitecustomize.py').is_file():
        raise FileNotFoundError('existing Python execution audit is required')
    inputs = {name: args.subject/'surf'/f'{args.hemi}.{name}'
              for name in ('sphere', 'smoothwm', 'sulc')}
    inputs['atlas'] = args.atlas
    summary = {'status': 'running', 'scope': 'same-input full registration; not raw-T1 recon-all',
               'stage_driver_sha256': sha(args.stage_driver), 'driver_sha256': sha(__file__),
               'execution_audit_sha256': sha(audit/'sitecustomize.py'),
               'inputs_sha256': {k: sha(v) for k, v in inputs.items()},
               'device': args.device, 'threads': args.threads,
               'affinity': sorted(os.sched_getaffinity(0)), 'runs': []}
    env = dict(os.environ)
    env.pop('PYTORCH_NO_CUDA_MEMORY_CACHING', None)
    env.pop('FS_LICENSE', None)
    env.pop('FREESURFER_HOME', None)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        env[key] = str(args.threads)
    try:
        for index, policy in enumerate(('baseline', 'candidate', 'candidate', 'baseline'), 1):
            source = getattr(args, policy + '_source')
            output = args.output_dir/f'{index:02d}-{policy}'
            child_env = dict(env, PYTHONPATH=os.pathsep.join((str(audit), str(source/'src'))),
                             FNIT_RECON_AUDIT_DIR=str(output/'execution-audit'),
                             NUMBA_CACHE_DIR=str(args.output_dir/f'{index:02d}-numba-cache'))
            command = [sys.executable, str(args.stage_driver), '--mode', 'full',
                       '--source', str(source), '--subject', str(args.subject),
                       '--atlas', str(args.atlas), '--output', str(output),
                       '--hemi', args.hemi, '--device', args.device,
                       '--threads', str(args.threads)]
            tick = time.perf_counter()
            with (args.output_dir/f'{index:02d}-{policy}.log').open('wb') as stream:
                completed = subprocess.run(command, env=child_env, stdout=stream, stderr=stream)
            row = {'policy': policy, 'returncode': completed.returncode,
                   'fresh_process_wall_seconds': time.perf_counter()-tick,
                   'report': str(output/'report.json'), 'source_binding_sha256':
                   sha(source/'STAGE_SOURCE_BINDING.json')}
            if (output/'report.json').is_file():
                report = json.loads((output/'report.json').read_text())
                row.update(status=report['status'], stage_api_seconds=
                           report.get('complete_stage_API_wall_seconds'))
            summary['runs'].append(row)
            if completed.returncode or row.get('status') != 'complete':
                raise RuntimeError(f'{policy} full registration failed; original report retained')

        if not evaluate_completed_reports(summary, args.output_dir):
            raise RuntimeError('existing or candidate registered mesh fails strict quality gate')
        summary['status'] = 'complete'
        summary['overall_metric_equivalence'] = 'not_assessed'
        summary['whole_pipeline_speedup'] = 'not_measured'
    except BaseException as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        summary['harness_wall_seconds_including_preflight_IO_and_comparison'] = time.perf_counter()-started
        (args.output_dir/'summary.private.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    main()
