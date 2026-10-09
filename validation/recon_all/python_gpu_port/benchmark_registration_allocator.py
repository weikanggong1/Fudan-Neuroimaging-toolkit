"""同自产网格的完整球面配准缓存ABBA；只改变fresh exec子进程分配策略。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def inputs(args):
    return {name: args.subject / 'surf' / f'{args.hemi}.{name}'
            for name in ('sphere', 'smoothwm', 'sulc')} | {'atlas': args.atlas}


def trajectory(api):
    return {key: [{name: value for name, value in row.items() if name != 'seconds'}
                  for row in api[key]['updates']]
            for key in ('sulc_pass', 'smoothwm_pass')} | {
                'negative_counts': api['smoothwm_pass']['negative_counts'],
                'rigid_angles': api['sulc_pass']['rigid_angles'],
                'rigid_score': api['sulc_pass']['rigid_score'],
                'rigid_evaluations': api['sulc_pass']['rigid_evaluations']}


def worker(args):
    tick = time.perf_counter()
    report = {'status': 'running', 'scope': 'complete_same_input_registration_not_whole',
              'code_version': args.code_version, 'allocator_requested': args.allocator,
              'device': args.device, 'threads': args.threads, 'pid': os.getpid()}
    try:
        import numpy as np
        import nibabel.freesurfer.io as fsio
        import numba
        import torch
        from fnit.recon_all.profiling import configure_cuda_allocator
        from fnit.recon_all.mris_register_run import run_register_sphere
        import fnit.recon_all.mris_register_run as source
        actual_source = Path(source.__file__).resolve()
        if not actual_source.is_relative_to(args.source.resolve()):
            raise RuntimeError('actual import is outside requested frozen source')
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        numba.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_grad_enabled(False)
        allocation = configure_cuda_allocator(args.device, args.allocator)
        device = torch.device(args.device)
        if device.type != 'cuda' or device.index is None:
            raise ValueError('device must be explicit cuda:N')
        torch.cuda.set_device(device)
        torch.cuda.synchronize(device)
        if allocation['torch_stats_known_valid']:
            torch.cuda.reset_peak_memory_stats(device)
        files = inputs(args)
        report.update(input_sha256={key: sha256(value) for key, value in files.items()},
                      module_sha256=sha256(actual_source), actual_source=str(actual_source),
                      allocator_details=allocation, cpu_affinity=sorted(os.sched_getaffinity(0)),
                      versions={'python': platform.python_version(), 'numpy': np.__version__,
                                'numba': numba.__version__, 'torch': torch.__version__,
                                'cuda': torch.version.cuda},
                      precision={'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                                 'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                                 'float32': True, 'autocast': torch.is_autocast_enabled('cuda'),
                                 'float16_or_bfloat16': False},
                      cuda_properties={'name': torch.cuda.get_device_properties(device).name,
                                       'uuid': str(getattr(torch.cuda.get_device_properties(device), 'uuid', 'unavailable'))},
                      numba_cache_dir=os.environ.get('NUMBA_CACHE_DIR'),
                      triton_cache_dir=os.environ.get('TRITON_CACHE_DIR'))
        api_tick = time.perf_counter()
        api = run_register_sphere(**files, output=args.output / f'{args.hemi}.sphere.reg',
                                  overlap_device='cpu', averaging_device=args.device)
        torch.cuda.synchronize(device)
        report['api_wall_seconds_including_io_jit_transfer_sync'] = time.perf_counter() - api_tick
        write_json(args.output / 'api.json', api)
        input_vertices, input_faces = fsio.read_geometry(str(files['sphere']))
        vertices, faces = fsio.read_geometry(str(api['output']))
        same_faces = np.array_equal(input_faces, faces)
        signed = np.einsum('ij,ij->i', np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                             vertices[faces[:, 2]] - vertices[faces[:, 0]]),
                           vertices[faces].mean(axis=1))
        structure = {'same_vertex_shape': vertices.shape == input_vertices.shape,
                     'ordered_faces_equal': bool(same_faces),
                     'finite_coordinates': bool(np.isfinite(vertices).all()),
                     'face_indices_valid': bool(((faces >= 0) & (faces < len(vertices))).all())}
        if not all(structure.values()):
            raise RuntimeError(f'invalid output structure: {structure}')
        report.update(status='complete', api=str(args.output / 'api.json'),
                      output_sha256=sha256(api['output']), trajectory=trajectory(api),
                      structure=structure, vertex_count=len(vertices), face_count=len(faces),
                      radial_orientation={'negative_face_count': int((signed < 0).sum()),
                                          'zero_face_count': int((signed == 0).sum()),
                                          'scope': 'signed cross-product dot face centroid about sphere origin'},
                      gpu_peak_allocated_bytes=(int(torch.cuda.max_memory_allocated(device))
                                                if allocation['torch_stats_known_valid'] else None),
                      gpu_peak_reserved_bytes=(int(torch.cuda.max_memory_reserved(device))
                                               if allocation['torch_stats_known_valid'] else None))
        if report['input_sha256'] != {key: sha256(value) for key, value in files.items()}:
            raise RuntimeError('input changed during benchmark')
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    report['worker_main_seconds'] = time.perf_counter() - tick
    report['worker_main_scope'] = 'argument parsing excluded; Python library imports, CUDA init, API and diagnostics included; parent exec wall additionally includes interpreter startup/exit'
    write_json(args.output / 'worker.json', report)
    return 0 if report['status'] == 'complete' else 1


def compare_surface(left, right):
    import numpy as np
    from nibabel.freesurfer.io import read_geometry
    a, af = read_geometry(str(left))
    b, bf = read_geometry(str(right))
    corresponding = a.shape == b.shape and np.array_equal(af, bf)
    result = {'left_sha256': sha256(left), 'right_sha256': sha256(right),
              'file_bytes_equal': sha256(left) == sha256(right),
              'same_vertex_shape': a.shape == b.shape, 'ordered_faces_equal': bool(np.array_equal(af, bf)),
              'index_correspondence': corresponding}
    if corresponding:
        distance = np.linalg.norm(a - b, axis=1)
        result.update(different_vertices=int(np.count_nonzero(distance)),
                      max_mm=float(distance.max(initial=0)), p99_mm=float(np.quantile(distance, .99)),
                      coordinates_equal=bool(np.array_equal(a, b)))
    return result


def analyze_completed_report(report_path, output_path):
    """由已结束四遍的原报告派生指标，不重写原件或重新标注执行状态。

    v1元数据同名覆盖导致median聚合失败，但四个worker仍独立完成。
    本函数只消费已记录比较与输入/线程等身份信息；不重新计算几何、
    不接触生产输出，也不将后处理称作一次新的完整配准。
    """
    original = json.loads(Path(report_path).read_text())
    runs = original['runs']
    if len(runs) != 4 or [row['allocator_requested'] for row in runs] != ['disabled', 'enabled', 'enabled', 'disabled']:
        raise ValueError('four complete ABBA workers required')
    if any(row['status'] != 'complete' or row['returncode'] != 0 for row in runs):
        raise ValueError('cannot analyze failed or partial workers as complete')
    if any(row['input_sha256'] != original['input_sha256'] for row in runs):
        raise ValueError('worker input identity mismatch')
    comparisons = original.get('comparisons_to_first')
    if not isinstance(comparisons, list) or len(comparisons) != 4:
        raise ValueError('original index correspondence and geometry comparison missing')
    medians = {policy: statistics.median(row['exec_wall_seconds'] for row in runs
                                        if row['allocator_requested'] == policy)
               for policy in ('disabled', 'enabled')}
    unchanged = all(row.get('coordinates_equal', False) and row['ordered_faces_equal']
                    for row in comparisons) and original['trajectories_all_equal']
    result = {'status': 'complete_posthoc_analysis_of_four_completed_workers',
              'scope': 'derived metrics; no new algorithm execution; original report status preserved',
              'original_report': str(report_path), 'original_report_sha256': sha256(report_path),
              'original_status': original['status'], 'original_error': original.get('error'),
              'analysis_script_sha256': sha256(__file__), 'input_sha256': original['input_sha256'],
              'exec_seconds': [row['exec_wall_seconds'] for row in runs],
              'allocator_order': [row['allocator_requested'] for row in runs],
              'exec_median_seconds': medians,
              'stage_speed_ratio_disabled_over_enabled': medians['disabled'] / medians['enabled'],
              'stage_reduction_percent': 100 * (1 - medians['enabled'] / medians['disabled']),
              'strict_reproduction': 'same_algorithm_output_exact' if unchanged else 'different',
              'optimization_regression': 'not_observed' if unchanged else 'failed_same_input_regression',
              'all_file_sha_equal': len({row['output_sha256'] for row in runs}) == 1,
              'comparisons_to_first': comparisons, 'trajectories_all_equal': original['trajectories_all_equal'],
              'previous_self_output_comparison': original.get('previous_self_output_comparison'),
              'whole_speedup': 'not_assessed', 'overall_metric_equivalence': 'not_assessed'}
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    write_json(output_path, result)
    return 0


def main():
    setup_tick = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=Path)
    parser.add_argument('--hemi', choices=('lh', 'rh'))
    parser.add_argument('--atlas', type=Path)
    parser.add_argument('--source', type=Path, help='frozen src directory')
    parser.add_argument('--profiling-module', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--code-version')
    parser.add_argument('--allocator', choices=('enabled', 'disabled'), default='disabled')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--previous-self-output', type=Path)
    parser.add_argument('--analyze-report', type=Path, help='derive metrics from four already completed ABBA workers')
    parser.add_argument('--analysis-output', type=Path, help='new derived JSON; original report unchanged')
    args = parser.parse_args()
    if args.analyze_report:
        if args.worker or args.analysis_output is None:
            parser.error('analysis requires --analysis-output and no --worker')
        return analyze_completed_report(args.analyze_report, args.analysis_output)
    for name in ('subject', 'hemi', 'atlas', 'source', 'profiling_module', 'output', 'code_version'):
        if getattr(args, name) is None:
            parser.error(f'--{name.replace("_", "-")} required for benchmark execution')
    if args.worker:
        return worker(args)
    if args.threads < 1 or not args.source.is_dir():
        parser.error('positive threads and existing frozen source required')
    for name, path in inputs(args).items():
        if not path.is_file():
            parser.error(f'missing {name}: {path}')
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('registration_benchmark_sampler', args.profiling_module)
    profiling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiling)
    report = {'status': 'running', 'scope': 'same-self-produced-input-complete-registration-cold-exec-ABBA;not-recon-whole',
              'code_version': args.code_version, 'subject': str(args.subject), 'hemi': args.hemi,
              'input_sha256': {key: sha256(value) for key, value in inputs(args).items()},
              'source_sha256': {str(path.relative_to(args.source)): sha256(path)
                                for path in sorted((args.source / 'fnit' / 'recon_all').rglob('*.py'))},
              'script_sha256': sha256(__file__), 'profiling_sha256': sha256(args.profiling_module),
              'threads': args.threads, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'host': platform.node(), 'cpu': platform.processor(), 'logical_device': args.device,
              'parent_allocator_environment': os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING'),
              'order': ['disabled', 'enabled', 'enabled', 'disabled'], 'runs': [],
              'timing_scope': 'fresh interpreter, Python/CUDA init, private cold JIT caches, full algorithm/load/transfer/IO, worker diagnostics/exit; parent comparison separately included in study wall',
              'strict_reproduction': 'pending', 'optimization_regression': 'pending',
              'overall_metric_equivalence': 'not_assessed', 'whole_speedup': 'not_assessed',
              'official_same_input_comparison': 'not_available_in_this_allocator_study; whole official comparison remains separate'}
    write_json(args.output / 'summary.json', report)
    sampler = profiling.ProcessTreeDeviceSampler(device=args.device, parent_pid=os.getpid(), interval=.5)
    try:
        for index, policy in enumerate(report['order']):
            run = args.output / f'{index + 1:02d}_{policy}'
            run.mkdir()
            environment = os.environ.copy()
            if policy == 'disabled':
                environment['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
            else:
                environment.pop('PYTORCH_NO_CUDA_MEMORY_CACHING', None)
            environment.update(NUMBA_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads),
                               OPENBLAS_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                               NUMBA_CACHE_DIR=str(run / 'numba_cache'), TRITON_CACHE_DIR=str(run / 'triton_cache'))
            environment['PYTHONPATH'] = str(args.source) + os.pathsep + environment.get('PYTHONPATH', '')
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--subject', str(args.subject),
                       '--hemi', args.hemi, '--atlas', str(args.atlas), '--source', str(args.source),
                       '--profiling-module', str(args.profiling_module), '--output', str(run),
                       '--device', args.device, '--threads', str(args.threads), '--code-version', args.code_version,
                       '--allocator', policy]
            exec_tick = time.perf_counter()
            with (run / 'worker.log').open('w') as stream:
                process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                sampler.add_worker(process.pid)
                while process.poll() is None:
                    sampler.sample_if_due()
                    time.sleep(.02)
            wall = time.perf_counter() - exec_tick
            result = json.loads((run / 'worker.json').read_text())
            row = {'index': index + 1, 'allocator': policy, 'returncode': process.returncode,
                   'exec_wall_seconds': wall, 'worker_report': str(run / 'worker.json'),
                   'worker_report_sha256': sha256(run / 'worker.json'), **result}
            report['runs'].append(row)
            write_json(args.output / 'summary.json', report)
            if process.returncode or result['status'] != 'complete':
                raise RuntimeError(f'worker {index + 1} failed')
        baseline = args.output / '01_disabled' / f'{args.hemi}.sphere.reg'
        comparisons = [compare_surface(baseline, args.output / f'{i + 1:02d}_{policy}' / f'{args.hemi}.sphere.reg')
                       for i, policy in enumerate(report['order'])]
        report['comparisons_to_first'] = comparisons
        report['trajectories_all_equal'] = all(row['trajectory'] == report['runs'][0]['trajectory'] for row in report['runs'])
        if args.previous_self_output:
            report['previous_self_output_comparison'] = compare_surface(baseline, args.previous_self_output)
        unchanged = all(row.get('coordinates_equal', False) and row['ordered_faces_equal'] for row in comparisons) and report['trajectories_all_equal']
        report.update(status='complete', strict_reproduction='same_algorithm_output_exact' if unchanged else 'different',
                      optimization_regression='not_observed' if unchanged else 'failed_same_input_regression')
        medians = {policy: statistics.median(row['exec_wall_seconds'] for row in report['runs'] if row['allocator_requested'] == policy)
                   for policy in ('enabled', 'disabled')}
        report['exec_median_seconds'] = medians
        report['stage_speed_ratio_disabled_over_enabled'] = medians['disabled'] / medians['enabled']
        report['stage_reduction_percent'] = 100 * (1 - medians['enabled'] / medians['disabled'])
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    report['device_process_tree'] = sampler.report()
    report['study_main_seconds'] = time.perf_counter() - setup_tick
    write_json(args.output / 'summary.json', report)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
