"""实际生产接线的smoothwm→inflated/sulc→standard sphere双侧完整配对。

只读取带SHA的公开自产smoothwm。control使用native+inherit，candidate
使用完整Torch inflation+fresh child cache enabled；球面算法/法向不改。
只读Python trace记录每轮坐标/梯度SHA、全部步长搜索和尺度状态，不改
函数局部变量或轨迹；双方包含同样观察开销。不是原始T1整例。
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    """只读JSON规范化；非有限诊断标明字符串，不替换为零。"""
    import numpy as np
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def callback(*, subject, hemi, device, threads, operation, native, assets,
             inflate_backend, expected_native_free_sha256):
    """调用实际_accurate_sphere_pair；只读逐轮诊断，不读取参考或改算法。"""
    import numpy as np
    import torch
    from fnit.recon_all import native_free
    from fnit.recon_all.sphere_standard_run import run_standard_sphere
    if sha(native_free.__file__) != expected_native_free_sha256:
        raise ValueError('native_free source SHA differs from frozen root overlay')
    if inflate_backend not in {'native', 'torch'}:
        raise ValueError('invalid explicit inflate backend')
    source_lines, first_line = inspect.getsourcelines(run_standard_sphere)
    markers = [first_line + index for index, line in enumerate(source_lines)
               if line.lstrip().startswith('updates.append(')]
    if len(markers) != 1:
        raise ValueError('cannot identify frozen standard-sphere update boundary')
    updates = []
    def observer(frame, event, arg):
        if frame.f_code is not run_standard_sphere.__code__:
            return None
        if event == 'line' and frame.f_lineno == markers[0]:
            values = frame.f_locals
            # CPython emits this line for method lookup and again after the
            # multiline dict arguments. Observe the same real update once.
            if updates and updates[-1]['index'] == values['index']:
                return observer
            updates.append({'index': values['index'], 'stage': values['stage'],
                'weight': values['weight'], 'averages': values['averages'],
                'steps_at_scale': values['steps_at_scale'], 'prior_scale': plain(values['prior_scale']),
                'next_scale': plain(values['next_scale']), 'ending_sse': plain(values['ending_sse']),
                'search': plain(values['search']),
                'coordinates_float32_sha256': hashlib.sha256(np.ascontiguousarray(values['xyz'], dtype=np.float32).tobytes()).hexdigest(),
                'gradient_float32_sha256': hashlib.sha256(np.ascontiguousarray(values['gradient'], dtype=np.float32).tobytes()).hexdigest()})
        return observer
    prior_trace = sys.gettrace()
    if prior_trace is not None:
        raise ValueError('benchmark requires no pre-existing Python trace')
    tick = time.perf_counter()
    try:
        sys.settrace(observer)
        timings, sphere = native_free._run_accurate_sphere_pair(
            inflate_binary=Path(native), subject=Path(subject), hemi=hemi,
            assets=Path(assets), device=device, normals_backend='numba',
            inflate_backend=inflate_backend)
    finally:
        sys.settrace(prior_trace)
    if len(updates) != len(sphere['updates']):
        raise RuntimeError('readonly trace did not observe every complete sphere update')
    modules = {name: sha(module.__file__) for name, module in sorted(sys.modules.items())
               if name.startswith('fnit.recon_all.') and getattr(module, '__file__', None)
               and Path(module.__file__).suffix == '.py'}
    return {'timings': timings, 'sphere': sphere, 'readonly_update_trace': updates,
        'trace_scope': 'coordinate/gradient SHA, complete search, SSE and schedule at each actual production standard-sphere update; no mutation',
        'callback_wall_seconds': time.perf_counter() - tick,
        'native_free_sha256': sha(native_free.__file__), 'source_sha256': modules,
        'actual_torch_threads': torch.get_num_threads(),
        'cache_disabled_environment': os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING'),
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'autocast': torch.is_autocast_enabled('cuda')}


def compare_sphere(candidate, control):
    import nibabel.freesurfer.io as fsio
    import numpy as np
    a, af, am = fsio.read_geometry(str(candidate), read_metadata=True)
    b, bf, bm = fsio.read_geometry(str(control), read_metadata=True)
    ordered = a.shape == b.shape and np.array_equal(af, bf)
    result = {'same_ordered_topology': bool(ordered), 'candidate_sha256': sha(candidate), 'control_sha256': sha(control),
              'vertices': len(a), 'faces': len(af),
              'volume_geometry_fields_equal': {key: bool(np.array_equal(am.get(key), bm.get(key))) for key in set(am) | set(bm)}}
    if ordered:
        distance = np.linalg.norm(a - b, axis=1)
        result.update(coordinates_equal=bool(np.array_equal(a, b)), different_vertices=int(np.count_nonzero(np.any(a != b, axis=1))),
                      max_mm=float(distance.max(initial=0)), p99_mm=float(np.percentile(distance, 99)))
    def quality(xyz, faces):
        corners = xyz[faces]
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        area = np.linalg.norm(cross, axis=1) * .5
        negative = np.einsum('ij,ij->i', cross, corners.mean(axis=1)) < 0
        return {'fp64_negative_faces': int(negative.sum()), 'fp64_negative_area_mm2': float(area[negative].sum()),
                'fp64_zero_area_faces': int(np.count_nonzero(area == 0)), 'all_finite': bool(np.isfinite(xyz).all()),
                'negative_face_indices_sha256': hashlib.sha256(np.flatnonzero(negative).astype(np.int64).tobytes()).hexdigest(),
                'quality_scope': 'radial orientation/area only; no full intersection check'}
    result['candidate_quality'] = quality(a, af); result['control_quality'] = quality(b, bf)
    return result


def main():
    tick = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--native-free-sha256', required=True)
    parser.add_argument('--code-version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append', help='明确case筛选，可重复；默认包全部双侧')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.threads < 2:
        raise ValueError('two workers require total threads >= 2')
    args.output.mkdir(parents=True)
    report = {'status': 'running', 'scope': 'actual production smoothwm-inflate-sulc-standard-sphere fixed-input bilateral chain; not whole recon-all',
              'code_version': args.code_version, 'expected_native_free_sha256': args.native_free_sha256,
              'device': args.device, 'threads_total': args.threads, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'rows': [], 'production_defaults_changed': False, 'overall_metric_equivalence': 'not assessed',
              'timing_scope': 'whole hemisphere group with readonly per-update trace, copy/exec/imports/CUDA/JIT/full chain/transfer/I/O/publish; independent cold JIT per strategy'}
    def save():
        temporary = args.output / 'summary.tmp'; temporary.write_text(json.dumps(plain(report), indent=2) + '\n')
        temporary.replace(args.output / 'summary.json')
    try:
        os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
        import torch
        import numba
        from fnit.recon_all import hemisphere_parallel, native_free
        from fnit.recon_all.profiling import configure_cuda_allocator
        from benchmark_complete import compare
        if sha(native_free.__file__) != args.native_free_sha256:
            raise ValueError('root overlay SHA changed before startup')
        selected = torch.device(args.device)
        if selected.type != 'cuda' or selected.index is None or torch.cuda.is_initialized():
            raise ValueError('fresh explicit cuda:N parent required')
        parent_allocator = configure_cuda_allocator(args.device, 'disabled')
        torch.set_num_threads(args.threads); torch.set_num_interop_threads(1); numba.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        live = torch.arange(4096, dtype=torch.float32, device=selected); torch.cuda.synchronize(selected)
        live_sha = hashlib.sha256(live.cpu().numpy().tobytes()).hexdigest()
        os.environ['PYTHONPATH'] = str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get('PYTHONPATH', '')
        manifest = json.loads((args.data / 'manifest.json').read_text()); entries = {}
        for item in manifest['cases']:
            if args.case and item['case'] not in args.case:
                continue
            if sha(args.data / item['surface']) != item['sha256']:
                raise ValueError('input SHA mismatch')
            entries.setdefault(item['case'], {})[item['hemisphere']] = item
        if not entries or any(set(row) != {'lh', 'rh'} for row in entries.values()):
            raise ValueError('selected cases require two exact hemispheres')
        report.update(data_manifest_sha256=sha(args.data / 'manifest.json'), native_sha256=sha(args.native),
            script_sha256=sha(__file__), hemisphere_parallel_sha256=sha(hemisphere_parallel.__file__),
            torch_version=torch.__version__, cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(selected),
            parent_allocator=parent_allocator, parent_cuda_initialized_before_group=True, parent_live_sha256=live_sha)
        save()
        for case, items in entries.items():
            row = {'case': case, 'input_sha256': {hemi: item['sha256'] for hemi, item in items.items()}, 'runs': []}
            report['rows'].append(row); destinations = []
            for backend, allocator in (('native', 'inherit'), ('torch', 'enabled')):
                directory = args.output / case / backend; subject = directory / 'subject'
                for folder in ('mri', 'surf', 'label', 'stats', 'scripts'):
                    (subject / folder).mkdir(parents=True)
                for hemi, item in items.items():
                    shutil.copyfile(args.data / item['surface'], subject / 'surf' / (hemi + '.smoothwm'))
                os.environ['NUMBA_CACHE_DIR'] = str(directory / 'cold_numba_cache')
                os.environ['TRITON_CACHE_DIR'] = str(directory / 'cold_triton_cache')
                begin = time.perf_counter()
                group = hemisphere_parallel.run_hemisphere_group(subject=subject, operation='accurate_sphere_chain',
                    device=args.device, threads=args.threads, workers=2, profile_stages=True,
                    callable_path='benchmark_sphere_prepare_group:callback',
                    kwargs={'native': str(args.native), 'assets': str(args.assets), 'inflate_backend': backend,
                            'expected_native_free_sha256': args.native_free_sha256},
                    startup_wait_seconds=30, cuda_allocator_cache=allocator)
                torch.cuda.synchronize(selected)
                checks = {'parent_live_sha_preserved': hashlib.sha256(live.cpu().numpy().tobytes()).hexdigest() == live_sha,
                    'parent_disabled_environment_preserved': os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') == '1',
                    'workers_actual_policy': all(group['workers'][hemi]['cuda_allocator']['effective'] == ('disabled' if backend == 'native' else 'enabled') for hemi in ('lh', 'rh')),
                    'threads_budget': all(group['values'][hemi]['actual_torch_threads'] == args.threads // 2 for hemi in ('lh', 'rh')),
                    'precision_preserved': all(group['values'][hemi]['matmul_tf32'] and group['values'][hemi]['cudnn_tf32'] and not group['values'][hemi]['autocast'] for hemi in ('lh', 'rh'))}
                row['runs'].append({'backend': backend, 'child_cache': allocator, 'group_wall_seconds': time.perf_counter() - begin,
                                    'group': group, 'checks': checks})
                destinations.append(subject / 'surf'); save()
                if not all(checks.values()):
                    raise RuntimeError('initialized parent/worker contract failed')
                print('DONE', case, backend, row['runs'][-1]['group_wall_seconds'], flush=True)
            row['comparisons'] = {hemi: {'inflated_sulc': compare(destinations[1], destinations[0], hemi),
                'sphere': compare_sphere(destinations[1] / (hemi + '.sphere'), destinations[0] / (hemi + '.sphere')),
                'every_update_trace_equal': row['runs'][0]['group']['values'][hemi]['readonly_update_trace'] == row['runs'][1]['group']['values'][hemi]['readonly_update_trace'],
                'update_count': len(row['runs'][0]['group']['values'][hemi]['readonly_update_trace']),
                'sphere_finish_history_equal': row['runs'][0]['group']['values'][hemi]['sphere']['negative_counts'] == row['runs'][1]['group']['values'][hemi]['sphere']['negative_counts']}
                for hemi in ('lh', 'rh')}
            row['observed_group_reduction_percent'] = (1 - row['runs'][1]['group_wall_seconds'] / row['runs'][0]['group_wall_seconds']) * 100
            row['strict_reproduction_passed'] = all(
                value['inflated_sulc']['strict_decoded_geometry_and_sulc']
                and all(value['inflated_sulc']['volume_geometry_fields_equal'].values())
                and value['sphere'].get('coordinates_equal', False)
                and all(value['sphere']['volume_geometry_fields_equal'].values())
                and value['every_update_trace_equal'] and value['sphere_finish_history_equal']
                for value in row['comparisons'].values())
            save()
        report['status'] = 'complete_actual_sphere_chain_pair'
        report['strict_reproduction'] = 'passed' if all(row['strict_reproduction_passed'] for row in report['rows']) else 'failed'
        report['optimization_regression'] = 'not_observed_in_this_pair' if report['strict_reproduction'] == 'passed' else 'difference_detected_requires_diagnosis'
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc()); traceback.print_exc()
    report['main_seconds'] = time.perf_counter() - tick; save()
    return 0 if report['status'] == 'complete_actual_sphere_chain_pair' else 1


if __name__ == '__main__':
    raise SystemExit(main())
