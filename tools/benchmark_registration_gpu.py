#!/usr/bin/env python3
"""Source-bound real BBR/FNIRT timing and offline FSL paired validation.

A server-local JSON supplies actual inputs and verified offline oracles. No
original executable is invoked. Cold/warm include complete function input reads;
independent full CUDA profiling never supplies performance-table time.
Raw outputs and input metadata remain private. Only report.safe.json is exported.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import nibabel as nib
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class Measure:
    """External clocks, operation counts and optional full CUDA-only CUPTI windows."""
    def __init__(self, torch, bbr, reg, output, full=False):
        self.torch, self.bbr, self.reg, self.output = torch, bbr, reg, output
        self.full = full
        self.times = defaultdict(lambda: dict(calls=0, wall_seconds=0.0, exclusive_wall_seconds=0.0))
        self.counts, self.api, self.events, self.cuda_us = Counter(), Counter(), Counter(), Counter()
        self.frames = []; self.profiler = None; self.windows = 0; self.work_units = 0

    @contextmanager
    def phase(self, name):
        frame = [name, time.perf_counter(), 0.0]; self.frames.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - frame[1]; self.frames.pop()
            record = self.times[name]; record['calls'] += 1; record['wall_seconds'] += elapsed
            record['exclusive_wall_seconds'] += elapsed - frame[2]
            if self.frames: self.frames[-1][2] += elapsed

    def wrap(self, obj, name, label=None, *, evaluation=False, work=False):
        original = getattr(obj, name)
        def call(*args, **kwargs):
            with self.phase(label or name):
                result = original(*args, **kwargs)
            if evaluation: self.counts[label or name] += 1
            if work: self.step()
            return result
        self.stack.enter_context(patch.object(obj, name, call))

    def step(self):
        self.work_units += 1
        if self.profiler: self.profiler.step()

    def collect(self, profiler):
        for event in profiler.events():
            name = event.name; category = None
            if event.device_type == self.torch.autograd.DeviceType.CUDA:
                low = name.lower()
                category = ('device_memcpy' if 'memcpy' in low else
                            'device_memset' if 'memset' in low else 'kernel')
                if 'htod' in low or 'h2d' in low: category = 'H2D'
                elif 'dtoh' in low or 'd2h' in low: category = 'D2H'
                elif 'dtod' in low or 'd2d' in low: category = 'D2D'
                self.events[category] += 1
                self.cuda_us[category] += event.device_time_total
            elif name.startswith(('cuda', 'cu')):
                self.events[name] += 1
        self.windows += 1
        progress = {**self.report(), 'elapsed_seconds': time.perf_counter() - self.started, 'complete': False}
        (self.output / 'profile.progress.safe.json').write_text(json.dumps(progress, indent=2) + '\n')
        print(json.dumps({'event': 'profile_progress', 'windows': self.windows,
                          'work_units': self.work_units, 'elapsed_seconds': progress['elapsed_seconds']}), flush=True)

    def __enter__(self):
        self.stack = ExitStack(); self.started = time.perf_counter()
        for obj, name, label in [(self.bbr, '_boundary', 'bbr_boundary_preparation'),
                                 (self.bbr, '_resample', 'bbr_final_resampling'),
                                 (self.reg, '_fsl_gaussian_blur', 'fnirt_smoothing'),
                                 (self.reg, '_force_jacobian_range', 'fnirt_topology')]:
            self.wrap(obj, name, label)
        if hasattr(self.reg, '_fsl_gaussian_blur_reference'):
            self.wrap(self.reg, '_fsl_gaussian_blur_reference', 'fnirt_smoothing_reference')
        self.wrap(self.reg._LevelSystem, 'evaluate', 'fnirt_cost_evaluation', evaluation=True)
        self.wrap(self.reg._LevelSystem, 'linearize', 'fnirt_linearization')
        self.wrap(self.reg._JointT1System, 'linearize', 'fnirt_joint_linearization')
        has_many = hasattr(self.bbr._BBRCost, 'evaluate')
        self.wrap(self.bbr._BBRCost, '__call__', 'bbr_scalar_cost_api' if has_many else 'bbr_cost_evaluation',
                  evaluation=not has_many, work=not has_many)
        if has_many:
            original_many = self.bbr._BBRCost.evaluate
            def many(instance, matrices, *args, **kwargs):
                with self.phase('bbr_batched_cost_evaluation'):
                    result = original_many(instance, matrices, *args, **kwargs)
                self.counts['bbr_batched_candidate_evaluations'] += 1 if np.ndim(matrices) == 2 else len(matrices)
                self.counts['bbr_batched_cost_calls'] += 1; self.step()
                return result
            self.stack.enter_context(patch.object(self.bbr._BBRCost, 'evaluate', many))
        # Aliases imported into registration, rather than unrelated call sites.
        for name in ['expand_coefficients', 'adjoint_field', 'design_diagonal']:
            self.wrap(self.reg, name, 'fnirt_' + name)
        for name in ['normal', 'energy', 'diagonal']:
            self.wrap(self.reg.BendingOperator, name, 'fnirt_bending_' + name)
        original = self.reg.preconditioned_conjugate_gradient
        def pcg(matvec, *args, **kwargs):
            def product(vector):
                with self.phase('fnirt_pcg_matvec'):
                    result = matvec(vector)
                self.counts['fnirt_pcg_matvec'] += 1; self.step()
                return result
            with self.phase('fnirt_pcg'):
                return original(product, *args, **kwargs)
        self.stack.enter_context(patch.object(self.reg, 'preconditioned_conjugate_gradient', pcg))
        if hasattr(self.bbr, 'minimize'):
            original_minimize = self.bbr.minimize
            def minimize(*args, **kwargs):
                step = 200 if self.counts['bbr_local_calls'] == 0 else 2
                self.counts['bbr_local_calls'] += 1
                with self.phase('bbr_local_step_' + str(step)):
                    return original_minimize(*args, **kwargs)
            self.stack.enter_context(patch.object(self.bbr, 'minimize', minimize))
        # The legacy module's single unqualified min is its 729-candidate coarse search.
        if not has_many and 'min' in self.bbr.register_bbr.__code__.co_names:
            import builtins
            def coarse(*args, **kwargs):
                with self.phase('bbr_coarse_search'):
                    return builtins.min(*args, **kwargs)
            self.stack.enter_context(patch.object(self.bbr, 'min', coarse, create=True))
        if self.full:
            for name in ['__float__', '__bool__', 'item', 'cpu', 'numpy']:
                original_api = getattr(self.torch.Tensor, name)
                def api(tensor, *args, _original=original_api, _name=name, **kwargs):
                    if tensor.device.type == 'cuda': self.api[_name] += 1
                    return _original(tensor, *args, **kwargs)
                self.stack.enter_context(patch.object(self.torch.Tensor, name, api))
            self.profiler = self.stack.enter_context(self.torch.profiler.profile(
                activities=[self.torch.profiler.ProfilerActivity.CUDA],
                schedule=self.torch.profiler.schedule(wait=0, warmup=0, active=64, repeat=0),
                on_trace_ready=self.collect, record_shapes=False, profile_memory=False))
        return self

    def __exit__(self, *exc):
        self.stack.__exit__(*exc)

    def report(self):
        return {'phase_timings': dict(self.times), 'operation_counts': dict(self.counts),
                'cuda_tensor_api_calls': dict(self.api), 'actual_cuda_event_counts': dict(self.events),
                'profiled_cuda_device_time_us': dict(self.cuda_us), 'profile_windows': self.windows,
                'profiler_scope': 'complete CUDA-only registration; window fences included' if self.full else 'not profiled',
                'phase_timing_scope': 'nested CPU wall clocks, no extra GPU fences; dispatched GPU work may finish in subsequent phases',
                'instrumentation_changes_timing': self.full}


def image_metrics(first, second, mask):
    x, y = np.asarray(first)[mask].astype(np.float64), np.asarray(second)[mask].astype(np.float64)
    delta = y - x
    return {'pearson_r': float(np.corrcoef(x, y)[0, 1]), 'mae': float(np.abs(delta).mean()),
            'rmse': float(np.sqrt(np.square(delta).mean()))}


def grid_checks(image, reference):
    return {'shape_equal': image.shape == reference.shape,
            'affine_equal': bool(np.array_equal(image.affine, reference.affine)),
            'pixdim_equal': bool(np.array_equal(image.header['pixdim'][1:4], reference.header['pixdim'][1:4]))}


def coefficient_checks(image, official):
    fields = ['intent_code', 'intent_p1', 'intent_p2', 'intent_p3',
              'qoffset_x', 'qoffset_y', 'qoffset_z', 'sform_code', 'qform_code']
    return {'shape_equal': image.shape == official.shape,
            'pixdim_equal': bool(np.array_equal(image.header['pixdim'][1:4], official.header['pixdim'][1:4])),
            'stored_affine_equal': bool(np.array_equal(image.get_sform(), official.get_sform())),
            **{field + '_equal': bool(np.array_equal(image.header[field], official.header[field])) for field in fields}}


def paired_bbr(result, config, epi, t1, wm, bbr, coordinates):
    official = nib.load(config['official_moved']); fixed = np.asarray(t1.dataobj)
    mask = fixed > 0; first = np.asarray(official.dataobj); second = np.asarray(result.moved.dataobj)
    official_matrix = np.loadtxt(config['official_matrix'])
    geometry = (epi.affine, t1.affine, epi.shape, t1.shape,
                epi.header.get_zooms()[:3], t1.header.get_zooms()[:3])
    worlds = [coordinates.flirt_to_world_affine(m, *geometry) for m in [official_matrix, result.matrix]]
    points = np.argwhere(mask)[::25]
    points = nib.affines.apply_affine(t1.affine, points)
    distances = np.linalg.norm(nib.affines.apply_affine(np.linalg.inv(worlds[0]), points) -
                               nib.affines.apply_affine(np.linalg.inv(worlds[1]), points), axis=1)
    grey, white, _ = bbr._boundary(wm, t1); cost = bbr._BBRCost(epi, grey, white, 'cuda:0')
    return {'image': image_metrics(first, second, mask),
            'inverse_fixed_brain_grid_displacement_mm': {'mean': float(distances.mean()), 'median': float(np.median(distances)),
                'p95': float(np.percentile(distances, 95)), 'rms': float(np.sqrt(np.square(distances).mean())), 'maximum': float(distances.max())},
            'fnit_initial_cost': result.initial_cost, 'fnit_final_cost': result.final_cost,
            'official_matrix_evaluated_with_fnit_cost': float(cost(official_matrix)),
            'grid_checks': {'reference_affine_matches': bool(np.array_equal(result.moved.affine, t1.affine)),
                            'official_grid_matches': bool(first.shape == second.shape and np.allclose(result.moved.affine, official.affine)),
                            'dtype_equal': result.moved.get_data_dtype() == official.get_data_dtype()},
            'boundary_points': result.boundary_points}


def paired_fnirt(result, config, reference, mask):
    official_image = nib.load(config['official_warped'])
    first = np.asarray(official_image.dataobj, dtype=np.float32)
    second = np.asarray(result.moved.dataobj, dtype=np.float32); brain = np.asarray(mask.dataobj) > 0.5
    a = first > 0.05 * np.percentile(first[brain & (first > 0)], 99)
    b = second > 0.05 * np.percentile(second[brain & (second > 0)], 99)
    coefficient = nib.load(config['official_coeff']); coefficients = np.asarray(coefficient.dataobj)
    equal_grid = coefficients.shape == np.asarray(result.coefficient_image.dataobj).shape
    contract = coefficient_checks(result.coefficient_image, coefficient)
    report = {'image': image_metrics(first, second, brain), 'support_dice': float(2 * (a & b).sum() / (a.sum() + b.sum())),
              'coefficient_shape_matches': equal_grid, 'coefficient_contract_checks': contract,
              'coefficient_grid_matches': all(contract.values()), 'coefficient_intent': int(result.coefficient_image.header['intent_code']),
              'physical_grid_checks': {'official_warped': grid_checks(official_image, reference),
                                       'fnit_warped': grid_checks(result.moved, reference), 'reference_mask': grid_checks(mask, reference)},
              'reference_grid_matches': bool(second.shape == reference.shape and np.array_equal(result.moved.affine, reference.affine)),
              'jacobian': {'minimum': float(np.asarray(result.full_pull_jacobian.dataobj).min()),
                           'maximum': float(np.asarray(result.full_pull_jacobian.dataobj).max()),
                           'nonpositive_fraction': float((np.asarray(result.full_pull_jacobian.dataobj) <= 0).mean())}}
    if equal_grid:
        report['coefficients'] = {**image_metrics(coefficients, np.asarray(result.coefficient_image.dataobj), np.ones(coefficients.shape, bool)),
                                  'metadata_contract_matches': all(contract.values())}
    if 'official_jacobian' in config:
        official_jacobian = nib.load(config['official_jacobian'])
        report['nonlinear_jacobian_paired'] = {**image_metrics(np.asarray(official_jacobian.dataobj),
                                                               np.asarray(result.nonlinear_jacobian.dataobj), brain),
                                              'official_grid_checks': grid_checks(official_jacobian, reference)}
    if all(k in config for k in ['official_pull_x', 'official_pull_y', 'official_pull_z']):
        grid = np.indices(reference.shape, dtype=np.float32).reshape(3, -1)
        world = (reference.affine[:3, :3].astype(np.float32) @ grid + reference.affine[:3, 3:4].astype(np.float32)).T.reshape((*reference.shape, 3))
        official_images = [nib.load(config['official_pull_' + k]) for k in 'xyz']
        report['physical_grid_checks']['official_pull_xyz'] = [grid_checks(image, reference) for image in official_images]
        official_world = np.stack([np.asarray(image.dataobj) for image in official_images], -1)
        distances = np.linalg.norm(world + np.asarray(result.pull_transform.dataobj) - official_world, axis=-1)[brain & a & b]
        report['pull_distance_mm'] = {'median': float(np.median(distances)), 'p95': float(np.percentile(distances, 95)),
                                      'mean': float(distances.mean()), 'compared_voxels': int(distances.size),
                                      'selection': 'reference mask and positive support in both warped images'}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-json', required=True); parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-dir', required=True); parser.add_argument('--function', choices=['bbr', 'bbr_chain', 'fnirt'], required=True)
    parser.add_argument('--profile-full', action='store_true'); parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--warm-repeats', type=int, default=1)
    parser.add_argument('--fnirt-blur-reference', action='store_true')
    parser.add_argument('--affine-geometry', choices=['affine_norm', 'header_pixdim'], default='affine_norm')
    parser.add_argument('--wm-header', choices=['legacy', 'reference'], default='legacy')
    parser.add_argument('--fnirt-execution', choices=['default', 'reference', 'optimized'], default='default')
    parser.add_argument('--fnirt-bending-reference', action='store_true', help='Diagnostic dense normal with otherwise optimized execution')
    parser.add_argument('--bbr-execution', choices=['default', 'reference', 'batched'], default='default')
    args = parser.parse_args()
    source, output = Path(args.source_root), Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source / 'src'))
    import torch
    torch.cuda.set_device(0); torch.cuda.init(); torch.set_num_threads(4)
    from fnit.fmri import bbr
    from fnit.fnirt import registration as reg
    from fnit.flirt import coordinates
    if args.fnirt_blur_reference:
        reg._fsl_gaussian_blur = reg._fsl_gaussian_blur_reference
    if args.fnirt_bending_reference:
        def dense_normal(operator, coefficients):
            return operator.adjoint(operator.forward(coefficients))
        reg.BendingOperator.normal = dense_normal
    support = module(Path(__file__).with_name('benchmark_flirt_gpu.py'), 'flirt_benchmark_support')
    config = json.loads(Path(args.case_json).read_text())
    if args.function.startswith('bbr'):
        moving, fixed, wm = [nib.load(config[k]) for k in ['epi', 't1', 'wmseg']]
        images = [moving, fixed, wm]; initial = np.loadtxt(config['init'])
    else:
        moving, fixed, mask = [nib.load(config[k]) for k in ['moving', 'reference', 'reference_mask']]
        images = [moving, fixed, mask]
        sizes = ((moving.header.get_zooms()[:3], fixed.header.get_zooms()[:3]) if args.affine_geometry == 'header_pixdim'
                 else (nib.affines.voxel_sizes(moving.affine), nib.affines.voxel_sizes(fixed.affine)))
        initial = coordinates.flirt_to_world_affine(np.loadtxt(config['affine']), moving.affine, fixed.affine, moving.shape, fixed.shape, *sizes)
    for image in images: image.get_fdata(dtype=np.float32)
    def invoke(measure):
        bbr_parameters = {} if args.bbr_execution == 'default' else {'execution': args.bbr_execution}
        if args.function == 'bbr_chain':
            from fnit.fast import TorchFAST
            with measure.phase('anatomical_fast'):
                segmentation = TorchFAST(device='cuda:0')(fixed)
            values = (np.asarray(segmentation.pve_wm.dataobj) >= 0.5)
            if args.wm_header == 'reference':
                header = fixed.header.copy(); header.set_data_dtype(np.uint8); header.set_slope_inter(1.0, 0.0)
                wm_now = nib.Nifti1Image(values.astype(np.uint8), fixed.affine, header)
            else:
                wm_now = nib.Nifti1Image(values.astype(np.float32), fixed.affine)
            from fnit.flirt import TorchFLIRT
            with measure.phase('bbr_initial_flirt'):
                init = TorchFLIRT(device='cuda:0', dof=6, cost='normmi')(moving, fixed).matrix
            with measure.phase('bbr_refinement'):
                return bbr.register_bbr(moving, fixed, wm_now, init=init, device='cuda:0', **bbr_parameters), wm_now
        if args.function == 'bbr':
            return bbr.register_bbr(moving, fixed, wm, init=initial, device='cuda:0', **bbr_parameters), wm
        parameters = {} if args.fnirt_execution == 'default' else {'execution': args.fnirt_execution}
        return reg.TorchFNIRT(device='cuda:0', config=reg.T1FNIRTConfig(), **parameters)(moving, fixed, initial, reference_mask=mask), None
    files = [*(source / 'src/fnit/fnirt').glob('*.py'), source / 'src/fnit/fmri/bbr.py', source / 'src/fnit/fmri/normalization.py',
             *(source / 'src/fnit/flirt').glob('*.py'), source / 'src/fnit/_nib.py', source / 'src/fnit/_transforms.py']
    if args.function == 'bbr_chain': files.extend((source / 'src/fnit/fast').glob('*.py'))
    if (source / 'src/fnit/fmri/_bbr_cuda.py').is_file(): files.append(source / 'src/fnit/fmri/_bbr_cuda.py')
    report = {'schema_version': 1, 'function': args.function, 'source_sha256': {str(p.relative_to(source)): sha(p) for p in files},
              'benchmark_tool_sha256': sha(__file__), 'support_tool_sha256': sha(Path(__file__).with_name('benchmark_flirt_gpu.py')),
              'environment': {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'gpu_name': torch.cuda.get_device_properties(0).name},
              'timing_scope': 'complete function including ArrayProxy reads/decompression and CPU output conversion; excludes output persistence and offline metrics; shared GPU',
              'scientific_conditions': {'fnirt_config': 't1' if args.function == 'fnirt' else None, 'same_fsl_initial_affine': args.function != 'bbr_chain',
                                        'same_fsl_wmseg': args.function == 'bbr', 'fnirt_execution': args.fnirt_execution, 'bbr_execution': args.bbr_execution,
                                        'input_arrayproxy_read_included': True, 'wm_header': args.wm_header, 'affine_geometry': args.affine_geometry,
                                        'fnirt_blur_reference': args.fnirt_blur_reference, 'fnirt_bending_reference': args.fnirt_bending_reference,
                                        'half_precision': False, 'default_tf32': True}, 'runs': []}
    iterations = ([] if args.profile_only else [False] * (args.warm_repeats + 1)) + ([True] if args.profile_full else [])
    for index, full in enumerate(iterations):
        torch.cuda.synchronize(0); torch.cuda.reset_peak_memory_stats(0)
        with support.GPUMonitor(torch, torch.device('cuda:0')) as monitor:
            started = time.perf_counter()
            with Measure(torch, bbr, reg, output, full=full) as measure:
                result, used_wm = invoke(measure); torch.cuda.synchronize(0)
            elapsed = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated(0)
        peak_reserved = torch.cuda.max_memory_reserved(0)
        tag = 'profiled' if full else ('cold_0' if index == 0 else 'warm_' + str(index))
        # Persist after clock/profiler closes. Nothing voxelwise enters the safe report.
        nib.save(result.moved, output / (tag + '.nii.gz'))
        if args.function.startswith('bbr'): np.savetxt(output / (tag + '.mat'), result.matrix, fmt='%.12g')
        else:
            for suffix, image in [('coeff', result.coefficient_image), ('pull', result.pull_transform),
                                  ('jacobian', result.full_pull_jacobian), ('nonlinear_jacobian', result.nonlinear_jacobian)]:
                nib.save(image, output / (tag + '_' + suffix + '.nii.gz'))
            (output / (tag + '.qc.private.json')).write_text(json.dumps(result.qc, indent=2, default=str))
        paired = paired_bbr(result, config, moving, fixed, used_wm, bbr, coordinates) if args.function.startswith('bbr') else paired_fnirt(result, config, fixed, mask)
        record = {'condition': tag, 'wall_seconds': elapsed, 'peak_allocated_bytes': peak_allocated,
                  'peak_reserved_bytes': peak_reserved, 'process_peak_cpu_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'whole_gpu_observation': monitor.report(), 'measurement': measure.report(), 'paired': paired}
        if args.function.startswith('bbr') and hasattr(result, 'phase_timings'):
            record['source_phase_timings'] = result.phase_timings
            record['source_cost_evaluations'] = result.cost_evaluations
            record['source_phase_cost_evaluations'] = result.phase_cost_evaluations
        if full:
            report['profile'] = record
        else:
            report['runs'].append(record)
        (output / 'report.safe.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'event': 'run_completed', 'function': args.function, 'condition': tag, 'wall_seconds': elapsed,
                          'pearson_r': paired['image']['pearson_r'], 'peak_allocated_bytes': record['peak_allocated_bytes']}), flush=True)


if __name__ == '__main__':
    main()
