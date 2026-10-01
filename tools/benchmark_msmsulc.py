#!/usr/bin/env python3
"""Source-bound newMSM paired timing and precision on server-local real inputs.

No official executable is invoked. The private JSON supplies identical prepared
MSMSulcInputs, saved official spheres and optional projection inputs/CIFTI oracle.
Cold and warm registrations use the full schedule; CUDA profiling is a separate
registration. Only report.safe.json contains exportable scalar summaries. Raw
spheres, time series and input paths remain in the private output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import nibabel as nib
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_support():
    path = Path(__file__).with_name('benchmark_flirt_gpu.py')
    spec = importlib.util.spec_from_file_location('msmsulc_benchmark_support', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sphere(path):
    image = nib.load(str(path))
    points = [array for array in image.darrays if array.intent == 1008]
    faces = [array for array in image.darrays if array.intent == 1009]
    if len(points) != 1 or len(faces) != 1:
        raise ValueError('sphere requires one pointset and one triangle array')
    vertices = np.asarray(points[0].data, np.float64)
    triangles = np.asarray(faces[0].data, np.int64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or
            triangles.ndim != 2 or triangles.shape[1] != 3 or
            not len(triangles) or triangles.min() < 0 or
            triangles.max() >= len(vertices) or not np.isfinite(vertices).all() or
            np.any(np.linalg.norm(vertices, axis=1) == 0)):
        raise ValueError('sphere has invalid vertices or triangles')
    return vertices, triangles


def summary(values):
    values = np.asarray(values, np.float64)
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'maximum': float(values.max())}


def sphere_metrics(actual_file, official_file, initial_file):
    actual, faces = sphere(actual_file)
    official, official_faces = sphere(official_file)
    initial, initial_faces = sphere(initial_file)
    if (actual.shape != official.shape or actual.shape != initial.shape or
            not np.array_equal(faces, official_faces) or
            not np.array_equal(faces, initial_faces)):
        raise ValueError('paired spheres must share native vertex order and topology')
    a = actual / np.linalg.norm(actual, axis=1, keepdims=True)
    b = official / np.linalg.norm(official, axis=1, keepdims=True)
    angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1),
                                  np.einsum('ij,ij->i', a, b)))
    def signs(vertices):
        triangle = vertices[faces]
        return np.einsum('ij,ij->i', np.cross(triangle[:, 1]-triangle[:, 0],
                                             triangle[:, 2]-triangle[:, 0]), triangle[:, 0])
    sign_initial = signs(initial)
    if np.any(sign_initial == 0):
        raise ValueError('initial sphere has degenerate triangles')
    return {'native_topology_exact': True, 'vertices': len(actual), 'faces': len(faces),
            'sphere_angle_deg': summary(angle),
            'chord_difference_mm': summary(np.linalg.norm(actual-official, axis=1)),
            'radius_mm': summary(np.linalg.norm(actual, axis=1)),
            'fnit_folded_faces': int(np.count_nonzero(signs(actual)*sign_initial <= 0)),
            'official_folded_faces': int(np.count_nonzero(signs(official)*sign_initial <= 0))}


def cifti_metrics(actual_file, official_file, chunk_size=4096):
    actual = nib.load(str(actual_file)); official = nib.load(str(official_file))
    if not isinstance(actual, nib.Cifti2Image) or not isinstance(official, nib.Cifti2Image):
        raise ValueError('paired functional outputs must be CIFTI files')
    if actual.shape != official.shape or actual.ndim != 2:
        raise ValueError('paired CIFTIs must share a frame/grayordinate shape')
    time_axis, axis = actual.header.get_axis(0), actual.header.get_axis(1)
    if time_axis != official.header.get_axis(0) or axis != official.header.get_axis(1):
        raise ValueError('paired CIFTI time and brain-model axes differ')
    result = {'frames': actual.shape[0], 'grayordinates': actual.shape[1],
              'time_axis_exact': True, 'brain_model_axis_exact': True,
              'correlation_definition': 'Pearson across time for each valid grayordinate, then arithmetic mean',
              'structures': {}}
    for name, selection, _ in axis.iter_structures():
        start, stop, step = selection.indices(actual.shape[1])
        if step != 1: raise ValueError('unexpected noncontiguous brain model')
        correlations = []; absolute_sum = 0.0; sample_count = 0; maximum = 0.0
        for offset in range(start, stop, chunk_size):
            end = min(offset+chunk_size, stop)
            x = np.asarray(actual.dataobj[:, offset:end], np.float64)
            y = np.asarray(official.dataobj[:, offset:end], np.float64)
            finite = np.isfinite(x).all(axis=0) & np.isfinite(y).all(axis=0)
            if not finite.all(): raise ValueError('nonfinite CIFTI time series')
            difference = np.abs(x-y)
            absolute_sum += float(difference.sum()); sample_count += difference.size
            maximum = max(maximum, float(difference.max()))
            x -= x.mean(axis=0); y -= y.mean(axis=0)
            xx = np.einsum('ij,ij->j', x, x); yy = np.einsum('ij,ij->j', y, y)
            valid = (xx > 0) & (yy > 0)
            correlations.append(np.einsum('ij,ij->j', x, y)[valid]/np.sqrt(xx[valid]*yy[valid]))
        values = np.concatenate(correlations)
        result['structures'][str(name)] = {
            'grayordinates': stop-start, 'valid_nonconstant_grayordinates': len(values),
            'mean_temporal_r': float(values.mean()) if len(values) else None,
            'median_temporal_r': float(np.median(values)) if len(values) else None,
            'mean_absolute_difference': absolute_sum/sample_count,
            'maximum_absolute_difference': maximum}
    return result


class Measure:
    """Profile without changing optimizer arguments, candidates, or acceptance."""
    def __init__(self, torch, msm, native, output, full=False):
        self.torch, self.msm, self.native, self.output = torch, msm, native, output
        self.full = full; self.frames = []; self.profiler = None
        self.times = defaultdict(lambda: {'calls': 0, 'wall_seconds': 0.0, 'exclusive_wall_seconds': 0.0})
        self.counts = Counter(); self.api = Counter(); self.events = Counter(); self.cuda_us = Counter()
        self.windows = 0; self.work_units = 0
        self.peak_allocated_before_source_reset = 0
        self.peak_reserved_before_source_reset = 0

    @contextmanager
    def phase(self, name):
        frame = [time.perf_counter(), 0.0]; self.frames.append(frame)
        try: yield
        finally:
            elapsed = time.perf_counter()-frame[0]; self.frames.pop()
            record = self.times[name]; record['calls'] += 1; record['wall_seconds'] += elapsed
            record['exclusive_wall_seconds'] += elapsed-frame[1]
            if self.frames: self.frames[-1][1] += elapsed

    def wrap(self, obj, name, label, work=False):
        if not hasattr(obj, name): return
        original = getattr(obj, name)
        @wraps(original)
        def wrapped(*args, **kwargs):
            with self.phase(label): result = original(*args, **kwargs)
            self.counts[label] += 1
            if label == 'face_costs':
                faces = args[3] if len(args) > 3 else kwargs['faces']
                self.counts['face_configuration_evaluations'] += len(faces)*(1 if kwargs.get('energy_only') else 8)
            if label == 'affine_source_wls':
                # The packed CPU buffer is already transferred by production.
                # Observing its byte size adds no CUDA operation or transfer.
                self.counts['affine_wls_host_payload_bytes'] += memoryview(args[0]).nbytes
                self.counts['affine_wls_query_slots'] += args[1]*args[2]
            if label == 'source_rotation_matrices':
                # The producer reuses its existing host prior-coordinate
                # copy. Buffer metadata observes no extra GPU operation.
                self.counts['control_rotation_points'] += args[2]
                self.counts['control_rotation_host_input_bytes'] += (
                    memoryview(args[0]).nbytes+memoryview(args[1]).nbytes)
                self.counts['control_rotation_host_result_bytes'] += len(result)
            if work:
                self.work_units += 1
                if self.profiler: self.profiler.step()
            return result
        self.stack.enter_context(patch.object(obj, name, wrapped))

    def collect(self, profiler):
        for event in profiler.events():
            name = event.name
            if event.device_type == self.torch.autograd.DeviceType.CUDA:
                low = name.lower()
                kind = 'device_memcpy' if 'memcpy' in low else ('device_memset' if 'memset' in low else 'kernel')
                if 'htod' in low or 'h2d' in low: kind = 'H2D'
                elif 'dtoh' in low or 'd2h' in low: kind = 'D2H'
                elif 'dtod' in low or 'd2d' in low: kind = 'D2D'
                self.events[kind] += 1; self.cuda_us[kind] += event.device_time_total
            elif name.startswith(('cuda', 'cu')): self.events[name] += 1
        self.windows += 1
        (self.output/'profile.progress.safe.json').write_text(json.dumps(self.report(), indent=2)+'\n')
        print(json.dumps({'event': 'profile_progress', 'windows': self.windows,
                          'work_units': self.work_units}), flush=True)

    def __enter__(self):
        self.stack = ExitStack()
        if self.torch.cuda.is_available():
            original_reset = self.torch.cuda.reset_peak_memory_stats
            def reset(device=None):
                self.peak_allocated_before_source_reset = max(
                    self.peak_allocated_before_source_reset,
                    self.torch.cuda.max_memory_allocated(device))
                self.peak_reserved_before_source_reset = max(
                    self.peak_reserved_before_source_reset,
                    self.torch.cuda.max_memory_reserved(device))
                return original_reset(device)
            self.stack.enter_context(patch.object(self.torch.cuda, 'reset_peak_memory_stats', reset))
        for name, label in [('_affine_initialization', 'affine_initialization'),
                            ('_adaptive_resample', 'adaptive_metric_resampling'),
                            ('_face_costs', 'face_costs'), ('_rotated_label', 'label_rotation'),
                            ('_rotation_matrices', 'label_rotation_preparation'),
                            ('_face_layout', 'face_layout'),
                            ('_unfold', 'topology_unfolding'), ('_sphere_warp', 'progressive_sphere_warp'),
                            ('_variance_normalize', 'variance_normalization')]:
            self.wrap(self.msm, name, label)
        if hasattr(self.msm, 'RadialSphereMap'):
            self.wrap(self.msm.RadialSphereMap, '__init__', 'radialspheremap_preparation')
            self.wrap(self.msm.RadialSphereMap, 'weights', 'radialspheremap_weights')
        affine_module = sys.modules.get(getattr(getattr(self.msm, '_affine_initialization', None), '__module__', ''))
        if affine_module is not None and hasattr(affine_module, '_RigidCost'):
            self.wrap(affine_module._RigidCost, 'evaluate_positions', 'rigid_cost_evaluation')
        self.wrap(self.native, 'optimize', 'hocr_fastpd', work=True)
        self.wrap(self.native, 'source_wls_cost', 'affine_source_wls')
        self.wrap(self.native, 'source_rotation_matrices', 'source_rotation_matrices')
        if self.full:
            for name in ['__float__', '__int__', '__bool__', 'item', 'cpu', 'numpy']:
                original = getattr(self.torch.Tensor, name)
                def api(tensor, *args, _original=original, _name=name, **kwargs):
                    if tensor.device.type == 'cuda': self.api[_name] += 1
                    return _original(tensor, *args, **kwargs)
                self.stack.enter_context(patch.object(self.torch.Tensor, name, api))
            self.profiler = self.stack.enter_context(self.torch.profiler.profile(
                activities=[self.torch.profiler.ProfilerActivity.CUDA],
                schedule=self.torch.profiler.schedule(wait=0, warmup=0, active=128, repeat=0),
                on_trace_ready=self.collect, record_shapes=False, profile_memory=False))
        return self

    def __exit__(self, *exc): self.stack.__exit__(*exc)

    def report(self):
        return {'phase_timings': dict(self.times), 'operation_counts': dict(self.counts),
                'cuda_tensor_api_calls': dict(self.api), 'actual_cuda_event_counts': dict(self.events),
                'profiled_cuda_device_time_us': dict(self.cuda_us), 'profile_windows': self.windows,
                'profiler_scope': 'complete CUDA registration, window fences included' if self.full else 'not profiled',
                'phase_timing_scope': 'nested host wall clocks without additional GPU fences; not isolated GPU kernel duration',
                'instrumentation_changes_timing': self.full}


def safe_registration_report(report):
    allowed = {'seconds', 'affine_angles_deg',
               'peak_allocated_gb', 'control_points', 'data_points', 'labels',
               'iterations', 'stages', 'changed', 'affine_seconds', 'affine',
               'folded_output_faces', 'energy', 'applied', 'converged', 'similarity',
               'folded_solver_faces', 'minimum_output_orientation_ratio',
               'minimum_solver_orientation_ratio', 'degenerate_input_faces',
               'maximum_iterations', 'source_unfold_updates', 'control_unfold_updates',
               'cost_evaluations', 'initial_similarity', 'final_best_similarity',
               'constant_similarity'}
    def scalars(value):
        if value is None or isinstance(value, (bool, int, float)): return value
        if isinstance(value, list): return [scalars(item) for item in value]
        if isinstance(value, dict):
            return {key: scalars(item) for key, item in value.items() if key in allowed}
        return None
    return {hemi: scalars(report[hemi]) for hemi in 'LR'}


def project(case, spheres, directory):
    """Replay the pipeline's sphere-dependent area-surface preparation."""
    from fnit.fmri.surface import SurfaceHemisphere
    from fnit.fmri.surface_fmriprep import run_fmriprep_surface_projection
    parameters = case['projection'].copy()
    parameters.pop('official_cifti', None)
    directory = Path(directory)
    area_directory = directory/'area_surfaces'
    area_directory.mkdir(parents=True, exist_ok=True)
    executable = shutil.which(str(parameters.get('wb_command', 'wb_command')))
    if executable is None:
        raise FileNotFoundError('Connectome Workbench is required for projection')
    for name, hemi in [('left', 'L'), ('right', 'R')]:
        geometry = SurfaceHemisphere(**parameters[name])
        atlas_mid = area_directory/(hemi+'.midthickness.32k_fsLR.surf.gii')
        subprocess.run([
            executable, '-surface-resample', str(geometry.midthickness),
            str(spheres[hemi]), str(geometry.atlas_sphere), 'BARYCENTRIC', str(atlas_mid),
        ], check=True, capture_output=True, text=True)
        parameters[name] = replace(geometry, registered_sphere=spheres[hemi],
                                   atlas_midthickness=atlas_mid)
    return run_fmriprep_surface_projection(**parameters, output_dir=directory)


def registration_options(case):
    """Interpret private configuration paths without placing them in a report."""
    options = dict(case.get('run_options', {}))
    if 'config' in options or 'device' in options:
        raise ValueError('use config_file/config_options and the device CLI option')
    requested = 'config_file' in case or 'config_options' in case
    if 'config_file' in case and 'config_options' in case:
        raise ValueError('supply only one of config_file and config_options')
    from fnit.msm.config import MSMSulcConfig
    if 'config_file' in case:
        configuration = MSMSulcConfig.from_file(case['config_file'])
    else:
        configuration = MSMSulcConfig(**case.get('config_options', {}))
    options['config'] = configuration
    display = {'requested': requested, 'supported_by_measured_source': True,
               'applied': True, 'values': configuration.to_dict()}
    return options, display


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-json', required=True); parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-dir', required=True); parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warm-repeats', type=int, default=1)
    parser.add_argument('--profile-full', action='store_true'); parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args()
    if args.warm_repeats < 0: parser.error('--warm-repeats must be nonnegative')
    if args.profile_only and not args.profile_full: parser.error('--profile-only requires --profile-full')
    source = Path(args.source_root).resolve(); output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False); sys.path.insert(0, str(source/'src'))
    import torch
    from fnit.msm import MSMSulcInputs, _fastpd_native
    from fnit.msm import msmsulc
    selected = torch.device(args.device)
    if selected.type == 'cuda': torch.cuda.set_device(selected); torch.cuda.init()
    torch.set_num_threads(4)
    case = json.loads(Path(args.case_json).read_text())
    inputs = {hemi: MSMSulcInputs(**{key: Path(value) for key, value in case['inputs'][hemi].items()}) for hemi in 'LR'}
    for entry in inputs.values():
        for name in entry.__dataclass_fields__:
            if not getattr(entry, name).is_file(): raise FileNotFoundError('missing prepared MSMSulc input')
    options, configuration = registration_options(case)
    files = sorted((source/'src/fnit/msm').glob('*.py')) + sorted((source/'src/fnit/msm/_fastpd_src').glob('*'))
    report = {'schema_version': 1, 'function': 'run_msmsulc',
              'source_sha256': {str(path.relative_to(source)): sha256(path) for path in files if path.is_file()},
              'benchmark_tool_sha256': sha256(__file__),
              'native_extension_sha256': sha256(_fastpd_native.__file__),
              'environment': {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                              'device': selected.type, 'cpu_threads': torch.get_num_threads(),
                              'gpu_name': torch.cuda.get_device_properties(selected).name if selected.type == 'cuda' else None},
              'timing_scope': 'complete paired registration including input reads and output sphere/report writes; excludes offline metrics and projection',
              'cold_definition': 'first full call after Python import and CUDA context initialization in this process',
              'configuration': configuration,
              'runs': []}
    official_projection = None
    if 'projection' in case and not args.profile_only:
        began = time.perf_counter()
        projected = project(case, case['official_spheres'], output/'official_projection')
        official_projection = projected.dtseries
        report['projection_reference'] = {
            'regenerated_from_official_spheres': True,
            'sphere_specific_barycentric_area_surfaces': True,
            'wall_seconds_including_area_surfaces': time.perf_counter()-began}
        existing = case['projection'].get('official_cifti')
        if existing:
            report['projection_reference']['existing_offline_oracle_comparison'] = cifti_metrics(
                official_projection, existing)
    calls = ([] if args.profile_only else [False]*(args.warm_repeats+1))+([True] if args.profile_full else [])
    for index, full in enumerate(calls):
        condition = 'profiled' if full else ('cold_0' if index == 0 else 'warm_'+str(index))
        directory = output/condition; directory.mkdir()
        if selected.type == 'cuda': torch.cuda.synchronize(selected); torch.cuda.reset_peak_memory_stats(selected)
        with load_support().GPUMonitor(torch, selected) as monitor:
            started = time.perf_counter()
            with Measure(torch, msmsulc, _fastpd_native, directory, full=full) as measure:
                spheres = msmsulc.run_msmsulc(inputs, directory/'spheres', device=args.device, **options)
                if selected.type == 'cuda': torch.cuda.synchronize(selected)
            elapsed = time.perf_counter()-started
        source_report = json.loads((directory/'spheres/registration_report.json').read_text())
        paired = {hemi: sphere_metrics(spheres[hemi], case['official_spheres'][hemi], inputs[hemi].rotated_sphere) for hemi in 'LR'}
        record = {'condition': condition, 'wall_seconds': elapsed,
                  'process_peak_cpu_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'measurement': measure.report(), 'whole_gpu_observation': monitor.report(),
                  'sphere_paired': paired, 'source_registration_report': safe_registration_report(source_report)}
        if selected.type == 'cuda':
            # Production may reset peak counters per hemisphere. Preserve its
            # two per-hemisphere peaks rather than calling the last reset global.
            record['peak_after_last_source_reset_bytes'] = torch.cuda.max_memory_allocated(selected)
            record['reserved_after_last_source_reset_bytes'] = torch.cuda.max_memory_reserved(selected)
            record['peak_allocated_bytes'] = max(record['peak_after_last_source_reset_bytes'],
                                                 measure.peak_allocated_before_source_reset)
            record['peak_reserved_bytes'] = max(record['reserved_after_last_source_reset_bytes'],
                                                measure.peak_reserved_before_source_reset)
        if 'projection' in case and not full:
            began = time.perf_counter(); projected = project(case, spheres, directory/'projection')
            record['projection_wall_seconds'] = time.perf_counter()-began
            record['cifti_paired'] = cifti_metrics(projected.dtseries, official_projection)
            record['projection_area_surfaces_recomputed_for_registered_spheres'] = True
        if full: report['profile'] = record
        else: report['runs'].append(record)
        (output/'report.safe.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'event': 'run_completed', 'condition': condition, 'wall_seconds': elapsed,
                          'L_median_angle_deg': paired['L']['sphere_angle_deg']['median'],
                          'R_median_angle_deg': paired['R']['sphere_angle_deg']['median']}), flush=True)


if __name__ == '__main__': main()
