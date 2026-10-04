#!/usr/bin/env python3
"""只读比较同例完整 MNI/CIFTI 终点；不运行 MRI、不拟合、不取轴交集。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import time
from types import SimpleNamespace

import nibabel as nib
from nibabel.cifti2.cifti2_axes import BrainModelAxis, SeriesAxis
import numpy as np


class ValidationError(ValueError):
    def __init__(self, code, details=None):
        self.code, self.details = code, details or {}
        super().__init__(code)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValidationError('json_object_required')
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def public_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', value):
        raise ValidationError('public_identifier_required')
    return value


def checked(entry, role, files):
    path = Path(entry['path']).expanduser().resolve(strict=True)
    digest = sha256(path)
    if not path.is_file() or digest != entry['sha256']:
        raise ValidationError('input_sha256_mismatch', {'role': role})
    files[role] = (path, digest)
    return path


def metric_module(entry, files):
    path = checked(entry, 'metric_helper', files)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('paired_endpoint_metric_definition', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('stream_statistics', 'distribution', 'brain_axes_equal', 'volume_grid',
                 'ras_orientation', 'fixed_brain_axis', 'STRUCTURES', 'SUBCORTEX'):
        if not hasattr(module, name):
            raise ValidationError('metric_helper_interface_mismatch', {'missing': name})
    return module


def get_field(value, dotted):
    for name in dotted.split('.'):
        value = value[name]
    return value


def run_binding(spec, role, case, raw_hashes, frames, tr, files):
    public_id(spec['label'])
    report_path = checked(spec['report'], role + '/report', files)
    report = load_json(report_path)
    fields = spec.get('report_fields', {})
    status = get_field(report, fields.get('status', 'status'))
    if status not in ('complete', 'scientific_complete'):
        raise ValidationError('run_not_scientifically_complete', {'role': role, 'status': str(status)[:80]})
    # Field names are explicit because the two original wrappers have different schemas.
    expected = {'case_id': case, 'frames': frames, 'tr_seconds': tr, 'raw_hashes': raw_hashes}
    for key, field in fields.items():
        if key not in expected:
            continue
        try:
            actual = get_field(report, field)
        except (KeyError, TypeError):
            raise ValidationError('run_identity_field_missing', {'role': role, 'field': key}) from None
        if key == 'raw_hashes':
            if not isinstance(actual, dict):
                raise ValidationError('run_raw_hash_mapping_required', {'role': role})
            missing = [name for name in ('t1w', 'bold') if name not in actual]
            if missing:
                raise ValidationError('run_raw_mri_hash_missing', {'role': role, 'missing': missing})
            # DeepPrep records two MRI hashes plus two JSON hashes; the original
            # fMRIPrep wrapper records the two MRI hashes. Pair MRI identities
            # explicitly; all additional bound raw files retain before/after SHA guards.
            actual = {name: actual[name] for name in ('t1w', 'bold')}
        equal = (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                 and np.isclose(actual, tr, rtol=1e-6, atol=1e-7)) if key == 'tr_seconds' else actual == expected[key]
        if not equal:
            raise ValidationError('run_identity_mismatch', {'role': role, 'field': key})
    if not all(key in fields for key in ('case_id', 'frames', 'raw_hashes')):
        raise ValidationError('explicit_run_identity_fields_required', {'role': role})
    timings = []
    for record in spec.get('performance', {}).get('timings', []):
        name = public_id(record['name'])
        seconds = record['seconds']
        if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                                    or not np.isfinite(seconds) or seconds < 0):
            raise ValidationError('invalid_reported_timing', {'role': role, 'name': name})
        if record.get('report_field') and seconds != get_field(report, record['report_field']):
            raise ValidationError('timing_report_binding_mismatch', {'role': role, 'name': name})
        boundary = public_id(record['boundary'])
        scope = public_id(record['scope'])
        timings.append({'name': name, 'seconds': seconds, 'boundary': boundary, 'scope': scope,
                        'report_field': record.get('report_field'), 'source_report_sha256': files[role + '/report'][1]})
    outputs = {key: checked(spec['outputs'][key], role + '/' + key, files)
               for key in ('preproc_mni', 'dtseries')}
    return outputs, {'label': spec['label'], 'status': status,
                     'report_sha256': files[role + '/report'][1], 'timings': timings}


def valid_nifti(image, frames, tr, role):
    if image.ndim != 4 or image.shape[3] != frames:
        raise ValidationError('full_frame_count_mismatch', {'role': role, 'observed_shape': list(image.shape), 'expected_frames': frames})
    xyz, temporal = image.header.get_xyzt_units()
    scale = {'sec': 1., 'msec': .001, 'usec': .000001}.get(temporal)
    if (xyz != 'mm' or scale is None or not np.isclose(image.header.get_zooms()[3] * scale, tr, rtol=1e-6, atol=1e-7)):
        raise ValidationError('nifti_units_or_tr_mismatch', {'role': role, 'space_unit': xyz, 'time_unit': temporal})


def check_brain_axis(brain, metric):
    structures = [str(name) for name, _, _ in brain.iter_structures()]
    expected = ['CIFTI_STRUCTURE_' + name for name in ('CORTEX_LEFT', 'CORTEX_RIGHT', *metric.SUBCORTEX)]
    if structures != expected:
        raise ValidationError('cifti_structure_order_mismatch', {'expected': expected, 'observed': structures})
    if (brain.nvertices != {'CIFTI_STRUCTURE_CORTEX_LEFT': 32492, 'CIFTI_STRUCTURE_CORTEX_RIGHT': 32492}
            or len(brain) != 91282):
        raise ValidationError('cifti_91k_model_count_mismatch')
    cortex_counts = {}
    for name, _, model in brain.iter_structures():
        name = str(name)
        if name.startswith('CIFTI_STRUCTURE_CORTEX_'):
            vertices = model.vertex
            if np.any(vertices < 0) or np.any(vertices >= 32492) or len(np.unique(vertices)) != len(vertices):
                raise ValidationError('cifti_invalid_cortical_indices', {'structure': name})
            cortex_counts[name] = len(model)
        elif np.any(model.voxel < 0) or len(np.unique(model.voxel, axis=0)) != len(model):
            raise ValidationError('cifti_invalid_voxel_indices', {'structure': name})
    if cortex_counts != {'CIFTI_STRUCTURE_CORTEX_LEFT': 29696, 'CIFTI_STRUCTURE_CORTEX_RIGHT': 29716}:
        raise ValidationError('cifti_cortical_count_mismatch', {'observed': cortex_counts})
    voxels = brain.voxel[brain.volume_mask]
    if (brain.volume_shape is None or brain.affine is None or not np.isfinite(brain.affine).all()
            or np.any(voxels >= np.asarray(brain.volume_shape))
            or len(np.unique(voxels, axis=0)) != len(voxels)):
        raise ValidationError('cifti_volume_indices_or_grid_invalid')


def validate_cifti_pair(images, frames, tr, grid, metric, expected_axis=None):
    axes = []
    for role, image in zip(('candidate', 'reference'), images):
        if not isinstance(image, nib.Cifti2Image) or image.shape != (frames, 91282):
            raise ValidationError('cifti_full_shape_mismatch', {'role': role, 'observed_shape': list(image.shape), 'expected_shape': [frames, 91282]})
        series, brain = image.header.get_axis(0), image.header.get_axis(1)
        if (not isinstance(series, SeriesAxis) or not isinstance(brain, BrainModelAxis)
                or series.unit != 'SECOND' or series.start != 0 or series.size != frames
                or not np.isclose(series.step, tr, rtol=1e-6, atol=1e-7)):
            raise ValidationError('cifti_time_axis_mismatch', {'role': role})
        check_brain_axis(brain, metric)
        shape, affine = metric.volume_grid(SimpleNamespace(shape=brain.volume_shape, affine=brain.affine))
        if shape != grid[0] or not np.allclose(affine, grid[1], rtol=0, atol=1e-4):
            raise ValidationError('cifti_mni_physical_grid_mismatch', {'role': role, 'observed_shape': list(shape), 'expected_shape': list(grid[0])})
        if expected_axis is not None and not metric.brain_axes_equal(brain, expected_axis):
            raise ValidationError('cifti_fixed_assets_axis_mismatch', {'role': role})
        axes.append((series, brain))
    a, b = axes[0][1], axes[1][1]
    if axes[0][0] != axes[1][0]:
        raise ValidationError('cifti_pair_series_axis_not_exact')
    if not metric.brain_axes_equal(a, b):
        raise ValidationError('cifti_pair_brain_axis_not_exact', {
            'name_mismatches': int(np.count_nonzero(a.name != b.name)),
            'vertex_mismatches': int(np.count_nonzero(a.vertex != b.vertex)),
            'voxel_mismatches': int(np.count_nonzero(np.any(a.voxel != b.voxel, axis=1))),
            'affine_exact': bool(np.array_equal(a.affine, b.affine))})
    return a


def point_statistics(reader, points, frames, metric, point_chunk=4096):
    """复用已有完整时序 centered-moment 算法，每次只生成 point_chunk×frames 误差。"""
    stored = {}
    for start in range(0, points, point_chunk):
        end = min(start + point_chunk, points)
        stats = metric.stream_statistics(lambda first, last: reader(start, end, first, last), end - start, frames)
        for key in ('r', 'mean_candidate', 'mean_reference', 'constant_candidate', 'constant_reference',
                    'zero_candidate', 'zero_reference', 'squared_error', 'reference_energy'):
            if key not in stored:
                stored[key] = np.empty(points, stats[key].dtype)
            stored[key][start:end] = stats[key]
        absolute = stats['absolute']
        if 'absolute_sum' not in stored:
            stored['absolute_sum'], stored['absolute_max'] = np.zeros(points), np.zeros(points)
        stored['absolute_sum'][start:end] = absolute.sum(axis=1)
        stored['absolute_max'][start:end] = absolute.max(axis=1)
    stored['frames'] = frames
    return stored


def summarize(stats, metric, selection=slice(None)):
    r = stats['r'][selection]
    cx, cy, zx, zy = (stats[key][selection] for key in ('constant_candidate', 'constant_reference', 'zero_candidate', 'zero_reference'))
    n = len(r) * stats['frames']
    mse = float(stats['squared_error'][selection].sum() / n)
    rms = float(np.sqrt(stats['reference_energy'][selection].sum() / n))
    valid = np.isfinite(r)
    return {'points': len(r), 'frames': stats['frames'], 'values': n,
            'defined_temporal_r_pairs': int(valid.sum()), 'undefined_temporal_r_pairs': int((~valid).sum()),
            'constant_candidate': int(cx.sum()), 'constant_reference': int(cy.sum()),
            'both_constant': int((cx & cy).sum()), 'one_side_constant': int((cx ^ cy).sum()),
            'zero_candidate': int(zx.sum()), 'zero_reference': int(zy.sum()),
            'both_zero': int((zx & zy).sum()), 'one_side_zero': int((zx ^ zy).sum()),
            'temporal_r': metric.distribution(r[valid]), 'rmse': float(np.sqrt(mse)),
            'reference_rms': rms, 'normalized_rmse': float(np.sqrt(mse) / rms) if rms > 0 else None,
            'normalized_rmse_defined': rms > 0,
            'absolute_difference': {'mean': float(stats['absolute_sum'][selection].sum() / n),
                                    'maximum': float(stats['absolute_max'][selection].max())},
            'temporal_mean_bias': metric.distribution(stats['mean_candidate'][selection] - stats['mean_reference'][selection]),
            'nonfinite_candidate_values': 0, 'nonfinite_reference_values': 0}


def volume_statistics(paths, mask, frames, tr, metric, scratch, point_chunk, time_chunk):
    images = [nib.load(str(path), keep_file_open=True) for path in paths]
    canonical = mask.as_reoriented(metric.ras_orientation(mask))
    if mask.ndim != 3 or mask.header.get_xyzt_units()[0] != 'mm':
        raise ValidationError('mask_3d_mm_required')
    mask_data = np.asarray(canonical.dataobj)
    if not np.isfinite(mask_data).all() or not np.any(mask_data > 0):
        raise ValidationError('empty_or_nonfinite_brain_mask')
    brain = mask_data > 0
    points = int(brain.sum())
    grid = (canonical.shape, canonical.affine)
    caches = []
    for role, image in zip(('candidate', 'reference'), images):
        valid_nifti(image, frames, tr, role)
        shape, affine = metric.volume_grid(image)
        if shape != grid[0] or not np.allclose(affine, grid[1], rtol=0, atol=1e-4):
            raise ValidationError('mni_physical_grid_mismatch', {'role': role, 'observed_shape': list(shape), 'expected_shape': list(grid[0])})
        cache = np.memmap(scratch / (role + '.f64'), dtype=np.float64, mode='w+', shape=(points, frames))
        for first in range(0, frames, time_chunk):
            last = min(frames, first + time_chunk)
            values = np.asanyarray(image.dataobj[..., first:last])
            count = int(np.count_nonzero(~np.isfinite(values)))
            if count:
                raise ValidationError('nonfinite_full_mni_image', {'role': role, 'count': count, 'first_frame': first, 'last_frame_exclusive': last})
            cache[:, first:last] = nib.orientations.apply_orientation(values, metric.ras_orientation(image))[brain]
        cache.flush()
        caches.append(cache)
    stats = point_statistics(lambda a, b, c, d: (caches[0][a:b, c:d], caches[1][a:b, c:d]), points, frames, metric, point_chunk)
    report = summarize(stats, metric)
    report.update(physical_grid_equal=True, index_reorientation_only=True, spatial_interpolation_performed=False,
                  canonical_ras_shape=list(grid[0]), canonical_ras_affine=grid[1].tolist(),
                  domain='entire_bound_MNI_brain_mask_including_zero_and_constant_series')
    del caches
    return report, stats, grid, brain


def comparison_csv(report, path):
    fields = ('case_id', 'status', 'domain', 'points', 'frames', 'temporal_r_mean', 'temporal_r_median',
              'defined_temporal_r_pairs', 'undefined_temporal_r_pairs', 'zero_candidate', 'zero_reference',
              'both_zero', 'one_side_zero', 'rmse', 'reference_rms', 'normalized_rmse', 'normalized_rmse_defined')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        if report['status'] != 'comparison_complete':
            writer.writerow({'case_id': report['case_id'], 'status': report['status']})
            return
        domains = {'MNI_brain': report['volume'], 'CIFTI_all_91k': report['cifti']}
        domains.update(report['cifti']['per_structure'])
        for domain, stats in domains.items():
            row = {key: stats.get(key) for key in fields if key in stats}
            row.update(case_id=report['case_id'], status=report['status'], domain=domain,
                       temporal_r_mean=stats['temporal_r']['mean'], temporal_r_median=stats['temporal_r']['median'])
            writer.writerow(row)


def timings_csv(report, path):
    fields = ('case_id', 'role', 'label', 'name', 'seconds', 'boundary', 'scope', 'source_report_sha256', 'report_field')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for role, run in report.get('runs', {}).items():
            for timing in run['timings']:
                writer.writerow(dict(timing, case_id=report['case_id'], role=role, label=run['label']))


def compare(manifest, output):
    started = time.perf_counter()
    files = {}
    created = False
    case = public_id(manifest['case_id'])
    cohort = public_id(manifest['cohort_id'])
    frames, tr = manifest['frames'], manifest['tr_seconds']
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1 or isinstance(tr, bool) or not np.isfinite(tr) or tr <= 0:
        raise ValidationError('invalid_expected_time_axis')
    point_chunk, time_chunk = manifest.get('point_chunk', 4096), manifest.get('time_chunk', 8)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (point_chunk, time_chunk)):
        raise ValidationError('invalid_chunk_size')
    report = {'schema': 1, 'cohort_id': cohort, 'case_id': case, 'status': 'validating',
              'scientific_equivalence': 'not_assessed', 'producer_sha256': sha256(__file__),
              'metric_definitions': {'temporal_r': 'Pearson over every original frame; only nonconstant pairs have defined r',
                  'normalized_rmse': 'RMSE / sqrt(mean(reference**2)); all zero/constant pairs remain in error metrics',
                  'constant_series': 'exact per-series min == max; undefined r is null/count, never substituted with zero',
                  'alignment': 'physical MNI grid only axis permutation/flip; CIFTI indices/order exactly equal; no intersection, fitting or resampling'},
              'runs': {}}
    files['endpoint_producer'] = (Path(__file__).resolve(), report['producer_sha256'])
    try:
        if manifest.get('_manifest_entry') is not None:
            checked(manifest['_manifest_entry'], 'comparison_manifest', files)
        raw = {key: checked(entry, 'raw/' + key, files) for key, entry in manifest['raw'].items()}
        if not {'t1w', 'bold'}.issubset(raw):
            raise ValidationError('paired_raw_t1w_bold_required')
        raw_image = nib.load(str(raw['bold']))
        if raw_image.ndim != 4 or raw_image.shape[3] != frames:
            raise ValidationError('raw_full_frame_count_mismatch')
        valid_nifti(raw_image, frames, tr, 'raw_bold')
        raw_hashes = {key: files['raw/' + key][1] for key in ('t1w', 'bold')}
        metric = metric_module(manifest['metric_helper'], files)
        mask_path = checked(manifest['brain_mask'], 'brain_mask', files)
        outputs = {}
        for role in ('candidate', 'reference'):
            outputs[role], report['runs'][role] = run_binding(manifest[role], role, case, raw_hashes, frames, tr, files)
        expected_axis = None
        if manifest.get('cifti_axis_assets') is not None:
            for key, entry in manifest['cifti_axis_assets'].items():
                checked(entry, 'cifti_axis_assets/' + key, files)
            expected_axis, _ = metric.fixed_brain_axis(manifest['cifti_axis_assets'])
        protected = [path.parent for path, _ in files.values()] + [Path(p).resolve() for p in manifest.get('protected_roots', [])]
        target = Path(output).expanduser().resolve()
        if target.exists() or target.is_symlink():
            raise FileExistsError('output_exists')
        if any(target.is_relative_to(p) or p.is_relative_to(target) for p in protected):
            raise ValidationError('output_overlaps_original_input')
        target.mkdir(parents=True, exist_ok=False)
        created = True
        tick = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='.paired-volume-', dir=target) as directory:
            volume, vs, grid, brain_mask = volume_statistics(
                [outputs[r]['preproc_mni'] for r in ('candidate', 'reference')], nib.load(str(mask_path)),
                frames, tr, metric, Path(directory), point_chunk, time_chunk)
        report['volume'], report['volume_comparison_seconds'] = volume, time.perf_counter() - tick
        tick = time.perf_counter()
        images = [nib.load(str(outputs[r]['dtseries'])) for r in ('candidate', 'reference')]
        axis = validate_cifti_pair(images, frames, tr, grid, metric, expected_axis)
        cs = point_statistics(lambda a, b, c, d: (images[0].dataobj[c:d, a:b].T, images[1].dataobj[c:d, a:b].T),
                              91282, frames, metric, point_chunk)
        cifti = summarize(cs, metric)
        cifti.update(brain_axis_exactly_equal=True, series_axis_exactly_equal=True,
                     fixed_original_assets_axis_exactly_equal=expected_axis is not None,
                     structures=21, cortical_points=59412, subcortical_points=31870,
                     per_structure={str(name): summarize(cs, metric, selection) for name, selection, _ in axis.iter_structures()})
        report['cifti'], report['cifti_comparison_seconds'] = cifti, time.perf_counter() - tick
        arrays = target / 'metrics.private.npz'
        np.savez_compressed(arrays, volume_r=vs['r'].astype(np.float32), volume_brain_mask=brain_mask,
                            volume_affine=grid[1], cifti_r=cs['r'].astype(np.float32),
                            cifti_squared_error=cs['squared_error'], cifti_reference_energy=cs['reference_energy'])
        report['private_metric_array_sha256'] = sha256(arrays)
        report['status'] = 'comparison_complete'
    except (ValidationError, metric.NonfiniteInput if 'metric' in locals() else ValidationError) as error:
        report['status'] = 'validation_failed'
        report['failure'] = {'code': getattr(error, 'code', 'nonfinite_timeseries'), 'details': getattr(error, 'details', getattr(error, 'public', {}))}
    # No paths in public reports; both sides are rehashed even for rejected axes.
    report['input_sha256_before'] = {key: value[1] for key, value in files.items()}
    report['input_sha256_after'] = {key: sha256(value[0]) if value[0].is_file() else None for key, value in files.items()}
    report['input_guards_equal'] = report['input_sha256_before'] == report['input_sha256_after']
    if not report['input_guards_equal']:
        report.update(status='validation_failed', failure={'code': 'input_changed_during_comparison', 'details': {}})
    report['comparison_seconds'] = time.perf_counter() - started
    report['comparison_timing_scope'] = 'posthoc_hashes_validation_disk_staging_and_metrics_excluding_MRI_wait_or_execution'
    target = Path(output).expanduser().resolve()
    # A preflight failure must not create directories in protected input trees.
    if created:
        write_json(target / 'comparison.public.json', report)
        comparison_csv(report, target / 'comparison.csv')
        timings_csv(report, target / 'timings.csv')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    payload = args.manifest.read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValidationError('json_object_required')
    manifest['_manifest_entry'] = {'path': str(args.manifest.resolve()), 'sha256': hashlib.sha256(payload).hexdigest()}
    manifest.setdefault('protected_roots', []).append(str(args.manifest.resolve().parent))
    report = compare(manifest, args.output_root)
    print(json.dumps(report if report['status'] != 'comparison_complete' else
                     {'case_id': report['case_id'], 'status': report['status'], 'comparison_seconds': report['comparison_seconds']}))
    return 0 if report['status'] == 'comparison_complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
