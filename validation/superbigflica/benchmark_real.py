"""Run the public API on private real images; publish aggregate checks only.

Supply SUBJECTS_ROOT CONFIG_JSON PHENOTYPES_CSV SUBJECTS_TXT OUTPUT_DIR
SUMMARY_JSON and --device cpu/cuda:0. Configuration and cohort files remain on
the validation host. No subject identifiers, individual labels, or data paths
are included in SUMMARY_JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.superbigflica import apply_model, run_superbigflica


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def sanitized_metrics(metrics: dict) -> dict:
    return {split: {target: {key: value for key, value in report.items()
                             if key != 'classes'}
                    for target, report in targets.items()}
            for split, targets in metrics.items()}


def verify_maps(model_dir: Path, metadata: dict) -> dict:
    checked = {}
    for name, spec in metadata['modalities'].items():
        mask_image = nib.load(str(model_dir / spec['mask']))
        mask = np.asarray(mask_image.dataobj) > 0
        count = 0
        for path in sorted((model_dir / 'maps' / name).glob('*.nii.gz')):
            image = nib.load(str(path))
            data = np.asarray(image.dataobj)
            if (image.shape != mask.shape or
                    not np.allclose(image.affine, mask_image.affine, atol=1e-4) or
                    image.get_data_dtype() != np.dtype('float32') or
                    not np.isfinite(data).all() or np.any(data[~mask] != 0)):
                raise ValueError(f'Invalid exported map for modality {name}')
            if path.name.endswith('_zstat.nii.gz'):
                component = int(path.name.split('_')[0].split('-')[1]) - 1
                expected = np.load(model_dir / f'{name}_zstat.npy', mmap_mode='r')[:, component]
                if not np.array_equal(data[mask], expected):
                    raise ValueError(f'NIfTI differs from saved z-stat array: {name}')
            count += 1
        if count != metadata['n_components'] * 2:
            raise ValueError(f'Unexpected number of NIfTI maps: {name}')
        checked[name] = {'shape': list(mask.shape), 'mask_voxels': int(mask.sum()),
                         'nifti_maps': count, 'dtype': 'float32', 'finite': True,
                         'outside_mask_zero': True, 'zstat_matches_array': True}
    return checked


def verify_apply(model_dir: Path, subjects_root: Path, metadata: dict) -> dict:
    with (model_dir / 'subj_course.tsv').open() as stream:
        courses = list(csv.DictReader(stream, delimiter='\t'))
    with (model_dir / 'predictions.csv').open() as stream:
        predictions = {row['subject_id']: row for row in csv.DictReader(stream)}
    row = next(row for row in courses if row['split'] == 'test')
    subject_dir = subjects_root / row['subject_id']
    expected_course = np.array([float(row[f'component_{i + 1:03d}'])
                                for i in range(metadata['n_components'])])
    reference = predictions[row['subject_id']]
    results = {}
    outputs = {}
    for device in ('cpu', 'cuda:0'):
        if device.startswith('cuda') and not torch.cuda.is_available():
            continue
        result = apply_model(model_dir, subject_dir, device=device)
        errors = {}
        for target in metadata['targets']:
            name = target['name']
            if target['type'] == 'continuous':
                errors[name] = abs(result['predictions'][name]['value'] -
                                   float(reference[f'{name}__prediction']))
            else:
                expected = np.array([float(reference[f'{name}__prob_{i:03d}'])
                                     for i in range(len(target['classes']))])
                actual = np.array([result['predictions'][name]['probabilities'][category]
                                   for category in target['classes']])
                errors[name] = float(np.max(np.abs(actual - expected)))
        results[device] = {'course_max_abs_error': float(np.max(np.abs(
            np.asarray(result['components']) - expected_course))),
                          'prediction_max_abs_errors': errors}
        outputs[device] = result
    if 'cuda:0' in outputs:
        results['frozen_cpu_gpu_course_max_abs_error'] = float(np.max(np.abs(
            np.asarray(outputs['cpu']['components']) - outputs['cuda:0']['components'])))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subjects_root', type=Path)
    parser.add_argument('config', type=Path)
    parser.add_argument('phenotypes_csv', type=Path)
    parser.add_argument('subjects_file', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('summary_json', type=Path)
    parser.add_argument('--device', required=True)
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    started = time.perf_counter()
    modalities = json.loads(args.config.read_text())['modalities']
    subjects = args.subjects_file.read_text().splitlines()
    progress = args.summary_json.with_name(args.summary_json.stem + '.progress.json')
    write_json(progress, {'status': 'hashing_inputs', 'subjects': len(subjects)})
    image_digest = hashlib.sha256()
    for subject in subjects:
        for spec in modalities.values():
            image_digest.update(bytes.fromhex(sha256(args.subjects_root / subject / spec['image'])))
    hashes = {'images_combined_sha256': image_digest.hexdigest(),
              'subjects_sha256': sha256(args.subjects_file),
              'matched_phenotypes_sha256': sha256(args.phenotypes_csv),
              'masks_sha256': {name: sha256(Path(spec['mask']))
                               for name, spec in modalities.items()}}
    import fnit.superbigflica.pipeline as implementation
    source_root = Path(implementation.__file__).parent
    implementation_hashes = {path.name: sha256(path)
                              for path in sorted(source_root.glob('*.py'))}
    implementation_hashes['benchmark_real.py'] = sha256(Path(__file__))
    write_json(progress, {'status': 'fitting', 'subjects': len(subjects), 'device': args.device})
    model_dir = run_superbigflica(
        args.subjects_root, modalities, args.phenotypes_csv,
        {'p20023_i0': 'continuous', 'Incident_I9_HYPTENS': 'categorical'},
        args.output_dir, n_components=3, id_column='eid', subjects=subjects,
        validation_fraction=.2, test_fraction=.2, max_epochs=args.max_epochs,
        batch_size=args.batch_size, random_state=0, device=args.device,
        max_gpu_gb=19, feature_block=2048, top_voxels=1000,
    )
    write_json(progress, {'status': 'verifying_outputs', 'subjects': len(subjects)})
    metadata = json.loads((model_dir / 'model.json').read_text())
    metrics = json.loads((model_dir / 'metrics.json').read_text())
    course = np.load(model_dir / 'subj_course.npy')
    if course.shape != (len(subjects), 3) or not np.isfinite(course).all():
        raise ValueError('Invalid real subject course output')
    report = {
        'dataset': '500 real UKB subjects; full-mask VBM/FA/MD; two measured targets',
        'scope': 'Public supervised API, held-out metrics and frozen inference checks',
        'targets': {'p20023_i0': {'type': 'continuous',
                                 'meaning': 'Mean time to correctly identify matches', 'unit': 'ms'},
                    'Incident_I9_HYPTENS': {'type': 'categorical'}},
        'subjects': len(subjects), 'components': 3, 'course_shape': list(course.shape),
        'splits': metadata['counts'], 'device': args.device, 'dtype': metadata['dtype'],
        'tf32': metadata['tf32'], 'max_epochs': args.max_epochs,
        'batch_size': args.batch_size, 'effective_batch_size': metadata['effective_batch_size'],
        'best_epoch': metadata['best_epoch'], 'random_state': 0,
        'metrics': sanitized_metrics(metrics), 'maps': verify_maps(model_dir, metadata),
        'frozen_apply': verify_apply(model_dir, args.subjects_root, metadata),
        'stage_timings_s': metadata['timings'], 'pipeline_wall_s': metadata['wall_time_s'],
        'harness_wall_s': time.perf_counter() - started,
        'peak_process_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
        'peak_gpu_allocated_gib': metadata['peak_gpu_allocated_gib'],
        'input_hashes': hashes, 'implementation_sha256': implementation_hashes,
        'model_sha256': sha256(model_dir / 'model.pt'),
        'hardware': {'hostname': __import__('socket').gethostname(),
                     'cpu_threads': torch.get_num_threads(),
                     'gpu': torch.cuda.get_device_name() if args.device.startswith('cuda') else None},
        'notes': 'Single observation on a shared host. Test metrics describe this cohort and setting; '
                 'not clinical validity or an independent external cohort.'
    }
    write_json(args.summary_json, report)
    write_json(progress, {'status': 'complete', 'subjects': len(subjects)})
    print(json.dumps({'status': 'complete', 'subjects': len(subjects),
                      'pipeline_wall_s': metadata['wall_time_s']}), flush=True)


if __name__ == '__main__':
    main()
