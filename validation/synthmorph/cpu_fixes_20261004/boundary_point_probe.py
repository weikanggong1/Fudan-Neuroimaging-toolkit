"""Capture actual FP32 image pulls at failed fixed boundary points."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from fnit.synthmorph import SynthMorph, models, pipeline


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('moving', 'fixed', 'candidate', 'reference', 'compare-worker', 'weights', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--extent', type=int, default=192)
    parser.add_argument('--hyper', type=float, default=.75)
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('comparison', args.compare_worker)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    reference = Path(args.reference)
    candidate = Path(args.candidate)
    mask = comparison.upper_boundary(args.moving, args.fixed, reference / 'forward.nii.gz', 'joint')
    expected = np.asanyarray(nib.load(reference / 'moved.nii.gz').dataobj)
    previous = np.asanyarray(nib.load(candidate / 'moved.nii.gz').dataobj)
    points = np.argwhere(mask & (expected != previous))
    if not len(points):
        raise ValueError('there are no failed boundary points to diagnose')
    source = nib.load(args.moving)
    target = nib.load(args.fixed)
    reference_warp = np.asanyarray(nib.load(reference / 'forward.nii.gz').dataobj)
    if reference_warp.ndim == 5:
        reference_warp = reference_warp[..., 0, :]
    world = nib.affines.apply_affine(target.affine, points) + reference_warp[tuple(points.T)]
    reference_coordinates = nib.affines.apply_affine(np.linalg.inv(source.affine), world)
    captured = []
    original_sampler = pipeline._resampled_image
    def sampler(image, pull, reference_image, device, *positional, **keywords):
        # The first call is the actual moving image's final sampler; consume
        # its raw displacement before the saved RAS conversion can round it.
        if not captured:
            indices = tuple(points.T)
            location = pull[0, :, *indices].T + torch.as_tensor(points, dtype=pull.dtype)
            normalized = torch.stack([
                location[:, axis] * (2 / (size - 1)) - 1
                for axis, size in enumerate(source.shape[:3])], -1)
            reconstructed = torch.stack([
                ((normalized[:, axis] + 1) / 2) * (size - 1)
                for axis, size in enumerate(source.shape[:3])], -1)
            captured.append({'raw_float32_source_coordinates': location.tolist(),
                             'normalized_float32_coordinates': normalized.tolist(),
                             'grid_sample_float32_roundtrip_coordinates': reconstructed.tolist()})
        return original_sampler(image, pull, reference_image, device, *positional, **keywords)
    features = []
    registration = SynthMorph(weights=args.weights, model='joint', device='cpu',
                              extent=args.extent, hyper=args.hyper, steps=args.steps)
    handle = registration.network.affine.detector.register_forward_hook(
        lambda module, inputs, result: features.append(result.detach().clone()))
    pipeline._resampled_image = sampler
    try:
        result = registration(args.moving, args.fixed)
    finally:
        handle.remove()
        pipeline._resampled_image = original_sampler
    stages = {}
    for index, feature in enumerate(features):
        center, mass = models._cpu_joint_barycenter(feature, (args.extent // 2,) * 3)
        stages[f'feature_{index}'] = feature.permute(0, 2, 3, 4, 1).numpy()
        stages[f'center_{index}'] = center.numpy()
        stages[f'mass_{index}'] = mass.numpy()
    np.savez_compressed(root / 'features.npz', **stages)
    rows = []
    for name, cli_name in [('moved', 'moved'), ('fixed_moved', 'fixed_moved'),
                           ('transform', 'forward'), ('inverse', 'inverse')]:
        actual = np.asarray(getattr(result, name).dataobj)
        saved = np.asanyarray(nib.load(candidate / (cli_name + '.nii.gz')).dataobj)
        rows.append({'output': cli_name, 'same_array': array_hash(actual) == array_hash(saved)})
    report = {'scope': 'actual native FP32 pull capture before RAS saving at failed fixed zero-band points',
              'points': points.tolist(), 'reference_source_coordinates_from_saved_RAS_float64': reference_coordinates.tolist(),
              'reference_intensity': expected[tuple(points.T)].tolist(),
              'candidate_intensity': previous[tuple(points.T)].tolist(),
              'source_shape': source.shape, 'sampler': captured[0], 'same_CLI_outputs': rows,
              'features_sha256': hashlib.sha256((root / 'features.npz').read_bytes()).hexdigest(),
              'source_sha256': {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                for name, module in [('models', models), ('pipeline', pipeline)]},
              'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    if not all(row['same_array'] for row in rows):
        raise RuntimeError('diagnostic changed the saved CLI output')


if __name__ == '__main__':
    main()
