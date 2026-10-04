"""Independent original SynthMorph branch reusing original VBM preprocessing.

Run in the original reference environment with Surfa, never in FNIT runtime.
The NumPy adapter implements coordinate conversion and the declared dense
finite-difference Jacobian. It does not import FNIT registration/converters.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import nibabel as nib
import numpy as np
import surfa as sf


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def voxel_to_fsl(image):
    """Original FSL scaled-mm axes: header pixdims plus radiological X flip."""
    zooms = image.header.get_zooms()[:3]
    result = np.diag((*zooms, 1.)).astype(np.float64)
    if np.linalg.det(image.affine[:3, :3]) > 0:
        result[0, 0] *= -1
        result[0, 3] = (image.shape[0] - 1) * zooms[0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--template', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--reuse-native-field', type=Path,
                        help='replay only postprocessing from an existing original RAS field')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    upstream_report = json.loads((args.upstream / 'reference_steps.private.json').read_text())
    for name in ('synthstrip', 'fast', 'flirt'):
        stage = next(row for row in upstream_report['stages'] if row['name'] == name)
        if stage.get('returncode') != 0:
            raise ValueError('original upstream stage is incomplete: ' + name)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir
    moving_path = args.upstream / 'T1_brain_pve_1.nii.gz'
    moving, fixed = nib.load(moving_path), nib.load(args.template)
    moving_sf, fixed_sf = sf.load_volume(moving_path), sf.load_volume(args.template)
    fsl_affine_path = args.upstream / 'gm_affine.mat'
    fsl_affine = np.loadtxt(fsl_affine_path)
    sm, st = voxel_to_fsl(moving), voxel_to_fsl(fixed)
    forward_world = (fixed.affine @ np.linalg.inv(st) @ fsl_affine @ sm @
                     np.linalg.inv(moving.affine))
    initial = output / 'original_flirt_initial.lta'
    sf.Affine(forward_world, source=moving_sf, target=fixed_sf,
              space='world').save(initial)
    loaded_world = sf.load_affine(initial).convert(space='world').matrix
    if not np.allclose(loaded_world, forward_world, atol=1e-8, rtol=0):
        raise ValueError('original FLIRT-to-world LTA roundtrip changed its matrix')
    # Validate forward direction on all source-grid corners, including oblique axes.
    corners = np.array(np.meshgrid(*[(0, size - 1) for size in moving.shape[:3]],
                                  indexing='ij')).reshape(3, -1)
    source = np.vstack((corners, np.ones((1, corners.shape[1]))))
    target_via_world = np.linalg.inv(fixed.affine) @ forward_world @ moving.affine @ source
    target_via_fsl = np.linalg.inv(st) @ fsl_affine @ sm @ source
    maximum_corner_error = float(np.max(np.linalg.norm(
        (fixed.affine @ target_via_world - fixed.affine @ target_via_fsl)[:3], axis=0)))
    if maximum_corner_error > 1e-8:
        raise ValueError('FLIRT forward coordinate reconstruction failed')
    fs = Path(os.environ['FREESURFER_HOME']) / 'bin'
    fsl = Path(os.environ['FSLDIR']) / 'bin'
    report = {'status': 'running', 'scope': 'original SynthMorph deform256 after original SynthStrip/FAST/FLIRT; NumPy coordinate/Jacobian adapter; original FSL applywarp and modulation',
              'upstream_reused': {name: digest(args.upstream / name) for name in (
                  'T1_brain.nii.gz', 'brain_mask.nii.gz', 'T1_brain_pve_1.nii.gz', 'gm_affine.mat')},
              'template_sha256': digest(args.template), 'weights_sha256': digest(args.weights),
              'threads': args.threads, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'initial_forward_world': forward_world.tolist(),
              'initial_corner_world_error_mm': maximum_corner_error,
              'initial_roundtrip_maximum_absolute_error': float(np.max(np.abs(loaded_world - forward_world))),
              'coordinate_adapter': 'NumPy float64 original NIfTI geometry with original header pixdims; output dense/Jacobian cast float32; no FNIT converter imported',
              'jacobian_definition': 'det(I + d residual_fsl / d target_fsl); residual=source_fsl-inv(original_FLIRT)@target_fsl; first-order boundary/central interior dense differences; nonlinear-only',
              'stages': [{'name': 'prepare_original_FLIRT_LTA',
                          'wall_seconds': time.perf_counter()-started, 'returncode': 0}]}
    report_path = output / 'reference_steps.private.json'

    def save():
        report['elapsed_seconds'] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    def run(name, command):
        row = {'name': name, 'argv': [str(item) for item in command]}
        report['stages'].append(row)
        save()
        with (output / (name + '.log')).open('wb') as stream:
            before = time.perf_counter()
            completed = subprocess.run(row['argv'], stdout=stream, stderr=subprocess.STDOUT)
            row.update(wall_seconds=time.perf_counter()-before, returncode=completed.returncode)
        save()
        if completed.returncode:
            report['status'] = 'failed';save();raise SystemExit(completed.returncode)

    native = args.reuse_native_field or (output / 'original_ras_pull.nii.gz')
    # The original network returns both antisymmetric directions regardless of
    # which fields are saved; no negative-velocity approximation is substituted.
    if args.reuse_native_field:
        report['reused_original_native_field_sha256'] = digest(native)
        report['stages'].append({'name': 'reuse_original_synthmorph_field', 'returncode': 0})
        save()
    else:
        run('synthmorph', [fs / 'mri_synthmorph', 'register', '-m', 'deform',
                          '-e', '256', '-r', '0.5', '-n', '7', '-j', args.threads,
                          '-i', initial, '-w', args.weights, '-t', native,
                          moving_path, args.template])
    before = time.perf_counter()
    warp = sf.load_warp(native).convert(format=sf.Warp.Format.disp_ras)
    displacement = np.asarray(warp.data, dtype=np.float64)
    source_affine, target_affine = warp.source.vox2world.matrix, warp.target.vox2world.matrix
    if tuple(warp.source.shape) != moving.shape[:3] or tuple(warp.target.shape) != fixed.shape[:3]:
        raise ValueError('original warp source/target grids do not match GM/template')
    if displacement.shape != (*fixed.shape[:3], 3):
        raise ValueError('original RAS displacement has unexpected component axis')
    source_error = float(np.max(np.abs(source_affine - moving_sf.geom.vox2world.matrix)))
    target_error = float(np.max(np.abs(target_affine - fixed_sf.geom.vox2world.matrix)))
    if source_error > 1e-3 or target_error > 1e-3:
        raise ValueError('original warp source/target affines differ from registration inputs')
    voxels = np.indices(fixed.shape[:3], dtype=np.float64).reshape(3, -1)
    homogeneous = np.vstack((voxels, np.ones((1, voxels.shape[1]))))
    # RAS displacements are already physical coordinates. The source geometry
    # serialized in a Warp extension can round voxel size/direction fields.
    # Consumption must use the actual input NIfTI grid, not that rounded copy.
    source_world = (fixed.affine @ homogeneous)[:3] + displacement.reshape(-1, 3).T
    source_voxels = (np.linalg.inv(moving.affine) @ np.vstack(
        (source_world, np.ones((1, source_world.shape[1])))))[:3]
    source_fsl = (sm @ np.vstack((source_voxels, np.ones((1, source_voxels.shape[1])))))[:3]
    target_fsl = (st @ homogeneous)[:3]
    affine_source_fsl = (np.linalg.inv(fsl_affine) @ np.vstack(
        (target_fsl, np.ones((1, target_fsl.shape[1])))))[:3]
    residual = (source_fsl - affine_source_fsl).reshape(3, *fixed.shape[:3])
    relative = (source_fsl - target_fsl).T.reshape(*fixed.shape[:3], 3)
    gradient_voxel = np.stack(np.gradient(residual, axis=(1, 2, 3), edge_order=1), axis=1)
    derivative = np.einsum('caxyz,ab->cbxyz', gradient_voxel, np.linalg.inv(st[:3, :3]))
    jacobian = np.linalg.det(np.moveaxis(derivative, (0, 1), (-2, -1)) + np.eye(3))
    dense_path = output / 'original_fsl_relative.nii.gz'
    header = fixed.header.copy();header.set_data_dtype(np.float32);header.set_intent(2006)
    nib.save(nib.Nifti1Image(relative.astype(np.float32), fixed.affine, header), dense_path)
    jacobian_path = output / 'T1_GM_JAC_nl.nii.gz'
    header = fixed.header.copy();header.set_data_dtype(np.float32);header.set_intent('none')
    nib.save(nib.Nifti1Image(jacobian.astype(np.float32), fixed.affine, header), jacobian_path)
    report['stages'].append({'name': 'numpy_field_and_dense_jacobian',
                             'wall_seconds': time.perf_counter()-before, 'returncode': 0})
    report['warp_geometry_check'] = {'source_affine_maximum_error': source_error,
                                      'target_affine_maximum_error': target_error,
                                      'source_shape': moving.shape, 'target_shape': fixed.shape,
                                      'source_affine_vs_nifti': float(np.max(np.abs(source_affine-moving.affine))),
                                      'target_affine_vs_nifti': float(np.max(np.abs(target_affine-fixed.affine)))}
    warped = output / 'T1_GM_to_template_GM.nii.gz'
    modulated = output / 'T1_GM_to_template_GM_mod.nii.gz'
    run('applywarp', [fsl / 'applywarp', '-i', moving_path, '-r', args.template,
                      '-w', dense_path, '-o', warped, '--rel', '--interp=trilinear', '--datatype=float'])
    run('modulation', [fsl / 'fslmaths', warped, '-mul', jacobian_path, modulated, '-odt', 'float'])
    report['outputs'] = {path.name: {'sha256': digest(path), 'size': path.stat().st_size}
                         for path in output.glob('*.nii.gz')}
    report['status'] = 'complete';save()
    print(json.dumps({'status': 'complete', 'seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
