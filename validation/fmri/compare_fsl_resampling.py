"""Validate real-frame spline resampling against isolated FSL applywarp.

FSL is used only as a benchmark oracle. Images, coordinates and program logs
remain in --private-output; the report contains anonymous metrics and hashes.
"""

import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion

from fnit.synthmorph.fsl_warp import convert_warp_to_fsl
from validate_resampling import agreement, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'candidate', 'reference', 'mask', 'pull', 'matrix',
                 'fsl-applywarp', 'private-output', 'report-out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--frames', type=int, default=8)
    args = parser.parse_args()
    args.private_output.mkdir(parents=True, exist_ok=True)
    source, target = nib.load(args.native), nib.load(args.reference)
    frames = min(args.frames, source.shape[3])
    data = np.asarray(source.dataobj, dtype=np.float32)[..., :frames]
    subset = nib.Nifti1Image(data, source.affine, source.header)
    source_path = args.private_output / 'source_frames.nii.gz'
    nib.save(subset, source_path)
    xyz = target.affine[:3, :3] @ np.indices(target.shape, dtype=np.float64).reshape(3, -1)
    xyz += target.affine[:3, 3:4]
    world = xyz + np.asarray(nib.load(args.pull).dataobj, dtype=np.float64).reshape(-1, 3).T
    matrix = np.loadtxt(args.matrix)
    epi_world = matrix[:3, :3] @ world + matrix[:3, 3:4]
    composed = (epi_world - xyz).T.reshape(*target.shape, 3).astype(np.float32)
    field = nib.Nifti1Image(composed, target.affine)
    field.header.set_intent('vector')
    fsl_field = convert_warp_to_fsl(field, moving=nib.Nifti1Image(data[..., 0], source.affine,
                                                               source.header), fixed=target)
    warp_path = args.private_output / 'composed_epi_to_mni_fsl.nii.gz'
    nib.save(fsl_field, warp_path)
    output = args.private_output / 'fsl_spline_frames.nii.gz'
    command = [str(args.fsl_applywarp), '--in=' + str(source_path), '--ref=' + str(args.reference),
               '--warp=' + str(warp_path), '--mask=' + str(args.mask), '--rel', '--interp=spline',
               '--out=' + str(output)]
    environment = dict(os.environ, FSLDIR=str(args.fsl_applywarp.parent.parent),
                       FSLOUTPUTTYPE='NIFTI_GZ', OMP_NUM_THREADS='8')
    environment['LD_LIBRARY_PATH'] = str(args.fsl_applywarp.parent.parent / 'lib') + ':' + environment.get('LD_LIBRARY_PATH', '')
    started = time.perf_counter()
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    wall = time.perf_counter() - started
    (args.private_output / 'applywarp.private.log').write_text(result.stdout + result.stderr)
    # Some FSL binaries on this host return 255 after writing a complete image.
    # Retain the exit code and validate the entire compressed output explicitly.
    with gzip.open(output, 'rb') as stream:
        while stream.read(8 * 1024 * 1024):
            pass
    image = nib.load(output)
    official = np.asarray(image.dataobj, dtype=np.float32)
    candidate_image = nib.load(args.candidate)
    candidate = np.asarray(candidate_image.dataobj, dtype=np.float32)[..., :frames]
    if official.shape != candidate.shape or not np.allclose(image.affine, target.affine,
                                                             atol=1e-4, rtol=0):
        raise ValueError('FSL output grid differs')
    if not np.isfinite(official).all():
        raise ValueError('nonfinite FSL output')
    mask = np.asarray(nib.load(args.mask).dataobj) > 0
    inverse = np.linalg.inv(source.affine)
    coords = inverse[:3, :3] @ epi_world + inverse[:3, 3:4]
    interior = binary_erosion(mask, iterations=2).ravel()
    for axis in range(3):
        interior &= (coords[axis] >= 4) & (coords[axis] <= source.shape[axis] - 5)
    interior = interior.reshape(target.shape)
    report = {'schema_version': 1, 'scope': 'Same real native BOLD frames and exact composed EPI-to-MNI warp.',
              'frames': frames, 'fsl_version': '6.0.7.22', 'fsl_exit_code': result.returncode,
              'fsl_command': 'applywarp --in=source_frames --ref=MNI --warp=composed_fsl --mask=MNI_mask --rel --interp=spline --out=reference',
              'checks': {'gzip_crc_passed': True, 'grid_matches': True, 'all_finite': True,
                         'shape': list(official.shape), 'interior_voxels': int(interior.sum()),
                         'outside_mask_max_abs': float(np.abs(official[~mask]).max())},
              'same_mask': agreement(candidate[mask], official[mask]),
              'interior': agreement(candidate[interior], official[interior]),
              'timing_seconds': {'fsl_process_including_io': wall},
              'sha256': {'fsl_executable': sha256(args.fsl_applywarp), 'native': sha256(args.native),
                         'candidate': sha256(args.candidate), 'reference': sha256(args.reference),
                         'pull': sha256(args.pull), 'matrix': sha256(args.matrix),
                         'composed_fsl_field': sha256(warp_path), 'fsl_output': sha256(output)},
              'limits': ['Fixed-warp real-frame comparison tests interpolation, not nonlinear registration or the complete UKB pipeline.',
                         'FNIT uses periodic spline boundaries; FSL applywarp uses its own source-edge policy.',
                         'Timing is comparable only for equal frame counts and the same input/warp, with different process/import boundaries noted.'],
              'privacy': 'Anonymous scalar metrics and hashes; images and logs remain private.'}
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
