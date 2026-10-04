"""Check the independent FSL field against original Surfa voxel coordinates."""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import surfa as sf


def scaled_mm(image):
    zooms = image.header.get_zooms()[:3]
    matrix = np.diag((*zooms, 1.)).astype(np.float64)
    if np.linalg.det(image.affine[:3, :3]) > 0:
        matrix[0, 0] *= -1
        matrix[0, 3] = (image.shape[0]-1) * zooms[0]
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--moving', required=True)
    parser.add_argument('--fixed', required=True)
    parser.add_argument('--original-ras-field', required=True)
    parser.add_argument('--fsl-relative-field', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    moving, fixed = nib.load(args.moving), nib.load(args.fixed)
    warp = sf.load_warp(args.original_ras_field)
    # Coordinate conversion uses actual registration grids. Serialized Warp
    # geometry is audited separately because its float fields can round.
    warp.source = sf.load_volume(args.moving).geom
    warp.target = sf.load_volume(args.fixed).geom
    original_voxel_displacement = warp.convert(format=sf.Warp.Format.disp_crs).data
    grid = np.indices(fixed.shape, dtype=np.float64).reshape(3, -1)
    expected_source_voxels = grid + original_voxel_displacement.reshape(-1, 3).T
    relative = np.asanyarray(nib.load(args.fsl_relative_field).dataobj)
    target_fsl = (scaled_mm(fixed) @ np.vstack((grid, np.ones((1, grid.shape[1])))))[:3]
    source_fsl = target_fsl + relative.reshape(-1, 3).T
    actual_source_voxels = (np.linalg.inv(scaled_mm(moving)) @ np.vstack(
        (source_fsl, np.ones((1, source_fsl.shape[1])))))[:3]
    error = moving.affine[:3, :3] @ (actual_source_voxels-expected_source_voxels)
    report = {'scope': 'every target voxel; independent NumPy/FSL field compared with original Surfa RAS-to-voxel conversion using actual input geometries',
              'target_voxels': int(grid.shape[1]),
              'maximum_world_vector_error_mm': float(np.max(np.linalg.norm(error, axis=0))),
              'maximum_world_component_error_mm': float(np.abs(error).max()),
              'world_component_rmse_mm': float(np.sqrt(np.mean(error ** 2))),
              'moving_affine_has_oblique_axes': bool(np.count_nonzero(np.abs(moving.affine[:3, :3]) > 1e-5) > 3)}
    report['acceptance'] = (report['maximum_world_component_error_mm'] <= 1e-3 and
                            report['world_component_rmse_mm'] <= 1e-4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
