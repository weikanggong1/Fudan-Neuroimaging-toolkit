"""Compare actual official binary masks, independently of zero T1 intensity."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def differences(value, expected):
    return {'different_voxels': int(np.count_nonzero(value != expected)),
            'candidate_nonzero': int(np.count_nonzero(value)),
            'official_nonzero': int(np.count_nonzero(expected))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t1', type=Path, required=True)
    parser.add_argument('--aseg', type=Path, required=True)
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new output path')
    from fnit.gems.context import SubregionContext
    from fnit.gems.recipes import ThalamusRecipe, HippoAmygdalaRecipe
    from fnit.gems.recipes.base import working_image
    context = SubregionContext.prepare(args.t1, need_coarse=True, need_parc=False,
                                       coarse_segmentation=args.aseg, device='cpu')
    report = {'scope': 'actual_binary_mask_only_no_fitting',
              'official_geometry_used_only_as_diagnostic': True, 'structures': {}}
    for name, recipe in [('thalamus', ThalamusRecipe('thalamus', '.')),
                         ('hippo-amygdala-left', HippoAmygdalaRecipe('left', '.')),
                         ('hippo-amygdala-right', HippoAmygdalaRecipe('right', '.'))]:
        image, _, _ = working_image(context, getattr(recipe, 'crop_ids', recipe.alignment_ids),
                                     recipe.resolution_mm)
        directory = args.official / name
        official_image = nib.load(str(directory / 'temporary/tempImage.mgz'))
        rounded = nib.MGHImage(np.asarray(image.dataobj), image.affine).header.get_affine()
        geometries = {'fnit_affine': image.affine, 'mgh_header_affine': rounded,
                      'official_affine_diagnostic': official_image.affine}
        merged = ndimage.binary_dilation(recipe.synthetic_labels(context.coarse_segmentation) > 1,
                                         structure=np.ones((3, 3, 3)), iterations=2)
        merged_reference = np.asarray(nib.load(str(
            directory / ('longMask.mgz' if name == 'thalamus' else
                         'temporary/asegModBinDilatedResampled.mgz'))).dataobj).squeeze() != 0
        row = {'nearest_merged': {}, 'hippo_linear_dilated': {}}
        if name != 'thalamus':
            hippo = np.isin(context.coarse_segmentation, recipe.alignment_ids).astype(np.float32)
            hippo_reference = np.asarray(nib.load(str(directory / 'longMask.mgz')).dataobj).squeeze() != 0
        for label, affine in geometries.items():
            transform = np.linalg.inv(context.image.affine) @ affine
            matrix, offset = transform[:3, :3], transform[:3, 3]
            assert np.max(np.abs(matrix - np.diag(np.diag(matrix)))) < 1e-8
            axes = [offset[d] + np.arange(image.shape[d]) * matrix[d, d] for d in range(3)]
            candidate = ndimage.affine_transform(merged.astype(np.uint8), matrix, offset,
                                                  output_shape=image.shape, order=0)
            row['nearest_merged'][label + '_scipy'] = differences(candidate != 0, merged_reference)
            for dtype in (np.float32, np.float64):
                grid_axes = [np.asarray(offset[d], dtype=dtype) + np.arange(image.shape[d], dtype=dtype)
                             * np.asarray(matrix[d, d], dtype=dtype) for d in range(3)]
                for rule in ('rint', 'half_up'):
                    indices = [np.rint(axis).astype(int) if rule == 'rint' else
                               np.floor(axis + .5).astype(int) for axis in grid_axes]
                    candidate = merged[np.ix_(*indices)]
                    row['nearest_merged'][label + '_' + np.dtype(dtype).name + '_' + rule] = differences(candidate, merged_reference)
            if name != 'thalamus':
                for dtype in (np.float32, np.float64):
                    grid_axes = [np.asarray(offset[d], dtype=dtype) + np.arange(image.shape[d], dtype=dtype)
                                 * np.asarray(matrix[d, d], dtype=dtype) for d in range(3)]
                    coordinates = np.stack(np.broadcast_arrays(grid_axes[0][:, None, None],
                                                               grid_axes[1][None, :, None],
                                                               grid_axes[2][None, None, :]))
                    sampled = ndimage.map_coordinates(hippo, coordinates, order=1, prefilter=False)
                    mask = ndimage.binary_dilation(sampled >= .5, structure=np.ones((3, 3, 3)),
                                                   iterations=int(round(3 / recipe.resolution_mm)))
                    row['hippo_linear_dilated'][label + '_' + np.dtype(dtype).name] = differences(mask, hippo_reference)
        if name != 'thalamus':
            for label, affine in geometries.items():
                transform = np.linalg.inv(context.image.affine.astype(np.float32)) @ affine.astype(np.float32)
                axis = [transform[d, 3] + np.arange(image.shape[d], dtype=np.float32) * transform[d, d]
                        for d in range(3)]
                coordinates = np.stack(np.broadcast_arrays(axis[0][:, None, None],
                                                           axis[1][None, :, None], axis[2][None, None, :]))
                sampled = ndimage.map_coordinates(hippo, coordinates, order=1, prefilter=False)
                mask = ndimage.binary_dilation(sampled >= .5, structure=np.ones((3, 3, 3)),
                                               iterations=int(round(3 / recipe.resolution_mm)))
                row['hippo_linear_dilated'][label + '_matrix32_coords32'] = differences(mask, hippo_reference)
                # Explicit trilinear FP32 sums test sampling arithmetic separately
                # from the coordinates. This is an isolated diagnostic only.
                low = [np.floor(a).astype(int) for a in axis]
                weights = [a - b for a, b in zip(axis, low)]
                sampled = np.zeros(image.shape, np.float32)
                for x in (0, 1):
                    for y in (0, 1):
                        for z in (0, 1):
                            blend = [(w if high else 1 - w).astype(np.float32)
                                     for w, high in zip(weights, (x, y, z))]
                            values = hippo[np.ix_(low[0] + x, low[1] + y, low[2] + z)]
                            sampled += values * blend[0][:, None, None] * blend[1][None, :, None] * blend[2][None, None, :]
                mask = ndimage.binary_dilation(sampled >= .5, structure=np.ones((3, 3, 3)),
                                               iterations=int(round(3 / recipe.resolution_mm)))
                row['hippo_linear_dilated'][label + '_matrix32_coords32_sum32'] = differences(mask, hippo_reference)
        report['structures'][name] = row
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
