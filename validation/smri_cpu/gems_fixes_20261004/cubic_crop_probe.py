"""Real-input ablation of cropped cubic coefficients, boundary and sign rules."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def metric(first, second):
    difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
    return {'different_values': int(np.count_nonzero(difference)),
            'above_1e_minus3': int(np.count_nonzero(difference > .001)),
            'max_abs': float(difference.max(initial=0)),
            'p99_abs': float(np.percentile(difference, 99)),
            'rmse': float(np.sqrt(np.mean(difference * difference)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t1', type=Path, required=True)
    parser.add_argument('--aseg', type=Path, required=True)
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new output directory')
    args.output.mkdir()
    from fnit.gems.context import SubregionContext
    from fnit.gems.recipes.base import working_image
    from fnit.gems.recipes import ThalamusRecipe, HippoAmygdalaRecipe
    context = SubregionContext.prepare(args.t1, need_coarse=True, need_parc=False,
                                       coarse_segmentation=args.aseg, device='cpu')
    report = {'scope': 'preprocessing_only_no_reference_labels_or_mesh_fit', 'structures': {}}
    for name, recipe in [('thalamus', ThalamusRecipe('thalamus', '.')),
                         ('hippo-amygdala-left', HippoAmygdalaRecipe('left', '.')),
                         ('hippo-amygdala-right', HippoAmygdalaRecipe('right', '.'))]:
        image, _, details = working_image(context, getattr(recipe, 'crop_ids', recipe.alignment_ids), recipe.resolution_mm)
        reference = nib.load(str(args.official / name / 'temporary/tempImage.mgz'))
        expected = np.asarray(reference.dataobj)
        low, high = (np.asarray(details[key]) for key in ['crop_start_native', 'crop_stop_native'])
        crop = tuple(slice(a, b) for a, b in zip(low, high))
        source = context.data[crop]
        relative = np.linalg.inv(context.image.affine) @ image.affine
        matrix, offset = relative[:3, :3], relative[:3, 3] - low
        axes = [offset[d] + np.arange(image.shape[d]) * matrix[d, d] for d in range(3)]
        valid = np.ones(image.shape, bool)
        for d, axis in enumerate(axes):
            rounded = np.rint(axis)
            local = (rounded >= 0) & (rounded < source.shape[d])
            axis_shape = [1, 1, 1]
            axis_shape[d] = len(axis)
            valid &= local.reshape(axis_shape)
        variants = {'current_full_image_nearest': np.asarray(image.dataobj)}
        variants['current_full_image_clip0'] = np.maximum(variants['current_full_image_nearest'], 0)
        mirror = ndimage.affine_transform(source, matrix, offset=offset, output_shape=image.shape,
                                          order=3, mode='mirror').astype(np.float32)
        variants['crop_mirror'] = mirror
        variants['crop_mirror_clip0_outside0'] = np.where(valid, np.maximum(mirror, 0), 0)
        coefficient = source
        for axis in range(3):
            coefficient = ndimage.spline_filter1d(coefficient, 3, axis=axis, output=np.float32, mode='mirror')
        sampled = ndimage.affine_transform(coefficient, matrix, offset=offset, output_shape=image.shape,
                                           order=3, mode='mirror', prefilter=False).astype(np.float32)
        variants['crop_fp32_coeff_clip0_outside0'] = np.where(valid, np.maximum(sampled, 0), 0)
        report['structures'][name] = {key: metric(value, expected) for key, value in variants.items()}
        report['structures'][name]['interior_current'] = metric(np.asarray(image.dataobj)[4:-4, 4:-4, 4:-4],
                                                               expected[4:-4, 4:-4, 4:-4])
        nib.save(nib.Nifti1Image(variants['crop_fp32_coeff_clip0_outside0'], image.affine), args.output / (name + '.nii.gz'))
    (args.output / 'report.public.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
