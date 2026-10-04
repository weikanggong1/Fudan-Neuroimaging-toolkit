"""Disentangle real thalamic cubic interpolation and nearest mask ties."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def metric(first, second):
    difference = first.astype(np.float64) - second.astype(np.float64)
    return {'different_values': int(np.count_nonzero(difference)),
            'max_abs': float(np.abs(difference).max(initial=0)),
            'rmse': float(np.sqrt(np.mean(difference * difference)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t1', type=Path, required=True)
    parser.add_argument('--aseg', type=Path, required=True)
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--fnit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new report file')
    from fnit.gems.context import SubregionContext
    from fnit.gems.recipes.base import working_image, _native
    from fnit.gems.recipes.thalamus import ThalamusRecipe
    context = SubregionContext.prepare(args.t1, need_coarse=True, need_parc=False,
                                       coarse_segmentation=args.aseg, device='cpu')
    image, _, details = working_image(context, (10, 49, 28, 60), .5)
    official = nib.load(str(args.official / 'processedImage.mgz'))
    unmasked = nib.load(str(args.official / 'temporary/tempImage.mgz'))
    a, b = np.asarray(image.dataobj), np.asarray(unmasked.dataobj)
    reference = np.asarray(official.dataobj)
    if reference.ndim == 4:
        reference = reference[..., 0]
    reference_mask = reference != 0
    mask = ndimage.binary_dilation(ThalamusRecipe('thalamus', '.').synthetic_labels(context.coarse_segmentation) > 1,
                                   structure=np.ones((3, 3, 3)), iterations=2)
    rows = {}
    for name, target in [('current_affine', image), ('official_affine', official),
                         ('mgh_header_roundtrip', nib.Nifti1Image(a, nib.MGHImage(a, image.affine).header.get_affine()))]:
        sampled = _native(mask.astype(np.uint8), context.image.affine, target, 0).astype(bool)
        rows[name] = metric(sampled, reference_mask)
    origin = np.asarray(details['crop_start_native'])
    for name, rounding in [('ties_upper', lambda p: np.floor(p + .5)),
                           ('ties_lower', lambda p: np.ceil(p - .5)),
                           ('ties_even', np.rint)]:
        axes = [rounding(origin[d] + np.arange(image.shape[d]) * .5).astype(int) for d in range(3)]
        sampled = mask[np.ix_(*axes)]
        rows[name] = metric(sampled, reference_mask)
    saved = np.asarray(nib.load(str(args.fnit)).dataobj)
    report = {'scope': 'real_preprocessing_only_not_fitting_or_speed',
              'unmasked_cubic': metric(a, b), 'mask_variants': rows,
              'final_masked': metric(saved, reference),
              'different_unmasked_where_both_masks_valid': metric(a[reference_mask & (saved != 0)], b[reference_mask & (saved != 0)]),
              'different_unmasked_above_1e_minus3': int(np.count_nonzero(np.abs(a.astype(np.float64) - b) > .001))}
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
