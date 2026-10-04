"""Compare real preprocessing before the first mesh evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t1', type=Path, required=True)
    parser.add_argument('--aseg', type=Path, required=True)
    parser.add_argument('--wmparc', type=Path, required=True)
    parser.add_argument('--atlas-root', type=Path, required=True)
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new output directory')
    args.output.mkdir(parents=True)
    from fnit.gems.context import SubregionContext
    from fnit.gems.recipes import make_recipe
    context = SubregionContext.prepare(args.t1, need_coarse=True, need_parc=True,
                                       coarse_segmentation=args.aseg, wmparc=args.wmparc, device='cpu')
    report = {'scope': 'first_mesh_preprocessing_difference_real_sameinput', 'inputs': {}, 'structures': {}}
    for field in ['t1', 'aseg', 'wmparc']:
        path = getattr(args, field)
        report['inputs'][field] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in ['thalamus', 'hippo-amygdala-left', 'hippo-amygdala-right']:
        recipe = make_recipe(name, args.atlas_root)
        recipe._preparation_device = torch.device('cpu')
        image, labels, details = recipe.prepare_working_image(context)
        nib.save(image, args.output / (name + '.nii.gz'))
        reference = nib.load(str(args.official / name / 'processedImage.mgz'))
        synthetic_reference = nib.load(str(args.official / name / 'synthImage.mgz'))
        synthetic = recipe.synthetic_labels(context.coarse_segmentation)
        synthetic_old = np.asarray(synthetic_reference.dataobj)
        first, second = np.asarray(image.dataobj), np.asarray(reference.dataobj)
        if second.ndim == 4 and second.shape[-1] == 1:
            second = second[..., 0]
        grid = np.linalg.inv(context.image.affine) @ image.affine
        official_grid = np.linalg.inv(context.image.affine) @ reference.affine
        row = {'fnit_shape': list(first.shape), 'official_shape': list(second.shape),
               'fnit_grid_in_native_voxels': grid.tolist(), 'official_grid_in_native_voxels': official_grid.tolist(),
               'origin_difference_native_voxels': (grid[:3, 3] - official_grid[:3, 3]).tolist(),
               'fnit_sha256': sha(first), 'official_sha256': sha(second),
               'synthetic_same_shape': synthetic.shape == synthetic_old.shape,
               'synthetic_exact': bool(np.array_equal(synthetic, synthetic_old)),
               'synthetic_different_voxels': int(np.count_nonzero(synthetic != synthetic_old)),
               'crop': details}
        if first.shape == second.shape:
            delta = first.astype(np.float64) - second.astype(np.float64)
            row['different_voxels'] = int(np.count_nonzero(delta))
            row['max_abs_difference'] = float(np.abs(delta).max(initial=0))
            row['rmse'] = float(np.sqrt(np.mean(delta ** 2)))
            row['nonzero_mask_difference'] = int(np.count_nonzero((first != 0) != (second != 0)))
        report['structures'][name] = row
    (args.output / 'report.public.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
