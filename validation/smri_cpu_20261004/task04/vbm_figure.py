"""Plot brain-masked real VBM outputs; do not publish raw T1 or templates."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def display_region(mask):
    occupied = np.argwhere(mask)
    low, high = occupied.min(axis=0), occupied.max(axis=0) + 1
    z = int(np.median(occupied[:, 2]))
    return (slice(low[0], high[0]), slice(low[1], high[1]), z)


def data(path, mask):
    array = np.asanyarray(nib.load(path).dataobj)
    if array.shape != mask.shape:
        raise ValueError('display mask and saved image must have the same grid')
    return np.where(mask, array, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-fnirt', required=True, type=Path)
    parser.add_argument('--candidate-morph', required=True, type=Path)
    parser.add_argument('--reference-fnirt', required=True, type=Path)
    parser.add_argument('--reference-morph', required=True, type=Path)
    parser.add_argument('--template', required=True)
    parser.add_argument('--reference-mask', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    native_mask = np.asanyarray(nib.load(args.reference_fnirt / 'brain_mask.nii.gz').dataobj) > 0
    template_mask = ((np.asanyarray(nib.load(args.reference_mask).dataobj) > 0) &
                     (np.asanyarray(nib.load(args.template).dataobj) > 0))
    native_region, standard_region = display_region(native_mask), display_region(template_mask)
    name = 'T1_brain_pve_1.nii.gz'
    reference = data(args.reference_fnirt / name, native_mask)
    candidate = data(args.candidate_fnirt / name, native_mask)
    scale = max(.001, float(np.percentile(np.abs(candidate-reference)[native_mask], 99)))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for column, (array, title) in enumerate(((reference, 'Original FAST GM'),
                                            (candidate, 'FNIT FAST GM'),
                                            (candidate-reference, 'FNIT - original'))):
        options = dict(cmap='gray', vmin=0, vmax=1) if column < 2 else dict(
            cmap='coolwarm', vmin=-scale, vmax=scale)
        shown = axes[column].imshow(array[native_region].T, origin='lower', **options)
        axes[column].set_title(title);axes[column].axis('off')
        if column == 2:
            fig.colorbar(shown, ax=axes[column], fraction=.046, pad=.02)
    fig.savefig(args.output_dir / 'native_gm.png', dpi=140);plt.close(fig)
    fig, axes = plt.subplots(3, 6, figsize=(20, 11), constrained_layout=True)
    rows = []
    for row, (name, label) in enumerate((
            ('T1_GM_to_template_GM.nii.gz', 'Warped GM'),
            ('T1_GM_JAC_nl.nii.gz', 'Nonlinear Jacobian'),
            ('T1_GM_to_template_GM_mod.nii.gz', 'Modulated GM'))):
        references = [data(root / name, template_mask) for root in (
            args.reference_fnirt, args.reference_morph)]
        candidates = [data(root / name, template_mask) for root in (
            args.candidate_fnirt, args.candidate_morph)]
        combined = np.concatenate([r[template_mask] for r in references])
        low = float(np.percentile(combined, 1)) if row == 1 else 0.
        high = float(np.percentile(combined, 99))
        differences = [c-r for c, r in zip(candidates, references)]
        scale = max(.001, float(np.percentile(np.concatenate(
            [np.abs(d)[template_mask] for d in differences]), 99)))
        for branch in range(2):
            for item, (array, title) in enumerate((
                    (references[branch], 'Original ' + ('FNIRT' if branch == 0 else 'Morph')),
                    (candidates[branch], 'FNIT ' + ('FNIRT' if branch == 0 else 'Morph')),
                    (differences[branch], 'FNIT - original'))):
                ax = axes[row, branch*3 + item]
                options = dict(cmap='gray', vmin=low, vmax=high) if item < 2 else dict(
                    cmap='coolwarm', vmin=-scale, vmax=scale)
                shown = ax.imshow(array[standard_region].T, origin='lower', **options)
                ax.set_title(label + '\n' + title);ax.axis('off')
                if item == 2:
                    fig.colorbar(shown, ax=ax, fraction=.046, pad=.02)
        rows.append({'image': name, 'display_value_min': low, 'display_value_max': high,
                     'display_error_limit': scale})
    fig.savefig(args.output_dir / 'standard_vbm.png', dpi=140);plt.close(fig)
    (args.output_dir / 'figure_scope.json').write_text(json.dumps({
        'scope': 'real CC0 T1 outputs; official brain masks applied before display; display boxes only, full-grid metrics unchanged',
        'native_slice_z': native_region[2], 'standard_slice_z': standard_region[2],
        'difference_color_limits': 'brain-region absolute difference P99; minimum 0.001; values outside color scale clipped',
        'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
