"""Display current real joint outputs within the original brain mask only."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'mask', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    base = Path(args.root) / 'runs/smri_cpu_20261004/remaining_20261004/morph'
    mask_image = nib.load(args.mask)
    mask = np.asarray(mask_image.dataobj) > .5
    occupied = np.argwhere(mask)
    low, high = occupied.min(0), occupied.max(0) + 1
    z = int(np.median(occupied[:, 2]))
    region = slice(low[0], high[0]), slice(low[1], high[1]), z
    figure, axes = plt.subplots(2, 3, figsize=(10, 7), constrained_layout=True)
    rows = []
    for row, (extent, original_folder, candidate_folder) in enumerate((
            (256, 'nodecw7_paired_v2/joint_reference', 'nodecw7_final_v29/joint'),
            (192, 'nodecw7_final_v7/joint_192_reference', 'nodecw7_final_v29/joint_192'))):
        paths = [base / folder / 'moved.nii.gz' for folder in (original_folder, candidate_folder)]
        images = [nib.load(path) for path in paths]
        if any(image.shape != mask.shape or not np.array_equal(image.affine, mask_image.affine) for image in images):
            raise ValueError('original mask must match the complete candidate/reference geometry')
        raw = [np.asarray(image.dataobj) for image in images]
        hashes = [hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() for array in raw]
        reference, candidate = [np.where(mask, array, 0) for array in raw]
        difference = candidate - reference
        vmax = float(np.percentile(reference[mask], 99))
        error_scale = max(.01, float(np.percentile(np.abs(difference[mask]), 99)))
        for column, (array, title) in enumerate(((reference, 'Official CPU'), (candidate, 'FNIT CPU v29'), (difference, 'FNIT - official'))):
            options = {'cmap': 'gray', 'vmin': 0, 'vmax': vmax} if column < 2 else {'cmap': 'coolwarm', 'vmin': -error_scale, 'vmax': error_scale}
            display = axes[row, column].imshow(array[region].T, origin='lower', **options)
            axes[row, column].set_title('extent ' + str(extent) + ' | ' + title)
            axes[row, column].axis('off')
            if column == 2:
                figure.colorbar(display, ax=axes[row, column], fraction=.046, pad=.02)
        rows.append({'extent': extent, 'source_array_sha256': hashes, 'source_file_sha256': [digest(path) for path in paths],
                     'brain_masked_before_display': True, 'axial_slice': z, 'display_error_scale': error_scale})
    output = Path(args.output)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    output.with_suffix('.json').write_text(json.dumps({
        'scope': 'OpenNeuro ds003138 CC0 full real T1 outputs; brain display only; precision gates use full FOV',
        'rows': rows, 'mask_file_sha256': digest(args.mask), 'worker_sha256': digest(__file__), 'figure_sha256': digest(output)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
