"""Render an unsmoothed, deidentified MNI-space temporal-SD comparison."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-maps', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--fsl-bold', type=Path, help='Optional same-grid, full-frame FSL spline output')
    parser.add_argument('--figure-out', type=Path, required=True)
    args = parser.parse_args()
    maps = np.load(args.private_maps)
    report = json.loads(args.report.read_text())
    mask = maps['mask']
    scale = np.percentile(maps['native_sd_in_mni'][mask], 98)
    rows = [('native_sd_in_mni', 'Native temporal SD mapped to MNI (amplitude reference)'),
            ('linear', 'Old: temporal SD after trilinear BOLD resampling'),
            ('spline', 'Fixed: temporal SD after FNIT cubic B-spline resampling')]
    maps = dict(maps)
    if args.fsl_bold:
        official = np.asarray(nib.load(args.fsl_bold).dataobj, dtype=np.float32)
        if official.shape != (*mask.shape, report['data']['frames']):
            raise ValueError('FSL frame count or grid shape differs')
        maps['fsl'] = official.std(axis=3)
        rows.append(('fsl', 'FSL applywarp --interp=spline: temporal SD'))
    fig = plt.figure(figsize=(12, 3 * len(rows) + 1), facecolor='white')
    layout = fig.add_gridspec(len(rows) + 1, 3, height_ratios=[1] * len(rows) + [.8], hspace=.25)
    for row, (name, title) in enumerate(rows):
        data = np.where(mask, maps[name], 0)
        for axis, plane_name in enumerate(('Sagittal', 'Coronal', 'Axial')):
            ax = fig.add_subplot(layout[row, axis])
            plane = np.rot90(np.take(data, data.shape[axis] // 2, axis=axis))
            view = ax.imshow(plane, cmap='gray', vmin=0, vmax=scale, interpolation='nearest')
            ax.axis('off')
            ax.set_title(title if axis == 1 else plane_name, fontsize=10)
        fig.colorbar(view, ax=ax, shrink=.7)
    ax = fig.add_subplot(layout[-1, :])
    for name, label in (('linear', 'Trilinear'), ('spline', 'Cubic B-spline')):
        bins = [b for b in report['phase_analysis'][name]['bins'] if b['voxels'] >= 100]
        x = [(b['weight_energy_low'] + b['weight_energy_high']) / 2 for b in bins]
        ax.plot(x, [b['mean_relative_sd'] for b in bins], 'o-', label=label)
    ax.set(xlabel='Trilinear weight energy (low = near half-voxel positions)',
           ylabel='Mean relative temporal SD')
    ax.legend()
    ax.grid(alpha=.2)
    fig.suptitle(f"Same real {report['data']['frames']}-frame BOLD, same exact warp; shared SD colour scale", fontsize=13)
    args.figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_out, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
