"""本轮完整490帧原指令与FNIT的模板空间temporal SD和时间相关图。"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def maps(first, second, mask):
    arrays = [np.asarray(nib.load(path).dataobj, dtype=np.float32) for path in (first, second)]
    if arrays[0].shape != arrays[1].shape or arrays[0].shape[3] != 490:
        raise ValueError('This latest figure requires paired full490 images')
    standard_deviations = [array.std(axis=3, dtype=np.float64).astype(np.float32) for array in arrays]
    left, right = [array[mask] for array in arrays]
    r = np.full(left.shape[0], np.nan, dtype=np.float32)
    for start in range(0, len(left), 4096):
        x, y = left[start:start + 4096].astype(np.float64), right[start:start + 4096].astype(np.float64)
        x -= x.mean(axis=1, keepdims=True)
        y -= y.mean(axis=1, keepdims=True)
        xx, yy, xy = np.square(x).sum(1), np.square(y).sum(1), (x * y).sum(1)
        valid = (xx > x.shape[1] * 1e-12) & (yy > y.shape[1] * 1e-12)
        r[start:start + len(x)][valid] = np.clip(xy[valid] / np.sqrt(xx[valid] * yy[valid]), -1, 1)
    volume = np.full(mask.shape, np.nan, dtype=np.float32)
    volume[mask] = r
    return standard_deviations, volume, int(np.isfinite(r).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--comparison-report', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--stages', nargs='+', choices=('preproc_mni', 'clean_mni'),
                        default=['preproc_mni', 'clean_mni'])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    comparison = json.loads(args.comparison_report.read_text())
    template_path = Path(manifest['input_files']['mni_template'])
    template = nib.load(template_path)
    anatomy = np.asarray(template.dataobj, dtype=np.float32)
    results = {}
    args.output_root.mkdir(parents=True, exist_ok=True)
    for stage, label in (('preproc_mni', 'Original FSL one-pass preproc'),
                         ('clean_mni', 'Original SynthStrip / FSL / ICA-AROMA clean')):
        if stage not in args.stages:
            continue
        item = manifest['images'][stage]
        region = np.ones(template.shape, dtype=bool)
        mask_hashes = {}
        for key in ('mask', 'candidate_mask', 'reference_mask'):
            if key in item:
                mask_image = nib.load(item[key])
                if mask_image.shape != template.shape or not np.allclose(mask_image.affine, template.affine, atol=1e-4, rtol=0):
                    raise ValueError('Figure mask does not match the template')
                region &= np.asarray(mask_image.dataobj) > 0
                mask_hashes[key] = sha256(item[key])
        for key in ('candidate', 'reference'):
            image = nib.load(item[key])
            if image.shape[:3] != template.shape or not np.allclose(image.affine, template.affine, atol=1e-4, rtol=0):
                raise ValueError('Figure source has a different template grid')
            if sha256(item[key]) != comparison['images'][stage]['sha256'][key]:
                raise ValueError('Figure source differs from the newest matched comparison')
        sd, correlation, valid = maps(item['candidate'], item['reference'], region)
        scale = float(np.percentile(np.concatenate([array[region] for array in sd]), 99))
        rendered = [anatomy, *sd, correlation]
        labels = ['MNI anatomical template', 'Current FNIT: temporal SD', label + ': temporal SD',
                  'FNIT / original: voxel temporal Pearson r']
        figure, axes = plt.subplots(4, 3, figsize=(10, 12), facecolor='white')
        for row, (array, row_label) in enumerate(zip(rendered, labels)):
            display = np.where(region, array, np.nan if row == 3 else 0)
            vmin, vmax, cmap = (0, float(np.percentile(anatomy[region], 99)), 'gray') if row == 0 else (0, scale, 'gray')
            if row == 3:
                vmin, vmax, cmap = -1, 1, 'coolwarm'
            for axis, name in enumerate(('Sagittal', 'Coronal', 'Axial')):
                plane = np.rot90(np.take(display, display.shape[axis] // 2, axis=axis))
                artist = axes[row, axis].imshow(plane, cmap=cmap, vmin=vmin, vmax=vmax)
                axes[row, axis].axis('off')
                axes[row, axis].set_title((row_label + '\n' if axis == 1 else '') + name, fontsize=10)
            figure.colorbar(artist, ax=axes[row, -1], shrink=.7)
        figure.suptitle('Same raw 490-frame BOLD; SD shares a scale; no additional smoothing', fontsize=12)
        figure.tight_layout()
        destination = args.output_root / (stage + '_fnit_original.png')
        figure.savefig(destination, dpi=140)
        plt.close(figure)
        results[stage] = {'png_sha256': sha256(destination), 'frames': 490,
            'candidate_sha256': sha256(item['candidate']), 'reference_sha256': sha256(item['reference']),
            'mask_sha256': mask_hashes, 'evaluation_voxels': int(region.sum()),
            'varying_time_series_for_r': valid, 'shared_sd_scale_0_to': scale,
            'r_scale': [-1, 1], 'slices_voxel_indices': [size // 2 for size in template.shape],
            'additional_smoothing': False}
    report = {'schema_version': 1, 'validated_on': '2026-10-02',
        'comparison_report_sha256': sha256(args.comparison_report),
        'template_sha256': sha256(template_path), 'figures': results,
        'figure_script_sha256': sha256(__file__),
        'privacy': 'Only authorised deidentified template-space PNG and anonymous hashes; no original images or per-voxel arrays published.'}
    (args.output_root / 'figures.public.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'rendered': list(results)}))


if __name__ == '__main__':
    main()
