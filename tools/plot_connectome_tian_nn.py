"""Plot saved real CPU/CUDA Tian labels; no source MRI pixels are exported."""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report-dir', type=Path, required=True)
parser.add_argument('--candidate-dir', type=Path)
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--subject', default='CON01')
parser.add_argument('--baseline-pattern', default='tian_s{level}_cpu.nii.gz')
parser.add_argument('--candidate-pattern', default='tian_s{level}_cuda.nii.gz')
parser.add_argument('--row-labels', nargs=2, default=('CPU', 'CUDA'))
args = parser.parse_args()
import json
output_dir = args.output_dir or args.report_dir
output_dir.mkdir(parents=True, exist_ok=True)
stats = {}
for level in (1, 4):
    cpu = np.asarray(nib.load(args.report_dir / args.baseline_pattern.format(level=level)).dataobj)
    gpu = np.asarray(nib.load((args.candidate_dir or args.report_dir) / args.candidate_pattern.format(level=level)).dataobj)
    if cpu.shape != gpu.shape: raise ValueError('label shape differs')
    stats[f's{level}'] = {'baseline_foreground': int(np.count_nonzero(cpu)), 'candidate_foreground': int(np.count_nonzero(gpu)), 'baseline_positive_labels': np.unique(cpu[cpu > 0]).astype(int).tolist(), 'candidate_positive_labels': np.unique(gpu[gpu > 0]).astype(int).tolist(), 'neq': int(np.count_nonzero(cpu != gpu))}
    positive = np.argwhere(cpu > 0)
    if not len(positive):
        raise RuntimeError('no foreground atlas for plot')
    # Choose the largest foreground cross-section independently on each voxel axis.
    slices = [int(np.argmax(np.count_nonzero(cpu, axis=tuple(a for a in range(3) if a != axis)))) for axis in range(3)]
    lower = np.maximum(positive.min(axis=0) - 5, 0)
    upper = np.minimum(positive.max(axis=0) + 6, cpu.shape)
    color_map = plt.get_cmap('tab20').copy()
    color_map.set_bad('black')
    fig, axes = plt.subplots(3, 3, figsize=(9, 8))
    for row, (label, data) in enumerate(((args.row_labels[0], cpu), (args.row_labels[1], gpu), ('difference', cpu != gpu))):
        for axis in range(3):
            plane = np.take(data, slices[axis], axis=axis)
            remaining = [a for a in range(3) if a != axis]
            cropped = plane[tuple(slice(lower[a], upper[a]) for a in remaining)].T
            shown = np.ma.masked_equal(cropped, 0) if row < 2 else cropped
            axes[row, axis].imshow(shown, origin='lower', interpolation='nearest',
                cmap=color_map if row < 2 else 'gray', vmin=0, vmax=int(cpu.max()) if row < 2 else 1)
            axes[row, axis].set_title(f'{label}: voxel axis {axis}, slice {slices[axis]}', fontsize=9)
            axes[row, axis].axis('off')
    fig.suptitle(f'New ds001226 {args.subject}, Tian S{level}: {args.row_labels[0]}/{args.row_labels[1]}\nForeground crop; black difference = equal labels')
    fig.tight_layout()
    fig.savefig(output_dir / f'tian_s{level}_NN.png', dpi=160)
    plt.close(fig)
(output_dir / 'label_stats.json').write_text(json.dumps(stats, indent=2) + '\n')
print('figure environment:', matplotlib.__version__, nib.__version__)
