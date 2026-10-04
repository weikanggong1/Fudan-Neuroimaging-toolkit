"""Render exported real brainstem slices with the existing local plotting library."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.input)
    colors = ListedColormap(['#e69f00', '#56b4e9', '#009e73', '#cc79a7'])
    ids = (173, 174, 175, 178)
    figure, axes = plt.subplots(3, 3, figsize=(10, 9), facecolor='white')
    for axis, title in enumerate(('Sagittal', 'Coronal', 'Axial')):
        image = data['image' + str(axis)]
        official = data['official' + str(axis)]
        candidate = data['candidate' + str(axis)]
        for row, labels in enumerate((official, candidate, official != candidate)):
            panel = axes[row, axis]
            panel.imshow(image, cmap='gray', vmin=0, vmax=float(data['upper']), origin='lower')
            if row < 2:
                indices = np.full(labels.shape, np.nan)
                for color, identifier in enumerate(ids):
                    indices[labels == identifier] = color
                panel.imshow(indices, cmap=colors, vmin=0, vmax=3, alpha=.65,
                             interpolation='nearest', origin='lower')
            else:
                different = labels.astype(float)
                different[different == 0] = np.nan
                panel.imshow(different, cmap=ListedColormap(['#ff3030']), vmin=0,
                             vmax=1, interpolation='nearest', origin='lower')
            panel.set_xticks([])
            panel.set_yticks([])
            if row == 0:
                panel.set_title(title)
            if axis == 0:
                panel.set_ylabel(('Official', 'FNIT CPU epsilon', 'Label difference')[row])
    figure.suptitle('Public sub-02, same-input brainstem stage; CPU candidate\n'
                   'Orange: Midbrain | Blue: Pons | Green: Medulla | Purple: SCP')
    figure.tight_layout(rect=(0, 0, 1, .93))
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    args.output.with_suffix('.json').write_text(json.dumps({
        'display_only': True, 'fixed_scoring_grid_unchanged': True,
        'slices_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'figure_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'plot_program_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
