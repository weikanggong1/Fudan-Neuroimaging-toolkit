#!/usr/bin/env python3
"""Plot one native-grid real DWI frame and same-frame official EDDY differences."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fnit', 'cpu', 'gpu', 'mask', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--frame', type=int, default=1)
    parser.add_argument('--slice', type=int, default=30, dest='slice_index')
    args = parser.parse_args()
    paths = {'FNIT': args.fnit, 'FSL CPU': args.cpu, 'FSL GPU': args.gpu}
    images = {name: nib.load(path) for name, path in paths.items()}
    first = images['FNIT']
    mask_image = nib.load(args.mask)
    for image in [*images.values(), mask_image]:
        if image.shape[:3] != first.shape[:3] or not np.array_equal(image.affine, first.affine):
            raise ValueError('all images must have exactly the same native grid and affine')
    if not 0 <= args.frame < first.shape[3] or not 0 <= args.slice_index < first.shape[2]:
        raise ValueError('frame or slice outside image')
    data = {name: np.asarray(image.dataobj[:, :, :, args.frame], dtype=np.float32)
            for name, image in images.items()}
    mask = np.asarray(mask_image.dataobj) > 0
    intensity_max = float(np.percentile(np.concatenate([value[mask] for value in data.values()]), 99))
    differences = [('FNIT - FSL CPU', data['FNIT'] - data['FSL CPU']),
                   ('FNIT - FSL GPU', data['FNIT'] - data['FSL GPU']),
                   ('FSL CPU - FSL GPU', data['FSL CPU'] - data['FSL GPU'])]
    difference_max = float(max(np.max(np.abs(value[mask])) for _, value in differences))
    canvas = Image.new('RGB', (990, 790), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f'CON03 native grid: frame {args.frame}, z={args.slice_index}; no resampling', fill='black')
    panels = [*data.items(), *differences]
    for index, (title, value) in enumerate(panels):
        row, column = divmod(index, 3)
        view = value[:, :, args.slice_index]
        if row == 0:
            level = np.clip(view / intensity_max, 0, 1)
            rgb = np.stack([level] * 3, axis=-1)
        else:
            level = np.clip(view / difference_max, -1, 1)
            rgb = np.stack([1 + np.minimum(level, 0), 1 - np.abs(level), 1 - np.maximum(level, 0)], axis=-1)
            rgb[~mask[:, :, args.slice_index]] = 0
        panel = Image.fromarray(np.asarray(np.rot90(rgb) * 255, dtype=np.uint8))
        panel = panel.resize((288, 288), Image.Resampling.NEAREST)
        canvas.paste(panel, (column * 330 + 21, 70 + row * 355))
        draw.text((column * 330 + 12, 45 + row * 355), title, fill='black')
        draw.text((column * 330 + 12, 365 + row * 355),
                  f'0 to {intensity_max:.2f}' if row == 0 else f'Signed: {-difference_max:.2f} to {difference_max:.2f}', fill='black')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    report = {'frame': args.frame, 'native_z': args.slice_index, 'resampling': False,
              'intensity_limits': [0, intensity_max], 'difference_limits': [-difference_max, difference_max],
              'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [*paths.values(), args.mask]}}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
