"""Plot actual CON01 T1 and cortical portions of its two volume templates."""

import argparse
import json
from pathlib import Path

import colorsys
from PIL import Image, ImageDraw, ImageFont
import nibabel as nib
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--bindings", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
entry = json.loads(args.bindings.read_text())["cases"]["sub-CON01"]
root = Path(entry["anatomy"]["directory"])
brain_image = nib.as_closest_canonical(nib.load(root / "mri/brain.mgz"))
brain = np.asarray(brain_image.dataobj)
images = [nib.as_closest_canonical(nib.load(root / f"mri/{name}.mgz"))
          for name in ("aparc+aseg", "aparc.a2009s+aseg")]
for image in images:
    assert image.shape == brain.shape and np.allclose(image.affine, brain_image.affine)
labels = [np.asarray(image.dataobj) for image in images]
coordinates = np.argwhere(brain > np.percentile(brain[brain > 0], 15))
center = np.rint(np.median(coordinates, axis=0)).astype(int)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 14)
except OSError:
    font = ImageFont.load_default()
canvas = Image.new("RGB", (900, 660), "black")
draw = ImageDraw.Draw(canvas)
draw.text((12, 12), "ds001226 CON01: actual T1 / template inputs (native grid)", fill="white", font=font)
vmax = np.percentile(brain[brain > 0], 99)
for row, axis in enumerate((2, 1)):
    background = np.flip(np.take(brain, center[axis], axis=axis).T, axis=0)
    gray = np.clip(background / vmax * 255, 0, 255).astype(np.uint8)
    for col in range(3):
        rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float64)
        if col:
            raw = np.flip(np.take(labels[col - 1], center[axis], axis=axis).T, axis=0)
            values = sorted(int(v) for v in np.unique(raw) if 1000 <= v < 3000 or 11000 <= v < 13000)
            for i, value in enumerate(values, 1):
                color = np.asarray(colorsys.hsv_to_rgb((i * .61803398875) % 1, .85, 1.)) * 255
                rgb[raw == value] = .25 * rgb[raw == value] + .75 * color
        panel = Image.fromarray(np.rint(rgb).astype(np.uint8)).resize((294, 294), resample=Image.Resampling.NEAREST)
        canvas.paste(panel, (col * 300 + 3, 65 + row * 294))
        if row == 0:
            draw.text((col * 300 + 5, 42), ("Actual recon-all brain", "aparc cortical labels", "a2009s cortical labels")[col],
                      fill="white", font=font)
args.output.parent.mkdir(parents=True, exist_ok=True)
canvas.save(args.output)
print(args.output.name)
