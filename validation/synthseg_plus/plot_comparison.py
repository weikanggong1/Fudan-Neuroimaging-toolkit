#!/usr/bin/env python3
"""Draw matched T1w and SynthSeg+ slices with Pillow."""

import argparse
import colorsys
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from PIL import Image, ImageDraw


def plane(data, axis, index):
    return np.rot90(np.take(data, index, axis=axis))


def panel(gray, labels=None, differences=None):
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if labels is not None:
        for label in np.unique(labels):
            if label == 0:
                continue
            color = np.array(colorsys.hsv_to_rgb((int(label) * 0.618034) % 1, 0.75, 1)) * 255
            rgb[labels == label] = (0.45 * rgb[labels == label] + 0.55 * color).astype(np.uint8)
    if differences is not None:
        rgb[differences] = (255, 45, 25)
    return Image.fromarray(rgb).resize((320, 320), Image.Resampling.NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--fnit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    official_image = nib.load(str(args.official))
    fnit_image = nib.load(str(args.fnit))
    if official_image.shape != fnit_image.shape or not np.allclose(
            official_image.affine, fnit_image.affine, atol=1e-5):
        raise ValueError("Official and FNIT label grids must match")
    official = np.asanyarray(official_image.dataobj)
    fnit = np.asanyarray(fnit_image.dataobj)
    t1 = resample_from_to(nib.load(str(args.image)), official_image, order=1).get_fdata(
        dtype=np.float32)
    foreground = t1[official != 0]
    low, high = np.percentile(foreground, (1, 99))
    gray = np.uint8(np.clip((t1 - low) / (high - low), 0, 1) * 255)
    center = np.median(np.argwhere(official != 0), axis=0).astype(int)
    difference = official != fnit

    canvas = Image.new("RGB", (4 * 320 + 5 * 12, 2 * 320 + 3 * 44), "white")
    draw = ImageDraw.Draw(canvas)
    titles = ("T1w input", "FreeSurfer SynthSeg+", "FNIT SynthSeg+", "Different voxels")
    for column, title in enumerate(titles):
        draw.text((12 + column * 332, 12), title, fill="black")
    for row, (axis, caption) in enumerate(((2, "Axial (center)"),
                                           (1, "Coronal (most differences)"))):
        index = int(center[axis]) if row == 0 else int(np.count_nonzero(difference, axis=(0, 2)).argmax())
        background = plane(gray, axis, index)
        images = (panel(background),
                  panel(background, labels=plane(official, axis, index)),
                  panel(background, labels=plane(fnit, axis, index)),
                  panel(background, differences=plane(difference, axis, index)))
        y = 44 + row * 364
        draw.text((12, y + 324), caption, fill="black")
        for column, image in enumerate(images):
            canvas.paste(image, (12 + column * 332, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == "__main__":
    main()
