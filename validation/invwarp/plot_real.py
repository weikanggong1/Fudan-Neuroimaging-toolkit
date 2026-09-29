"""Draw the real DWI FSL/FNIT inverse-mask comparison."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fa", required=True)
    parser.add_argument("--fsl-mask", required=True)
    parser.add_argument("--fnit-mask", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fa = nib.load(args.fa).get_fdata(dtype=np.float32)
    fsl = nib.load(args.fsl_mask).get_fdata() > 0
    fnit = nib.load(args.fnit_mask).get_fdata() > 0
    if fa.shape != fsl.shape or fa.shape != fnit.shape:
        raise ValueError("FA and masks must share the diffusion grid")
    difference_by_slice = (fsl ^ fnit).sum(axis=(0, 1))
    z = int(np.argmax(difference_by_slice if difference_by_slice.any()
                      else (fsl | fnit).sum(axis=(0, 1))))
    background = np.clip(fa[:, :, z] / max(float(np.percentile(fa[fa > 0], 99)), 1e-6), 0, 1)
    gray = (background * 220).astype(np.uint8)
    canvas = Image.new("RGB", (3 * 356, 704), "white")
    draw = ImageDraw.Draw(canvas)
    points = np.argwhere((fsl | fnit)[:, :, z])
    lower = np.maximum(points.min(axis=0) - 5, 0)
    upper = np.minimum(points.max(axis=0) + 6, np.asarray(fsl.shape[:2]))
    for index, (title, mask, color) in enumerate((
        ("FSL inverse + applywarp", fsl[:, :, z], (255, 65, 48)),
        ("FNIT inverse + applywarp", fnit[:, :, z], (36, 190, 255)),
        ("Different voxels", (fsl ^ fnit)[:, :, z], (255, 185, 30)),
    )):
        rgb = np.repeat(gray[..., None], 3, axis=2)
        rgb[mask] = (0.45 * rgb[mask] + 0.55 * np.asarray(color)).astype(np.uint8)
        panel = Image.fromarray(np.rot90(rgb)).resize((324, 300), Image.Resampling.NEAREST)
        detail = Image.fromarray(np.rot90(rgb[lower[0]:upper[0], lower[1]:upper[1]])).resize(
            (324, 300), Image.Resampling.NEAREST)
        canvas.paste(panel, (index * 356 + 16, 32))
        canvas.paste(detail, (index * 356 + 16, 368))
        draw.text((index * 356 + 16, 10), title, fill=(0, 0, 0))
    draw.text((16, 342), "Enlarged ROI", fill=(0, 0, 0))
    draw.text((16, 684), f"Real FA, diffusion slice {z}; FSL {fsl.sum()} voxels; FNIT {fnit.sum()} voxels", fill=(0, 0, 0))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


if __name__ == "__main__":
    main()
