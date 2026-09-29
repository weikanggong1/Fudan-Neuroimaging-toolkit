"""绘制真实诊断 ROI 中 FSL 与 FNIT 的次要纤维方向轴。"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load(folder, name):
    return np.asarray(nib.load(str(folder / f"{name}.nii.gz")).dataobj)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsl-dir", type=Path, required=True)
    parser.add_argument("--fnit-dir", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mask = np.asarray(nib.load(str(args.mask)).dataobj) > 0
    canvas = Image.new("RGB", (1720, 960), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    title_font = ImageFont.truetype("DejaVuSans.ttf", 25)
    draw.text((40, 20), "Real dMRI diagnostic ROI: secondary-fibre axes", fill="#202936", font=title_font)
    draw.text((40, 58), "Shared support: FSL and FNIT posterior mean fraction >= 0.1", fill="#596273", font=font)

    for row, fibre in enumerate((2, 3)):
        fsl_fraction = load(args.fsl_dir, f"mean_f{fibre}samples")
        fnit_fraction = load(args.fnit_dir, f"mean_f{fibre}samples")
        fsl_axis = load(args.fsl_dir, f"dyads{fibre}")
        fnit_axis = load(args.fnit_dir, f"dyads{fibre}")
        supported = mask & (fsl_fraction >= 0.1) & (fnit_fraction >= 0.1)
        coords = np.argwhere(supported)
        a, b = fsl_axis[supported], fnit_axis[supported]
        cosine = np.abs(np.sum(a * b, axis=1)) / np.maximum(
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8)
        angles = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
        median = float(np.median(angles)) if len(angles) else float("nan")
        for col, heading in enumerate(("FSL xfibres", "FNIT H100", "acute axis difference")):
            x0, y0 = 40 + col * 560, 100 + row * 425
            draw.rounded_rectangle((x0, y0, x0 + 530, y0 + 395), radius=12,
                                   fill="#f7f9fc", outline="#c8d0da", width=2)
            draw.text((x0 + 18, y0 + 15), f"f{fibre}: {heading}", fill="#202936", font=title_font)
            if col == 2:
                draw.text((x0 + 18, y0 + 50), f"n={len(coords)}, median={median:.2f} deg",
                          fill="#596273", font=font)
            for coord, fsl_vector, fnit_vector, angle in zip(coords, a, b, angles):
                x, y, z = coord
                px = x0 + 105 + 27 * x + 17 * y
                py = y0 + 200 + 16 * y - 16 * z
                if col == 2:
                    shade = min(255, int(45 + angle * 9))
                    color = (shade, 80, 255 - shade // 2)
                    draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
                    continue
                vector = fsl_vector if col == 0 else fnit_vector
                fraction = fsl_fraction[tuple(coord)] if col == 0 else fnit_fraction[tuple(coord)]
                color = (25, min(210, int(75 + fraction * 300)), 165)
                dx = 17 * vector[0] + 10 * vector[1]
                dy = 14 * vector[1] - 14 * vector[2]
                draw.line((px - dx, py - dy, px + dx, py + dy), fill=color, width=3)
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)


if __name__ == "__main__":
    main()
