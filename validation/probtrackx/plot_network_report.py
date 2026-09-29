"""从私有配对报告生成不含原始矩阵数值的五区连接图。"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.report.read_text())["cases"]["network_gpu"]
    canvas = Image.new("RGB", (1240, 820), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 17)
    large = ImageFont.truetype("DejaVuSans.ttf", 23)
    draw.text((30, 15), "Real dMRI: five-region directed network", fill="#202936", font=large)
    draw.text((30, 50), "FSL probtrackx2_gpu / FNIT H100; 2000 tracks per seed voxel",
              fill="#596273", font=font)
    labels = ("CCg", "CST-R", "CST-L", "SLF-R", "SLF-L")
    short = ("CCg", "CR", "CL", "SR", "SL")
    for row, (key, label) in enumerate((("matrix", "Length-weighted path density"),
                                         ("mean_length_matrix", "Mean path length (mm)"))):
        fsl = np.asarray(case[key]["fsl"], dtype=float)
        fnit = np.asarray(case[key]["fnit"], dtype=float)
        scale = max(float(fsl.max()), float(fnit.max()), 1e-8)
        difference = fnit - fsl
        diff_scale = max(float(np.max(np.abs(difference))), 1e-8)
        for col, (title, data) in enumerate((("FSL", fsl), ("FNIT", fnit), ("FNIT - FSL", difference))):
            x0, y0 = 25 + 405 * col, 90 + 355 * row
            draw.rounded_rectangle((x0, y0, x0 + 385, y0 + 335), radius=12,
                                   fill="#f7f9fc", outline="#c8d0da", width=2)
            draw.text((x0 + 15, y0 + 12), f"{label}: {title}", fill="#202936", font=font)
            for i in range(5):
                draw.text((x0 + 10, y0 + 80 + 47 * i), labels[i], fill="#596273", font=font)
                draw.text((x0 + 85 + 47 * i, y0 + 305), short[i], fill="#596273", font=font)
                for j in range(5):
                    value = float(data[i, j])
                    if col == 2:
                        strength = min(1.0, abs(value) / diff_scale)
                        color = ((255, int(245 - 170 * strength), int(245 - 170 * strength))
                                 if value >= 0 else
                                 (int(245 - 170 * strength), int(245 - 170 * strength), 255))
                    else:
                        strength = min(1.0, value / scale)
                        color = (int(235 - 190 * strength), int(245 - 130 * strength),
                                 int(250 - 30 * strength))
                    left, top = x0 + 82 + 47 * j, y0 + 77 + 47 * i
                    draw.rectangle((left, top, left + 44, top + 44), fill=color)
    draw.text((30, 800), "ROI labels are generic; source images and numeric matrices remain on the research server.",
              fill="#596273", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)


if __name__ == "__main__":
    main()
