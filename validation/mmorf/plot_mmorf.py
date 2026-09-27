#!/usr/bin/env python3
"""Create a de-identified real-data MMORF comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


def load(path):
    data = np.asarray(nib.load(path).dataobj, dtype=np.float32).squeeze()
    if data.ndim != 3:
        raise ValueError(f"{path} does not contain one 3D image")
    return data


def grayscale(data, limit):
    values = np.clip(data / max(limit, np.finfo(np.float32).eps), 0.0, 1.0)
    gray = np.rint(255.0 * values).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def difference_color(data, limit):
    values = np.clip(data / max(limit, np.finfo(np.float32).eps), 0.0, 1.0)
    red = np.rint(255.0 * values)
    green = np.rint(255.0 * np.clip(2.0 * values - 1.0, 0.0, 1.0))
    return np.stack((red, green, np.zeros_like(red)), axis=2).astype(np.uint8)


def panel(data, *, limit, difference=False, size=(340, 390)):
    pixels = difference_color(data, limit) if difference else grayscale(data, limit)
    image = Image.fromarray(pixels, mode="RGB")
    image.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, "black")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def metrics(candidate, reference, support):
    x = candidate[support].astype(np.float64)
    y = reference[support].astype(np.float64)
    return float(np.corrcoef(x, y)[0, 1]), float(np.mean(np.abs(x - y)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-scalar", required=True)
    parser.add_argument("--fnit-scalar", required=True)
    parser.add_argument("--official-fa", required=True)
    parser.add_argument("--fnit-fa", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    official_scalar = load(args.official_scalar)
    fnit_scalar = load(args.fnit_scalar)
    official_fa = load(args.official_fa)
    fnit_fa = load(args.fnit_fa)
    if not (official_scalar.shape == fnit_scalar.shape == official_fa.shape == fnit_fa.shape):
        raise ValueError("all images must share one 3D grid")

    scalar_support = (official_scalar != 0) | (fnit_scalar != 0)
    fa_support = (official_fa != 0) | (fnit_fa != 0)
    if not scalar_support.any() or not fa_support.any():
        raise ValueError("comparison support is empty")
    index = int(np.argmax((scalar_support | fa_support).sum(axis=(0, 1))))
    scalar_r, scalar_mae = metrics(fnit_scalar, official_scalar, scalar_support)
    fa_r, fa_mae = metrics(fnit_fa, official_fa, fa_support)
    scalar_diff = np.abs(fnit_scalar - official_scalar)
    fa_diff = np.abs(fnit_fa - official_fa)
    scalar_limit = float(np.percentile(np.concatenate((official_scalar[scalar_support], fnit_scalar[scalar_support])), 99.5))
    scalar_diff_limit = float(np.percentile(scalar_diff[scalar_support], 99.5))
    fa_diff_limit = float(np.percentile(fa_diff[fa_support], 99.5))

    panel_size = (340, 390)
    header_height = 50
    title_height = 32
    figure = Image.new("RGB", (3 * panel_size[0], header_height + 2 * (title_height + panel_size[1])), "white")
    draw = ImageDraw.Draw(figure)
    draw.text((12, 10), f"Matched real data; T1 r={scalar_r:.4f}, MAE={scalar_mae:.2f}; FA r={fa_r:.4f}, MAE={fa_mae:.4f}", fill="black")
    rows = (
        (official_scalar, fnit_scalar, scalar_diff, scalar_limit, scalar_diff_limit, "warped T1"),
        (official_fa, fnit_fa, fa_diff, 1.0, fa_diff_limit, "warped FA"),
    )
    for row, (official, candidate, difference, limit, diff_limit, label) in enumerate(rows):
        top = header_height + row * (title_height + panel_size[1])
        panels = (
            (official, f"Official warp: {label}", False, limit),
            (candidate, f"TorchMMORF: {label}", False, limit),
            (difference, "absolute difference", True, diff_limit),
        )
        for column, (data, title, is_difference, scale) in enumerate(panels):
            left = column * panel_size[0]
            draw.text((left + 10, top + 9), title, fill="black")
            sliced = np.rot90(data[:, :, index])
            figure.paste(panel(sliced, limit=scale, difference=is_difference, size=panel_size), (left, top + title_height))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output)


if __name__ == "__main__":
    main()
