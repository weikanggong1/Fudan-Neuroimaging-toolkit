"""Plot a de-identified FSL/FNIT TBSS FA comparison from matched outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


def _load(path):
    data = np.asarray(nib.load(path).dataobj, dtype=np.float32).squeeze()
    if data.ndim != 3:
        raise ValueError(f"{path} does not contain one 3D image")
    return data


def _slice(data, axis, index):
    return np.rot90(np.take(data, index, axis=axis))


def _grayscale(data):
    values = np.clip(data, 0.0, 1.0)
    return np.repeat(
        np.rint(255.0 * values)[..., None].astype(np.uint8), 3, axis=2
    )


def _difference_color(data, limit):
    values = np.clip(
        data / max(limit, np.finfo(np.float32).eps), 0.0, 1.0
    )
    red = np.rint(255.0 * values)
    green = np.rint(255.0 * np.clip(2.0 * values - 1.0, 0.0, 1.0))
    blue = np.zeros_like(red)
    return np.stack((red, green, blue), axis=2).astype(np.uint8)


def _panel(data, *, difference_limit=None, size=(360, 420)):
    pixels = (
        _grayscale(data)
        if difference_limit is None
        else _difference_color(data, difference_limit)
    )
    image = Image.fromarray(pixels, mode="RGB")
    image.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, "black")
    canvas.paste(
        image,
        ((size[0] - image.width) // 2, (size[1] - image.height) // 2),
    )
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True, help="FSL standard-space FA")
    parser.add_argument("--fnit", required=True, help="FNIT standard-space FA")
    parser.add_argument("--output", required=True, help="output PNG")
    args = parser.parse_args(argv)

    official = _load(args.official)
    candidate = _load(args.fnit)
    if official.shape != candidate.shape:
        raise ValueError("official and FNIT images must have the same shape")
    support = (official != 0) | (candidate != 0)
    if not np.any(support):
        raise ValueError("comparison support is empty")
    x = official[support].astype(np.float64)
    y = candidate[support].astype(np.float64)
    pearson = float(np.corrcoef(x, y)[0, 1])
    mae = float(np.mean(np.abs(x - y)))
    difference = np.abs(candidate - official)
    difference_limit = float(np.percentile(difference[support], 99.5))

    views = (
        (2, official.shape[2] // 2, "axial"),
        (1, official.shape[1] // 2, "coronal"),
    )
    panel_size = (360, 420)
    title_height = 34
    header_height = 48
    figure = Image.new(
        "RGB",
        (
            3 * panel_size[0],
            header_height + 2 * (title_height + panel_size[1]),
        ),
        "white",
    )
    draw = ImageDraw.Draw(figure)
    draw.text(
        (12, 14),
        f"Matched real FA: r={pearson:.6f}, MAE={mae:.6f}",
        fill="black",
    )
    for row, (axis, index, label) in enumerate(views):
        panels = (
            (official, "FSL/UKB", None),
            (candidate, "FNIT TorchTBSS", None),
            (difference, "absolute difference", difference_limit),
        )
        top = header_height + row * (title_height + panel_size[1])
        for column, (data, title, limit) in enumerate(panels):
            left = column * panel_size[0]
            draw.text(
                (left + 10, top + 10),
                f"{title} ({label})",
                fill="black",
            )
            figure.paste(
                _panel(
                    _slice(data, axis, index),
                    difference_limit=limit,
                    size=panel_size,
                ),
                (left, top + title_height),
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output)


if __name__ == "__main__":
    main()
