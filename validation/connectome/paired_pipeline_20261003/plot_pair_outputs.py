"""CPU Pillow QC from actual saved pair matrices, labels and shared-core FA.

No tracking, image resampling or source input templates are substituted. Output
is anonymous; existing image/matrix files remain unchanged. Only publish an
image after checking the source dataset's redistribution/privacy permission.
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _published_core(output, checkpoint_root):
    state = json.loads((output / "pairs_run_state.json").read_text())
    key = state["cache_status"]["core_key"]
    root = checkpoint_root / "shared/core" / key
    marker = json.loads((root / "complete.json").read_text())
    if marker["stage"] != "core" or marker["key"] != key:
        raise ValueError("shared core marker and saved pair output disagree")
    generation = root / marker["generation"]
    # Numeric NPY only; no pickle, no CUDA or reconstruction.
    fa = np.load(generation / "fa.npy", mmap_mode="r", allow_pickle=False)
    affine = np.load(generation / "dwi_affine.npy", allow_pickle=False)
    image = nib.as_closest_canonical(nib.Nifti1Image(np.asarray(fa), affine))
    return state, image


def _fit(panel, size):
    canvas = Image.new("RGB", size, "black")
    scale = min(size[0] / panel.width, size[1] / panel.height)
    resized = panel.resize((max(1, round(panel.width * scale)),
                            max(1, round(panel.height * scale))),
                           resample=Image.Resampling.NEAREST)
    canvas.paste(resized, ((size[0] - resized.width) // 2,
                          (size[1] - resized.height) // 2))
    return canvas


def _brain_panel(fa, labels, spacing, center, axis, maximum):
    background = np.flip(np.take(fa, center[axis], axis=axis).T, axis=0)
    selected = np.flip(np.take(labels, center[axis], axis=axis).T, axis=0)
    gray = np.clip(np.nan_to_num(background, nan=0, posinf=0, neginf=0)
                   / maximum * 255, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float64)
    for label in np.unique(selected):
        if label <= 0:
            continue
        color = 255 * np.asarray(colorsys.hsv_to_rgb(
            (int(label) * .61803398875) % 1., .8, 1.))
        rgb[selected == label] = .3 * rgb[selected == label] + .7 * color
    panel = Image.fromarray(np.rint(rgb).astype(np.uint8))
    plane_axes = [i for i in range(3) if i != axis]
    # Preserve physical aspect ratio for an anisotropic DWI grid.
    target = tuple(max(1, round(fa.shape[i] * spacing[i] * 3))
                   for i in plane_axes)
    return _fit(panel.resize(target, Image.Resampling.NEAREST), (210, 205))


def _heatmap(count, maximum):
    normalized = np.log1p(count.astype(np.float64)) / maximum
    # Black zero edges, then increasing blue→cyan→yellow intensity.
    rgb = np.stack((np.clip(2 * normalized - .4, 0, 1),
                    np.clip(1.8 * normalized, 0, 1),
                    np.clip(2.5 * normalized, 0, 1)
                    * (1 - .65 * normalized)), axis=-1)
    panel = Image.fromarray(np.rint(rgb * 255).astype(np.uint8))
    return _fit(panel, (465, 205))


def render(output_dir, output_png, *, checkpoint_root=None, pairs=("SS", "VV", "SV"),
           case_label="anonymous subject"):
    output = Path(output_dir)
    checkpoint = Path(checkpoint_root) if checkpoint_root else output / "checkpoints"
    state, fa_image = _published_core(output, checkpoint)
    fa = np.asarray(fa_image.dataobj)
    valid = np.isfinite(fa) & (fa > 0)
    if not valid.any():
        raise ValueError("actual shared-core FA has no positive finite voxels")
    center = np.rint(np.median(np.argwhere(valid), axis=0)).astype(int)
    maximum = max(float(np.percentile(fa[valid], 99)), np.finfo(np.float32).tiny)
    world = nib.affines.apply_affine(fa_image.affine, center)
    rows = []
    for name in pairs:
        folder = output / "pairs" / name
        metadata = json.loads((folder / "pair.json").read_text())
        labels = []
        for side in ("first", "second"):
            image = nib.as_closest_canonical(nib.load(folder / f"{side}_atlas_dwi.nii.gz"))
            if image.shape != fa.shape or not np.allclose(
                    image.affine, fa_image.affine, rtol=0, atol=1e-5):
                raise ValueError("actual pair labels and core FA grids differ; QC does not resample")
            labels.append(np.asarray(image.dataobj))
        count = np.loadtxt(folder / "connectome_count.csv", delimiter=",", ndmin=2)
        if tuple(count.shape) != tuple(metadata["shape"]):
            raise ValueError("actual count shape and node metadata disagree")
        if not np.isfinite(count).all() or (count < 0).any() or (count != np.floor(count)).any():
            raise ValueError("saved count matrix must contain finite nonnegative integers")
        rows.append((name, metadata, labels, count))
    count_max = max(float(row[3].max(initial=0)) for row in rows)
    heat_max = max(float(np.log1p(count_max)), np.finfo(np.float64).tiny)
    canvas = Image.new("RGB", (1430, 110 + 275 * len(rows)), "black")
    draw, font, small = ImageDraw.Draw(canvas), _font(21), _font(15)
    draw.text((15, 12), f"Actual paired SC outputs: {case_label}", fill="white", font=font)
    draw.text((15, 43), f"Whole-brain ACT: {state['seed_attempts']:,} attempts; "
              f"{state['accepted_streamlines']:,} accepted | FA background, saved DWI labels", fill="white", font=small)
    draw.text((15, 65), f"Neurological display | axial z={world[2]:.1f} mm, coronal y={world[1]:.1f} mm | "
              f"count heatmaps share log1p scale (max {count_max:g})", fill="white", font=small)
    for i, (name, meta, labels, count) in enumerate(rows):
        top = 105 + 275 * i
        for column, (side, atlas) in enumerate(zip(("first", "second"), labels)):
            left = 15 + column * 450
            spec = meta[side]
            draw.text((left, top), f"{name} {side}: {spec['name']} ({spec['kind']})", fill="white", font=small)
            for plane, axis in enumerate((2, 1)):
                canvas.paste(_brain_panel(fa, atlas, fa_image.header.get_zooms()[:3],
                                         center, axis, maximum), (left + plane * 215, top + 28))
            draw.text((left, top + 235), "L               axial               R       coronal", fill="white", font=small)
        draw.text((935, top), f"{name}: count [{count.shape[0]} x {count.shape[1]}], "
                  f"radius {meta['radius_mm']:g} mm", fill="white", font=small)
        canvas.paste(_heatmap(count, heat_max), (940, top + 28))
        draw.text((935, top + 235), f"rows=first; columns=second; edge contributions={int(count.sum()):,}",
                  fill="white", font=small)
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)
    return {"case_label": case_label, "pairs": [row[0] for row in rows],
            "scope": "actual saved DWI label overlays and rectangular count matrices; no resampling",
            "count_scale": "shared log1p", "seed_attempts": state["seed_attempts"],
            "accepted_streamlines": state["accepted_streamlines"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--pairs", nargs="+", default=["SS", "VV", "SV"])
    parser.add_argument("--case-label", default="anonymous subject")
    options = parser.parse_args()
    receipt = render(options.output_dir, options.output_png,
                     checkpoint_root=options.checkpoint_root,
                     pairs=options.pairs, case_label=options.case_label)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
