"""绘制真实 T1 表面叠加、逐脑区误差及最差分区边界；不判断指标等效。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import nibabel as nib
from nibabel.freesurfer.io import read_geometry
import numpy as np


def _section(vertices: np.ndarray, faces: np.ndarray, axis: int,
             position: int) -> np.ndarray:
    """在 conform 体素平面切三角面，返回二维线段；不按顶点配对。"""
    triangles = vertices[faces]
    signed = triangles[:, :, axis] - position
    selected = (signed.min(axis=1) < 0) & (signed.max(axis=1) > 0)
    triangles, signed = triangles[selected], signed[selected]
    first = triangles[:, [0, 1, 2]]
    second = triangles[:, [1, 2, 0]]
    a, b = signed, signed[:, [1, 2, 0]]
    crossing = a * b < 0
    complete = crossing.sum(axis=1) == 2
    a, b, first, second, crossing = (
        value[complete] for value in (a, b, first, second, crossing))
    fraction = np.divide(a, a - b, out=np.zeros_like(a), where=a != b)
    points = (first + fraction[:, :, None] * (second - first))[crossing]
    dimensions = [index for index in range(3) if index != axis]
    return points.reshape(-1, 2, 3)[:, :, dimensions]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--region-report", type=Path, required=True)
    parser.add_argument("--dice-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    files = [args.region_report, args.dice_report]
    image_path = args.candidate / "mri/orig.mgz"
    files.append(image_path)
    image = nib.load(str(image_path))
    data = np.asarray(image.dataobj)
    reference_image_path = args.reference / "mri/orig.mgz"
    files.append(reference_image_path)
    reference_image = nib.load(str(reference_image_path))
    if not np.allclose(image.header.get_vox2ras_tkr(),
                       reference_image.header.get_vox2ras_tkr(), atol=1e-6, rtol=0):
        raise ValueError("surface RAS to conform voxel transforms differ")
    inverse = np.linalg.inv(image.header.get_vox2ras_tkr())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = {("reference", "white"): "cyan", ("reference", "pial"): "lime",
              ("candidate", "white"): "orange", ("candidate", "pial"): "red"}
    for axis, panel in enumerate(axes):
        position = data.shape[axis] // 2
        panel.imshow(np.take(data, position, axis=axis).T, cmap="gray", origin="lower")
        for label, root in (("reference", args.reference), ("candidate", args.candidate)):
            for hemi in ("lh", "rh"):
                for surface in ("white", "pial"):
                    path = root / "surf" / f"{hemi}.{surface}"
                    if axis == 0:
                        files.append(path)
                    vertices, faces = read_geometry(str(path))
                    voxels = vertices @ inverse[:3, :3].T + inverse[:3, 3]
                    segments = _section(voxels, faces, axis, position)
                    panel.add_collection(LineCollection(segments, colors=colors[label, surface],
                                                        linewidths=.65))
        panel.set_title(f"Conform voxel axis {axis}, slice {position}")
        panel.set_axis_off()
    fig.legend([Line2D([], [], color=color) for color in colors.values()],
               [f"{label} {surface}" for label, surface in colors], loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(args.output_dir / "t1_surface_overlay.png", dpi=160)
    plt.close(fig)

    report = json.loads(args.region_report.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    for panel, field in zip(axes, ("SurfArea", "GrayVol", "ThickAvg")):
        metric = report["aparc_68"][field]
        names = metric["worst_regions_by_relative_error"]
        values = [metric["per_region"][name]["signed_relative_difference_percent"]
                  for name in names]
        panel.barh(names, values, color=["darkred" if value > 0 else "steelblue"
                                        for value in values])
        panel.axvline(0, color="black", linewidth=.6)
        panel.invert_yaxis()
        panel.set_title(field)
        panel.set_xlabel("Signed difference from reference (%)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "region_errors.png", dpi=160)
    plt.close(fig)

    dice = json.loads(args.dice_report.read_text())
    filename = "aparc.a2009s+aseg.mgz"
    row = dice["files"][filename]
    label = row["worst_labels"][0]
    masks = []
    for root in (args.reference, args.candidate):
        path = root / "mri" / filename
        files.append(path)
        volume = nib.load(str(path))
        if volume.shape != image.shape or not np.allclose(volume.affine, image.affine,
                                                         atol=1e-6, rtol=0):
            raise ValueError("parcellation and T1 conform grids differ")
        masks.append(np.asarray(volume.dataobj) == int(label))
    coordinates = np.argwhere(masks[0] | masks[1])
    center = np.median(coordinates, axis=0).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for axis, panel in enumerate(axes):
        panel.imshow(np.take(data, center[axis], axis=axis).T, cmap="gray", origin="lower")
        for mask, color in zip(masks, ("cyan", "red")):
            section = np.take(mask, center[axis], axis=axis).T
            if section.any():
                panel.contour(section, levels=[.5], colors=[color], linewidths=1)
        dimensions = [index for index in range(3) if index != axis]
        panel.set_xlim(coordinates[:, dimensions[0]].min() - 8,
                       coordinates[:, dimensions[0]].max() + 8)
        panel.set_ylim(coordinates[:, dimensions[1]].min() - 8,
                       coordinates[:, dimensions[1]].max() + 8)
        panel.set_title(f"Voxel axis {axis}, slice {center[axis]}")
    region = row["per_label"][label]
    fig.suptitle(f"{region['name']} ({label}), Dice={region['dice']:.6f}\n"
                 "cyan: reference; red: candidate")
    fig.tight_layout()
    fig.savefig(args.output_dir / "local_region_boundary.png", dpi=160)
    plt.close(fig)
    provenance = {"code_commit": args.code_commit, "candidate": str(args.candidate),
                  "reference": str(args.reference), "space": "conform voxel slices",
                  "purpose": "qualitative localization; no equivalence gates applied",
                  "inputs_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in files},
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
