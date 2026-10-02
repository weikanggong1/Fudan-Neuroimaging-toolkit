#!/usr/bin/env python3
"""Plot a public HCP reference feature on supplied fsLR32k surfaces with CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-map", required=True)
    parser.add_argument("--left-surface", required=True)
    parser.add_argument("--right-surface", required=True)
    parser.add_argument("--component", type=int, default=10, help="One-based reference-map column")
    parser.add_argument("--color-limit", type=float, help="Symmetric range; defaults to the cortical absolute 98th percentile")
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.component < 1 or args.threshold < 0 or (args.color_limit is not None and
            (args.color_limit <= 0 or args.threshold >= args.color_limit)):
        parser.error("component and color-limit must be positive; threshold must be in [0, color-limit)")
    reference = nib.load(args.reference_map)
    axis = reference.header.get_axis(1)
    if not isinstance(axis, nib.cifti2.BrainModelAxis) or args.component > reference.shape[0]:
        raise ValueError("reference map must contain the selected component and a BrainModelAxis")
    values = np.asarray(reference.dataobj)[args.component - 1]
    surfaces = {}
    for hemisphere, filename in (("LEFT", args.left_surface), ("RIGHT", args.right_surface)):
        mesh = nib.load(filename)
        points = np.asarray(next(array.data for array in mesh.darrays if array.intent == 1008))
        faces = np.asarray(next(array.data for array in mesh.darrays if array.intent == 1009))
        structure = f"CIFTI_STRUCTURE_CORTEX_{hemisphere}"
        if axis.nvertices.get(structure) != len(points):
            raise ValueError("surface topology does not match the reference BrainModel")
        scalar = np.full(len(points), np.nan)
        selected = axis.name == structure
        scalar[axis.vertex[selected]] = values[selected]
        surfaces[hemisphere] = points, faces, scalar
    if args.color_limit is None:
        cortical = np.concatenate([scalar[np.isfinite(scalar)] for _, _, scalar in surfaces.values()])
        args.color_limit = float(np.percentile(np.abs(cortical), 98))
    if args.color_limit <= args.threshold:
        raise ValueError("reference feature has no range above the requested threshold")
    norm = Normalize(-args.color_limit, args.color_limit)
    cmap = plt.get_cmap("RdBu_r")
    figure = plt.figure(figsize=(12, 7), facecolor="white")
    views = [("LEFT", 180, "Left lateral"), ("RIGHT", 0, "Right lateral"),
             ("LEFT", 0, "Left medial"), ("RIGHT", 180, "Right medial")]
    for index, (hemisphere, azimuth, title) in enumerate(views):
        panel = figure.add_subplot(2, 2, index + 1, projection="3d")
        points, faces, scalar = surfaces[hemisphere]
        face_values = scalar[faces].mean(1)
        colors = cmap(norm(np.nan_to_num(face_values)))
        inactive = ~np.isfinite(face_values) | (np.abs(face_values) < args.threshold)
        colors[inactive] = (0.79, 0.80, 0.82, 1)
        triangles = points[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        light = np.array([np.cos(np.deg2rad(azimuth)), 0.25, 0.55])
        light /= np.linalg.norm(light)
        shade = 0.72 + 0.28 * np.abs(normals @ light)
        colors[:, :3] *= shade[:, None]
        panel.add_collection3d(Poly3DCollection(triangles, facecolors=colors,
                                              edgecolors="none", rasterized=True))
        for setter, dimension in ((panel.set_xlim, 0), (panel.set_ylim, 1), (panel.set_zlim, 2)):
            setter(points[:, dimension].min(), points[:, dimension].max())
        panel.set_box_aspect(np.ptp(points, axis=0))
        panel.view_init(elev=0, azim=azimuth)
        panel.set_proj_type("ortho")
        panel.set_axis_off()
        panel.set_title(title, fontsize=12, pad=-8)
    figure.suptitle(f"Public HCP reference RSN {args.component}: fsLR32k", fontsize=17, y=0.97)
    figure.text(0.5, 0.035, "Reference-feature illustration; no individual MRI or fitted map", ha="center", fontsize=10)
    color_axis = figure.add_axes([0.34, 0.12, 0.32, 0.022])
    figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis,
                   orientation="horizontal", label="Reference feature value")
    figure.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.16, hspace=-0.15, wspace=-0.15)
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    provenance = {"kind": "public reference illustration", "component_one_based": args.component,
                  "color_limit": args.color_limit, "threshold": args.threshold,
                  "renderer": "Matplotlib CPU", "private_data_used": False,
                  "public_resource_sha256": {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                      for path in (args.reference_map, args.left_surface, args.right_surface)}}
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
