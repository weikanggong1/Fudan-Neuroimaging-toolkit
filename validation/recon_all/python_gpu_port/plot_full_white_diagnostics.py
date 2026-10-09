"""绘制冻结同输入表面的真实MRI叠加、同索引误差与局部边界；默认white。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plane_lines(vertices, faces, slice_y):
    """沿体素y平面切三角面，返回(x,z)线段；仅用于可视化。"""
    triangles = vertices[faces]
    edges = ((0, 1), (1, 2), (2, 0))
    intersections = np.zeros((len(faces), 3, 2), np.float64)
    valid = np.zeros((len(faces), 3), bool)
    for edge, (first, second) in enumerate(edges):
        a, b = triangles[:, first], triangles[:, second]
        dy = b[:, 1]-a[:, 1]
        crossing = (dy != 0) & ((a[:, 1] <= slice_y) != (b[:, 1] <= slice_y))
        fraction = np.zeros(len(faces), np.float64)
        np.divide(slice_y-a[:, 1], dy, out=fraction, where=crossing)
        position = a + fraction[:, None]*(b-a)
        intersections[:, edge] = position[:, (0, 2)]
        valid[:, edge] = crossing
    selected = valid.sum(axis=1) == 2
    return intersections[selected][valid[selected]].reshape(-1, 2, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--reference-surface", type=Path, required=True)
    parser.add_argument("--brain-mri", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--surface-kind", choices=("white.preaparc", "pial.T1"), default="white.preaparc")
    parser.add_argument("--annotation", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("new figure and metadata paths required")
    vertices, faces = fs.read_geometry(str(args.candidate_surface))
    reference, reference_faces = fs.read_geometry(str(args.reference_surface))
    if vertices.shape != reference.shape or not np.array_equal(faces, reference_faces):
        raise ValueError("same-index error figure requires equal vertex counts and ordered faces")
    brain = nib.load(str(args.brain_mri))
    if len(brain.shape) != 3 or not hasattr(brain.header, "get_vox2ras_tkr"):
        raise ValueError("three-dimensional conformed MGH/MGZ with surface-RAS transform required")
    image = np.asarray(brain.dataobj)
    inverse_tkr = np.linalg.inv(brain.header.get_vox2ras_tkr())
    candidate_voxel = nib.affines.apply_affine(inverse_tkr, vertices)
    reference_voxel = nib.affines.apply_affine(inverse_tkr, reference)
    distances = np.linalg.norm(vertices-reference, axis=1)
    maximum_vertex = int(np.argmax(distances))
    annotation_name = None
    if args.annotation is not None:
        labels, _, names = fs.read_annot(str(args.annotation))
        if len(labels) != len(vertices):
            raise ValueError("annotation must address this same vertex order")
        label_index = int(labels[maximum_vertex])
        annotation_name = (names[label_index].decode("utf8") if 0 <= label_index < len(names)
                           else "unannotated")
    target = candidate_voxel[maximum_vertex]
    slice_y = int(np.clip(round(float(target[1])), 0, brain.shape[1]-1))
    candidate_lines = plane_lines(candidate_voxel, faces, slice_y)
    reference_lines = plane_lines(reference_voxel, faces, slice_y)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax in axes[:2]:
        ax.imshow(image[:, slice_y, :].T, origin="lower", cmap="gray", vmin=0, vmax=130,
                  extent=(-.5, brain.shape[0]-.5, -.5, brain.shape[2]-.5))
        ax.add_collection(LineCollection(reference_lines, colors="cyan", linewidths=.65, alpha=.9))
        ax.add_collection(LineCollection(candidate_lines, colors="magenta", linewidths=.65, alpha=.9))
        ax.set_xlabel("conformed voxel x")
        ax.set_ylabel("conformed voxel z")
    axes[0].set_title(f"MRI plane y={slice_y}: reference / FNIT")
    axes[1].set_title("Local boundary near maximum error" +
                      (f"\n{annotation_name}" if annotation_name is not None else ""))
    axes[1].set_xlim(target[0]-18, target[0]+18)
    axes[1].set_ylim(target[2]-18, target[2]+18)
    near_plane = np.abs(candidate_voxel[:, 1]-slice_y) < 1.0
    points = axes[1].scatter(candidate_voxel[near_plane, 0], candidate_voxel[near_plane, 2],
                            c=distances[near_plane], s=7, cmap="inferno", vmin=0,
                            vmax=max(float(distances.max()), 1e-12))
    fig.colorbar(points, ax=axes[1], label="same-index distance (mm)", shrink=.7)
    stride = max(1, len(vertices)//100000)
    error_points = axes[2].scatter(vertices[::stride, 0], vertices[::stride, 2],
                                  c=distances[::stride], s=.6, cmap="inferno", vmin=0,
                                  vmax=max(float(distances.max()), 1e-12), rasterized=True)
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("surface RAS x (mm)")
    axes[2].set_ylabel("surface RAS z (mm)")
    axes[2].set_title("Vertex error: all values used in statistics")
    fig.colorbar(error_points, ax=axes[2], label="distance (mm)", shrink=.7)
    fig.suptitle(f"{args.label}: frozen-input {args.surface_kind}, not recon-all equivalence", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    metadata = {
        "scope": f"frozen_same_input_{args.surface_kind}_MRI_overlay_and_local_boundary_not_recon_all_equivalence",
        "script_sha256": sha(__file__),
        "input_sha256": {"candidate_surface": sha(args.candidate_surface),
                         "reference_surface": sha(args.reference_surface),
                         "brain_MRI": sha(args.brain_mri), "comparison_report": sha(args.comparison_report)},
        "figure_sha256": sha(args.output), "same_ordered_faces": True,
        "MRI_space": "conformed voxel grid; surface coordinates transformed with inverse vox2ras_tkr",
        "surface_space": "surface RAS in mm", "slice_axis": "voxel y", "slice_index": slice_y,
        "plane_selection": "nearest coronal voxel plane to maximum same-index error; no algorithm change",
        "maximum_vertex_index": maximum_vertex,
        "surface_kind": args.surface_kind,
        "maximum_error_region_name": annotation_name,
        "mean_distance_mm": float(distances.mean()), "p99_distance_mm": float(np.percentile(distances, 99)),
        "max_distance_mm": float(distances.max()), "vertices_over_0_1_mm": int(np.count_nonzero(distances > .1)),
        "raster_visualization_stride": stride,
    }
    if args.annotation is not None:
        metadata["input_sha256"]["annotation"] = sha(args.annotation)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
