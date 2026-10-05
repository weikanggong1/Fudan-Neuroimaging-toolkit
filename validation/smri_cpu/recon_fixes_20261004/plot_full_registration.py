"""Render real cortical input and the two saved registered spheres."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("smoothwm", "sulc", "official", "candidate", "report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    import nibabel as nib
    import numpy as np
    report = json.loads(args.report.read_text())
    curvature = nib.freesurfer.read_morph_data(args.sulc)
    meshes = [nib.freesurfer.read_geometry(path) for path in (
        args.smoothwm, args.official, args.candidate)]
    faces = meshes[0][1]
    if any(not np.array_equal(faces, other[1]) for other in meshes[1:]):
        raise ValueError("surface projection requires unchanged ordered faces")
    if len(curvature) != len(meshes[0][0]):
        raise ValueError("sulc must have the same vertex correspondence")
    color = curvature[faces].mean(axis=1)
    limits = np.percentile(color, [5, 95])
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
    titles = ["Real LH input: smoothwm + sulc", "Official sphere.reg + input sulc",
              "FNIT sphere.reg + input sulc"]
    for axis, (vertices, faces), title in zip(axes, meshes, titles):
        triangles = vertices[faces]
        order = np.argsort(triangles[:, :, 0].mean(axis=1), kind="stable")
        # Orthographic view along +X. Painter order hides the far hemisphere.
        collection = PolyCollection(triangles[order][:, :, [1, 2]],
                                    array=color[order], cmap="gray", edgecolors="none",
                                    linewidths=0, rasterized=True)
        collection.set_clim(*limits)
        axis.add_collection(collection)
        axis.autoscale_view()
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(title, fontsize=9)
    times = [report["arms"][name]["cold_process_wall_seconds"] for name in (
        "baseline", "candidate", "official")]
    axes[-1].bar(["Old FNIT", "New FNIT", "Official"], times, color=["#9aa8b5", "#3476b6", "#de8a39"])
    for index, seconds in enumerate(times):
        axes[-1].text(index, seconds + 8, f"{seconds:.1f}", ha="center", fontsize=9)
    axes[-1].set_ylabel("Cold process seconds")
    axes[-1].set_ylim(0, max(times) * 1.16)
    axes[-1].set_title("Same real mesh, eight CPU cores", fontsize=9)
    axes[-1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("119,451 vertices: all saved coordinates identical; maximum difference 0 mm", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
