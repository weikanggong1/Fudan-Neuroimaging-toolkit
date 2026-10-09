"""Render a frozen real smoothwm mesh and curvature derivative differences."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import nibabel.freesurfer.io as fsio
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--candidate-prefix", type=Path, required=True)
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    xyz, faces = fsio.read_geometry(str(args.surface))
    C = fsio.read_morph_data(str(args.candidate_prefix) + ".C.crv")
    FI = np.asarray(fsio.read_morph_data(str(args.candidate_prefix) + ".FI.crv"), dtype=np.float32)
    reference = np.asarray(fsio.read_morph_data(str(args.reference_prefix) + ".FI.crv"), dtype=np.float32)
    if len(C) != len(xyz) or len(FI) != len(xyz) or len(reference) != len(xyz):
        raise ValueError("frozen surface and maps must share vertex order")
    bits_a, bits_b = FI.view(np.int32).astype(np.int64), reference.view(np.int32).astype(np.int64)
    ordered_a = np.where(bits_a < 0, -2147483648 - bits_a, bits_a)
    ordered_b = np.where(bits_b < 0, -2147483648 - bits_b, bits_b)
    ulps = np.abs(ordered_a - ordered_b)
    corners = xyz[faces]
    face_normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    selected = np.flatnonzero(face_normal[:, 0] < 0)
    selected = selected[np.argsort(corners[selected, :, 0].mean(axis=1))[::-1]]
    polygons = corners[selected][:, :, [1, 2]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, data, name, cmap, maximum in (
        (axes[0], C[faces[selected]].mean(axis=1), "Curvedness C (mm$^{-1}$)", "viridis", np.percentile(C, 99)),
        (axes[1], ulps[faces[selected]].max(axis=1), "FI native vs Torch (ULP, face maximum)", "magma", max(1, ulps.max())),
    ):
        collection = PolyCollection(polygons, array=data, cmap=cmap, edgecolors="none", rasterized=True)
        collection.set_clim(0, maximum)
        axis.add_collection(collection)
        axis.autoscale_view()
        axis.set_aspect("equal")
        axis.set_xlabel("surface RAS y (mm)")
        axis.set_ylabel("surface RAS z (mm)")
        axis.set_title(name)
        fig.colorbar(collection, ax=axis, shrink=0.75)
    fig.suptitle("ds000114 sub-07 LH: frozen smoothwm + K1/K2 derivatives")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
