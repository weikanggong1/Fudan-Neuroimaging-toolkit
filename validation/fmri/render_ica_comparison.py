"""Render original and FNIT thresholded IC maps on an identical MNI grid."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("current", "reference", "template", "pairs", "figure-out", "report-out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--components", type=int, nargs="+", default=[1, 20, 50])
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    current, reference, template = (nib.load(path) for path in
                                    (args.current, args.reference, args.template))
    for image in (current, reference):
        if (image.shape[:3] != template.shape or
                not np.allclose(image.affine, template.affine, atol=1e-4, rtol=0)):
            raise ValueError("all maps must have the same MNI template grid")
    pair = np.load(args.pairs)
    underlay = np.asarray(template.dataobj, dtype=np.float32)
    brain = underlay > 0
    anatomical_scale = np.percentile(underlay[brain], 99)
    fig, axes = plt.subplots(3, len(args.components), figsize=(4 * len(args.components), 9),
                             facecolor="white", squeeze=False)
    metadata = []
    for column, reference_id in enumerate(args.components):
        pairing_index = int(np.flatnonzero(pair["native"] == reference_id - 1)[0])
        candidate_id = int(pair["fnit"][pairing_index])
        sign = float(pair["sign"][pairing_index])
        original = np.asarray(reference.dataobj[..., reference_id - 1], dtype=np.float32)
        candidate = np.asarray(current.dataobj[..., candidate_id], dtype=np.float32) * sign
        support_by_slice = (original != 0).sum(axis=(0, 1))
        axial_index = int(np.argmax(support_by_slice))
        ras_z = float((template.affine @ np.array([0, 0, axial_index, 1]))[2])
        maps = (original, candidate, np.abs(original - candidate))
        for row, values in enumerate(maps):
            background = np.rot90(underlay[:, :, axial_index])
            if row < 2:
                axes[row, column].imshow(background, cmap="gray", vmin=0, vmax=anatomical_scale)
            else:
                axes[row, column].imshow(np.zeros_like(background), cmap="gray", vmin=0, vmax=1)
                axes[row, column].contour(np.rot90(brain[:, :, axial_index]), levels=[0.5],
                                         colors=["0.4"], linewidths=0.5)
            plane = np.rot90(values[:, :, axial_index])
            plane = np.ma.masked_where((plane == 0) | ~np.rot90(brain[:, :, axial_index]), plane)
            scale = ("coolwarm", -8, 8) if row < 2 else ("magma", 0, 0.01)
            image = axes[row, column].imshow(plane, cmap=scale[0], vmin=scale[1], vmax=scale[2])
            method = ("Original MELODIC", "FNIT", "Absolute difference")[row]
            axes[row, column].set_title(f"{method}: IC {reference_id}\nMNI axial z={ras_z:g} mm", fontsize=10)
            axes[row, column].axis("off")
            if column == len(args.components) - 1:
                fig.colorbar(image, ax=axes[row, column], fraction=0.045, pad=0.04)
        metadata.append({"reference_ic_1based": reference_id,
                         "candidate_ic_1based": candidate_id + 1, "candidate_sign": sign,
                         "axial_voxel_index": axial_index, "mni_ras_z_mm": ras_z})
    fig.suptitle("Real 490-frame BOLD: identical pre-ICA input and mask\n"
                 "Original and FNIT thresholded maps through the same original MNI warp", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_out, dpi=150)
    plt.close(fig)
    report = {
        "schema_version": 1, "scope": "Authorized anonymous template-space IC comparison PNG.",
        "ica_source_sha256": args.source_sha256, "renderer_sha256": sha256(__file__),
        "input_sha256": {"current_mni_thresholded": sha256(args.current),
                         "reference_mni_thresholded": sha256(args.reference),
                         "template": sha256(args.template), "private_pairing": sha256(args.pairs)},
        "components": metadata, "image_sha256": sha256(args.figure_out),
        "map_display_range": [-8, 8], "absolute_difference_display_range": [0, 0.01],
        "method": "Same axial slice and color scale for each matched component; no additional smoothing.",
        "privacy": "MNI template-space PNG and aggregate metadata only; raw maps and private pairing stay on server.",
    }
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
