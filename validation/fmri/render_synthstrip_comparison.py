"""将同输入的原版/FNIT 脑提取结果通过同一场展示在 MNI 模板空间。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import torch

from fnit.fmri import normalization
from fnit.synthstrip import pipeline as synthstrip


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_grid(image, reference, label):
    if image.ndim != 3 or image.shape != reference.shape or not np.allclose(
        image.affine, reference.affine, atol=1e-4, rtol=0
    ):
        raise ValueError(f"{label} does not match its brain image grid")


def mask_summary(current, reference):
    current = np.asarray(current, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    denominator = int(current.sum()) + int(reference.sum())
    return {
        "dice": 2 * int(np.count_nonzero(current & reference)) / denominator,
        "changed_voxels": int(np.count_nonzero(current != reference)),
        "current_only_voxels": int(np.count_nonzero(current & ~reference)),
        "reference_only_voxels": int(np.count_nonzero(reference & ~current)),
        "current_voxels": int(current.sum()),
        "reference_voxels": int(reference.sum()),
    }


def plane(values, axis, index):
    return np.rot90(np.take(values, index, axis=axis))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "current-brain", "reference-brain", "current-mask", "reference-mask",
        "template", "pull-ras", "private-output", "figure-out", "report-out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    candidate = nib.load(args.current_brain)
    reference = nib.load(args.reference_brain)
    candidate_mask = nib.load(args.current_mask)
    reference_mask = nib.load(args.reference_mask)
    template = nib.load(args.template)
    field = nib.load(args.pull_ras)
    check_grid(candidate, reference, "current brain")
    check_grid(candidate_mask, candidate, "current mask")
    check_grid(reference_mask, reference, "reference mask")
    if template.ndim != 3 or field.shape != (*template.shape, 3) or not np.allclose(
        field.affine, template.affine, atol=1e-4, rtol=0
    ):
        raise ValueError("pull field must match the MNI template grid")
    if not np.allclose(template.header.get_zooms()[:3], 2, atol=1e-3):
        raise ValueError("the public example requires an MNI152 2-mm template")
    native_candidate_mask = np.asarray(candidate_mask.dataobj) > 0
    native_reference_mask = np.asarray(reference_mask.dataobj) > 0
    if not native_candidate_mask.any() or not native_reference_mask.any():
        raise ValueError("empty input brain mask")
    args.private_output.mkdir(parents=True, exist_ok=True)
    mapped = {}
    for label, source, interpolation in (
        ("current_brain", args.current_brain, "spline"),
        ("reference_brain", args.reference_brain, "spline"),
        ("current_mask", args.current_mask, "nearest"),
        ("reference_mask", args.reference_mask, "nearest"),
    ):
        output = args.private_output / f"{label}_MNI152_2mm.nii.gz"
        # 两幅图共用原版 FNIRT 的 RAS pull；已包含线性部分，无额外 affine。
        normalization.resample_world(
            source, args.template, np.eye(4), output,
            pre_affine_pull_ras=args.pull_ras, interpolation=interpolation,
            device=args.device,
        )
        image = nib.load(output)
        check_grid(image, template, label)
        data = np.asarray(image.dataobj, dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError(f"nonfinite {label}")
        mapped[label] = data
    current_mask = mapped["current_mask"] > .5
    reference_mask = mapped["reference_mask"] > .5
    union = current_mask | reference_mask
    difference = mapped["current_brain"][union].astype(np.float64) - mapped["reference_brain"][union]
    display_scale = float(np.percentile(mapped["reference_brain"][reference_mask], 99))
    if display_scale <= 0:
        raise ValueError("nonpositive reference display scale")
    # 显示固定的 MNI 坐标，避免以某一实现的差异点挑选切面。
    coordinates = ((2, 24.0, "Axial z"), (1, -10.0, "Coronal y"), (0, 0.0, "Sagittal x"))
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), facecolor="white")
    slices = []
    for column, (axis, world_coordinate, label) in enumerate(coordinates):
        world = np.array([0., 0., 0., 1.])
        world[axis] = world_coordinate
        index = int(np.round((np.linalg.inv(template.affine) @ world)[axis]))
        if not 0 <= index < template.shape[axis]:
            raise ValueError("requested MNI slice outside template")
        actual_world = float((template.affine @ np.array([
            index if axis == 0 else 0, index if axis == 1 else 0,
            index if axis == 2 else 0, 1,
        ]))[axis])
        for row, name in enumerate(("reference_brain", "current_brain")):
            axes[row, column].imshow(plane(mapped[name], axis, index), cmap="gray",
                                     vmin=0, vmax=display_scale)
            axes[row, column].set_title(
                f"{('Original SynthStrip', 'FNIT')[row]}\n{label}={actual_world:g} mm", fontsize=10
            )
        original_plane = plane(reference_mask, axis, index)
        current_plane = plane(current_mask, axis, index)
        changed = np.zeros_like(original_plane, dtype=np.uint8)
        changed[original_plane & ~current_plane] = 1
        changed[current_plane & ~original_plane] = 2
        axes[2, column].imshow(np.zeros_like(changed), cmap="gray", vmin=0, vmax=1)
        axes[2, column].contour(original_plane, levels=[.5], colors=["0.5"], linewidths=.7)
        axes[2, column].imshow(np.ma.masked_equal(changed, 0),
                              cmap=ListedColormap(["#32b8df", "#f18c36"]), vmin=1, vmax=2)
        axes[2, column].set_title(f"Mask difference\n{label}={actual_world:g} mm", fontsize=10)
        slices.append({"axis": axis, "voxel_index": index, "mni_ras_mm": actual_world})
        for row in range(3):
            axes[row, column].axis("off")
    fig.legend(handles=[Patch(color="#32b8df", label="Original only"),
                        Patch(color="#f18c36", label="FNIT only"),
                        Patch(color="0.5", label="Original outline")],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Real T1 brain extraction: identical input and original MNI pull field\n"
                 "Same spline brain / nearest-mask sampling; fixed MNI slices", fontsize=12)
    fig.tight_layout(rect=(0, .045, 1, .94))
    args.figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_out, dpi=150)
    plt.close(fig)
    input_paths = {"current_brain": args.current_brain, "reference_brain": args.reference_brain,
                   "current_mask": args.current_mask, "reference_mask": args.reference_mask,
                   "template": args.template, "original_pull_ras": args.pull_ras}
    report = {
        "schema_version": 1,
        "scope": "Authorized anonymous SynthStrip brain extraction PNG in template space; fixed original transform, not registration accuracy.",
        "candidate_source_revision": args.source_revision,
        "synthstrip_source_sha256": sha256(synthstrip.__file__),
        "sampling_source_sha256": sha256(normalization.__file__),
        "renderer_sha256": sha256(__file__),
        "input_sha256": {key: sha256(path) for key, path in input_paths.items()},
        "native_grid_mask": mask_summary(native_candidate_mask, native_reference_mask),
        "mni_grid_mask": mask_summary(current_mask, reference_mask),
        "mni_brain_union_rmse": float(np.sqrt(np.mean(difference * difference))),
        "mni_brain_union_max_abs": float(np.max(np.abs(difference))),
        "template_shape": list(template.shape), "slices": slices,
        "brain_display_range": [0, display_scale],
        "sampling": {"device": args.device, "threads": args.threads,
                     "brain": "spline", "mask": "nearest",
                     "field": "same original MNI-to-T1 RAS-mm relative pull", "affine": "identity"},
        "figure_sha256": sha256(args.figure_out),
        "rendering_wall_seconds": time.perf_counter() - started,
        "privacy": "Only this MNI PNG, anonymous summaries and file hashes are public; source and resampled image arrays remain private.",
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
