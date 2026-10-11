"""真实全体积线程差异的三正交脑图；误差最多切面规则写入provenance。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(*, fnit_root: Path, output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import nibabel as nib
    run = fnit_root / "runs/n4_fitting_threads_20261010_v2/ABBA"
    complete = json.loads((run / "COMPLETE.json").read_bytes())
    if complete["status"] != "complete" or complete["runs"] != 8:
        raise ValueError("ABBA incomplete")
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"scope": "public same-input N4 stage, not whole-pipeline/QC equivalence",
        "plot_script_sha256": digest(Path(__file__)),
        "slice_rule": "for each XYZ axis select the slice with most fit1-versus-fit4 uint8 changes; totals always use entire volume",
        "cases": []}
    for case in ("sub07", "sub06"):
        first, second = run / (case + "-1-fit1"), run / (case + "-2-fit4")
        record = json.loads((first / "run.json").read_bytes())
        comparison = json.loads((run / (case + "_comparisons.json")).read_bytes())
        original = fnit_root / "runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1" / (case + "-candidate/subject/mri/orig.mgz")
        if digest(original) != record["input_sha256_before"]:
            raise ValueError("plot input changed")
        image = np.asarray(nib.load(str(original)).dataobj)
        a, b = (np.asarray(nib.load(str(path / "nu0.mgz")).dataobj) for path in (first, second))
        difference = b.astype(np.int16) - a.astype(np.int16)
        mask = difference != 0
        total = int(mask.sum())
        if total != comparison["comparisons"]["nu0"]["A1_vs_B1"]["different_voxels"]:
            raise ValueError("plot and report differ")
        slices = [int(np.argmax(mask.sum(axis=tuple(j for j in range(3) if j != axis)))) for axis in range(3)]
        fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
        for axis, index in enumerate(slices):
            plane = np.take(image, index, axis=axis).T
            delta = np.take(difference, index, axis=axis).T
            axes[0, axis].imshow(plane, cmap="gray", origin="lower", vmin=0, vmax=160)
            axes[0, axis].set_title(f"orig: XYZ axis {axis}, slice {index}")
            axes[1, axis].imshow(plane, cmap="gray", origin="lower", vmin=0, vmax=160)
            overlay = axes[1, axis].imshow(np.ma.masked_equal(delta, 0), cmap="coolwarm", origin="lower", vmin=-1, vmax=1, interpolation="nearest")
            axes[1, axis].set_title(f"fit4 - fit1: {int(np.count_nonzero(delta))} changed pixels")
            for row in range(2): axes[row, axis].axis("off")
        fig.colorbar(overlay, ax=list(axes[1, :]), shrink=.65, ticks=[-1, 0, 1], label="uint8 intensity difference")
        fig.suptitle(f"{case}: complete-volume {total:,} changed voxels; max {int(np.abs(difference).max())}; N4 stage only")
        path = output / (case + "_fit_threads.png")
        fig.savefig(path, dpi=160); plt.close(fig)
        provenance["cases"].append({"case": case, "slice_indices_XYZ": slices,
            "complete_volume_changed_voxels": total, "maximum_uint8_error": int(np.abs(difference).max()),
            "input_sha256": digest(original), "fit1_nu0_sha256": digest(first / "nu0.mgz"),
            "fit4_nu0_sha256": digest(second / "nu0.mgz"), "comparison_sha256": digest(run / (case + "_comparisons.json")),
            "png_sha256": digest(path)})
    (output / "plot_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(**vars(args))))
