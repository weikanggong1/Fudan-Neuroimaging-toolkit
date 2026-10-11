"""真实连续前段T1/WM/filled差异切面；不发布影像或数组。"""
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
    import nibabel as nib
    import numpy as np
    run = fnit_root / "runs/n4_fitting_prefix_20261010_v2"
    complete = json.loads((run / "prefixes/COMPLETE.json").read_bytes())
    if complete["status"] != "complete_prefixes" or complete["full_workflow_complete"]:
        raise ValueError("continuous prefixes not complete")
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"scope": "same raw T1 continuous volume-prefix differences; not whole recon-all or metric equivalence",
        "slice_rule": "each volume and stored XYZ axis selects its slice with most changed voxels; all errors computed over full volume",
        "plot_script_sha256": digest(Path(__file__)), "cases": []}
    for case in ("sub07", "sub06"):
        comparison_path = run / "comparisons" / (case + ".json")
        comparison = json.loads(comparison_path.read_bytes())
        directories = [run / "prefixes" / (case + "-fit" + str(fit)) / "subject/mri" for fit in (1, 4)]
        orig = np.asarray(nib.load(str(directories[0] / "orig.mgz")).dataobj)
        fig, axes = plt.subplots(4, 3, figsize=(12, 12), constrained_layout=True)
        volume_records = []
        for row, name in enumerate(("tmp/nu0.mgz", "T1.mgz", "wm.mgz", "filled.mgz")):
            paths = [directory / name for directory in directories]
            arrays = [np.asarray(nib.load(str(path)).dataobj) for path in paths]
            if arrays[0].shape != orig.shape or arrays[1].shape != orig.shape:
                raise ValueError("all plot inputs must share conformed grid")
            error = arrays[1].astype(np.float64) - arrays[0].astype(np.float64)
            mask = error != 0
            hashes = [digest(path) for path in paths]
            detail = comparison["volumes"][name]
            if (int(mask.sum()) != detail["different_voxels"]
                    or hashes != [detail["file_sha256"][side] for side in ("reference", "candidate")]):
                raise ValueError("plot array/hash differs from comparison")
            slices = [int(np.argmax(mask.sum(axis=tuple(j for j in range(3) if j != axis)))) for axis in range(3)]
            for axis, index in enumerate(slices):
                plane = np.take(orig, index, axis=axis).T
                delta = np.take(error, index, axis=axis).T
                axes[row, axis].imshow(plane, cmap="gray", origin="lower", vmin=0, vmax=160)
                if name == "filled.mgz":
                    axes[row, axis].imshow(np.ma.masked_equal(delta != 0, False), cmap="autumn", origin="lower", vmin=0, vmax=1, interpolation="nearest")
                else:
                    limit = max(1, float(np.abs(error).max()))
                    axes[row, axis].imshow(np.ma.masked_equal(delta, 0), cmap="coolwarm", origin="lower", vmin=-limit, vmax=limit, interpolation="nearest")
                axes[row, axis].set_title(f"{name}: XYZ {axis}, slice {index}\n{int(np.count_nonzero(delta))} changed pixels")
                axes[row, axis].axis("off")
            volume_records.append({"volume": name, "slice_indices_XYZ": slices,
                "full_volume_changed_voxels": int(mask.sum()), "max_abs": float(np.abs(error).max()), "files_sha256": hashes,
                "overlay": "yellow boolean label mismatch" if name == "filled.mgz" else "blue/red signed fit4-minus-fit1 intensity, zero masked"})
        fig.suptitle(f"{case}: fit4 minus fit1, complete raw T1-to-filled prefix only", fontsize=14)
        path = output / (case + "_prefix_differences.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        provenance["cases"].append({"case": case, "comparison_sha256": digest(comparison_path),
            "volumes": volume_records, "png_sha256": digest(path)})
    (output / "plot_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(plot(**vars(parser.parse_args()))))
