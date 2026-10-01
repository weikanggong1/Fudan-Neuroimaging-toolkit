"""Plot a matched FAST comparison after applying the same archived MNI pull.

The private JSON manifest has template, pull, native_gm and fnit_gm fields.
Only the template-space PNG and anonymous hash/metric report are public.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from fnit.fmri.normalization import resample_world


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.private_output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in ("native", "fnit"):
        path = args.private_output / f"{name}_gm_mni.nii.gz"
        resample_world(manifest[f"{name}_gm"], manifest["template"], np.eye(4), path,
                       pre_affine_pull_ras=manifest["pull"], device=args.device)
        outputs[name] = nib.as_closest_canonical(nib.load(path))
    template = nib.as_closest_canonical(nib.load(manifest["template"]))
    background = np.asarray(template.dataobj, dtype=np.float32)
    reference = np.asarray(outputs["native"].dataobj, dtype=np.float32)
    candidate = np.asarray(outputs["fnit"].dataobj, dtype=np.float32)
    difference = np.abs(candidate - reference)
    mask = (reference + candidate) > .01
    center = np.round(np.median(np.argwhere(mask), axis=0)).astype(int)
    background_limits = np.percentile(background[mask], (1, 99))
    slices = ((2, int(center[2]), "Axial"), (1, int(center[1]), "Coronal"))
    figure, axes = plt.subplots(2, 4, figsize=(10.5, 5.5), constrained_layout=True)
    titles = ("MNI152 template", "FSL FAST GM", "FNIT FAST: fsl", "Absolute GM difference")
    difference_panel = None
    for row, (axis, index, name) in enumerate(slices):
        def view(array):
            return np.rot90(np.take(array, index, axis=axis))
        axes[row, 0].imshow(view(background), cmap="gray", vmin=background_limits[0], vmax=background_limits[1])
        axes[row, 0].set_ylabel(name)
        for column, gm in ((1, reference), (2, candidate)):
            axes[row, column].imshow(view(background), cmap="gray", vmin=background_limits[0], vmax=background_limits[1])
            overlay = np.ma.masked_where(view(gm) <= .05, view(gm))
            axes[row, column].imshow(overlay, cmap="magma", vmin=0, vmax=1, alpha=.8)
        difference_panel = axes[row, 3].imshow(view(difference), cmap="viridis", vmin=0, vmax=.01)
        for column, panel in enumerate(axes[row]):
            panel.set_xticks([])
            panel.set_yticks([])
            if row == 0:
                panel.set_title(titles[column], fontsize=10)
    figure.colorbar(difference_panel, ax=axes[:, 3], shrink=.8, label="GM fraction")
    figure.suptitle("Same real T1 brain and MNI warp; ordered FAST execution")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=180, facecolor="white")
    plt.close(figure)
    report = {
        "scope": "De-identified GM example; native and FNIT maps sampled with the same archived native MNI pull; this figure does not benchmark registration",
        "input_sha256": {name: sha256(path) for name, path in manifest.items()},
        "driver_sha256": sha256(__file__),
        "canonical_template_shape": list(reference.shape),
        "slices": {name.lower(): index for _, index, name in slices},
        "template_gm_pearson": float(np.corrcoef(candidate[mask].astype(np.float64), reference[mask].astype(np.float64))[0, 1]),
        "template_gm_rmse": float(np.sqrt(np.mean((candidate[mask].astype(np.float64) - reference[mask].astype(np.float64)) ** 2))),
        "template_gm_max_abs": float(difference.max()),
        "figure_sha256": sha256(args.figure),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
