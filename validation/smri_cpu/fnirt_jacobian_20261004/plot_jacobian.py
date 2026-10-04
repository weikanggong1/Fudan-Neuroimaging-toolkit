"""Render already computed real Jacobians, with equal display scales."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: args.input_directory / (name + ".nii.gz") for name in
             ["official_jacobian", "old_jacobian", "new_jacobian", "mask"]}
    values = {name: np.asanyarray(nib.load(path).dataobj) for name, path in paths.items()}
    mask = values["mask"] > 0
    official, old, new = [values[name] for name in ["official_jacobian", "old_jacobian", "new_jacobian"]]
    cuts = tuple(int((size - 1) // 2) for size in official.shape)
    rows = [("Official analytic", official), ("Previous dense", old), ("New CPU analytic", new),
            ("Previous absolute error", np.abs(old - official)), ("New absolute error", np.abs(new - official))]
    fig, axes = plt.subplots(5, 3, figsize=(9.5, 12), layout="constrained")
    for row, (title, data) in enumerate(rows):
        for axis in range(3):
            plane = np.take(data, cuts[axis], axis=axis).T
            shown_mask = np.take(mask, cuts[axis], axis=axis).T
            shown = np.ma.masked_where(~shown_mask, plane)
            cmap = plt.get_cmap("viridis" if row < 3 else "magma").copy()
            cmap.set_bad("black")
            image = axes[row, axis].imshow(shown, origin="lower", interpolation="nearest", cmap=cmap,
                                           vmin=.5 if row < 3 else 0., vmax=1.5 if row < 3 else .08)
            axes[row, axis].set_axis_off()
            if axis == 0:
                axes[row, axis].set_title(title, loc="left", fontsize=10)
        fig.colorbar(image, ax=axes[row, :].tolist(), shrink=.75)
    fig.suptitle("CPU GM-FNIRT Jacobian definition repair\nSame fitted field; nonlinear estimation mismatch remains", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    manifest = {"scope": "derived real Jacobians from the same CPU fit, central orthogonal planes",
                "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
                "shape": official.shape, "cuts_voxel_indices": cuts,
                "display_mask": "explicit dilated reference mask; numerical metrics separately use mask AND positive GM template",
                "display_ranges": {"jacobian": [.5, 1.5], "absolute_error": [0., .08]},
                "clipping": "display only, all maxima retained in cpu.public.json",
                "png_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(".public.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
