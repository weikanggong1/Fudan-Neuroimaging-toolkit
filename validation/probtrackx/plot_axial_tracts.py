"""Plot real ProbtrackX densities on several axial slices without saving source volumes."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load_canonical(path):
    image = nib.as_closest_canonical(nib.load(path))
    return image, np.asarray(image.dataobj, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--ac-mask", type=Path, required=True)
    parser.add_argument("--fsl-original", type=Path, required=True)
    parser.add_argument("--fsl-union", type=Path, required=True)
    parser.add_argument("--fnit-union", type=Path, required=True)
    parser.add_argument("--slices", type=int, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names = ("background", "ac_mask", "fsl_original", "fsl_union", "fnit_union")
    images = {name: load_canonical(getattr(args, name)) for name in names}
    reference = images["background"][0]
    for image, _ in images.values():
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
            raise ValueError("all volumes must have the same voxel grid and affine")
    if any(index < 0 or index >= reference.shape[2] for index in args.slices):
        raise ValueError("axial slice index out of range")

    background = images["background"][1]
    ac = images["ac_mask"][1] > 0
    original = images["fsl_original"][1]
    fsl = images["fsl_union"][1]
    fnit = images["fnit_union"][1]
    scale = max(float(original.max()), float(fsl.max()), float(fnit.max()))
    difference = fnit - fsl
    values = np.abs(difference[difference != 0])
    diff_scale = float(np.percentile(values, 99)) if values.size else 1.0
    brain_values = background[background > 0]
    background_max = float(np.percentile(brain_values, 99))

    fig, axes = plt.subplots(4, 4, figsize=(14.5, 13.8), facecolor="#101821")
    fig.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.07,
                        hspace=0.035, wspace=0.025)
    labels = ("FSL: last avoid only", "FSL: AC + brainstem",
              "FNIT: AC + brainstem", "FNIT - FSL: union")
    density_handle = None
    difference_handle = None
    for row in range(4):
        fig.text(0.015, 0.813 - row * 0.213, labels[row], color="white",
                 fontsize=12, rotation=90, va="center")
        for column, index in enumerate(args.slices):
            ax = axes[row, column]
            ax.set_facecolor("#101821")
            ax.imshow(background[:, :, index].T, origin="lower", cmap="gray",
                      vmin=0, vmax=background_max, interpolation="nearest")
            if row < 3:
                density = (original, fsl, fnit)[row][:, :, index].T
                density_handle = ax.imshow(
                    np.ma.masked_where(density <= 0, np.log1p(density)),
                    origin="lower", cmap="magma", vmin=0, vmax=np.log1p(scale),
                    alpha=0.82, interpolation="nearest")
            else:
                delta = difference[:, :, index].T
                difference_handle = ax.imshow(
                    np.ma.masked_where(delta == 0, delta), origin="lower",
                    cmap="coolwarm", vmin=-diff_scale, vmax=diff_scale,
                    alpha=0.82, interpolation="nearest")
            if ac[:, :, index].any():
                ax.contour(ac[:, :, index].T.astype(float), levels=[0.5],
                           colors=["#38d6d0"], linewidths=0.75)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"native axial slice {index}", color="white", fontsize=12)
    for handle, targets, label in (
        (density_handle, axes[:3, :], "log(1 + visits)"),
        (difference_handle, axes[3, :], "FNIT - FSL visits (clipped at p99)"),
    ):
        colorbar = fig.colorbar(handle, ax=targets.ravel().tolist(),
                                fraction=0.014, pad=0.012)
        colorbar.set_label(label, color="white")
        colorbar.ax.tick_params(colors="white")
    fig.suptitle("Real dMRI: NbM to Cingulum, 5000 samples per seed voxel",
                 color="white", fontsize=16)
    fig.text(0.12, 0.032,
             "Cyan outline: AC mask. Native oblique axial slices shown in RAS orientation. Density maps are unthresholded.",
             color="#d4dce5", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=fig.get_facecolor(),
                metadata={"Software": "FNIT validation"})
    plt.close(fig)


if __name__ == "__main__":
    main()
