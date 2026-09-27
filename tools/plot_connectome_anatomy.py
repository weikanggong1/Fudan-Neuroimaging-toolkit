"""Render the fixed ds004666 FreeSurfer/5TT/atlas example without modifying inputs."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap


def data(path):
    return np.asarray(nib.as_closest_canonical(nib.load(str(path))).dataobj)


def plane(array, fraction=0.52):
    return np.rot90(array[:, :, int(array.shape[2] * fraction)])


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("t1", "aparc", "five-tt", "gmwmi", "b0", "cortical", "subcortical", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    t1, aparc, five, gmwmi, b0 = (data(path) for path in
                                      (args.t1, args.aparc, args.five_tt, args.gmwmi, args.b0))
    cortical, sub = data(args.cortical).astype(np.int32), data(args.subcortical).astype(np.int32)
    tissue = np.argmax(five, axis=-1) + 1
    tissue[five.sum(-1) == 0] = 0
    combined = np.where(cortical > 0, cortical, np.where(sub > 0, sub + 2, 0))
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.4), constrained_layout=True)
    axes[0, 0].imshow(plane(t1), cmap="gray", vmin=np.percentile(t1, 5), vmax=np.percentile(t1, 99))
    axes[0, 0].set_title("Official FreeSurfer T1 brain")
    axes[0, 1].imshow(plane(t1), cmap="gray")
    axes[0, 1].imshow(np.ma.masked_equal(plane(aparc), 0), cmap="tab20", alpha=0.55, interpolation="nearest")
    axes[0, 1].set_title("Official aparc+aseg")
    axes[0, 2].imshow(plane(tissue), cmap=ListedColormap(["black", "#ef6a4b", "#e6b443", "#57add1", "#6b71cc", "#a979bd"]), vmin=0, vmax=5, interpolation="nearest")
    axes[0, 2].set_title("MRtrix 5TT = PyTorch 5TT")
    axes[0, 3].imshow(plane(t1), cmap="gray")
    axes[0, 3].imshow(np.ma.masked_where(plane(gmwmi) <= 0, plane(gmwmi)), cmap="autumn", alpha=0.8, vmin=0, vmax=1.5)
    axes[0, 3].set_title("GMWMI = PyTorch GMWMI")
    for ax, atlas, title in zip(axes[1], (None, cortical, sub, combined),
                                ("Corrected b0 brain", "Cortical atlas in DWI", "Subcortical atlas in DWI", "Combined atlas in DWI")):
        ax.imshow(plane(b0), cmap="gray", vmin=np.percentile(b0[b0 > 0], 2), vmax=np.percentile(b0, 99))
        if atlas is not None:
            ax.imshow(np.ma.masked_equal(plane(atlas), 0), cmap="tab10", alpha=0.67, interpolation="nearest", vmin=0, vmax=4)
        ax.set_title(title)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("ds004666 sub-01/ses-2mm · same T1 and corrected DWI · atlas and 5TT stage", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
