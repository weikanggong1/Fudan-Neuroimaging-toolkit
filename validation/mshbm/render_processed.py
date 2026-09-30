"""在服务器用公开模板几何绘制匿名 MS-HBM 对照图；不导出影像数组。"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np


COLORS = np.asarray([[.55, .55, .55, 1.]] +
                    [[(i * 37 % 255) / 255., (i * 73 % 255) / 255.,
                      (i * 109 % 255) / 255., 1.] for i in range(1, 18)])


def surface(root, left, right, output):
    candidate = np.load(root / "surface_candidate_current/labels_fslr32k_64984.npy")
    reference = np.load(root / "surface_ukb_release/labels_fslr32k_64984.npy")
    figure = plt.figure(figsize=(13, 7))
    for row, path in enumerate((left, right)):
        image = nib.load(str(path))
        points = image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data
        faces = image.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")[0].data
        selection = slice(row * 32492, (row + 1) * 32492)
        for col, (values, title) in enumerate(((candidate[selection], "FNIT"),
                                              (reference[selection], "UKB release: MSMAll"),
                                              ((candidate[selection] != reference[selection]),
                                               "Disagreement (red)"))):
            axes = figure.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
            colors = (np.where(values[faces[:, 0], None], [1., .12, .10, 1.],
                               [.75, .75, .75, 1.]) if col == 2 else COLORS[values[faces[:, 0]]])
            collection = Poly3DCollection(points[faces], facecolors=colors,
                                            edgecolors="none", linewidths=0)
            axes.add_collection3d(collection)
            axes.set_xlim(-80, 80); axes.set_ylim(-110, 75); axes.set_zlim(-60, 90)
            axes.set_box_aspect((160, 185, 150), zoom=1.35)
            axes.view_init(elev=10, azim=180 if row == 0 else 0)
            axes.set_axis_off()
            axes.set_title(("Left: " if row == 0 else "Right: ") + title)
    figure.suptitle("MS-HBM HCP_40 17 networks — same real 490-frame scan", fontsize=15)
    figure.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=.02, top=.89)
    figure.savefig(output / "mshbm_surface_release.png", dpi=150)
    plt.close(figure)


def volume(root, template, output):
    candidate = np.asarray(nib.load(root / "volume_candidate/labels_mni.nii.gz").dataobj)
    reference = np.asarray(nib.load(root / "volume_ukb_fix/labels_mni.nii.gz").dataobj)
    anatomy = np.asarray(nib.load(str(template)).dataobj)
    figure, axes = plt.subplots(3, 3, figsize=(10, 10))
    for row, (values, title) in enumerate(((candidate, "FNIT"), (reference, "UKB FIX + FSL warp"),
                                           ((candidate != reference), "Disagreement"))):
        for col, z in enumerate((40, 54, 66)):
            axis = axes[row, col]
            axis.imshow(anatomy[:, :, z].T, origin="lower", cmap="gray", vmin=0, vmax=8000)
            layer = np.ma.masked_equal(values[:, :, z].T, 0)
            axis.imshow(layer, origin="lower", cmap=(ListedColormap(COLORS) if row < 2
                                                     else ListedColormap(["none", "red"])),
                        vmin=0, vmax=17 if row < 2 else 1, interpolation="nearest", alpha=.8)
            axis.set_axis_off()
            axis.set_title(f"{title}; MNI z={2*z-72} mm")
    figure.suptitle("MNI 2 mm → fsLR32k MS-HBM → cortical volume", fontsize=15)
    figure.tight_layout()
    figure.savefig(output / "mshbm_volume_release.png", dpi=150)
    plt.close(figure)


def connectivity(root, output):
    figure, axes = plt.subplots(2, 3, figsize=(11, 7))
    for row, name in enumerate(("surface_release_comparison", "volume_release_comparison")):
        data = np.load(root / f"{name}.public.private.npz")
        candidate, reference = data["candidate_fc"], data["reference_fc"]
        for col, (matrix, title) in enumerate(((candidate, "FNIT r"), (reference, "UKB reference r"),
                                               (candidate - reference, "FNIT - reference"))):
            shown = axes[row, col].imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
            axes[row, col].set_title(("Surface: " if row == 0 else "Volume: ") + title)
            axes[row, col].set_xticks([0, 8, 16], [1, 9, 17])
            axes[row, col].set_yticks([0, 8, 16], [1, 9, 17])
            axes[row, col].set_xlabel("HCP_40 network")
    colorbar_axis = figure.add_axes([.91, .2, .02, .55])
    figure.colorbar(shown, cax=colorbar_axis)
    figure.suptitle("17-network Pearson FC — fixed UKB-reference MS-HBM partition")
    figure.subplots_adjust(wspace=.35, hspace=.5, right=.86, top=.87)
    figure.savefig(output / "mshbm_release_connectivity.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("private-root", "left-surface", "right-surface", "template", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    root = Path(args.private_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    surface(root, args.left_surface, args.right_surface, output)
    volume(root, args.template, output)
    connectivity(root, output)


if __name__ == "__main__":
    main()
