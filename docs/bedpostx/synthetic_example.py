"""Generate shareable synthetic DWI and plot FSL/TorchBEDPOSTX posterior maps.

Usage: python synthetic_example.py OUTPUT_DIR [--plot]
Run the two fitters between the generation and --plot calls; see README.md.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def generate(root):
    subject = root / "subject"
    subject.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260927)
    index = np.arange(100, dtype=np.float64)
    z = 1 - 2 * (index + 0.5) / 100
    azimuth = index * np.pi * (3 - np.sqrt(5))
    radial = np.sqrt(1 - z * z)
    vectors = np.column_stack((radial * np.cos(azimuth),
                               radial * np.sin(azimuth), z))
    bvecs = np.vstack((np.zeros((5, 3)), vectors))
    bvals = np.array([0] * 5 + [1000] * 50 + [2000] * 50)
    x, y = np.indices((8, 8))
    f1 = 0.25 + 0.30 * x / 7
    f2 = 0.15 + 0.15 * y / 7
    d, d_std, s0 = 0.0015, 0.0004, 100.0
    exponent = -(d / d_std) ** 2
    scale = bvals * d_std**2 / d
    ball = (1 + scale) ** exponent
    stick1 = (1 + scale * bvecs[:, 0] ** 2) ** exponent
    stick2 = (1 + scale * bvecs[:, 1] ** 2) ** exponent
    signal = s0 * ((1 - f1[..., None] - f2[..., None]) * ball
                   + f1[..., None] * stick1 + f2[..., None] * stick2)
    signal = np.clip(signal + rng.normal(0, 1.0, signal.shape), 0.1, None)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(signal[:, :, None].astype(np.float32), affine),
             str(subject / "data.nii.gz"))
    nib.save(nib.Nifti1Image(np.ones((8, 8, 1), np.uint8), affine),
             str(subject / "nodif_brain_mask.nii.gz"))
    np.savetxt(subject / "bvals", bvals[None], fmt="%d")
    np.savetxt(subject / "bvecs", bvecs.T, fmt="%.8f")
    np.savez(root / "ground_truth.npz", f1=f1, f2=f2)


def plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = np.load(root / "ground_truth.npz")
    original = root / "fsl_cpu"
    replica = root / "fnit_gpu"
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4), constrained_layout=True)
    for row, fibre in enumerate((1, 2)):
        reference = truth[f"f{fibre}"]
        fsl = np.squeeze(nib.load(str(original / f"mean_f{fibre}samples.nii.gz")).get_fdata())
        torch = np.squeeze(nib.load(str(replica / f"mean_f{fibre}samples.nii.gz")).get_fdata())
        images = (reference, fsl, torch, np.abs(fsl - torch))
        for col, (image, title) in enumerate(zip(images,
                                                  ("Ground truth", "Original FSL",
                                                   "TorchBEDPOSTX", "Absolute difference"))):
            ax = axes[row, col]
            vmax = (max(reference.max(), fsl.max(), torch.max()) if col < 3
                    else max(float(np.percentile(np.abs(fsl - torch), 99)), 1e-6))
            im = ax.imshow(image.T, origin="lower", vmin=0, vmax=vmax,
                           cmap="viridis" if col < 3 else "magma")
            label = f"{title} · f{fibre}"
            if col == 3:
                label += f" (p99={vmax:.4f})"
            ax.set_title(label)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(root / "synthetic_example.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.plot:
        plot(args.output_dir)
    else:
        generate(args.output_dir)
