"""Generate a public, deterministic bedpostX-format single-fibre tube."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
out = args.output_dir.resolve()
samples = out / "bedpostX"
samples.mkdir(parents=True, exist_ok=True)

shape = (40, 40, 40)
frames = 20
random_seed = 20260927
angle_noise_sd = 0.12
affine = np.array([[-2.0, 0, 0, 40],
                   [0, 2.0, 0, -40],
                   [0, 0, 2.0, -40],
                   [0, 0, 0, 1]])
x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]
mask = (x >= 4) & (x <= 35) & ((y - 20) ** 2 + (z - 20) ** 2 <= 16)

rng = np.random.default_rng(random_seed)
dy = rng.normal(0, angle_noise_sd, size=shape + (frames,))
dz = rng.normal(0, angle_noise_sd, size=shape + (frames,))
norm = np.sqrt(1 + dy**2 + dz**2)
theta = np.arccos(dz / norm).astype(np.float32)
phi = np.arctan2(dy, np.ones_like(dy)).astype(np.float32)
fraction = np.full(shape + (frames,), 0.8, dtype=np.float32)

def save(path, array):
    nib.save(nib.Nifti1Image(array, affine), str(path))


save(samples / "nodif_brain_mask.nii.gz", mask.astype(np.uint8))
for key, values in (("th", theta), ("ph", phi), ("f", fraction)):
    values[~mask] = 0
    save(samples / f"merged_{key}1samples.nii.gz", values)

rois = []
for number, center in enumerate(((10, 20, 20), (26, 20, 20)), start=1):
    roi = mask & ((x - center[0]) ** 2 + (y - center[1]) ** 2
                  + (z - center[2]) ** 2 <= 1)
    path = out / f"roi_{number:02d}.nii.gz"
    save(path, roi.astype(np.uint8))
    rois.append({"path": str(path), "center_ijk": center, "voxel_count": int(roi.sum())})

(out / "seed_list.txt").write_text("\n".join(roi["path"] for roi in rois) + "\n")
(out / "synthetic_metadata.json").write_text(json.dumps({
    "synthetic": True,
    "shape": shape,
    "affine": affine.tolist(),
    "posterior_samples": frames,
    "fibre_fraction": 0.8,
    "direction": "+x with independent Gaussian y/z components",
    "angular_component_sd": angle_noise_sd,
    "random_seed": random_seed,
    "tube_center_yz": [20, 20],
    "tube_radius_voxels": 4,
    "tube_x_range_inclusive": [4, 35],
    "rois": rois,
}, indent=2) + "\n")
print(out / "synthetic_metadata.json")
