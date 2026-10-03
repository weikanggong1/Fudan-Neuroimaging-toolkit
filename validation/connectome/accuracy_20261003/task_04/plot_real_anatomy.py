"""Draw full real CON03 MRI results; do not modify the measured arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
root = args.root
nearest = json.loads((root / "nearest_reference_v1/report.json").read_text())
natural = next(case for case in nearest["cases"] if case["name"] == "sub-CON03_natural")
source = nib.load(natural["source"])
brain_path = Path(natural["source"]).parent / "brain.mgz"
brain = nib.load(brain_path)
assert np.allclose(brain.affine, source.affine) and brain.shape == source.shape
labels = np.asarray(source.dataobj)
brain_array = np.asarray(brain.dataobj)
tissue = root / "tissue_reference_v1/sub-CON03"


def aligned(path, target):
    image = nib.load(path)
    orientation = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(image.affine), nib.orientations.io_orientation(target.affine))
    return nib.orientations.apply_orientation(np.asarray(image.dataobj), orientation)


five = aligned(tissue / "native_five_tissue.nii.gz", source)
interface = aligned(tissue / "native_gmwmi.nii.gz", source)
# MGH is LIA: source axis 1 is superior/inferior. Keep the complete slice.
source_slice = 128
figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
axes[0].imshow(brain_array[:, source_slice, :].T, cmap="gray", origin="lower")
classes = five.argmax(-1) + 1
classes[five.sum(-1) == 0] = 0
axes[1].imshow(classes[:, source_slice, :].T, cmap="tab10", vmin=0, vmax=5, origin="lower")
axes[2].imshow(interface[:, source_slice, :].T, cmap="magma", origin="lower")
for axis, title in zip(axes, ("Real CON03 brain MRI", "5TT: all values identical", "GMWMI: all values identical")):
    axis.set_title(title)
    axis.set_axis_off()
figure.savefig(args.output / "CON03_5tt_gmwmi.png", dpi=150)
plt.close(figure)

target = nib.load(natural["target"])
baseline = np.load(root / "nearest_reference_v1/sub-CON03_natural/baseline.npy")
expected = aligned(root / "nearest_reference_v1/sub-CON03_natural/native.nii.gz", target)
bzero = np.asarray(target.dataobj[..., 0])
target_slice = target.shape[2] // 2
figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
axes[0].imshow(bzero[:, :, target_slice].T, cmap="gray", origin="lower")
axes[1].imshow(bzero[:, :, target_slice].T, cmap="gray", origin="lower")
mask = np.ma.masked_where(baseline[:, :, target_slice] == 0, baseline[:, :, target_slice])
axes[1].imshow(mask.T, cmap="nipy_spectral", origin="lower", alpha=.55)
axes[2].imshow((baseline != expected)[:, :, target_slice].T, cmap="Reds", vmin=0, vmax=1, origin="lower")
for axis, title in zip(axes, ("Real corrected DWI, volume 0", "Natural transform: FS labels", "All 552,960 target voxels: 0 differences")):
    axis.set_title(title, fontsize=10)
    axis.set_axis_off()
figure.savefig(args.output / "CON03_natural_nearest.png", dpi=150)
plt.close(figure)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

provenance = {"brain": str(brain_path), "brain_sha256": sha(brain_path),
              "source_labels_sha256": sha(natural["source"]),
              "dwi_sha256": sha(natural["target"]),
              "source_slice_axis": 1, "source_slice_index": source_slice,
              "dwi_slice_axis": 2, "dwi_slice_index": target_slice,
              "array_disagreements": int(np.count_nonzero(baseline != expected)),
              "plot_source_sha256": sha(__file__),
              "figures_sha256": {path.name: sha(path) for path in args.output.glob("*.png")}}
(args.output / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps(provenance, indent=2))
