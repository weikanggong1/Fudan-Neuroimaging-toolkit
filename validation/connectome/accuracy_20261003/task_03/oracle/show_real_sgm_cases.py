"""Record actual SGM arc coordinates, 5TT state and SH; plot real anatomy."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
for name in ("images", "cases", "source", "official", "output"):
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(8)
package = types.ModuleType("sgm_figure")
package.__path__ = [str(args.source)]
sys.modules[package.__name__] = package
for name in ("fod", "tracking"):
    spec = importlib.util.spec_from_file_location(
        package.__name__ + "." + name, args.source / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

five_image = nib.load(args.images / "five_tissue_act.nii.gz")
fod_image = nib.load(args.images / "wm_fod.nii.gz")
five = torch.as_tensor(np.asarray(five_image.dataobj, dtype=np.float32))
fod = torch.as_tensor(np.asarray(fod_image.dataobj, dtype=np.float32))
five_inverse = torch.as_tensor(np.linalg.inv(five_image.affine), dtype=torch.float64)
fod_inverse = torch.as_tensor(np.linalg.inv(fod_image.affine), dtype=torch.float64)
arc_cases = np.load(args.cases / "cases.npz")["arc_cases"]
official = np.array([[float(x) for x in line.split()[1:]]
                     for line in args.official.read_text().splitlines()
                     if line.startswith("A ")])
positions = torch.as_tensor(official[:, [3, 4, 5, 11, 12, 13]].reshape(-1, 2, 3),
                            dtype=torch.float32)
tissue = module._five_tissue_mrtrix(five, positions.reshape(-1, 3), five_inverse)
tissue = tissue.reshape(-1, 2, 5).numpy()
cgm, sgm, wm, csf, pathology = np.moveaxis(tissue, -1, 0)
both_sgm = ((tissue.sum(-1) >= .5) & (sgm > cgm) & (cgm + sgm >= wm) &
            (cgm + sgm > csf) & (cgm + sgm > pathology) &
            ~((csf >= cgm) & (csf >= sgm) & (csf >= wm) & (csf >= pathology))).all(-1)
old_choice = official[:, [9, 17]].argmin(-1)
new_choice = official[:, [10, 18]].argmin(-1)
changed = np.flatnonzero(both_sgm & (old_choice != new_choice))
records = []
for index in changed:
    selected_points = positions[index]
    coefficients = module._sample(fod, selected_points, fod_inverse).numpy()
    records.append({
        "arc_case": int(index), "start_ras_mm": arc_cases[index, :3].tolist(),
        "prior_direction": arc_cases[index, 3:6].tolist(),
        "proposal_end_direction": arc_cases[index, 6:9].tolist(),
        "vertices_ras_mm": selected_points.tolist(),
        "tangents": official[index, [6, 7, 8, 14, 15, 16]].reshape(2, 3).tolist(),
        "five_tissue_cgm_sgm_wm_csf_pathology": tissue[index].tolist(),
        "wm_fod_sh_coefficients": coefficients.tolist(),
        "official_tangent_metric": official[index, [9, 17]].tolist(),
        "official_chord_metric": official[index, [10, 18]].tolist(),
        "baseline_minimum_vertex": int(old_choice[index]),
        "official_and_candidate_minimum_vertex": int(new_choice[index]),
    })
report = {"scope": "real CON03 SGM internal arc diagnostic; coordinates in RAS mm",
          "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "official_sha256": hashlib.sha256(args.official.read_bytes()).hexdigest(),
          "changed_cases": records}
(args.output / "changed_sgm_cases.json").write_text(json.dumps(report, indent=2) + "\n")

# Image panels use the real 5TT anatomy, centred on a measured changed arc.
example = int(changed[0])
vertices = positions[example].numpy()
voxel = nib.affines.apply_affine(np.linalg.inv(five_image.affine), vertices)
center = np.rint(voxel.mean(0)).astype(int)
anatomy = (five.numpy() * np.array([.5, .25, .9, .05, .4])).sum(-1)
figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
for axis, displayed, caption in ((0, (1, 2), "Sagittal"),
                                  (1, (0, 2), "Coronal"),
                                  (2, (0, 1), "Axial")):
    ax = axes[axis]
    section = np.take(anatomy, center[axis], axis=axis)
    ax.imshow(section.T, origin="lower", cmap="gray", vmin=0, vmax=1)
    horizontal, vertical = displayed
    ax.plot(voxel[:, horizontal], voxel[:, vertical], color="#40b7e7", lw=2)
    ax.scatter(*voxel[old_choice[example], [horizontal, vertical]],
               color="#f39c34", marker="x", s=100, lw=2, label="Baseline tangent")
    ax.scatter(*voxel[new_choice[example], [horizontal, vertical]],
               facecolors="none", edgecolors="#e3416c", s=90, lw=2,
               label="Official / candidate chord")
    ax.set_xlim(center[horizontal] - 30, center[horizontal] + 30)
    ax.set_ylim(center[vertical] - 30, center[vertical] + 30)
    ax.set_title(caption)
    ax.set_xlabel("Voxel index")
    ax.set_ylabel("Voxel index")
axes[-1].legend(fontsize=7, loc="lower right")
figure.suptitle(f"CON03 actual SGM arc {example}: FOD minimum uses incoming chord")
figure.savefig(args.output / "sgm_chord_real_CON03.png", dpi=180)
plt.close(figure)
print(json.dumps({"changed_cases": len(changed), "example": example,
                  "output": str(args.output)}))
