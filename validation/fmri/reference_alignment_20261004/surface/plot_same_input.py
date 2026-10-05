"""Plot real cortical means and sampler differences on the own prepared geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np

from run_same_input import sha256, write_json


def plot(existing_root, manifest_path, proof_path, output_root):
    existing, output = Path(existing_root).resolve(), Path(output_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    proof_path = Path(proof_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    proof = json.loads(proof_path.read_text())
    if proof["case_id"] != manifest["case_id"] or proof.get("input_guards_equal") is not True:
        raise ValueError("A guarded same-case numerical proof is required")
    if output.exists():
        raise FileExistsError("A fresh figure directory is required")
    prepared = json.loads((existing / "prepared_inputs.private.json").read_text())
    guard_paths = {"manifest": manifest_path, "proof": proof_path,
                   "plot_source": Path(__file__).resolve(),
                   "helper_source": Path(__file__).with_name("run_same_input.py"),
                   "prepared_binding": existing / "prepared_inputs.private.json"}
    bindings = {}
    for hemi in ("L", "R"):
        bindings[hemi] = {
            "mid": prepared[hemi + ".atlas_midthickness"],
            "roi": manifest["hemispheres"][hemi]["atlas_roi"],
            **{role: {"path": str(existing / role / f"{hemi}.32k.func.gii"),
                      "sha256": proof["input_sha256_before"][role + "." + f"{hemi}.32k.func.gii"]}
               for role in ("FNIT", "reference")},
        }
        for key, entry in bindings[hemi].items():
            guard_paths[hemi + "." + key] = Path(entry["path"])
            if sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("Figure bytes differ from the numerical proof")
    before = {key: sha256(path) for key, path in guard_paths.items()}
    rows = {}
    for hemi in ("L", "R"):
        mid = Path(prepared[hemi + ".atlas_midthickness"]["path"])
        roi = Path(manifest["hemispheres"][hemi]["atlas_roi"]["path"])
        fnit, reference = [existing / role / f"{hemi}.32k.func.gii" for role in ("FNIT", "reference")]
        for key, path in (("mid", mid), ("roi", roi), ("FNIT", fnit), ("reference", reference)):
            guard_paths[hemi + "." + key] = path
        geometry = nib.load(mid)
        vertices = np.asarray(geometry.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data)
        faces = np.asarray(geometry.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")[0].data)
        mask = np.asarray(nib.load(roi).darrays[0].data) > 0
        a = np.stack([d.data for d in nib.load(fnit).darrays])
        b = np.stack([d.data for d in nib.load(reference).darrays])
        if a.shape != (manifest["expected_frames"], 32492) or a.shape != b.shape:
            raise ValueError("Requires the complete real cortical time series")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite cortical values")
        rows[hemi] = (vertices, faces, mask, a.mean(axis=0), np.abs(a.astype(np.float64)-b).max(axis=0))
    mean_values = np.concatenate([row[3][row[2]] for row in rows.values()])
    mean_norm = Normalize(*np.percentile(mean_values, [2, 98]))
    error_norm = Normalize(0, 1e-6)
    figure = plt.figure(figsize=(10, 7), facecolor="white")
    for row_index, hemi in enumerate(("L", "R")):
        vertices, faces, mask, mean, error = rows[hemi]
        cortical_faces = faces[mask[faces].all(axis=1)]
        for column, (values, cmap, norm, label) in enumerate((
                (mean, "viridis", mean_norm, "Mean preproc BOLD"),
                (error, "magma", error_norm, "Maximum absolute difference"))):
            ax = figure.add_subplot(2, 2, row_index*2+column+1, projection="3d")
            background = Poly3DCollection(vertices[faces], facecolor="#dedede", edgecolor="none")
            ax.add_collection3d(background)
            collection = Poly3DCollection(vertices[cortical_faces], edgecolor="none")
            collection.set_facecolor(plt.get_cmap(cmap)(norm(values[cortical_faces].mean(axis=1))))
            ax.add_collection3d(collection)
            for setter, bounds in ((ax.set_xlim, vertices[:, 0]), (ax.set_ylim, vertices[:, 1]), (ax.set_zlim, vertices[:, 2])):
                setter(bounds.min(), bounds.max())
            ax.set_box_aspect(np.ptp(vertices, axis=0))
            ax.view_init(elev=0, azim=180 if hemi=="L" else 0)
            ax.set_axis_off()
            ax.set_title(f"{hemi}: {label}", fontsize=11)
    figure.suptitle(f"{manifest['case_id']} | Same input, {manifest['expected_frames']} frames | WB 2.1.0 vs 2.0.1", fontsize=13)
    mean_bar = figure.colorbar(plt.cm.ScalarMappable(norm=mean_norm, cmap="viridis"),
                              ax=figure.axes[0::2], fraction=.03, pad=.01)
    mean_bar.set_label("BOLD intensity", fontsize=9)
    error_bar = figure.colorbar(plt.cm.ScalarMappable(norm=error_norm, cmap="magma"),
                               ax=figure.axes[1::2], fraction=.03, pad=.01)
    error_bar.set_label("Absolute error (0 to declared 1e-6)", fontsize=9)
    output.mkdir(parents=True)
    image = output / f"{manifest['case_id']}.same_input.surface.png"
    figure.savefig(image, dpi=160, bbox_inches="tight")
    plt.close(figure)
    after = {key: sha256(path) for key, path in guard_paths.items()}
    write_json(output / "figure.public.json", {
        "case_id": manifest["case_id"], "frames": manifest["expected_frames"],
        "scope": "same-input cortical signal and per-vertex maximum difference on newly prepared own individual fsLR midthickness",
        "scientific_equivalence": "not_assessed", "input_guards_equal": before == after,
        "input_sha256_before": before, "input_sha256_after": after,
        "image_sha256": sha256(image),
        "maximum_cortical_absolute_error": float(max(row[4].max() for row in rows.values())),
        "geometry_is_original_temp_recovery": False,
        "renderer_versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "nibabel": nib.__version__},
    })
    if before != after:
        raise ValueError("Figure inputs changed")
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--proof", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(plot(args.existing_root, args.manifest, args.proof, args.output_root))
