"""Run a real 2,050-subject, three-modality public-API CPU smoke test.

Usage (private paths remain on the validation server)::

    python benchmark_public_cpu_real2050.py SUBJECTS_ROOT CONFIG_JSON \
        SUBJECTS_TXT OUTPUT_DIR SUMMARY_JSON

The first 2,050 listed subjects train the model. The next subject is held out
for ``apply_model``. Only aggregate results are written to SUMMARY_JSON.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from fnit.bigflica.pipeline import apply_model, run_bigflica


N_SUBJECTS = 2050
MODALITIES = ("vbm", "fa", "md")


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    root, config_file, subjects_file, output, summary_file = map(Path, sys.argv[1:])
    specs = json.loads(config_file.read_text(encoding="utf-8"))["modalities"]
    modalities = {name: specs[name] for name in MODALITIES}
    subjects = subjects_file.read_text(encoding="utf-8").splitlines()
    if len(subjects) <= N_SUBJECTS:
        raise ValueError("Need 2,050 training subjects and one held-out subject")
    selected = subjects[:N_SUBJECTS]
    held_out = root / subjects[N_SUBJECTS]
    if len(set(selected)) != N_SUBJECTS or held_out.name in selected:
        raise ValueError("Training and held-out subject lists overlap")
    if not all((held_out / modalities[name]["image"]).is_file()
               for name in MODALITIES):
        raise ValueError("Held-out subject lacks a requested modality")

    started = time.perf_counter()
    attempts = []
    result = None
    for components in (3, 2):
        try:
            result = run_bigflica(
                root, modalities, output, n_components=components,
                migp_dim=100, dicl_dim=200, subjects=selected, device="cpu",
                dicl_max_iter=1000, flica_max_iter=100, top_voxels=1000,
                random_state=0, max_gpu_gb=20.0, feature_block=2048,
                flica_lambda_dims="R",
            )
            break
        except ValueError as error:
            attempts.append({"components": components, "error": str(error)})
            if "FLICA collapsed or pruned requested components" not in str(error):
                raise
    if result is None:
        raise RuntimeError("Neither C3 nor C2 passed the FLICA rank gate")

    model = json.loads((result / "model.json").read_text(encoding="utf-8"))
    reconstruction = json.loads(
        (result / "flica_reconstruction.json").read_text(encoding="utf-8"))
    course = np.load(result / "subj_course.npy", mmap_mode="r")
    components = model["n_components"]
    if course.shape != (N_SUBJECTS, components) or not np.isfinite(course).all():
        raise ValueError("Invalid subject course output")
    maps = {}
    for name in MODALITIES:
        z = np.load(result / f"{name}_zstat.npy", mmap_mode="r")
        mask = np.asarray(nib.load(str(result / f"{name}_mask.nii.gz")).dataobj) > 0
        if z.shape != (int(mask.sum()), components) or not np.isfinite(z).all():
            raise ValueError(f"Invalid {name} z-stat matrix")
        map_dir = result / "maps" / name
        for index in range(components):
            prefix = f"component-{index + 1:03d}"
            z_image = nib.load(str(map_dir / f"{prefix}_zstat.nii.gz"))
            image_values = np.asarray(z_image.dataobj)[mask]
            if not np.array_equal(image_values, z[:, index]):
                raise ValueError(f"{name} z-stat NIfTI differs from its array")
            if not (map_dir / f"{prefix}_top-1000.nii.gz").is_file() or not (
                map_dir / f"{prefix}_top-1000.png").is_file():
                raise ValueError(f"Missing {name} thresholded map or plot")
        maps[name] = {"voxels": int(mask.sum()), "components": components,
                      "nifti_array_max_abs_error": 0.0}
    held_out_scores = apply_model(result, held_out, device="cpu",
                                  output_file=output / "held_out_course.tsv")
    if held_out_scores.shape != (components,) or not np.isfinite(held_out_scores).all():
        raise ValueError("Invalid held-out course")
    report = {
        "dataset": "2050 real subjects; full-mask VBM/FA/MD; task excluded",
        "backend": "cpu_public_api_cold_start",
        "parameters": {"migp_dim": 100, "dicl_dim": 200,
                       "dicl_max_iter": 1000, "flica_max_iter": 100,
                       "flica_lambda_dims": "R", "feature_block": 2048},
        "requested_components_attempts": attempts + [{"components": components,
                                                      "status": "accepted"}],
        "effective_rank": reconstruction["component_rank"],
        "reconstruction_norm_ratio": reconstruction["per_modality_ratio"],
        "course_shape": list(course.shape), "maps": maps,
        "held_out_apply_model": {"finite": True, "components": components,
                                 "score_norm": float(np.linalg.norm(held_out_scores))},
        "stage_timings_s": model["timings"],
        "total_wall_s": time.perf_counter() - started,
        "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "scope": "C3 or C2 functional smoke; C20 scientific acceptance remains separate",
    }
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
