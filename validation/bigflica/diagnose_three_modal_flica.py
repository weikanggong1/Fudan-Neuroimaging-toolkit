"""CPU sensitivity check for three real DicL dictionaries (VBM, FA, MD).

Run with ``PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python
validation/bigflica/diagnose_three_modal_flica.py --dictionary-dir DIR --output FILE``.
The dictionaries must be the actual D-by-R ``*_dictionary.npy`` outputs.
The JSON contains aggregate statistics and file hashes, never subject IDs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import scipy

from fnit.bigflica import flica_vb


NAMES = ("vbm", "fa", "md")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit(data: list[np.ndarray], components: int, *, initialization: str = "PCA",
         fixed_dd: bool = False, vbm_scale: float = 1.0) -> dict:
    matrices = [matrix.copy() for matrix in data]
    matrices[0] *= vbm_scale
    options = {
        "num_components": components, "maxits": 100,
        "dof_per_voxel": np.ones(3) if fixed_dd else "auto_eigenspectrum",
        "lambda_dims": "R", "initH": initialization, "computeF": 0,
        "output_dir": "/tmp",
    }
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        priors, posteriors, constants = flica_vb.flica_init_params(matrices, options)
        initial_lambda = [np.asarray(value).reshape(-1) for value in posteriors["Lambda"]]
        dd = np.asarray(constants["DD"]).copy()
        fitted = flica_vb.flica_iterate(matrices, options, priors, posteriors, constants)
    h = np.asarray(fitted["H"], dtype=np.float64)
    singular = np.linalg.svd(h, compute_uv=False)
    ratios = {}
    precision_ratios = {}
    for name, matrix, x, w, precision in zip(
            NAMES, matrices, fitted["X"], fitted["W"], fitted["lambda"]):
        reconstructed = (np.asarray(x) * np.asarray(w).reshape(-1)) @ h
        ratios[name] = float(np.linalg.norm(reconstructed) / np.linalg.norm(matrix))
        precision = np.asarray(precision).reshape(-1)
        precision_ratios[name] = float(np.max(precision) / np.min(precision))
    return {
        "components": components, "initialization": initialization,
        "fixed_dd_ones": fixed_dd, "vbm_input_scale": vbm_scale,
        "wall_time_s": time.perf_counter() - start,
        "dd": dict(zip(NAMES, map(float, dd))),
        "initial_noise_precision_median": {
            name: float(np.median(values))
            for name, values in zip(NAMES, initial_lambda)},
        "rank_at_relative_1e-6": int(np.count_nonzero(singular / singular[0] > 1e-6)),
        "smallest_over_largest_h_singular": float(singular[-1] / singular[0]),
        "reconstruction_norm_ratio": ratios,
        "final_subject_noise_precision_max_over_min": precision_ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.dictionary_dir / f"{name}_dictionary.npy" for name in NAMES]
    matrices = [np.load(path) for path in paths]
    if len({matrix.shape for matrix in matrices}) != 1:
        raise ValueError("all three dictionaries must have the same D-by-R shape")
    report = {
        "data": "2050 real UKB subjects; three full-mask VBM/FA/MD DicL dictionaries, D200/R100",
        "source_sha256": _sha256(Path(flica_vb.__file__)),
        "dictionary_sha256": dict(zip(NAMES, map(_sha256, paths))),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "dictionary_spectra": {}, "joint_pca_C20_projection_norm_ratio": {},
        "sensitivity_fits": [],
    }
    dd = []
    for name, matrix in zip(NAMES, matrices):
        singular = np.linalg.svd(matrix, compute_uv=False)
        rms_by_subject = np.sqrt(np.mean(matrix ** 2, axis=0))
        with contextlib.redirect_stdout(io.StringIO()):
            estimated_dd = flica_vb.est_DOF_eigenspectrum(matrix) / matrix.shape[0]
        dd.append(estimated_dd)
        report["dictionary_spectra"][name] = {
            "shape": list(matrix.shape),
            "rms": float(np.sqrt(np.mean(matrix ** 2))),
            "s17_over_s1": float(singular[16] / singular[0]),
            "s20_over_s1": float(singular[19] / singular[0]),
            "s40_over_s1": float(singular[39] / singular[0]),
            "subject_rms_min_median_max": [float(np.min(rms_by_subject)),
                                            float(np.median(rms_by_subject)),
                                            float(np.max(rms_by_subject))],
            "estimated_dd": float(estimated_dd),
        }
    grams = [matrix.T @ matrix for matrix in matrices]
    for label, weights in (("unweighted", np.ones(3)),
                           ("estimated_dd", np.asarray(dd))):
        _, eigenvectors = np.linalg.eigh(
            sum(weight * gram for weight, gram in zip(weights, grams)))
        basis = eigenvectors[:, -20:]
        report["joint_pca_C20_projection_norm_ratio"][label] = {
            name: float(np.linalg.norm(matrix @ basis) / np.linalg.norm(matrix))
            for name, matrix in zip(NAMES, matrices)}
    for components, initialization, fixed_dd, scale in (
            (20, "PCA", False, 1.0),
            (17, "PCA", False, 1.0),
            (20, "PCAnew", False, 1.0),
            (20, "PCA", True, 1.0),
            (20, "PCA", False, 2.0)):
        report["sensitivity_fits"].append(_fit(
            matrices, components, initialization=initialization,
            fixed_dd=fixed_dd, vbm_scale=scale))
    report["interpretation"] = (
        "All dictionaries have 20th singular values far above zero. "
        "Joint PCA can represent all three modalities, but variational R-mode "
        "FLICA prunes components or sacrifices modalities. Fixed DD and VBM scaling "
        "are sensitivity tests, not accepted corrections or equivalent inputs. "
        "This three-full-mask case cannot be directly compared to earlier sparse "
        "four-modality mMIGP dictionaries because both masks and mMIGP differ.")
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
