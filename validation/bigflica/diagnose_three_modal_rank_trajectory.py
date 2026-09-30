"""Track FLICA pruning on real D200/R100 dictionaries without subject IDs.

PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python3 \
    validation/bigflica/diagnose_three_modal_rank_trajectory.py \
    --dictionary-dir PRIVATE_DIR --output PRIVATE_OR_SUMMARY_JSON
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from fnit.bigflica import flica_vb


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    names = ("vbm", "fa", "md")
    paths = [arguments.dictionary_dir / f"{name}_dictionary.npy" for name in names]
    matrices = [np.load(path) for path in paths]
    report = {
        "input": "2050 real subjects, full-mask VBM/FA/MD; CPU sklearn D200/R100 dictionaries",
        "dictionary_sha256": {name: _sha256(path) for name, path in zip(names, paths)},
        "mode": "FLICA lambda_dims=R, C=20, auto_eigenspectrum DD, PCA init",
        "fits": [],
    }
    for updates in (1, 10, 20, 30, 50, 101):
        data = [matrix.copy() for matrix in matrices]
        options = {"num_components": 20, "maxits": updates - 1,
                   "dof_per_voxel": "auto_eigenspectrum", "lambda_dims": "R",
                   "initH": "PCA", "computeF": 0, "output_dir": "/tmp"}
        with contextlib.redirect_stdout(io.StringIO()):
            priors, posteriors, constants = flica_vb.flica_init_params(data, options)
            fitted = flica_vb.flica_iterate(data, options, priors, posteriors, constants)
        h = np.asarray(fitted["H"], dtype=np.float64)
        singular = np.linalg.svd(h, compute_uv=False)
        ratios = {}
        for name, matrix, x, w in zip(names, data, fitted["X"], fitted["W"]):
            reconstruction = (np.asarray(x) * np.asarray(w).reshape(-1)) @ h
            ratios[name] = float(np.linalg.norm(reconstruction) / np.linalg.norm(matrix))
        report["fits"].append({
            "actual_updates": updates,
            "effective_rank_relative_1e-6": int(np.sum(singular > singular[0] * 1e-6)),
            "relative_h_singular_values": (singular / singular[0]).tolist(),
            "reconstruction_norm_ratio": ratios,
        })
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
