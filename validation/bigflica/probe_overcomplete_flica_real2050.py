"""Bounded C28/R-mode stability check on existing real three-modality dictionaries.

Usage::

    python probe_overcomplete_flica_real2050.py DICTIONARY_DIR OUTPUT_DIR \
        LABEL REPORT_JSON MAX_ITER [MAX_ITER ...]

The private dictionary and output paths stay on the validation server.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from fnit.bigflica.pipeline import _fit_flica


def main() -> None:
    if len(sys.argv) < 6:
        raise SystemExit(__doc__)
    dictionaries_dir, output, label, report_file = (
        Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
    iterations = [int(value) for value in sys.argv[5:]]
    names = ("vbm", "fa", "md")
    files = {name: dictionaries_dir / f"{name}_dictionary.npy" for name in names}
    dictionaries = {name: np.load(path) for name, path in files.items()}
    report = {
        "dataset": "2050 real subjects; full-mask VBM/FA/MD; task excluded",
        "dictionary_source": label,
        "dictionary_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in files.items()},
        "requested_components": 28, "flica_lambda_dims": "R", "runs": [],
        "scope": "FLICA-only overcomplete stability, not official C20 equivalence",
    }
    for max_iter in iterations:
        directory = output / f"maxits_{max_iter}"
        start = time.perf_counter()
        try:
            _fit_flica(dictionaries, 28, max_iter, directory, "cpu", "R")
            status = "accepted"
        except ValueError as error:
            if "FLICA collapsed or pruned requested components" not in str(error):
                raise
            status = "rejected_rank_gate"
        reconstruction = json.loads(
            (directory / "flica_reconstruction.json").read_text(encoding="utf-8"))
        norms = np.asarray(reconstruction["component_row_norms"], dtype=np.float64)
        relative = norms / norms.max()
        kept = np.flatnonzero(relative > 1e-6).tolist()
        report["runs"].append({
            "flica_max_iter": max_iter, "actual_updates": max_iter + 1,
            "status": status, "fit_wall_s": time.perf_counter() - start,
            "effective_rank": reconstruction["component_rank"],
            "near_zero_component_count": int(np.count_nonzero(relative <= 1e-6)),
            "relative_component_norms_sorted": np.sort(relative).tolist(),
            "retained_indices_zero_based": kept,
            "retained_indices_sha256": hashlib.sha256(
                json.dumps(kept).encode()).hexdigest(),
            "reconstruction_norm_ratio": reconstruction["per_modality_ratio"],
            "overall_reconstruction_norm_ratio": reconstruction["overall_ratio"],
        })
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
