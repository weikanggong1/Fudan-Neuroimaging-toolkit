"""Plot four real-data connectomes from a completed paired FNIT benchmark.

Input is the benchmark's candidate directory (report.json plus four CSVs)
and the exact MRtrix reference CSV directory. Output is a 4x3 PNG with a
matching JSON file containing input hashes. This displays data; it does not
change or rescale the quantitative benchmark metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NAMES = ("count", "sift2_fbc", "mean_length", "mean_fa")
UNITS = ("streamlines", "weight sum", "mm", "FA")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads((args.candidate_dir / "report.json").read_text())
    fig, axes = plt.subplots(4, 3, figsize=(12, 15), layout="constrained")
    sources = {}
    for row, (name, unit) in enumerate(zip(NAMES, UNITS)):
        candidate_path = args.candidate_dir / f"candidate_{name}.csv"
        reference_path = args.reference_dir / f"connectome_{name}.csv"
        if not reference_path.is_file():
            reference_path = args.reference_dir / f"{name}.csv"
        reference_hash = _sha(reference_path)
        if reference_hash != report["reference_sha256"][name]:
            raise ValueError(f"{name}: reference CSV differs from benchmark")
        candidate = np.loadtxt(candidate_path, delimiter=",")
        reference = np.loadtxt(reference_path, delimiter=",")
        if candidate.shape != reference.shape or candidate.ndim != 2:
            raise ValueError(f"{name}: matrix shapes differ")
        delta = candidate - reference
        if name in ("count", "sift2_fbc"):
            shown = (np.log1p(reference), np.log1p(candidate),
                     np.sign(delta) * np.log1p(np.abs(delta)))
            scale_note = "log1p"
        else:
            shown = (reference, candidate, delta)
            scale_note = "linear"
        high = max(float(shown[0].max()), float(shown[1].max()), 1e-12)
        limit = max(float(np.abs(shown[2]).max()), 1e-12)
        for col, values in enumerate(shown):
            ax = axes[row, col]
            image = ax.imshow(values, origin="lower", interpolation="nearest",
                              cmap="RdBu_r" if col == 2 else "viridis",
                              vmin=-limit if col == 2 else 0,
                              vmax=limit if col == 2 else high)
            ax.set_title(("MRtrix", "FNIT", "FNIT − MRtrix")[col])
            ax.set_xlabel("region index")
            ax.set_ylabel(f"{name} ({unit}; {scale_note})" if col == 0 else "region index")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        sources[name] = {"candidate_sha256": _sha(candidate_path),
                         "reference_sha256": reference_hash,
                         "shape": list(reference.shape)}
    fig.suptitle(f"ds004666 paired T1/DWI, FNIT seed {report['seed']}, 10,000 attempts", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps({
        "dataset": report["dataset"], "seed": report["seed"],
        "report_sha256": _sha(args.candidate_dir / "report.json"),
        "matrices": sources,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
