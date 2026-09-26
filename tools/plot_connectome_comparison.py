"""Plot four measured reference/PyTorch connectomes and their JSON metrics.

Requires the report produced by compare_connectome_matrices.py for these exact
CSV bytes. Writes connectome_comparison.png and connectome_metrics.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_connectome_matrices import NAMES, _load


LABELS = {
    "count": "Streamline count",
    "sift2_fbc": "SIFT2 FBC",
    "mean_length": "Mean length (mm)",
    "mean_fa": "Mean FA",
}


def plot_comparison(reference_dir: Path, candidate_dir: Path,
                    report_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Plot exact CSVs named in a prior quantitative comparison report."""
    reference_dir = Path(reference_dir).resolve()
    candidate_dir = Path(candidate_dir).resolve()
    report = json.loads(Path(report_path).read_text())
    reference, reference_hashes = _load(reference_dir)
    candidate, candidate_hashes = _load(candidate_dir)
    if (Path(report["reference"]["directory"]).resolve() != reference_dir
            or report["reference"]["sha256"] != reference_hashes):
        raise ValueError("reference CSVs do not match the quantitative report")
    matching = [entry for entry in report["candidates"]
                if Path(entry["directory"]).resolve() == candidate_dir]
    if len(matching) != 1 or matching[0]["sha256"] != candidate_hashes:
        raise ValueError("candidate CSVs do not match the quantitative report")
    metrics = matching[0]["metrics"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(14, 15), layout="constrained")
    for row, name in enumerate(NAMES):
        original, current = reference[name], candidate[name]
        if original.shape != current.shape:
            raise ValueError(f"{name}: reference and candidate shapes differ")
        difference = current - original
        if name in ("count", "sift2_fbc"):
            original_display = np.log1p(original)
            current_display = np.log1p(current)
            difference_display = np.sign(difference) * np.log1p(np.abs(difference))
            scale_label = "log1p scale"
        else:
            original_display, current_display, difference_display = original, current, difference
            scale_label = "raw scale"
        low = min(0.0, float(original_display.min()), float(current_display.min()))
        high = max(float(original_display.max()), float(current_display.max()))
        if high <= low:
            high = low + 1.0
        limit = max(float(np.abs(difference_display).max()), 1e-12)
        original_image = axes[row, 0].imshow(original_display, cmap="viridis", vmin=low, vmax=high,
                                              interpolation="nearest")
        axes[row, 1].imshow(current_display, cmap="viridis", vmin=low, vmax=high,
                            interpolation="nearest")
        difference_image = axes[row, 2].imshow(difference_display, cmap="RdBu_r", vmin=-limit,
                                                vmax=limit, interpolation="nearest")
        axes[row, 0].set_ylabel(f"{LABELS[name]}\nRegion", fontsize=10)
        for col, title in enumerate(("MRtrix reference", "PyTorch", "PyTorch − reference")):
            axes[row, col].set_title(f"{LABELS[name]}: {title} ({scale_label})", fontsize=10)
            axes[row, col].set_xlabel("Region")
        fig.colorbar(original_image, ax=axes[row, :2], shrink=.72, pad=.01)
        fig.colorbar(difference_image, ax=axes[row, 2], shrink=.72, pad=.01)
    fig.suptitle("Region–region connectomes: same scale within each matrix pair", fontsize=15)
    comparison_path = output_dir / "connectome_comparison.png"
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)

    labels = [LABELS[name] for name in NAMES]
    positions = np.arange(len(NAMES))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
    correlation_series = (("Pearson", "pearson", "#277DA1"),
                          ("Spearman", "spearman", "#4D908E"),
                          ("Support Dice", "dice", "#F9C74F"))
    for offset, (label, key, color) in zip((-.24, 0., .24), correlation_series):
        values = [metrics[name]["nonzero_support"]["dice"] if key == "dice"
                  else metrics[name]["values"][key] for name in NAMES]
        axes[0].barh(positions + offset, [0 if value is None else value for value in values],
                     height=.21, color=color, label=label)
        for row, value in enumerate(values):
            if value is None:
                axes[0].text(.02, positions[row] + offset, "undefined", va="center", fontsize=7)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-1.05, 1.05)
    axes[0].axvline(0, color="black", linewidth=.6)
    axes[0].set_xlabel("Correlation / nonzero-edge Dice")
    axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3, fontsize=8)

    error_series = (("Normalized MAE", "normalized_mae", "#F3722C"),
                    ("Normalized RMSE", "normalized_rmse", "#B5174F"))
    finite_errors = []
    for offset, (label, key, color) in zip((-.15, .15), error_series):
        values = [metrics[name]["values"][key] for name in NAMES]
        finite_errors.extend(value for value in values if value is not None)
        axes[1].barh(positions + offset, [0 if value is None else value for value in values],
                     height=.27, color=color, label=label)
        for row, value in enumerate(values):
            if value is None:
                axes[1].text(.02, positions[row] + offset, "undefined", va="center", fontsize=7)
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(0.01, max(finite_errors, default=0.0) * 1.2))
    axes[1].set_xlabel("Error / mean nonzero reference value")
    axes[1].legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, fontsize=8)
    fig.suptitle("Verified report: upper triangle; length/FA values on shared count edges", fontsize=14)
    metrics_path = output_dir / "connectome_metrics.png"
    fig.savefig(metrics_path, dpi=180)
    plt.close(fig)
    return comparison_path, metrics_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    for path in plot_comparison(args.reference_dir, args.candidate_dir,
                                args.report, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
