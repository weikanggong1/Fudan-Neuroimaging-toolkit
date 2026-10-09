"""Plot actual native/reference tracking bound to this run's private report.

CPU interpolation is for the displayed brain background only. It never changes
tracks, FA, atlases or matrices. The output JSON contains an explicit whitelist;
no private paths, commands, accounts, process IDs or individual points are copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_path(path, expected):
    result = Path(path).resolve(strict=True)
    if not isinstance(expected, str) or len(expected) != 64 or sha256(result) != expected:
        raise ValueError("actual file does not match the measured report SHA-256")
    return result


def load_tracks(path, *, seed, limit):
    loaded = nib.streamlines.load(str(path), lazy_load=False)
    tracks = loaded.streamlines
    indices = np.sort(np.random.default_rng(seed).choice(len(tracks), min(limit, len(tracks)), replace=False))
    selected = [np.asarray(tracks[int(index)], dtype=np.float32) for index in indices]
    return selected, dict(actual_track_count=len(tracks), displayed_track_count=len(selected),
                          header_total_count=str(loaded.header.get("total_count")))


def matrix_summary(report):
    result = {}
    for atlas, metrics in report["fixed_tck_reference"]["atlas_matrices"].items():
        result[atlas] = {}
        for name, values in metrics.items():
            result[atlas][name] = {key: values[key] for key in
                                  ("matrix_shape", "pearson", "relative_l1", "max_abs", "support_dice")}
            result[atlas][name]["full_matrix_unequal_values"] = values["full_matrix"]["unequal_values"]
            result[atlas][name]["nonfinite_mismatch"] = values["full_matrix"]["nonfinite_mismatch"]
    return result


def execute(arguments):
    report_bytes = arguments.report.read_bytes()
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    if arguments.expected_report_sha256 is not None and report_sha != arguments.expected_report_sha256:
        raise ValueError("private report differs from the supplied SHA-256")
    report = json.loads(report_bytes)
    if report["status"] != "completed":
        raise ValueError("QC requires the completed actual full-call and reference report")
    exported = report["exported"]
    candidate = verified_path(exported["tracks"]["path"], exported["tracks"]["sha256"])
    five_record = exported["images"]["five_tissue"]
    five = verified_path(five_record["path"], five_record["sha256"])
    stock_row = next(row for row in report["tracking_reference"]["threaded_independent_repeats"] if row["seed"] == 0)
    stock_command = report["tracking_reference"]["commands"]["seed_0"]
    if stock_command["returncode"] != 0:
        raise ValueError("stock reference seed 0 command was not completed")
    stock = verified_path(stock_command["command"][2], stock_row["reference_tck_sha256"])
    images = nib.load(str(five))
    tissue = np.asarray(images.dataobj, dtype=np.float32)[..., :3].sum(axis=-1)
    background = resample_to_output(nib.Nifti2Image(tissue, images.affine),
                                    voxel_sizes=(2., 2., 2.), order=1)
    background_values = np.asarray(background.dataobj, dtype=np.float32)
    tracks, counts = [], []
    for path in (candidate, stock):
        selected, record = load_tracks(path, seed=arguments.plot_seed, limit=arguments.max_tracks)
        tracks.append(selected)
        counts.append(record)
    if counts[0]["actual_track_count"] != report["accepted_streamlines"]:
        raise ValueError("candidate record count differs from the real pipeline report")
    if counts[1]["actual_track_count"] != stock_row["accepted_streamlines"]:
        raise ValueError("stock record count differs from the real reference report")
    figures, axes = plt.subplots(2, 3, figsize=(12, 9), facecolor="white")
    views = ((0, 1, 2, "Axial", "X / R", "Y / A"),
             (0, 2, 1, "Coronal", "X / R", "Z / S"),
             (1, 2, 0, "Sagittal", "Y / A", "Z / S"))
    for row in range(2):
        for column, (x, y, hidden, title, xlabel, ylabel) in enumerate(views):
            axis = axes[row, column]
            origin, spacing = background.affine[:3, 3], np.diag(background.affine)[:3]
            extent = [origin[x] - spacing[x] / 2, origin[x] + (background.shape[x] - .5) * spacing[x],
                      origin[y] - spacing[y] / 2, origin[y] + (background.shape[y] - .5) * spacing[y]]
            projection = background_values.max(axis=hidden).T
            axis.imshow(projection, cmap="gray", vmin=0, vmax=1, origin="lower", extent=extent)
            axis.add_collection(LineCollection([path[:, [x, y]] for path in tracks[row]],
                                               colors="#4ce0d2", linewidths=.25, alpha=.17))
            axis.set(xlim=extent[:2], ylim=extent[2:], aspect="equal", xlabel=xlabel + " (mm)",
                     ylabel=ylabel + " (mm)", title=title if row == 0 else "")
            axis.tick_params(labelsize=7)
        axes[row, 0].text(-.32, .5, ("FNIT pinned tckgen" if row == 0 else "Stock MRtrix tckgen")
                         + f"\n{counts[row]['actual_track_count']:,} accepted", transform=axes[row, 0].transAxes,
                         rotation=90, va="center", ha="center", fontsize=10)
    matrices = matrix_summary(report)
    differences = "; ".join(f"{name}: count unequal={values['count']['full_matrix_unequal_values']}, "
                            f"SIFT2 r={values['sift2_fbc']['pearson']}" for name, values in matrices.items())
    figures.suptitle(arguments.dataset_label + " — real corrected DWI / supplied recon-all", fontsize=12)
    figures.text(.5, .045, "Same iFOD2/ACT algorithm; multi-thread scheduling yields independent track populations.\n"
                 "Tracks are not paired; 2 mm background interpolation is display only.\n" + differences,
                 ha="center", va="center", fontsize=8)
    figures.subplots_adjust(left=.12, right=.98, top=.90, bottom=.11, wspace=.34, hspace=.24)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    image = arguments.output_dir / "native_tracking_qc.png"
    if image.exists() or (arguments.output_dir / "native_tracking_qc.public.json").exists():
        raise FileExistsError("QC outputs already exist; choose a new output directory")
    figures.savefig(image, dpi=180)
    plt.close(figures)
    if sha256(arguments.report) != report_sha:
        raise RuntimeError("report changed while generating QC")
    for path, expected in ((candidate, exported["tracks"]["sha256"]),
                           (stock, stock_row["reference_tck_sha256"]), (five, five_record["sha256"])):
        verified_path(path, expected)
    public = dict(scope="actual end-to-end native tracking example; matrix comparisons use fixed candidate TCK",
                  dataset_label=arguments.dataset_label, dataset_cc0=arguments.dataset_cc0,
                  license_assertion="explicit argument; not inferred from filenames", license_source=arguments.license_source,
                  verified_report_sha256=report_sha, script_sha256=sha256(Path(__file__)), figure_sha256=sha256(image),
                  input_sha256=dict(candidate_tck=sha256(candidate), stock_tck=sha256(stock), five_tissue=sha256(five)),
                  counts=dict(candidate=counts[0], stock=counts[1]), matrix_summary=matrices,
                  plot_seed=arguments.plot_seed, maximum_tracks_per_row=arguments.max_tracks,
                  selection="independent uniform subsets without replacement; no trajectory pairing",
                  background_display_only=dict(voxel_size_mm=2., interpolation_order=1, channels="cGM+sGM+WM"),
                  matplotlib=matplotlib.__version__, nibabel=nib.__version__, numpy=np.__version__)
    (arguments.output_dir / "native_tracking_qc.public.json").write_text(json.dumps(public, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-report-sha256")
    parser.add_argument("--dataset-label", default="Real corrected-DWI case")
    parser.add_argument("--dataset-cc0", action="store_true")
    parser.add_argument("--license-source")
    parser.add_argument("--max-tracks", type=int, default=2000)
    parser.add_argument("--plot-seed", type=int, default=20261009)
    options = parser.parse_args()
    if options.max_tracks < 1 or options.plot_seed < 0:
        parser.error("max-tracks must be positive and plot-seed nonnegative")
    execute(options)
