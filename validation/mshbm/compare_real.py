#!/usr/bin/env python3
"""Build the public real-data MS-HBM comparison report and surface figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import loadmat

from fnit.mshbm import load_assets
from fnit.mshbm.core import _unit_columns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _time_record(path: Path) -> dict[str, float | int]:
    text = path.read_text(errors="replace")
    wall_match = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)",
        text,
    )
    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if wall_match is None or rss_match is None:
        raise ValueError(f"Incomplete /usr/bin/time -v log: {path}")
    fields = [float(value) for value in wall_match.group(1).strip().split(":")]
    wall = sum(value * 60**power for power, value in enumerate(reversed(fields)))
    return {"wall_seconds": wall, "maximum_rss_kbytes": int(rss_match.group(1))}


def _candidate_binary(series: np.ndarray, seed_positions: np.ndarray) -> np.ndarray:
    target = _unit_columns(series)
    correlation = np.asarray(target.T @ target[:, seed_positions], dtype=np.float32)
    rank = int(np.floor(64984 * len(seed_positions) * 0.1 + 0.5))
    cutoff = np.partition(correlation.ravel(), correlation.size - rank)[correlation.size - rank]
    return correlation >= cutoff


def _profile_metrics(timeseries: Path, profile_paths: list[Path], assets: dict) -> list[dict]:
    series = np.asarray(np.load(timeseries, mmap_mode="r", allow_pickle=False), dtype=np.float32)
    if series.shape[1] == 64984:
        series = series[:, assets["cortex_mask"]]
    if series.shape[1] != 59412:
        raise ValueError("Expected time x 59412 or time x 64984 input")
    positions = np.searchsorted(np.flatnonzero(assets["cortex_mask"]), assets["seed_vertices"])
    half = len(series) // 2
    parts = (series[:half], series[half:])
    if len(profile_paths) != 2:
        raise ValueError("This paired benchmark requires two CBIG pseudo-session profiles")
    records = []
    for index, (part, profile_path) in enumerate(zip(parts, profile_paths, strict=True), start=1):
        candidate = _candidate_binary(part, positions)
        reference = np.asarray(loadmat(profile_path)["profile_mat"])[assets["cortex_mask"]].astype(bool)
        different = int(np.count_nonzero(candidate != reference))
        records.append({
            "session": index,
            "frames": int(len(part)),
            "values": int(reference.size),
            "different_values": different,
            "agreement": float(1 - different / reference.size),
        })
    return records


def _load_fnit_labels(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return np.asarray(archive["labels"]).ravel()
    return np.asarray(np.load(path, allow_pickle=False)).ravel()


def _reference_labels(path: Path) -> np.ndarray:
    content = loadmat(path)
    return np.concatenate((content["lh_labels"].ravel(), content["rh_labels"].ravel()))


def _network_dice(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    result = {}
    for label in range(1, 18):
        left = reference == label
        right = candidate == label
        denominator = int(left.sum() + right.sum())
        result[str(label)] = float(2 * np.count_nonzero(left & right) / denominator)
    return result


def _surface_figure(reference: np.ndarray, candidate: np.ndarray,
                    left_path: Path, right_path: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    colors = np.vstack(([0.75, 0.75, 0.75, 1.0], plt.get_cmap("tab20")(np.arange(17))))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 18.5), cmap.N)
    surfaces = []
    for path in (left_path, right_path):
        image = nib.load(str(path))
        surfaces.append((np.asarray(image.darrays[0].data),
                         np.asarray(image.darrays[1].data, dtype=np.int64)))

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    rows = (("CBIG MATLAB", reference), ("FNIT", candidate))
    for row, (name, labels) in enumerate(rows):
        for hemi, (vertices, faces) in enumerate(surfaces):
            ax = fig.add_subplot(2, 2, row * 2 + hemi + 1, projection="3d")
            hemi_labels = labels[:32492] if hemi == 0 else labels[32492:]
            face_labels = np.rint(np.median(hemi_labels[faces], axis=1)).astype(np.int16)
            collection = Poly3DCollection(vertices[faces], linewidths=0, antialiased=False)
            collection.set_facecolor(cmap(norm(face_labels)))
            ax.add_collection3d(collection)
            lower, upper = vertices.min(axis=0), vertices.max(axis=0)
            center = (lower + upper) / 2
            radius = float((upper - lower).max() / 2)
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=0, azim=180 if hemi == 0 else 0)
            ax.set_axis_off()
            ax.set_title(f"{name} · {'left hemisphere' if hemi == 0 else 'right hemisphere'}")
    fig.suptitle("MSC02 real five-minute resting-state data: 17-network labels", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeseries", type=Path, required=True)
    parser.add_argument("--fnit-labels", type=Path, required=True)
    parser.add_argument("--cbig-labels", type=Path, required=True)
    parser.add_argument("--cbig-profile", type=Path, action="append", required=True)
    parser.add_argument("--fnit-full-time", type=Path, required=True)
    parser.add_argument("--fnit-matched-time", type=Path, required=True)
    parser.add_argument("--cbig-matched-time", type=Path, required=True)
    parser.add_argument("--left-surface", type=Path, required=True)
    parser.add_argument("--right-surface", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()

    assets = load_assets()
    candidate = _load_fnit_labels(args.fnit_labels).astype(np.uint8)
    reference = _reference_labels(args.cbig_labels).astype(np.uint8)
    if candidate.shape != (64984,) or reference.shape != (64984,):
        raise ValueError("Expected 64,984 fsLR32k labels")
    different = int(np.count_nonzero(candidate != reference))
    cortex = assets["cortex_mask"]
    dice = _network_dice(reference, candidate)
    report = {
        "schema_version": 1,
        "date": "2026-09-27",
        "feature": "FNIT MS-HBM versus CBIG Kong2019 MS-HBM",
        "data": {
            "subjects": 1,
            "case": "MSC02",
            "kind": "real 5-minute resting-state fMRI, fsLR32k cortex",
            "frames": int(np.load(args.timeseries, mmap_mode="r").shape[0]),
            "timeseries_shape": list(np.load(args.timeseries, mmap_mode="r").shape),
            "input_sha256": _sha256(args.timeseries),
            "input_in_repository": False,
        },
        "reference": {
            "software": "CBIG Kong2019 MS-HBM under MATLAB R2018b",
            "commit": "b69b822a15e2a94f1e439606552fc44b6858cf3c",
            "mesh": "fs_LR_32k",
            "sessions": 2,
            "networks": 17,
            "w": 200,
            "c": 50,
        },
        "output_contract": {
            "labels_shape": [64984],
            "left_vertices": 32492,
            "right_vertices": 32492,
            "cortical_vertices": int(cortex.sum()),
            "medial_wall_label": 0,
            "network_labels": "1..17 in CBIG HCP_40 order",
        },
        "profiles": _profile_metrics(args.timeseries, args.cbig_profile, assets),
        "labels": {
            "all_vertex_agreement": float(1 - different / 64984),
            "cortex_agreement": float(np.mean(candidate[cortex] == reference[cortex])),
            "different_all_vertices": different,
            "different_cortical_vertices": int(np.count_nonzero(candidate[cortex] != reference[cortex])),
            "per_network_dice": dice,
            "minimum_network_dice": float(min(dice.values())),
        },
        "timing": {
            "hardware": "headcw, Intel Xeon Gold 6418H, 8 BLAS/MATLAB threads",
            "fnit_raw_timeseries_to_labels": _time_record(args.fnit_full_time),
            "fnit_saved_profiles_to_labels": _time_record(args.fnit_matched_time),
            "cbig_saved_profiles_to_labels": _time_record(args.cbig_matched_time),
            "comparison_note": "The matched timing starts from the same two saved binary profiles and ends after labels; each process includes interpreter startup and file I/O.",
        },
        "precision": {
            "fnit": "NumPy/SciPy CPU; float32 profiles and float64 inference state",
            "gpu_used": False,
            "float16_used": False,
        },
        "source_sha256": {
            str(path.relative_to(args.source_root)): _sha256(path)
            for path in (
                args.source_root / "src/fnit/mshbm/core.py",
                args.source_root / "src/fnit/mshbm/cli.py",
                args.source_root / "src/fnit/mshbm/assets/hcp40_fslr32k_17.npz",
                args.source_root / "validation/mshbm/run_fnit_from_cbig_profiles.py",
                args.source_root / "validation/mshbm/compare_real.py",
            )
        },
        "figure": "docs/mshbm/figures/mshbm_cbig_comparison.png",
        "scope": "One real subject and one five-minute acquisition; exact agreement here is not a multi-cohort accuracy claim.",
    }
    _surface_figure(reference, candidate, args.left_surface, args.right_surface,
                    args.output_figure)
    report["figure_sha256"] = _sha256(args.output_figure)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
