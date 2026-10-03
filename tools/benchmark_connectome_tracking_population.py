"""读取已有真实 TCK，比较接受率、长度、端点和采样点访问直方图。"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import ks_2samp

try:
    from .connectome_repeat_common import (
        envelope, fnit_repeat_envelope, overall_status, pairwise, seed_labels, sha256,
    )
except ImportError:
    from connectome_repeat_common import (
        envelope, fnit_repeat_envelope, overall_status, pairwise, seed_labels, sha256,
    )

_hash = sha256  # historical helper used by external analysis scripts


def _read(path, affine, shape):
    """保留既有定义：保存点逐个舍入到公共网格；不是 tckmap 的密度定义。"""
    tractogram = nib.streamlines.load(str(path), lazy_load=False)
    tracks = list(tractogram.streamlines)
    if any(len(track) < 2 or not np.isfinite(track).all() for track in tracks):
        raise ValueError(f"{path}: tracks require at least two finite world-coordinate points")
    lengths = np.asarray([np.linalg.norm(np.diff(track, axis=0), axis=1).sum()
                          for track in tracks])
    endpoints = (np.concatenate([np.stack((track[0], track[-1])) for track in tracks])
                 if tracks else np.empty((0, 3)))
    points = np.concatenate(tracks) if tracks else np.empty((0, 3))
    voxel = nib.affines.apply_affine(np.linalg.inv(affine), points)
    index = np.rint(voxel).astype(np.int32)
    inside = ((index >= 0) & (index < np.asarray(shape))).all(-1)
    tdi = np.zeros(shape, dtype=np.int32)
    np.add.at(tdi, tuple(index[inside].T), 1)
    return lengths, endpoints, tdi


def _correlation(left, right):
    support = (left + right) > 0
    a, b = left[support], right[support]
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else None


def _coarse_tdi(tdi):
    if any(size % 4 for size in tdi.shape):
        raise ValueError("FOD grid must divide into four-voxel spatial blocks")
    x, y, z = (size // 4 for size in tdi.shape)
    return tdi.reshape(x, 4, y, 4, z, 4).sum(axis=(1, 3, 5)).ravel()


def compare(official_paths, fnit_paths, grid, *, dataset="unspecified", n_seeds=100000,
            official_seeds=None, fnit_seeds=None, return_data=False):
    """至少三份官方和一份 FNIT；只比较既有轨迹，不调用追踪程序。"""
    if len(official_paths) < 3 or not fnit_paths or n_seeds < 1:
        raise ValueError("at least three official repeats, one FNIT run and positive n_seeds are required")
    paths = [Path(path) for path in (*official_paths, *fnit_paths)]
    if len(set(path.resolve() for path in paths)) != len(paths):
        raise ValueError("duplicate resolved TCK paths cannot count as independent repeats")
    official_seeds = seed_labels(official_seeds, len(official_paths), "official")
    fnit_seeds = seed_labels(fnit_seeds, len(fnit_paths), "fnit")
    image = nib.load(str(grid))
    shape = image.shape[:3]
    if any(size % 4 for size in shape):
        raise ValueError("FOD grid must divide into four-voxel spatial blocks")
    official_names = [f"official_{i}" for i in range(len(official_paths))]
    fnit_names = [f"fnit_{i}" for i in range(len(fnit_paths))]
    names = official_names + fnit_names
    data = {name: _read(path, image.affine, shape) for name, path in zip(names, paths)}
    if any(len(item[0]) > n_seeds for item in data.values()):
        raise ValueError("accepted streamline count exceeds declared attempted seeds")
    all_endpoints = np.concatenate([item[1] for item in data.values()])
    bins = ([np.arange(all_endpoints[:, axis].min() - 8,
                       all_endpoints[:, axis].max() + 16, 8.) for axis in range(3)]
            if len(all_endpoints) else None)
    histograms = {name: np.histogramdd(item[1], bins=bins)[0].ravel()
                  if bins is not None else np.zeros(1) for name, item in data.items()}
    coarse = {name: _coarse_tdi(item[2]) for name, item in data.items()}

    def metrics(left, right):
        first, second = data[left], data[right]
        return {
            "accepted_fraction_absolute_difference": abs(len(first[0]) - len(second[0])) / n_seeds,
            "length_ks": (float(ks_2samp(first[0], second[0]).statistic)
                          if len(first[0]) and len(second[0]) else None),
            "endpoint_8mm_histogram_pearson": _correlation(histograms[left], histograms[right]),
            "tdi_native_voxel_pearson": _correlation(first[2].ravel(), second[2].ravel()),
            "tdi_four_voxel_block_pearson": _correlation(coarse[left], coarse[right]),
        }

    grouped = pairwise(official_names, fnit_names, metrics)
    fields = tuple(metrics(names[0], names[1]))
    ranges = {field: envelope([item["metrics"][field] for item in grouped["official"]],
                             [item["metrics"][field] for item in grouped["cross"]],
                             similarity=field.endswith("pearson")) for field in fields}
    self_ranges = {field: fnit_repeat_envelope(
        [item["metrics"][field] for item in grouped["official"]],
        [item["metrics"][field] for item in grouped["fnit"]],
        similarity=field.endswith("pearson")) for field in fields}
    comparisons = [*itertools.combinations(official_names, 2),
                   *itertools.product(official_names, fnit_names),
                   *itertools.combinations(fnit_names, 2)]
    report = {
        "dataset": dataset, "n_seed_attempts": n_seeds,
        "design": f"{len(official_paths)} supplied official repeats and {len(fnit_paths)} FNIT runs; existing TCK only",
        "official_seeds": official_seeds, "fnit_seeds": fnit_seeds,
        "input_paths": dict(zip(names, (str(path.resolve()) for path in paths))),
        "input_sha256": {name: sha256(path) for name, path in zip(names, paths)},
        "grid_path": str(Path(grid).resolve()), "grid_sha256": sha256(grid),
        "grid_shape": list(shape), "grid_affine": image.affine.tolist(),
        "four_voxel_block_axes_mm": (4 * nib.affines.voxel_sizes(image.affine)).tolist(),
        "endpoint_bin_edges_mm": [value.tolist() for value in bins] if bins is not None else None,
        "metric_policy": {
            "acceptance": "errors <= maximum observed official-pair error; similarities >= minimum observed official-pair similarity",
            "sampling": "finite observed repeat range, not a population confidence interval; no post-hoc tolerance",
            "seed_attempts": "caller supplies attempted seed count; TCK total_count is not the denominator",
            "length_ks": "two-sample KS on stored polyline lengths; scipy.stats.ks_2samp default method",
            "endpoint_histogram": "all first/last world-coordinate points; common 8 mm bins from all runs",
            "tdi": "stored point-visit counts, np.rint world-to-common-grid; Pearson on union nonzero support; not MRtrix tckmap",
            "coarse_tdi": "sum nonoverlapping 4 x 4 x 4 voxel blocks; physical size comes from supplied grid",
            "undefined": "null and not_assessed; never zero or a pass",
            "scope": "descriptive tractogram population comparison, not a proof of identical RNG or full pipeline parity",
        },
        "track_counts": {name: len(item[0]) for name, item in data.items()},
        "accepted_fractions": {name: len(item[0]) / n_seeds for name, item in data.items()},
        "length_quantiles_mm": {name: np.quantile(item[0], [0, .1, .25, .5, .75, .9, 1]).tolist()
                                if len(item[0]) else None for name, item in data.items()},
        "pairs": {f"{left}_vs_{right}": metrics(left, right) for left, right in comparisons},
        "pairwise": grouped, "comparison_counts": {name: len(value) for name, value in grouped.items()},
        "ranges": ranges, "population_envelope_status": overall_status(list(ranges.values())),
        "fnit_reproducibility_ranges": self_ranges,
        "fnit_reproducibility_status": overall_status(list(self_ranges.values())),
    }
    return (report, data) if return_data else report


def figure(path, report, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(data)
    fnit_name = next(name for name in names if name.startswith("fnit_"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for name in names:
        axes[0, 0].hist(data[name][0], bins=np.arange(0, 255, 5), density=True,
                        histtype="step", linewidth=1.5, label=name)
    axes[0, 0].set(xlabel="Length (mm)", ylabel="Density", title="Accepted streamline lengths")
    axes[0, 0].legend(frameon=False)
    z = report["grid_shape"][2] // 2
    official_tdi = data["official_0"][2][:, :, max(0, z - 2):z + 3].sum(-1).T
    fnit_tdi = data[fnit_name][2][:, :, max(0, z - 2):z + 3].sum(-1).T
    vmax = np.quantile(np.concatenate((official_tdi.ravel(), fnit_tdi.ravel())), .995)
    for axis, matrix, title in ((axes[0, 1], official_tdi, "MRtrix point visits"),
                                (axes[1, 0], fnit_tdi, "FNIT point visits")):
        axis.imshow(matrix, origin="lower", vmin=0, vmax=vmax, cmap="magma")
        axis.set(title=title, xlabel="FOD voxel x", ylabel="FOD voxel y")
    difference = fnit_tdi - official_tdi
    limit = np.quantile(np.abs(difference), .995)
    axes[1, 1].imshow(difference, origin="lower", vmin=-limit, vmax=limit, cmap="coolwarm")
    axes[1, 1].set(title="FNIT minus MRtrix point visits", xlabel="FOD voxel x", ylabel="FOD voxel y")
    fig.suptitle(f"{report['dataset']} · {report['n_seed_attempts']:,} attempted seeds")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, nargs="+", required=True, help="至少三份官方真实 TCK")
    parser.add_argument("--fnit", type=Path, nargs="+", required=True, help="一份或多份 FNIT 真实 TCK")
    parser.add_argument("--fnit-repeat", type=Path, nargs="+", help="兼容旧调用：追加 FNIT TCK")
    parser.add_argument("--grid", type=Path, required=True, help="公共 FOD NIfTI 网格")
    parser.add_argument("--dataset", default="unspecified (provide --dataset)")
    parser.add_argument("--n-seeds", type=int, default=100000, help="每轮实际尝试的种子数")
    parser.add_argument("--official-seeds", type=int, nargs="+")
    parser.add_argument("--fnit-seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True, help="指标 JSON")
    parser.add_argument("--figure", type=Path, help="可选：长度和真实 point-visit 脑图 PNG")
    args = parser.parse_args(argv)
    report, data = compare(args.official, [*args.fnit, *(args.fnit_repeat or [])], args.grid,
                           dataset=args.dataset, n_seeds=args.n_seeds,
                           official_seeds=args.official_seeds, fnit_seeds=args.fnit_seeds,
                           return_data=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.figure:
        figure(args.figure, report, data)
    print(json.dumps({"population_envelope_status": report["population_envelope_status"],
                      "fnit_reproducibility_status": report["fnit_reproducibility_status"],
                      "track_counts": report["track_counts"]}, allow_nan=False))


if __name__ == "__main__":
    main()
