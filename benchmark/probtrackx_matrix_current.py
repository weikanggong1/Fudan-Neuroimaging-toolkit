"""Compare FSL and FNIT sparse ProbtrackX matrices on matched real DWI.

FSL .dot files are 1-based row/column/value triples followed by a
nrows ncols 0 size row. Rows and columns are aligned by their saved voxel
coordinate tables before comparison. Optional per-edge TSV files and raw ROI
matrices in the JSON stay with the private benchmark run. Sanitize the JSON
before any public release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def _coordinates(path: Path) -> list[tuple[int, int, int]]:
    table = np.loadtxt(path, ndmin=2)
    if table.shape[1] < 3 or not np.isfinite(table[:, :3]).all():
        raise ValueError(f"invalid voxel coordinates: {path}")
    xyz = np.rint(table[:, :3]).astype(np.int64)
    if not np.allclose(table[:, :3], xyz, atol=1e-4):
        raise ValueError(f"only volume voxel coordinates are supported: {path}")
    coordinates = [tuple(map(int, row)) for row in xyz]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError(f"duplicate voxel coordinates: {path}")
    return coordinates


def _dot(path: Path) -> tuple[tuple[int, int], dict[tuple[int, int], float]]:
    entries = {}
    previous = None
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"expected row col value in {path}:{number}")
            row, col = int(fields[0]), int(fields[1])
            value = float(fields[2])
            if row < 1 or col < 1 or not np.isfinite(value):
                raise ValueError(f"invalid sparse entry in {path}:{number}")
            if previous is not None:
                key = previous[:2]
                if key in entries:
                    raise ValueError(f"duplicate sparse entry in {path}: {key}")
                if previous[2] != 0:
                    entries[key] = previous[2]
            previous = (row, col, value)
    if previous is None or previous[2] != 0:
        raise ValueError(f"missing FSL .dot size row in {path}")
    shape = (previous[0], previous[1])
    if any(row > shape[0] or col > shape[1] for row, col in entries):
        raise ValueError(f"sparse entry outside declared shape in {path}")
    return shape, entries


def _coordinate_files(directory: Path, number: int) -> tuple[Path, Path]:
    first = directory / f"coords_for_fdt_matrix{number}"
    if number == 2:
        second = directory / "tract_space_coords_for_fdt_matrix2"
    elif number == 3 and (directory / "tract_space_coords_for_fdt_matrix3").exists():
        second = directory / "tract_space_coords_for_fdt_matrix3"
    else:
        second = first
    return first, second


def _mapped_edges(directory: Path, number: int):
    shape, entries = _dot(directory / f"fdt_matrix{number}.dot")
    row_path, col_path = _coordinate_files(directory, number)
    rows, cols = _coordinates(row_path), _coordinates(col_path)
    if shape != (len(rows), len(cols)):
        raise ValueError(f"matrix{number} shape {shape} disagrees with coordinate tables in {directory}")
    edges = {(rows[row - 1], cols[col - 1]): value
             for (row, col), value in entries.items()}
    return shape, rows, cols, edges


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or not a.std() or not b.std():
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _compare_matrix(fsl_dir: Path, fnit_dir: Path, number: int, detail_dir: Path | None):
    fsl_shape, fsl_rows, fsl_cols, fsl_edges = _mapped_edges(fsl_dir, number)
    fnit_shape, fnit_rows, fnit_cols, fnit_edges = _mapped_edges(fnit_dir, number)
    if set(fsl_rows) != set(fnit_rows) or set(fsl_cols) != set(fnit_cols):
        raise ValueError(f"matrix{number} FSL/FNIT coordinate sets differ")
    keys = sorted(fsl_edges.keys() | fnit_edges.keys())
    a = np.fromiter((fsl_edges.get(key, 0.0) for key in keys), dtype=np.float64)
    b = np.fromiter((fnit_edges.get(key, 0.0) for key in keys), dtype=np.float64)
    common = len(fsl_edges.keys() & fnit_edges.keys())
    if detail_dir is not None:
        detail_dir.mkdir(parents=True, exist_ok=True)
        with (detail_dir / f"matrix{number}_edge_differences.tsv").open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(("row_x", "row_y", "row_z", "col_x", "col_y", "col_z",
                             "fsl", "fnit", "fnit_minus_fsl"))
            for key, left, right in zip(keys, a, b):
                writer.writerow((*key[0], *key[1], f"{left:.8g}", f"{right:.8g}",
                                 f"{right - left:.8g}"))
    return {
        "shape": list(fsl_shape),
        "fsl_and_fnit_coordinate_order_identical": fsl_rows == fnit_rows and fsl_cols == fnit_cols,
        "fsl_nonzero_edges": len(fsl_edges),
        "fnit_nonzero_edges": len(fnit_edges),
        "common_nonzero_edges": common,
        "nonzero_support_dice": (2 * common / (len(fsl_edges) + len(fnit_edges))
                                 if fsl_edges or fnit_edges else 1.0),
        "pearson_on_nonzero_union": _pearson(a, b),
        "mean_abs_error_on_nonzero_union": float(np.abs(a - b).mean()) if len(keys) else 0.0,
        "max_abs_error": float(np.abs(a - b).max()) if len(keys) else 0.0,
        "fsl_edge_value_sum": float(a.sum()),
        "fnit_edge_value_sum": float(b.sum()),
    }


def _wall(path: Path) -> float | None:
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("wall_s="):
            return float(line.split("=", 1)[1].split()[0])
    raise ValueError(f"missing wall_s= in {path}")


def _network(fsl_dir: Path, fnit_dir: Path, roi_list: Path, nsamples: int):
    if nsamples < 1:
        raise ValueError("--nsamples must be positive")
    rois = [Path(line) if Path(line).is_absolute() else roi_list.parent / line
            for line in roi_list.read_text().splitlines() if line.strip()]
    launched = np.asarray([int(np.count_nonzero(np.asarray(nib.load(str(path)).dataobj)))
                           * nsamples for path in rois], dtype=np.float64)
    if not len(launched) or np.any(launched == 0):
        raise ValueError("ROI list must contain nonempty masks")
    matrices = {}
    for name, directory in (("fsl", fsl_dir), ("fnit", fnit_dir)):
        counts = np.loadtxt(directory / "fdt_network_matrix", ndmin=2)
        if counts.shape != (len(rois), len(rois)):
            raise ValueError(f"network matrix shape differs from ROI list: {directory}")
        directed = counts / launched[:, None]
        symmetric = (directed + directed.T) / 2
        np.fill_diagonal(symmetric, 0)
        matrices[name] = {"raw": counts, "directed_per_launched_particle": directed,
                          "symmetric_mean": symmetric}
        if name == "fnit":
            for filename, expected in (("fdt_network_matrix_probability", directed),
                                       ("fdt_network_matrix_symmetric", symmetric)):
                saved = np.loadtxt(directory / filename, ndmin=2)
                if saved.shape != expected.shape or not np.allclose(saved, expected, atol=1e-8):
                    raise ValueError(f"FNIT saved network normalization differs: {filename}")
    result = {"roi_seed_voxels": (launched / nsamples).astype(int).tolist(),
              "nsamples_per_seed_voxel": nsamples,
              "formula": "directed[i,j] = raw[i,j] / (ROI_seed_voxels[i] * nsamples); symmetric[i,j] = (directed[i,j] + directed[j,i]) / 2; diagonal = 0",
              "requires_unweighted_count_matrices": True}
    for key in ("raw", "directed_per_launched_particle", "symmetric_mean"):
        a, b = matrices["fsl"][key], matrices["fnit"][key]
        result[key] = {"fsl": a.tolist(), "fnit": b.tolist(),
                       "fnit_minus_fsl": (b - a).tolist(),
                       "mean_abs_error": float(np.abs(a - b).mean())}
    return result



def _plot_matrices(fsl_dir: Path, fnit_dir: Path, numbers: list[int], output: Path):
    """Save a compact real-data comparison without publishing subject images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(numbers), 3, figsize=(11, 3.4 * len(numbers)),
                               squeeze=False, constrained_layout=True)
    for row, number in enumerate(numbers):
        _, row_coords, col_coords, fsl_edges = _mapped_edges(fsl_dir, number)
        _, _, _, fnit_edges = _mapped_edges(fnit_dir, number)
        index_row = {coord: i for i, coord in enumerate(row_coords)}
        index_col = {coord: i for i, coord in enumerate(col_coords)}
        arrays = []
        for edges in (fsl_edges, fnit_edges):
            values = np.zeros((len(row_coords), len(col_coords)), dtype=np.float32)
            for (source, target), weight in edges.items():
                values[index_row[source], index_col[target]] = weight
            arrays.append(values)
        limit = max(float(array.max()) for array in arrays) or 1.0
        panels = ((arrays[0], "FSL", "viridis", 0, limit),
                  (arrays[1], "FNIT", "viridis", 0, limit),
                  (arrays[1] - arrays[0], "FNIT − FSL", "coolwarm",
                   -limit, limit))
        for col, (values, title, cmap, low, high) in enumerate(panels):
            axis = axes[row, col]
            image = axis.imshow(values, cmap=cmap, vmin=low, vmax=high,
                                interpolation="nearest", aspect="auto")
            axis.set_title(f"matrix{number}: {title}")
            axis.set_xlabel("column voxel")
            axis.set_ylabel("row voxel")
            figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsl-dir", type=Path, required=True)
    parser.add_argument("--fnit-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="directory containing current pipeline.py, _triton.py, cli.py, matrix_io.py")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path,
                        help="optional FSL/FNIT/difference heatmaps")
    parser.add_argument("--detail-dir", type=Path,
                        help="private directory for every raw sparse-edge difference")
    parser.add_argument("--fsl-time", type=Path)
    parser.add_argument("--fnit-time", type=Path)
    parser.add_argument("--network-fsl-dir", type=Path)
    parser.add_argument("--network-fnit-dir", type=Path)
    parser.add_argument("--roi-list", type=Path)
    parser.add_argument("--nsamples", type=int)
    args = parser.parse_args()
    report = {
        "reference": "FSL probtrackx2 sparse .dot files; 1-based row/column/value plus size row",
        "data": "matched real DWI posterior; source images and per-edge details remain private",
        "source_sha256": {name: hashlib.sha256((args.source_dir / name).read_bytes()).hexdigest()
                          for name in ("pipeline.py", "_triton.py", "cli.py", "matrix_io.py")},
        "wall_seconds_including_load_and_write": {
            "fsl": _wall(args.fsl_time or Path(str(args.fsl_dir) + ".time")),
            "fnit": _wall(args.fnit_time or Path(str(args.fnit_dir) + ".time")),
        },
        "matrices": {},
    }
    for number in (1, 2, 3):
        left = args.fsl_dir / f"fdt_matrix{number}.dot"
        right = args.fnit_dir / f"fdt_matrix{number}.dot"
        if left.exists() != right.exists():
            raise ValueError(f"matrix{number} exists on only one side")
        if left.exists():
            report["matrices"][f"matrix{number}"] = _compare_matrix(
                args.fsl_dir, args.fnit_dir, number, args.detail_dir)
    if not report["matrices"]:
        raise ValueError("no paired matrix1/2/3 .dot output found")
    network_args = (args.network_fsl_dir, args.network_fnit_dir,
                    args.roi_list, args.nsamples)
    if any(value is not None for value in network_args):
        if any(value is None for value in network_args):
            raise ValueError("network comparison needs both network dirs, --roi-list and --nsamples")
        report["network_counts"] = _network(args.network_fsl_dir,
                                            args.network_fnit_dir,
                                            args.roi_list, args.nsamples)
        report["network_counts"]["wall_seconds_including_load_and_write"] = {
            "fsl": _wall(Path(str(args.network_fsl_dir) + ".time")),
            "fnit": _wall(Path(str(args.network_fnit_dir) + ".time")),
        }
    if args.output_figure:
        _plot_matrices(args.fsl_dir, args.fnit_dir,
                       [int(name.removeprefix("matrix")) for name in report["matrices"]],
                       args.output_figure)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output_json)


if __name__ == "__main__":
    main()
