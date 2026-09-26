"""Quantify a native-free candidate against a same-T1 FreeSurfer subject."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np
from scipy.spatial import cKDTree


MAIN_SURFACES = ("orig", "white", "pial", "inflated", "sphere", "sphere.reg")
MAPS = ("thickness", "area", "area.pial", "area.mid", "volume", "curv",
        "curv.pial", "sulc", "white.preaparc.H", "white.preaparc.K",
        "inflated.H", "inflated.K", "smoothwm.H.crv", "smoothwm.K.crv",
        "smoothwm.K1.crv", "smoothwm.K2.crv")
LABEL_VOLUMES = {"aseg.mgz", "aseg.auto.mgz", "aseg.presurf.mgz",
                 "synthseg.rca.mgz", "aparc+aseg.mgz",
                 "aparc.a2009s+aseg.mgz", "aparc.DKTatlas+aseg.mgz",
                 "wmparc.mgz"}


def _numbers(reference: np.ndarray, candidate: np.ndarray) -> dict:
    left, right = np.asarray(reference, np.float64), np.asarray(candidate, np.float64)
    result = {"reference_count": int(left.size), "candidate_count": int(right.size),
              "reference_mean": float(np.mean(left)), "candidate_mean": float(np.mean(right)),
              "reference_median": float(np.median(left)), "candidate_median": float(np.median(right)),
              "reference_sum": float(np.sum(left)), "candidate_sum": float(np.sum(right)),
              "mean_difference": float(np.mean(right) - np.mean(left)),
              "sum_difference": float(np.sum(right) - np.sum(left))}
    if left.shape == right.shape:
        delta = np.abs(left - right)
        result.update(ordered_comparable=True, exact=int(np.count_nonzero(left == right)),
                      mae=float(delta.mean()), p95=float(np.quantile(delta, .95)),
                      max_abs=float(delta.max(initial=0)))
    else:
        result["ordered_comparable"] = False
    return result


def _label_dice(reference: np.ndarray, candidate: np.ndarray) -> dict:
    a, b = np.asarray(reference, np.int32), np.asarray(candidate, np.int32)
    counts_a = np.bincount(a.ravel())
    counts_b = np.bincount(b.ravel())
    common = np.bincount(a[a == b])
    result = {}
    for code in np.union1d(np.unique(a), np.unique(b)):
        code = int(code)
        na = int(counts_a[code]) if code < len(counts_a) else 0
        nb = int(counts_b[code]) if code < len(counts_b) else 0
        both = int(common[code]) if code < len(common) else 0
        result[str(code)] = {"reference_voxels": na, "candidate_voxels": nb,
                             "difference_voxels": nb - na,
                             "dice": 2 * both / (na + nb)}
    return result


def _volume(reference: Path, candidate: Path) -> dict:
    a, b = nib.load(str(reference)), nib.load(str(candidate))
    left, right = np.asarray(a.dataobj), np.asarray(b.dataobj)
    result = {"reference_shape": list(left.shape), "candidate_shape": list(right.shape),
              "reference_dtype": str(left.dtype), "candidate_dtype": str(right.dtype),
              "affine_max_abs": float(np.max(np.abs(a.affine - b.affine)))}
    if left.shape == right.shape:
        result.update(_numbers(left, right))
        result["different_voxels"] = int(np.count_nonzero(left != right))
        if reference.name in LABEL_VOLUMES:
            result["per_label"] = _label_dice(left, right)
    return result


def _surface(reference: Path, candidate: Path) -> dict:
    a, fa = fs.read_geometry(str(reference))
    b, fb = fs.read_geometry(str(candidate))
    result = {"reference_vertices": len(a), "candidate_vertices": len(b),
              "reference_faces": len(fa), "candidate_faces": len(fb),
              "ordered_faces_equal": bool(np.array_equal(fa, fb))}
    if len(a) == len(b) and result["ordered_faces_equal"]:
        displacement = np.linalg.norm(a - b, axis=1)
        result["ordered_vertex_displacement_mm"] = {
            "mean": float(displacement.mean()), "p95": float(np.quantile(displacement, .95)),
            "max": float(displacement.max())}
    else:
        nearest_a = cKDTree(a).query(b, workers=4)[0]
        nearest_b = cKDTree(b).query(a, workers=4)[0]
        result["no_vertex_correspondence"] = True
        result["symmetric_nearest_mm"] = {
            "candidate_to_reference_mean": float(nearest_a.mean()),
            "reference_to_candidate_mean": float(nearest_b.mean()),
            "candidate_to_reference_p95": float(np.quantile(nearest_a, .95)),
            "reference_to_candidate_p95": float(np.quantile(nearest_b, .95))}
    return result


def _map(reference: Path, candidate: Path, *, vertex_correspondence: bool,
         nearest_reference: np.ndarray | None = None) -> dict:
    a, b = fs.read_morph_data(str(reference)), fs.read_morph_data(str(candidate))
    result = _numbers(a, b)
    if not vertex_correspondence:
        result.pop("exact", None)
        result.pop("mae", None)
        result.pop("p95", None)
        result.pop("max_abs", None)
        result["ordered_comparable"] = False
        result["reason"] = "mesh vertex order/topology differs; spatial nearest is not true vertex correspondence"
        if nearest_reference is not None and len(b) == len(nearest_reference):
            difference = np.abs(a[nearest_reference].astype(np.float64) - b.astype(np.float64))
            result["spatial_nearest_error"] = {
                "mae": float(difference.mean()), "p95": float(np.quantile(difference, .95)),
                "max_abs": float(difference.max(initial=0))}
    return result


def _annotation(reference: Path, candidate: Path,
                *, vertex_correspondence: bool,
                nearest_reference: np.ndarray | None = None) -> dict:
    left, _, left_names = fs.read_annot(str(reference))
    right, _, right_names = fs.read_annot(str(candidate))
    names = sorted(set(left_names) | set(right_names))
    rows = {}
    for name in names:
        ia = left_names.index(name) if name in left_names else -2
        ib = right_names.index(name) if name in right_names else -2
        a, b = left == ia, right == ib
        row = {"reference_vertices": int(a.sum()),
               "candidate_vertices": int(b.sum()),
               "difference_vertices": int(b.sum() - a.sum())}
        if vertex_correspondence and len(a) == len(b) and (a.sum() + b.sum()):
            row["dice"] = float(2 * np.count_nonzero(a & b) / (a.sum() + b.sum()))
        elif nearest_reference is not None and len(b) == len(nearest_reference):
            matched = a[nearest_reference]
            if matched.sum() + b.sum():
                row["spatial_nearest_dice"] = float(
                    2 * np.count_nonzero(matched & b) / (matched.sum() + b.sum()))
        rows[name.decode(errors="replace")] = row
    return {"reference_vertices": len(left), "candidate_vertices": len(right),
            "vertex_correspondence": vertex_correspondence and len(left) == len(right),
            "regions": rows}


def _statistics(reference: Path, candidate: Path) -> dict:
    def parse(path: Path) -> tuple[dict[str, float], dict[str, dict[str, float]], list[str]]:
        measures, rows, columns = {}, {}, []
        for line in path.read_text().splitlines():
            if line.startswith("# Measure "):
                parts = [field.strip() for field in line.split(",")]
                if len(parts) >= 4:
                    try:
                        measures[parts[1]] = float(parts[3])
                    except ValueError:
                        pass
            elif line.startswith("# ColHeaders "):
                columns = line.split()[2:]
            elif line and not line.startswith("#"):
                fields = line.split()
                names = columns if len(columns) == len(fields) else [f"column_{i}" for i in range(len(fields))]
                key = fields[names.index("StructName")] if "StructName" in names else fields[0]
                numeric = {}
                for name, value in zip(names, fields):
                    try:
                        numeric[name] = float(value)
                    except ValueError:
                        pass
                if numeric:
                    rows[key] = numeric
        return measures, rows, columns

    am, ar, ac = parse(reference)
    bm, br, bc = parse(candidate)
    measures = {name: {"reference": am[name], "candidate": bm[name],
                       "difference": bm[name] - am[name]}
                for name in am.keys() & bm.keys()}
    rows = {}
    for name in ar.keys() & br.keys():
        rows[name] = {column: {"reference": ar[name][column],
                               "candidate": br[name][column],
                               "difference": br[name][column] - ar[name][column]}
                      for column in ar[name].keys() & br[name].keys()}
    return {"reference_rows": len(ar), "candidate_rows": len(br),
            "reference_columns": ac, "candidate_columns": bc,
            "missing_reference_regions": sorted(br.keys() - ar.keys()),
            "missing_candidate_regions": sorted(ar.keys() - br.keys()),
            "global_measure_differences": measures, "regions": rows}


def _model_volumes(reference: Path, candidate: Path) -> dict:
    def read(path: Path) -> dict[str, float]:
        with path.open(newline="") as stream:
            row = next(csv.DictReader(stream))
        numeric = {}
        for name, value in row.items():
            try:
                numeric[name] = float(value)
            except (TypeError, ValueError):
                pass
        return numeric

    a, b = read(reference), read(candidate)
    return {"reference_columns": len(a), "candidate_columns": len(b),
            "missing_candidate_columns": sorted(a.keys() - b.keys()),
            "values": {name: {"reference": a[name], "candidate": b[name],
                              "difference": b[name] - a[name]}
                       for name in a.keys() & b.keys()}}


def compare(reference: str | Path, candidate: str | Path) -> dict:
    reference, candidate = Path(reference), Path(candidate)
    result: dict = {"reference": str(reference), "candidate": str(candidate),
                    "volumes": {}, "surfaces": {}, "vertex_maps": {},
                    "statistics": {}, "annotations": {}, "model_volumes": {},
                    "missing_candidate": []}
    for ref in sorted((reference / "mri").rglob("*.mgz")):
        name = str(ref.relative_to(reference))
        got = candidate / name
        if got.is_file():
            result["volumes"][name] = _volume(ref, got)
        else:
            result["missing_candidate"].append(name)
    for hemi in ("lh", "rh"):
        ordered = {}
        for name in MAIN_SURFACES:
            key = f"surf/{hemi}.{name}"
            ref, got = reference / key, candidate / key
            if not got.is_file():
                result["missing_candidate"].append(key)
            elif ref.is_file():
                surface = _surface(ref, got)
                result["surfaces"][key] = surface
                ordered[name] = surface["ordered_faces_equal"] and (
                    surface["reference_vertices"] == surface["candidate_vertices"])
        nearest = None
        if not ordered.get("white") and (reference / f"surf/{hemi}.white").is_file() and (
            candidate / f"surf/{hemi}.white").is_file():
            ref_xyz, _ = fs.read_geometry(str(reference / f"surf/{hemi}.white"))
            got_xyz, _ = fs.read_geometry(str(candidate / f"surf/{hemi}.white"))
            nearest = cKDTree(ref_xyz).query(got_xyz, workers=4)[1]
        for name in MAPS:
            key = f"surf/{hemi}.{name}"
            ref, got = reference / key, candidate / key
            if not got.is_file():
                result["missing_candidate"].append(key)
            elif ref.is_file():
                result["vertex_maps"][key] = _map(
                    ref, got, vertex_correspondence=bool(ordered.get("white")),
                    nearest_reference=nearest)
        for name in ("aparc", "aparc.a2009s", "aparc.DKTatlas",
                     "BA_exvivo", "BA_exvivo.thresh", "mpm.vpnl"):
            key = f"label/{hemi}.{name}.annot"
            ref, got = reference / key, candidate / key
            if not got.is_file():
                result["missing_candidate"].append(key)
            elif ref.is_file():
                result["annotations"][key] = _annotation(
                    ref, got, vertex_correspondence=bool(ordered.get("white")),
                    nearest_reference=nearest)
        for name in ("aparc", "aparc.a2009s", "aparc.DKTatlas"):
            key = f"stats/{hemi}.{name}.stats"
            ref, got = reference / key, candidate / key
            if not got.is_file():
                result["missing_candidate"].append(key)
            elif ref.is_file():
                result["statistics"][key] = _statistics(ref, got)
    for name in ("aseg.stats", "wmparc.stats", "brainvol.stats"):
        key = f"stats/{name}"
        ref, got = reference / key, candidate / key
        if ref.is_file():
            if got.is_file():
                result["statistics"][key] = _statistics(ref, got)
            else:
                result["missing_candidate"].append(key)
    key = "stats/synthseg.vol.csv"
    ref, got = reference / key, candidate / key
    if ref.is_file():
        if got.is_file():
            result["model_volumes"][key] = _model_volumes(ref, got)
        else:
            result["missing_candidate"].append(key)
    key = "stats/synthseg.tiv.dat"
    ref, got = reference / key, candidate / key
    if ref.is_file():
        if got.is_file():
            a, b = float(ref.read_text()), float(got.read_text())
            result["model_volumes"][key] = {"reference": a, "candidate": b,
                                            "difference": b - a}
        else:
            result["missing_candidate"].append(key)
    result["compared_files"] = sum(len(result[key]) for key in
                                   ("volumes", "surfaces", "vertex_maps", "annotations", "statistics",
                                    "model_volumes"))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.reference, args.candidate)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"compared {report['compared_files']} files; "
          f"{len(report['missing_candidate'])} candidate files missing")


if __name__ == "__main__":
    main()
