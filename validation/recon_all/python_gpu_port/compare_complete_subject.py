"""Compare the fixed recon-all profile's final outputs, including every vertex."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
from nibabel import freesurfer as fs
import numpy as np


from fnit.recon_all.expected_outputs import VOLUMES, SURFACES, MORPHS, ANNOTS, STATS

SURFACE_ATOL_MM = 1e-5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _error(reference: np.ndarray, candidate: np.ndarray, absolute: float,
           relative: float = 0.0) -> dict:
    if reference.shape != candidate.shape:
        return {"pass": False, "reference_shape": list(reference.shape),
                "candidate_shape": list(candidate.shape)}
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    limit = absolute + relative * np.abs(reference.astype(np.float64))
    failed = ~np.isfinite(delta) | (delta > limit)
    finite = delta[np.isfinite(delta)]
    return {"pass": not bool(np.any(failed)), "elements": int(reference.size),
            "exact": int(np.count_nonzero(reference == candidate)),
            "outliers": int(np.count_nonzero(failed)),
            "max_abs": float(finite.max(initial=0)),
            "p99_abs": float(np.quantile(finite, .99)) if finite.size else 0.0,
            "first_outlier": np.argwhere(failed)[0].tolist() if np.any(failed) else None}


def _volume(reference: Path, candidate: Path) -> dict:
    ref, got = nib.load(str(reference)), nib.load(str(candidate))
    a, b = np.asarray(ref.dataobj), np.asarray(got.dataobj)
    integer = np.issubdtype(a.dtype, np.integer)
    values = _error(a, b, 0.0 if integer else 1e-6)
    affine = _error(ref.affine, got.affine, 1e-6)
    header = ref.header.binaryblock == got.header.binaryblock
    dtype = a.dtype == b.dtype
    return {"pass": values["pass"] and affine["pass"] and header and dtype,
            "dtype": [str(a.dtype), str(b.dtype)], "voxels": values,
            "affine": affine, "header_exact": header}


def _surface(reference: Path, candidate: Path) -> dict:
    a, fa, va = fs.read_geometry(str(reference), read_metadata=True)
    b, fb, vb = fs.read_geometry(str(candidate), read_metadata=True)
    xyz = _error(a, b, SURFACE_ATOL_MM)
    faces = _error(fa, fb, 0.0)
    keys = set(va) | set(vb)
    geometry = all(np.array_equal(va.get(key), vb.get(key))
                   for key in keys if key != "filename")
    filename = np.array_equal(va.get("filename"), vb.get("filename"))
    return {"pass": xyz["pass"] and faces["pass"] and geometry,
            "coordinates_mm": xyz, "ordered_faces": faces,
            "volume_geometry_data_exact": geometry,
            "volume_geometry_filename_equal": filename,
            "volume_geometry_exact": geometry and filename}


def _morph(reference: Path, candidate: Path) -> dict:
    name = reference.name.split(".", 1)[1]
    absolute = .001 if name.startswith("area") else .005
    return _error(fs.read_morph_data(str(reference)),
                  fs.read_morph_data(str(candidate)), absolute, .001)


def _annot(reference: Path, candidate: Path) -> dict:
    a, ca, na = fs.read_annot(str(reference), orig_ids=True)
    b, cb, nb = fs.read_annot(str(candidate), orig_ids=True)
    ids = _error(a, b, 0.0)
    table = np.array_equal(ca, cb) and na == nb
    return {"pass": ids["pass"] and table, "vertex_ids": ids,
            "color_table_and_names_exact": table}


def _stats(reference: Path, candidate: Path) -> dict:
    def rows(path: Path) -> list[str]:
        return [line.strip() for line in path.read_text().splitlines()
                if line.strip() and (not line.startswith("#")
                                     or line.startswith("# Measure "))]

    a, b = rows(reference), rows(candidate)
    failures = []
    maximum = 0.0
    for index, (left, right) in enumerate(zip(a, b)):
        comma = left.startswith("# Measure ")
        left_fields = [item.strip() for item in left.split(",")] if comma else left.split()
        right_fields = [item.strip() for item in right.split(",")] if comma else right.split()
        if len(left_fields) != len(right_fields):
            failures.append(index)
            continue
        for x, y in zip(left_fields, right_fields):
            if x == y:
                continue
            try:
                difference = abs(float(x) - float(y))
            except ValueError:
                if x != y:
                    failures.append(index)
                    break
            else:
                maximum = max(maximum, difference)
                if not np.isfinite(difference) or difference > .005:
                    failures.append(index)
                    break
    passed = len(a) == len(b) and not failures
    return {"pass": passed, "reference_rows": len(a), "candidate_rows": len(b),
            "exact_rows": sum(x == y for x, y in zip(a, b)),
            "outlier_rows": len(failures), "max_numeric_abs": maximum,
            "first_outlier": failures[0] if failures else None}


def compare(reference: Path, candidate: Path) -> dict:
    expected = {name: _volume for name in VOLUMES}
    expected.update({f"surf/{hemi}.{name}": _surface
                     for hemi in ("lh", "rh") for name in SURFACES})
    expected.update({f"surf/{hemi}.{name}": _morph
                     for hemi in ("lh", "rh") for name in MORPHS})
    expected.update({f"label/{hemi}.{name}.annot": _annot
                     for hemi in ("lh", "rh") for name in ANNOTS})
    expected.update({f"stats/{hemi}.{name}.stats": _stats
                     for hemi in ("lh", "rh")
                     for name in ("aparc", "aparc.a2009s", "aparc.DKTatlas",
                                  "aparc.pial", "BA_exvivo", "BA_exvivo.thresh",
                                  "curv", "w-g.pct")})
    expected.update({f"stats/{name}": _stats for name in STATS})
    rows = {}
    for name, check in expected.items():
        left, right = reference / name, candidate / name
        if not left.is_file() or not right.is_file():
            rows[name] = {"pass": False, "missing_reference": not left.is_file(),
                          "missing_candidate": not right.is_file()}
        else:
            rows[name] = check(left, right)
            rows[name]["reference_sha256"] = _sha256(left)
            rows[name]["candidate_sha256"] = _sha256(right)
    return {"reference": str(reference), "candidate": str(candidate),
            "comparator_sha256": _sha256(Path(__file__)),
            "checked": len(rows), "passed": sum(bool(row["pass"]) for row in rows.values()),
            "all_pass": all(row["pass"] for row in rows.values()), "files": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference.resolve(), args.candidate.resolve())
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['passed']}/{result['checked']} final outputs passed")
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
