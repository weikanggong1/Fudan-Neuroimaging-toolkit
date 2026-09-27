"""Generate candidate LH cortical labels, then compare with FreeSurfer 8.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from fnit.recon_all.label_cortex_fix_ga_python import label_cortex_fix_ga
from fnit.recon_all.label_cortex_python import label_cortex


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_rows(path: Path) -> tuple[int, list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    rows = lines[2:]
    return int(lines[1]), rows, np.asarray([int(row.split()[0]) for row in rows])


def compare_label(candidate: Path, official: Path) -> dict:
    count, rows, ids = label_rows(candidate)
    ref_count, ref_rows, ref_ids = label_rows(official)
    first_difference = next((index for index, (a, b) in
                             enumerate(zip(rows, ref_rows)) if a != b), None)
    if first_difference is None and len(rows) != len(ref_rows):
        first_difference = min(len(rows), len(ref_rows))
    candidate_tokens = [row.split() for row in rows]
    official_tokens = [row.split() for row in ref_rows]
    paired = list(zip(candidate_tokens, official_tokens))
    coordinates = np.asarray([[float(value) for value in row[1:4]]
                              for row in candidate_tokens])
    reference_coordinates = np.asarray([[float(value) for value in row[1:4]]
                                        for row in official_tokens])
    coordinate_distance = np.linalg.norm(coordinates - reference_coordinates, axis=1)
    return {
        "header_and_count_text_equal": candidate.read_text().splitlines()[:2] ==
                                      official.read_text().splitlines()[:2],
        "id_and_stat_fields_equal": all(a[0] == b[0] and a[4] == b[4]
                                        for a, b in paired),
        "coordinate_text_rows_different": sum(a[1:4] != b[1:4]
                                              for a, b in paired),
        "coordinate_text_mean_euclidean_mm": float(np.mean(coordinate_distance)),
        "coordinate_text_p99_euclidean_mm": float(np.percentile(coordinate_distance, 99)),
        "coordinate_text_max_euclidean_mm": float(np.max(coordinate_distance)),
        "candidate_path": str(candidate.resolve()),
        "candidate_sha256": sha256(candidate),
        "official_sha256": sha256(official),
        "byte_equal": candidate.read_bytes() == official.read_bytes(),
        "declared_rows": count, "actual_rows": len(rows),
        "official_rows": ref_count, "official_actual_rows": len(ref_rows),
        "ordered_vertex_ids_equal": bool(np.array_equal(ids, ref_ids)),
        "candidate_unique_vertices": len(np.unique(ids)),
        "official_unique_vertices": len(np.unique(ref_ids)),
        "vertex_id_symmetric_difference": len(set(ids) ^ set(ref_ids)),
        "first_different_row_index": first_difference,
        "different_text_rows": sum(a != b for a, b in zip(rows, ref_rows))
                               + abs(len(rows) - len(ref_rows)),
    }


def paired_mri(candidate: Path, official: Path) -> dict:
    a, b = nib.load(str(candidate)), nib.load(str(official))
    return {
        "candidate_realpath": str(candidate.resolve()),
        "candidate_sha256": sha256(candidate),
        "official_sha256": sha256(official),
        "different_voxels": int(np.count_nonzero(
            np.asarray(a.dataobj) != np.asarray(b.dataobj))),
        "affine_max_abs_mm": float(np.max(np.abs(a.affine - b.affine))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_subject", type=Path)
    parser.add_argument("candidate_entowm", type=Path)
    parser.add_argument("official_subject", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--compare-existing", action="store_true")
    args = parser.parse_args()
    candidate, official = args.candidate_subject.resolve(), args.official_subject.resolve()
    surface = candidate / "surf/lh.white.preaparc"
    aseg = candidate / "mri/aseg.presurf.mgz"
    entowm = candidate / "mri/entowm.mgz"
    if not entowm.exists():
        entowm.symlink_to(args.candidate_entowm.resolve())
    if entowm.resolve() != args.candidate_entowm.resolve():
        raise ValueError("subject EntoWM differs from candidate input")
    for source in (surface, aseg, entowm):
        if not source.is_file() or source.resolve().is_relative_to(official):
            raise ValueError(f"missing or official reconstruction input: {source}")
    labels = candidate / "label"
    labels.mkdir(exist_ok=True)
    cortex = labels / "lh.cortex.label"
    hipamyg = labels / "lh.cortex+hipamyg.label"
    for output in (cortex, hipamyg):
        if args.compare_existing:
            if not output.is_file():
                raise FileNotFoundError(output)
        elif output.exists():
            raise FileExistsError(output)
    report = {
        "candidate_subject": str(candidate),
        "official_subject": str(official),
        "inputs": {
            "white.preaparc": {"candidate_realpath": str(surface.resolve()),
                                "candidate_sha256": sha256(surface),
                                "official_sha256": sha256(official / "surf/lh.white.preaparc")},
            "aseg.presurf": paired_mri(aseg, official / "mri/aseg.presurf.mgz"),
            "entowm": paired_mri(entowm, official / "mri/entowm.mgz"),
        },
        "outputs": {}, "timings_seconds": {},
    }
    if args.compare_existing:
        previous = json.loads(args.report.read_text())
        report["timings_seconds"] = previous["timings_seconds"]
        base_count = previous["outputs"]["cortex.label"]["base_rows"]
        ga_count = previous["outputs"]["cortex.label"]["appended_ga_rows"]
        hip_count = previous["outputs"]["cortex+hipamyg.label"]["selected_vertices"]
    else:
        start = time.perf_counter()
        base, ga = label_cortex_fix_ga(surface, aseg, entowm, "lh", cortex)
        report["timings_seconds"]["cortex_fix_ga"] = time.perf_counter() - start
        base_count, ga_count = len(base), len(ga)
        start = time.perf_counter()
        hip = label_cortex(surface, aseg, hipamyg, keep_hip_amyg=True)
        report["timings_seconds"]["cortex_hipamyg"] = time.perf_counter() - start
        hip_count = len(hip)
    report["outputs"]["cortex.label"] = compare_label(
        cortex, official / "label/lh.cortex.label") | {
            "base_rows": base_count, "appended_ga_rows": ga_count}
    report["outputs"]["cortex+hipamyg.label"] = compare_label(
        hipamyg, official / "label/lh.cortex+hipamyg.label") | {
            "selected_vertices": hip_count}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(args.report)


if __name__ == "__main__":
    main()
