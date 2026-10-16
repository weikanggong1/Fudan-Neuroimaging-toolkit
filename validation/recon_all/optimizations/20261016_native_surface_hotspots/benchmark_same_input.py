"""同一 FNIT 自产 premesh 的原生 remesh/相交清理诊断，不发布生产文件。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import nibabel.freesurfer.io as fs
import numpy as np
from scipy.spatial import cKDTree

from fnit.recon_all.mris_remove_intersection_python import mark_intersections
from fnit.recon_all.place_surface_intersection_marking import mark_source_intersections
from fnit.recon_all.place_surface_quality import mesh_structure, placement_structure_passed


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mesh(path: Path, *, max_pair_candidates: int) -> dict:
    vertices, faces, volume_info = fs.read_geometry(str(path), read_metadata=True)
    structure = mesh_structure(vertices=vertices, faces=faces)
    triangles = np.asarray(vertices, np.float64)[faces]
    centers = triangles.mean(axis=1)
    radius = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    maximum_radius = float(radius.max()) if len(radius) else 0.0
    tree = cKDTree(centers)
    # The existing geometric marker materializes query_pairs at this radius.
    # Count first so a long triangle cannot allocate an unbounded pair array.
    search_radius = 2 * maximum_radius + 1e-5
    pair_count = (int(tree.count_neighbors(tree, search_radius)) - len(faces)) // 2
    row = {
        "path": str(path), "sha256": _sha(path), "structure": structure,
        "structure_passed": placement_structure_passed(structure),
        "maximum_face_radius_mm": maximum_radius,
        "geometric_candidate_pairs_upper_bound": pair_count,
        "volume_info": {key: np.asarray(value).tolist() for key, value in (volume_info or {}).items()},
    }
    _, source_count = mark_source_intersections(vertices, faces, predicate_backend="numba")
    row["source_intersecting_faces"] = int(source_count)
    if pair_count > max_pair_candidates:
        row["geometric_intersecting_faces"] = None
        row["geometric_status"] = "resource_guard_blocked"
    else:
        _, geometric_count = mark_intersections(vertices, faces)
        row["geometric_intersecting_faces"] = int(geometric_count)
        row["geometric_status"] = "complete"
    return row


def _run(binary: Path, arguments: list[str], log: Path, *, assets: Path) -> float:
    env = dict(os.environ, FREESURFER_HOME=str(assets))
    started = time.perf_counter()
    with log.open("w") as stream:
        subprocess.run([str(binary), *arguments], check=True, env=env,
                       stdout=stream, stderr=subprocess.STDOUT)
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--premesh", required=True, type=Path)
    parser.add_argument("--remesh-binary", required=True, type=Path)
    parser.add_argument("--intersection-binary", required=True, type=Path)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-orig", type=Path)
    parser.add_argument("--preclean", action="store_true",
                        help="先对同一自产 premesh 执行原生清理，保留原输入")
    parser.add_argument("--max-pair-candidates", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.max_pair_candidates < 1:
        raise ValueError("max-pair-candidates must be positive")
    for path in (args.premesh, args.remesh_binary, args.intersection_binary):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.assets_dir.is_dir():
        raise NotADirectoryError(args.assets_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    remeshed = args.output_dir / "orig.native-remesh"
    cleaned = args.output_dir / "orig.native-cleaned"
    report_path = args.output_dir / "report.json"
    report = {
        "scope": "frozen_same_FNIT_premesh_stage_only",
        "status": "running", "premesh_sha256": _sha(args.premesh),
        "remesh_binary_sha256": _sha(args.remesh_binary),
        "intersection_binary_sha256": _sha(args.intersection_binary),
        "max_pair_candidates": args.max_pair_candidates,
        "paths": {"premesh": str(args.premesh), "output_dir": str(args.output_dir)},
    }
    started = time.perf_counter()
    try:
        remesh_input = args.premesh
        if args.preclean:
            remesh_input = args.output_dir / "premesh.native-preclean"
            report["preclean_seconds_including_io"] = _run(
                args.intersection_binary, [str(args.premesh), str(remesh_input)],
                args.output_dir / "preclean.log", assets=args.assets_dir)
            before, before_faces, before_info = fs.read_geometry(str(args.premesh), read_metadata=True)
            after, after_faces, after_info = fs.read_geometry(str(remesh_input), read_metadata=True)
            before_info, after_info = before_info or {}, after_info or {}
            metadata_equal = before_info.keys() == after_info.keys() and all(
                np.array_equal(before_info[key], after_info[key]) for key in before_info)
            if (len(before) != len(after) or not np.array_equal(before_faces, after_faces)
                    or not np.isfinite(after).all() or not metadata_equal):
                raise RuntimeError("native preclean changed ordered faces or MRI geometry")
            report["preclean_output_sha256"] = _sha(remesh_input)
        report["remesh_seconds_including_io"] = _run(
            args.remesh_binary,
            ["-i", str(remesh_input), "-o", str(remeshed), "--remesh", "--iters", "3"],
            args.output_dir / "remesh.log", assets=args.assets_dir)
        report["remeshed"] = _mesh(remeshed, max_pair_candidates=args.max_pair_candidates)
        report["intersection_seconds_including_io"] = _run(
            args.intersection_binary, [str(remeshed), str(cleaned)],
            args.output_dir / "intersection.log", assets=args.assets_dir)
        report["cleaned"] = _mesh(cleaned, max_pair_candidates=args.max_pair_candidates)
        if args.python_orig is not None:
            candidate, candidate_faces = fs.read_geometry(str(cleaned))
            baseline, baseline_faces = fs.read_geometry(str(args.python_orig))
            report["python_baseline"] = {
                "sha256": _sha(args.python_orig), "vertices": len(baseline),
                "faces": len(baseline_faces),
                "ordered_faces_equal": bool(np.array_equal(candidate_faces, baseline_faces)),
                "same_index_coordinate_max_mm": (
                    float(np.linalg.norm(candidate - baseline, axis=1).max())
                    if np.array_equal(candidate_faces, baseline_faces) and len(candidate) == len(baseline)
                    else None),
            }
        final = report["cleaned"]
        report["quality_passed"] = bool(
            final["structure_passed"]
            and final["source_intersecting_faces"] == 0
            and final["geometric_intersecting_faces"] == 0)
        report["status"] = "complete" if report["quality_passed"] else "quality_failed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_seconds_including_checks_and_io"] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["quality_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
