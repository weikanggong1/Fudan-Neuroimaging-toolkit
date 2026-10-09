"""固定真实orig：比较初始化相交清理与已验证的有序Möller谓词。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import nibabel.freesurfer.io as fs
import numpy as np
from scipy.spatial import cKDTree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    if args.output_directory.exists():
        raise FileExistsError(args.output_directory)
    args.output_directory.mkdir(parents=True)
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.candidate_directory.resolve()))
    from fnit.recon_all import place_surface_final_cleanup as cleanup
    from fnit.recon_all import mris_remove_intersection_python as marking
    from fnit.recon_all.place_surface_smoothing import average_vertex_positions
    from fnit.recon_all.place_surface_collision_torch import _source_pairs
    report = {"scope": "real_white_initial_cleanup_diagnostic_only", "hostname": platform.node(),
              "code_commit": args.code_commit, "cuda": "not_initialized", "hemisphere": args.hemisphere,
              "script_sha256": sha(__file__), "source_sha256": {}, "cleanup_calls": []}
    for name in ("place_surface_final_cleanup.py", "mris_remove_intersection_python.py",
                 "place_surface_smoothing.py", "place_surface_collision.py", "place_surface_collision_torch.py"):
        report["source_sha256"][name] = sha(args.candidate_directory / name)
    orig = args.subject / f"surf/{args.hemisphere}.orig"
    report["input_sha256"] = sha(orig)
    vertices, faces = fs.read_geometry(str(orig))
    started = time.perf_counter()
    smooth = average_vertex_positions(vertices, faces, 5)
    report["smoothing_seconds"] = time.perf_counter() - started
    original = cleanup.mark_intersections

    def observed_mark(vertices, faces):
        tick = time.perf_counter()
        marked, count = original(vertices, faces)
        report["cleanup_calls"].append({"count": count, "marked_vertices": int(marked.sum()),
            "seconds": time.perf_counter() - tick,
            "coordinate_sha256": hashlib.sha256(vertices.tobytes()).hexdigest()})
        (args.output_directory / "report.json").write_text(json.dumps(report, indent=2))
        return marked, count

    cleanup.mark_intersections = observed_mark
    started = time.perf_counter()
    result, diagnostic = cleanup.repair_intersections(smooth, faces, np.zeros(len(smooth), dtype=bool))
    report["repair_seconds"] = time.perf_counter() - started
    report["cleanup"] = diagnostic
    np.savez(args.output_directory / "initial_and_cleaned.npz", initial=smooth, cleaned=result, faces=faces)
    # Conservative sphere + AABB candidates, complete in bounded face blocks.
    xyz = np.asarray(result, dtype=np.float64)
    triangles = xyz[faces]
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    tree = cKDTree(centers)
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    marks_numpy = np.zeros(len(faces), dtype=bool)
    marks_source = np.zeros(len(faces), dtype=bool)
    different_pairs, reverse_differences, candidates = 0, 0, 0
    mismatch_faces = []
    started = time.perf_counter()
    for offset in range(0, len(faces), 1024):
        end = min(offset + 1024, len(faces))
        blocks = tree.query_ball_point(centers[offset:end], radii[offset:end] + radii.max() + 1e-5)
        for index, block in enumerate(blocks, offset):
            other = np.asarray(block, dtype=np.int64)
            other = other[other > index]
            keep = np.sum((centers[other] - centers[index]) ** 2, axis=1) <= (radii[other] + radii[index] + 1e-5) ** 2
            other = other[keep]
            keep = np.all(low[index] <= high[other] + 1e-5, axis=1) & np.all(low[other] <= high[index] + 1e-5, axis=1)
            other = other[keep]
            other = other[np.all(faces[index, :, None] != faces[other, None, :], axis=(1, 2))]
            if len(other) == 0:
                continue
            first = np.repeat(triangles[index][None], len(other), axis=0)
            second = triangles[other]
            source = _source_pairs(first, second)
            reverse = _source_pairs(second, first)
            old = np.asarray([marking._triangles_intersect(a, b) for a, b in zip(first, second)])
            marks_numpy[index] |= old.any()
            marks_numpy[other[old]] = True
            marks_source[index] |= source.any()
            marks_source[other[reverse]] = True
            different_pairs += int(np.count_nonzero(old != source))
            reverse_differences += int(np.count_nonzero(source != reverse))
            candidates += len(other)
            mismatch_faces.extend((index, int(other[j]), bool(old[j]), bool(source[j]), bool(reverse[j]))
                                  for j in np.flatnonzero((old != source) | (source != reverse)))
    report.update(predicate_comparison_seconds=time.perf_counter() - started,
                  candidates=candidates, numpy_intersecting_faces=int(marks_numpy.sum()),
                  source_ordered_intersecting_faces=int(marks_source.sum()),
                  different_pairs=different_pairs, reversal_differences=reverse_differences,
                  mismatch_pairs=mismatch_faces, execution_status="complete")
    (args.output_directory / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
