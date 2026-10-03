#!/usr/bin/env python3
"""独立完整网格自交扫描：分半径组/分块枚举，复用冻结 FNIT 的判定。

输入私有具名 manifest；输出匿名数值、SHA 和私有标记数组。
不运行或修改 MRI pipeline；墙钟独立于生产 benchmark。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checked_binding(entry):
    path = Path(entry["path"]).expanduser().resolve()
    if sha256(path) != entry["sha256"]:
        raise ValueError("bound input SHA-256 differs")
    return path


class ScanTimeout(RuntimeError):
    pass


def scan(vertices, faces, predicate, *, threads=4, block_faces=256,
         maximum_pairs_per_block=2_000_000, timeout_seconds=900):
    """返回完整扫描标记；超时抛出异常，不把部分结果视为通过。

    每对半径分组独立使用保守邻域 r_i + max(r_group) + 1e-5。
    每个无序面片对最多出现一次；随后沿用成熟实现的三个过滤和判定。
    """
    import numpy as np
    from scipy.spatial import cKDTree

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3
            or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices)))):
        raise ValueError("invalid finite triangle mesh")
    triangles = vertices[faces]
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    # 零半径面单独归组；1 mm 仅为候选分组的边界，不是相交阈值。
    groups = np.floor(np.log2(np.maximum(radii, np.finfo(np.float64).tiny))).astype(np.int32)
    bins = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    trees = [cKDTree(centers[index]) for index in bins]
    maxima = [float(radii[index].max()) for index in bins]
    marked_faces = np.zeros(len(faces), dtype=bool)
    counts = {"bounding_sphere_candidates": 0, "individual_radius_candidates": 0,
              "bbox_candidates": 0, "disjoint_vertex_candidates": 0,
              "intersecting_face_pairs": 0, "query_blocks": 0,
              "largest_query_block_pairs": 0}
    started = time.perf_counter()

    def check_deadline():
        if time.perf_counter() - started > timeout_seconds:
            raise ScanTimeout("incomplete_time_budget")

    for first_bin, first in enumerate(bins):
        for second_bin in range(first_bin, len(bins)):
            second, tree = bins[second_bin], trees[second_bin]
            pending = [first[start:start + block_faces]
                       for start in range(0, len(first), block_faces)]
            while pending:
                check_deadline()
                indices = pending.pop()
                query_radii = radii[indices] + maxima[second_bin] + 1e-5
                lengths = tree.query_ball_point(centers[indices], query_radii,
                                               workers=threads, return_length=True)
                total = int(np.sum(lengths))
                if total > maximum_pairs_per_block and len(indices) > 1:
                    middle = len(indices) // 2
                    pending.extend((indices[:middle], indices[middle:]))
                    continue
                # 一个面片最多查询该组所有面；无全局配对数组或预算截断。
                neighbors = tree.query_ball_point(centers[indices], query_radii,
                                                 workers=threads, return_sorted=True)
                counts["query_blocks"] += 1
                counts["largest_query_block_pairs"] = max(counts["largest_query_block_pairs"], total)
                for face_a, local_neighbors in zip(indices, neighbors):
                    check_deadline()
                    candidates = second[np.asarray(local_neighbors, dtype=np.intp)]
                    if first_bin == second_bin:
                        candidates = candidates[candidates > face_a]
                    counts["bounding_sphere_candidates"] += len(candidates)
                    if not len(candidates):
                        continue
                    displacement = centers[candidates] - centers[face_a]
                    near = np.sum(displacement ** 2, axis=1) <= (
                        radii[face_a] + radii[candidates] + 1e-5) ** 2
                    candidates = candidates[near]
                    counts["individual_radius_candidates"] += len(candidates)
                    overlap = np.all(low[face_a] <= high[candidates] + 1e-5, axis=1) & np.all(
                        low[candidates] <= high[face_a] + 1e-5, axis=1)
                    candidates = candidates[overlap]
                    counts["bbox_candidates"] += len(candidates)
                    disjoint = np.all(faces[face_a, :, None] != faces[candidates, None, :], axis=(1, 2))
                    candidates = candidates[disjoint]
                    counts["disjoint_vertex_candidates"] += len(candidates)
                    for face_b in candidates:
                        check_deadline()
                        # 原 query_pairs 按全局面片编号升序，保留判定输入次序。
                        first_face, second_face = sorted((int(face_a), int(face_b)))
                        if predicate(triangles[first_face], triangles[second_face]):
                            marked_faces[face_a] = marked_faces[face_b] = True
                            counts["intersecting_face_pairs"] += 1
    marked_vertices = np.zeros(len(vertices), dtype=bool)
    marked_vertices[faces[marked_faces].ravel()] = True
    return marked_vertices, marked_faces, {**counts, "radius_groups": len(bins),
        "maximum_triangle_radius_mm": float(radii.max()) if len(radii) else None,
        "seconds": time.perf_counter() - started}


def run(manifest_path, output_root, *, threads=4, block_faces=256,
        maximum_pairs_per_block=2_000_000, timeout_seconds=900):
    manifest_path = Path(manifest_path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("cohort_id") != "formal-v4"
            or manifest.get("case_id") not in ("CON01", "CON03", "CON04", "CON05", "CON06", "CON07", "CON08", "CON09", "CON10", "CON11")
            or manifest.get("role") not in ("reference", "candidate")
            or manifest.get("hemisphere") not in ("lh", "rh")
            or manifest.get("surface_kind") not in ("white", "pial")):
        raise ValueError("explicit formal cohort/case/role/hemisphere/surface required")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    source_root = Path(manifest["source_root"]).resolve()
    source_path = source_root / "src/fnit/recon_all/mris_remove_intersection_python.py"
    if sha256(source_path) != manifest["predicate_module_sha256"]:
        raise ValueError("frozen predicate source differs")
    surface_path = checked_binding(manifest["surface"])
    run_report_path = checked_binding(manifest["run_report"])
    run_report = json.loads(run_report_path.read_text())
    if (run_report.get("status") != "complete" or run_report.get("subject") != manifest["case_id"]
            or run_report.get("source_unchanged_during_run") is not True):
        raise ValueError("completed exact formal run required")
    if manifest["role"] == "candidate" and run_report.get("source_revision") != manifest["candidate_source_revision"]:
        raise ValueError("candidate revision differs from its completed report")
    protected = [source_root, *(Path(path).resolve() for path in manifest["protected_roots"])]
    output = Path(output_root).resolve()
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in protected):
        raise ValueError("QC output must be separate from protected frozen inputs")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    baseline_path = checked_binding(manifest["baseline"]) if manifest.get("baseline") else None
    source_before = sha256(source_path)
    script_before = sha256(__file__)
    guards = {"surface": sha256(surface_path), "run_report": sha256(run_report_path),
              "manifest": hashlib.sha256(manifest_bytes).hexdigest(), "predicate_module": source_before,
              "scanner": script_before}
    if baseline_path:
        guards["baseline"] = sha256(baseline_path)
    result = {"schema_version": 1, "cohort_id": manifest["cohort_id"],
              "case_id": manifest["case_id"], "role": manifest["role"],
              "hemisphere": manifest["hemisphere"], "surface_kind": manifest["surface_kind"],
              "candidate_source_revision": manifest["candidate_source_revision"],
              "status": "running", "input_sha256": guards,
              "threads": threads, "device": "cpu", "block_faces": block_faces,
              "maximum_pairs_per_block": maximum_pairs_per_block,
              "timeout_seconds": timeout_seconds,
              "scope": "independent self-intersection QC; excluded from MRI pipeline clocks",
              "method": "exhaustive radius-bin bounding spheres, individual radii, bounding boxes and shared-vertex exclusion; frozen native-style triangle predicate unchanged"}
    write_json(output / "report.public.json", result)
    started = time.perf_counter()
    try:
        import nibabel.freesurfer.io as fsio
        import nibabel
        import numpy as np
        import scipy
        spec = importlib.util.spec_from_file_location("frozen_self_predicate", source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        vertices, faces = fsio.read_geometry(str(surface_path))
        vertex_marks, face_marks, details = scan(vertices, faces, module._triangles_intersect,
            threads=threads, block_faces=block_faces, maximum_pairs_per_block=maximum_pairs_per_block,
            timeout_seconds=timeout_seconds)
        marked_vertices, intersecting_faces = int(vertex_marks.sum()), int(face_marks.sum())
        np.savez_compressed(output / "marks.private.npz", vertices=vertex_marks, faces=face_marks)
        result.update(status="measured", vertices=len(vertices), faces=len(faces),
                      intersecting_faces=intersecting_faces, marked_vertices=marked_vertices,
                      predicate_result="detected_intersections" if intersecting_faces else "no_detected_intersections",
                      ordered_faces_sha256=hashlib.sha256(np.asarray(faces, dtype="<i4").tobytes()).hexdigest(),
                      vertex_marks_sha256=hashlib.sha256(vertex_marks.tobytes()).hexdigest(),
                      face_marks_sha256=hashlib.sha256(face_marks.tobytes()).hexdigest(),
                      scan=details, software_versions={"python":sys.version.split()[0],"numpy":np.__version__,
                          "scipy":scipy.__version__,"nibabel":nibabel.__version__})
        if baseline_path:
            baseline_provenance = json.loads(baseline_path.read_text())
            baseline_key = f"{manifest['role']}/{manifest['hemisphere']}.{manifest['surface_kind']}"
            baseline = baseline_provenance["results"][baseline_key]
            if (baseline_provenance.get("inputs_unchanged") is not True
                    or baseline_provenance.get("case_id") != manifest["case_id"]
                    or baseline_provenance.get("candidate_source_revision") != manifest["candidate_source_revision"]
                    or baseline_provenance["input_sha256"].get(baseline_key) != guards["surface"]
                    or baseline_provenance["input_after_sha256"].get(baseline_key) != guards["surface"]
                    or baseline.get("mesh_sha256") != guards["surface"]
                    or baseline.get("ordered_faces_sha256") != result["ordered_faces_sha256"]
                    or baseline.get("vertices") != len(vertices) or baseline.get("faces") != len(faces)):
                raise ValueError("mature baseline does not bind the same complete actual mesh")
            if baseline.get("status") != "measured":
                raise ValueError("baseline is incomplete; cannot assert oracle equivalence")
            equal = (baseline["intersecting_faces"] == intersecting_faces
                     and baseline["marked_vertices"] == marked_vertices
                     and baseline["source_sha256"] == source_before)
            result["mature_baseline"] = {"counts_equal": equal,
                "bitexact_zero_marks_proven": equal and marked_vertices == 0 and intersecting_faces == 0,
                "same_actual_mesh_and_ordered_faces_proven": True,
                "scope": "same full actual mesh and unchanged predicate; zero counts prove both Boolean masks all false"}
            if not equal:
                raise ValueError("complete actual-mesh mature oracle counts differ")
    except ScanTimeout:
        result.update(status="incomplete_time_budget", predicate_result="not_assessed")
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__,
                      error_sha256=hashlib.sha256(str(error).encode()).hexdigest())
        (output / "error.private.txt").write_text(str(error) + "\n")
    finally:
        after = {"surface": sha256(surface_path), "run_report": sha256(run_report_path),
                 "manifest": sha256(manifest_path), "predicate_module": sha256(source_path),
                 "scanner": sha256(__file__)}
        if baseline_path:
            after["baseline"] = sha256(baseline_path)
        result.update(input_sha256_after=after, inputs_and_sources_unchanged=after == guards,
                      posthoc_wall_seconds=time.perf_counter() - started)
        if after != guards:
            result.update(status="failed", error_type="ChangedInputOrSource")
        write_json(output / "report.public.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--block-faces", type=int, default=256)
    parser.add_argument("--maximum-pairs-per-block", type=int, default=2_000_000)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    args = parser.parse_args()
    if min(args.threads, args.block_faces, args.maximum_pairs_per_block, args.timeout_seconds) <= 0:
        parser.error("positive execution budgets required")
    result = run(args.manifest, args.output_root, threads=args.threads,
        block_faces=args.block_faces, maximum_pairs_per_block=args.maximum_pairs_per_block,
        timeout_seconds=args.timeout_seconds)
    raise SystemExit(0 if result["status"] == "measured" else 1)


if __name__ == "__main__":
    main()
