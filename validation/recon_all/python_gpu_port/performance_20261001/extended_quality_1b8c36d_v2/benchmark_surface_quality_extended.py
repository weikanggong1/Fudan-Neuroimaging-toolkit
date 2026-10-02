"""只读候选整例表面，补查连通性、顶点 link、球面翻折及 white/pial 穿越。

--subject 是已完成 FNIT 被试目录；读取 orig/white/pial/sphere/sphere.reg
（surface RAS，mm）、cortex label 和运行报告，不读取官方结果或写回表面。
--output 是新目录，写 JSON；--code-version 绑定被测源码。--threads 默认4，
--cross-timeout-seconds 默认180秒（每半球），--max-bbox-pairs 默认20000000。
复用已安装 FNIT 的 topology、signed_sphere 和三角碰撞内核；新空间查询
使用完整包围球与 bbox 候选，不使用最近顶点。预算不足明确写 incomplete。
归一化平面分类容差固定1e-6 mm，空间边界保护1e-5 mm；保留原碰撞内核
1e-6非归一化平面容差。proper transverse hit 与接触/重合分开报告，
不把有意固定到 white 的 pial 内侧壁接触称为穿越，也不据此证明全局包含。
只是独立 benchmark，没有独立官方等价命令；不修改生产质量判定或阈值。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import time

import nibabel.freesurfer as fs
import numpy as np
from numba import njit, set_num_threads
from scipy.spatial import cKDTree
import torch

from fnit.recon_all.compare_subject import _topology
from fnit.recon_all.mris_register_nonlinear import face_area_normals
from fnit.recon_all.place_surface_collision import _interval, triangles_intersect


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@njit(cache=True)
def _link_status(offsets, opposite_a, opposite_b):
    # 每个顶点的 link 必须为单个环：所有节点度2，且只有一个连通分量。
    status = np.zeros(len(offsets) - 1, dtype=np.int8)
    for vertex in range(len(status)):
        first, stop = offsets[vertex], offsets[vertex + 1]
        count = stop - first
        if count == 0:
            status[vertex] = 3
            continue
        nodes = np.empty(count * 2, dtype=np.int32)
        parents = np.arange(count * 2, dtype=np.int32)
        degree = np.zeros(count * 2, dtype=np.int32)
        used = 0
        for edge in range(first, stop):
            ends = np.array([opposite_a[edge], opposite_b[edge]], dtype=np.int32)
            ids = np.empty(2, dtype=np.int32)
            for corner in range(2):
                index = 0
                while index < used and nodes[index] != ends[corner]:
                    index += 1
                if index == used:
                    nodes[used] = ends[corner]
                    used += 1
                ids[corner] = index
                degree[index] += 1
            a, b = ids[0], ids[1]
            while parents[a] != a:
                a = parents[a]
            while parents[b] != b:
                b = parents[b]
            parents[a] = b
        components = 0
        for index in range(used):
            if parents[index] == index:
                components += 1
        if np.any(degree[:used] != 2):
            status[vertex] = 1
        elif components != 1:
            status[vertex] = 2
    return status


def vertex_links(faces, vertices):
    centers = np.concatenate([faces[:, corner] for corner in range(3)])
    opposite_a = np.concatenate([faces[:, (corner + 1) % 3] for corner in range(3)])
    opposite_b = np.concatenate([faces[:, (corner + 2) % 3] for corner in range(3)])
    order = np.argsort(centers, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(centers, minlength=vertices))]
    status = _link_status(offsets, opposite_a[order], opposite_b[order])
    return {"checked_vertices": vertices, "noncycle_links": int((status != 0).sum()),
            "degree_not_two": int((status == 1).sum()), "disconnected_cycles": int((status == 2).sum()),
            "isolated_vertices": int((status == 3).sum()),
            "anomaly_vertex_ids_first_100": np.flatnonzero(status)[:100].tolist()}


@njit(cache=True)
def _classify_pairs(white, pial, rows, indices):
    result = np.zeros(len(rows), dtype=np.int8)
    for pair in range(len(rows)):
        a, b = white[rows[pair]], pial[indices[pair]]
        if not triangles_intersect(a, b):
            continue
        aa, bb = a.astype(np.float64), b.astype(np.float64)
        na = np.cross(aa[1] - aa[0], aa[2] - aa[0])
        nb = np.cross(bb[1] - bb[0], bb[2] - bb[0])
        la, lb = np.linalg.norm(na), np.linalg.norm(nb)
        if la == 0 or lb == 0:
            result[pair] = 2
            continue
        da = (bb - aa[0]) @ (na / la)
        db = (aa - bb[0]) @ (nb / lb)
        proper = (np.min(da) < -1e-6 and np.max(da) > 1e-6 and
                  np.min(db) < -1e-6 and np.max(db) > 1e-6)
        if proper:
            line = np.cross(na / la, nb / lb)
            length = np.linalg.norm(line)
            if length == 0:
                proper = False
            else:
                axis = np.argmax(np.abs(line))
                a0, a1, ac = _interval(aa[:, axis], db)
                b0, b1, bc = _interval(bb[:, axis], da)
                # 单点/端点接触不能称穿越；正长度需超过预先固定的1e-6 mm。
                overlap = (min(a1, b1) - max(a0, b0)) / (abs(line[axis]) / length)
                proper = not ac and not bc and overlap > 1e-6
        result[pair] = 1 if proper else 2
    return result


def transverse_crossings(white, pial, faces, cortex, threads, timeout, pair_budget):
    start = time.perf_counter()
    w, p = white[faces].astype(np.float32), pial[faces].astype(np.float32)
    wc, pc = w.astype(np.float64).mean(axis=1), p.astype(np.float64).mean(axis=1)
    wr = np.linalg.norm(w - wc[:, None], axis=2).max(axis=1)
    pr = np.linalg.norm(p - pc[:, None], axis=2).max(axis=1)
    wl, wh, pl, ph = w.min(axis=1), w.max(axis=1), p.min(axis=1), p.max(axis=1)
    tree = cKDTree(pc)
    counts = {"raw_radius_pairs": 0, "bbox_pairs": 0, "proper_transverse_pairs": 0,
              "native_nonproper_hit_pairs": 0, "proper_pairs_with_any_cortex_vertex": 0,
              "proper_pairs_with_both_faces_fully_in_cortex": 0,
              "proper_pairs_with_no_cortex_vertex": 0}
    ids = []
    same = np.all(white == pial, axis=1)
    counts["coincident_same_index_faces"] = int(np.all(same[faces], axis=1).sum())
    status = "complete"
    completed = 0
    for first in range(0, len(faces), 1024):
        if time.perf_counter() - start > timeout:
            status = "incomplete_time_budget"
            break
        stop = min(first + 1024, len(faces))
        nearby = tree.query_ball_point(wc[first:stop], wr[first:stop] + pr.max() + 1e-5,
                                       workers=threads, return_sorted=True)
        size = np.fromiter((len(row) for row in nearby), dtype=np.int64, count=len(nearby))
        b = np.fromiter((index for row in nearby for index in row), dtype=np.int32,
                        count=int(size.sum()))
        a = np.repeat(np.arange(first, stop, dtype=np.int32), size)
        counts["raw_radius_pairs"] += len(a)
        close = np.sum((wc[a] - pc[b]) ** 2, axis=1) <= (wr[a] + pr[b] + 1e-5) ** 2
        a, b = a[close], b[close]
        overlap = np.all(wl[a] <= ph[b] + 1e-5, axis=1) & np.all(pl[b] <= wh[a] + 1e-5, axis=1)
        a, b = a[overlap], b[overlap]
        counts["bbox_pairs"] += len(a)
        if counts["bbox_pairs"] > pair_budget:
            status = "incomplete_candidate_budget"
            break
        kind = _classify_pairs(w, p, a, b)
        counts["proper_transverse_pairs"] += int((kind == 1).sum())
        counts["native_nonproper_hit_pairs"] += int((kind == 2).sum())
        aa, bb = a[kind == 1], b[kind == 1]
        cortical = np.any(cortex[faces[aa]], axis=1) | np.any(cortex[faces[bb]], axis=1)
        fully_cortical = np.all(cortex[faces[aa]], axis=1) & np.all(cortex[faces[bb]], axis=1)
        counts["proper_pairs_with_any_cortex_vertex"] += int(cortical.sum())
        counts["proper_pairs_with_both_faces_fully_in_cortex"] += int(fully_cortical.sum())
        counts["proper_pairs_with_no_cortex_vertex"] += int((~cortical).sum())
        ids.extend([[int(x), int(y)] for x, y in zip(aa[:max(0, 100 - len(ids))], bb)])
        completed = stop
    return {"status": status, **counts, "white_faces_examined": completed,
            "white_faces_total": len(faces), "proper_pair_face_ids_first_100": ids,
            "seconds": time.perf_counter() - start,
            "scope": "complete bounding sphere/bbox candidates for transverse hits; nonproper hits are native-kernel diagnostic, not exhaustive touch counting; no containment proof"}


def semantic_checks():
    tetra = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    assert vertex_links(tetra, 4)["noncycle_links"] == 0
    pinch = np.concatenate([tetra, np.where(tetra == 0, 0, tetra + 3)])
    assert vertex_links(pinch, 7)["disconnected_cycles"] == 1
    a = np.array([[[-1, -1, 0], [1, -1, 0], [0, 1, 0]]], dtype=np.float32)
    b = np.array([[[0, -.5, -1], [0, -.5, 1], [0, .5, 0]]], dtype=np.float32)
    assert _classify_pairs(a, b, np.array([0]), np.array([0]))[0] == 1
    assert _classify_pairs(a, b + 10, np.array([0]), np.array([0]))[0] == 0
    contact = np.array([[[0, 1, 0], [0, 2, -1], [0, 3, 1]]], dtype=np.float32)
    assert _classify_pairs(a, contact, np.array([0]), np.array([0]))[0] != 1
    return {"tetrahedron_links": "passed", "pinched_vertex_two_cycles": "passed",
            "transverse_triangle_and_separated_control": "passed", "endpoint_touch_not_transverse": "passed",
            "scope": "synthetic semantic checks, not real-data benchmark"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cross-timeout-seconds", type=float, default=180)
    parser.add_argument("--max-bbox-pairs", type=int, default=20_000_000)
    args = parser.parse_args()
    if args.threads < 1 or args.cross_timeout_seconds <= 0 or args.max_bbox_pairs < 1:
        parser.error("线程及资源预算必须为正值")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    set_num_threads(args.threads)
    run_path = args.subject / "fnit-native-free-run.json"
    run = json.loads(run_path.read_text())
    if run.get("status") != "complete":
        raise ValueError("候选整例未完成")
    report = {"subject": str(args.subject), "code_version": args.code_version,
              "host": socket.gethostname(), "threads": args.threads, "device": "cpu",
              "cuda_initialized": torch.cuda.is_initialized(),
              "script_sha256": sha256(__file__), "candidate_run_sha256": sha256(run_path),
              "coordinate_space": "surface RAS", "coordinate_unit": "mm",
              "parameters": {"bbox_guard_mm": 1e-5, "normalized_plane_classifier_tolerance_mm": 1e-6,
                             "intersection_line_overlap_tolerance_mm": 1e-6,
                             "sphere_negative_area_threshold_mm2": 0,
                             "cross_timeout_seconds_per_hemisphere": args.cross_timeout_seconds,
                             "maximum_bbox_pairs_per_hemisphere": args.max_bbox_pairs},
              "library_versions": {name: importlib.metadata.version(name) for name in
                                   ("torch", "numpy", "scipy", "numba", "nibabel")},
              "source_paths": {}, "source_sha256": {},
              "semantic_checks": semantic_checks(), "hemispheres": {}}
    report["predicate_semantics"] = {
        "proper_transverse": "native triangle hit AND strict straddling of both normalized planes beyond 1e-6 mm AND intersection-line overlap greater than 1e-6 mm",
        "endpoint_edge_and_coplanar": "not proper transverse; native nonproper hit counter is diagnostic, not exhaustive contact enumeration; exact same-index coincident faces reported separately",
        "cortex": "all surface pairs checked first; any-cortex subset means at least one of six triangle corners in cortex label; fully-cortex subset requires all six; neither subset excludes pairs from full-surface results",
        "sphere_orientation": "FNIT signed_sphere float32 area; outward means triangle cross dot sum of corner positions >=0 about fixed origin, threshold<0; zero/nonfinite areas separate",
    }
    for function in (_topology, face_area_normals, triangles_intersect):
        module = __import__(function.__module__, fromlist=["__file__"])
        report["source_sha256"][function.__module__] = sha256(module.__file__)
        report["source_paths"][function.__module__] = module.__file__
    for hemi in ("lh", "rh"):
        tick = time.perf_counter()
        paths = {name: args.subject / "surf" / f"{hemi}.{name}"
                 for name in ("orig", "white", "pial", "sphere", "sphere.reg")}
        meshes = {name: fs.read_geometry(str(path)) for name, path in paths.items()}
        vertices, faces = meshes["orig"]
        identical = all(v.shape == vertices.shape and np.array_equal(f, faces)
                        for v, f in meshes.values())
        row = {"input_sha256": {name: sha256(path) for name, path in paths.items()},
               "ordered_faces_and_vertex_counts_preserved": identical,
               "topology": _topology(vertices, faces), "vertex_links": vertex_links(faces, len(vertices)),
               "previous_pipeline_mesh_validation": run["mesh_validation"][hemi], "sphere_orientation": {}}
        for name in ("sphere", "sphere.reg"):
            xyz, triangles = meshes[name]
            area, _ = face_area_normals(torch.as_tensor(xyz, dtype=torch.float32),
                                        torch.as_tensor(triangles.astype(np.int64)), signed_sphere=True)
            area = area.numpy()
            row["sphere_orientation"][name] = {"negative_faces": int((area < 0).sum()),
                    "zero_area_faces": int((area == 0).sum()), "nonfinite_areas": int((~np.isfinite(area)).sum()),
                    "minimum_signed_area_mm2": float(area.min()),
                    "negative_face_ids_first_100": np.flatnonzero(area < 0)[:100].tolist()}
        report["hemispheres"][hemi] = row
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        label_path = args.subject / "label" / f"{hemi}.cortex.label"
        cortex = np.zeros(len(vertices), dtype=bool)
        cortex[fs.read_label(str(label_path))] = True
        row["input_sha256"]["cortex_label"] = sha256(label_path)
        row["white_pial_crossings"] = (transverse_crossings(meshes["white"][0], meshes["pial"][0], faces,
                                            cortex, args.threads, args.cross_timeout_seconds, args.max_bbox_pairs)
                                       if identical else {"status": "not_assessed_topology_mismatch"})
        row["seconds_including_io"] = time.perf_counter() - tick
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"hemisphere": hemi, **row}), flush=True)
    report["scope"] = "independent extended diagnostics; original production mesh gate unchanged; no overall-equivalence judgement"
    report["status"] = ("measured" if all(h["white_pial_crossings"]["status"] == "complete"
                        for h in report["hemispheres"].values()) else "partially_measured")
    report["cuda_initialized_after"] = torch.cuda.is_initialized()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
