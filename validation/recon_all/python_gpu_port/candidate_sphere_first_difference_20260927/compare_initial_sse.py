"""比较同一候选 smoothwm 和首次原生 SSE 内存快照；只读，不执行 sphere 优化。

用法：python compare_initial_sse.py --smoothwm ... --sphere0000 ... --capture-dir ... --native-log ... --report-json ...
"""

import argparse
import gzip
import hashlib
import json
import math
import re
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
from numba import njit

from fnit.recon_all.sphere_standard_metric import (
    average_standard_metric, sample_standard_metric_matrix,
)
from fnit.recon_all.sphere_standard_unfold import _sphere_radius_units, _spherical_distance


@njit
def current_distances(xyz, offsets, ids):
    radius, unit = _sphere_radius_units(xyz)
    result = np.empty(len(ids), np.float32)
    for vertex in range(len(xyz)):
        for index in range(offsets[vertex], offsets[vertex + 1]):
            result[index] = _spherical_distance(xyz, radius, unit, vertex, ids[index])
    return result


@njit
def weighted_distance_sse(current, target, offsets, scale, weight):
    total = 0.0
    for vertex in range(len(offsets) - 1):
        subtotal = 0.0
        for index in range(offsets[vertex], offsets[vertex + 1]):
            delta = scale * np.float64(current[index]) - np.float64(target[index])
            subtotal += delta * delta
        total += subtotal
    return total * weight


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compare(a, b, offsets):
    if a.shape != b.shape:
        return {"same_shape": False, "native_count": len(a), "python_count": len(b)}
    equal = a == b
    first = np.flatnonzero(~equal)
    result = {"same_shape": True, "count": len(a), "exact": int(equal.sum()),
              "first_mismatch": None}
    if len(first):
        index = int(first[0])
        row = int(np.searchsorted(offsets, index, side="right") - 1)
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        result.update({"first_mismatch": {
            "flat_index": index, "vertex": row,
            "row_position": index - int(offsets[row]),
            "native": float(a[index]), "python": float(b[index]),
            "native_bits": hex(int(a.view(np.uint32)[index])),
            "python_bits": hex(int(b.view(np.uint32)[index]))},
            "mean_abs": float(delta.mean()),
            "p99_abs": float(np.quantile(delta, .99)),
            "max_abs": float(delta.max())})
    return result


def first_log_terms(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as stream:
        text = stream.read()
    block = text.split("logSSE:1", 1)[1].split("logSSE:2", 1)[0]
    def value(name):
        found = re.search(rf"\b{re.escape(name)}\s*:\s*([-+0-9.eE]+)", block)
        if not found:
            raise ValueError(f"native first SSE log lacks {name}")
        return float(found.group(1))
    return {name: value(name) for name in
            ("mris->orig_area", "mris->total_area", "mris->neg_area", "new sse_dist")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoothwm", required=True, type=Path, help="候选 lh.smoothwm 输入")
    parser.add_argument("--sphere0000", required=True, type=Path, help="原生首次投影网格输入")
    parser.add_argument("--capture-dir", required=True, type=Path, help="GDB 原生数组目录")
    parser.add_argument("--native-log", required=True, type=Path, help="首次 SSE 原生日志，可为 gzip")
    parser.add_argument("--report-json", required=True, type=Path, help="JSON 审计输出")
    args = parser.parse_args()
    t0 = time.perf_counter()
    smoothwm, faces = fsio.read_geometry(str(args.smoothwm))
    checkpoint_xyz, sphere_faces = fsio.read_geometry(str(args.sphere0000))
    if not np.array_equal(faces, sphere_faces):
        raise ValueError("smoothwm and sphere0000 face order differs")
    cap = args.capture_dir
    provenance = json.loads((cap / "native_capture.json").read_text())
    native_offsets = np.fromfile(cap / "native_offsets.bin", dtype="<u8")
    native_target = np.fromfile(cap / "native_target.bin", dtype="<f4")
    native_current = np.fromfile(cap / "native_current.bin", dtype="<f4")
    native_xyz = np.fromfile(cap / "native_xyz.bin", dtype="<f4").reshape(-1, 3)
    if len(native_offsets) != len(checkpoint_xyz) + 1 or native_offsets[-1] != len(native_target):
        raise ValueError("native matrix dimensions are inconsistent")
    if len(native_current) != len(native_target):
        raise ValueError("native current/target lengths differ")
    vertex_offsets = np.arange(len(checkpoint_xyz) + 1, dtype=np.int64) * 3
    xyz_compare = compare(native_xyz.ravel(), checkpoint_xyz.ravel(), vertex_offsets)
    xyz = native_xyz  # SSE 求值须使用同一次 GDB 捕获的状态。

    t1 = time.perf_counter()
    offsets, ids, raw, sampling = sample_standard_metric_matrix(smoothwm, faces)
    target, matched, averaging_seconds = average_standard_metric(offsets, ids, raw)
    metric_seconds = time.perf_counter() - t1
    offset_equal = np.array_equal(native_offsets, offsets)
    row_counts = compare(np.diff(native_offsets).astype(np.int32),
                         np.diff(offsets).astype(np.int32),
                         np.arange(len(xyz) + 1, dtype=np.int64))
    result = {"input_sha256": {"smoothwm": sha256(args.smoothwm),
                                "sphere0000": sha256(args.sphere0000),
                                "native_log": sha256(args.native_log)},
              "native_capture": provenance,
              "native_initial_xyz": xyz_compare,
              "native_target_count": len(native_target),
              "python_target_count": len(target),
              "row_counts": row_counts,
              "offsets_exact": bool(offset_equal),
              "python_metric_seconds_including_jit": metric_seconds,
              "python_sampling": sampling,
              "python_reciprocal_pairs": matched,
              "python_averaging_seconds": averaging_seconds,
              "source_sha256": {
                  "metric": sha256(Path(sample_standard_metric_matrix.__code__.co_filename)),
                  "comparison": sha256(Path(__file__))},
              "native_filter_state": {
                  "target_nonfinite": int(np.count_nonzero(~np.isfinite(native_target))),
                  "target_zero": int(np.count_nonzero(native_target == 0)),
                  "target_over_10000": int(np.count_nonzero(native_target >= 10000)),
                  "current_nonfinite": int(np.count_nonzero(~np.isfinite(native_current)))},
              "native_log_terms": first_log_terms(args.native_log)}
    rip_path = cap / "native_vertex_ripflags.bin"
    if rip_path.exists():
        flag_size = provenance["layout"]["vertex_ripflag_size"]
        flags = np.fromfile(rip_path, dtype={1: "u1", 2: "<u2", 4: "<u4"}[flag_size])
        result["native_filter_state"]["ripflags_captured"] = True
        result["native_filter_state"]["ripped_vertices"] = int(np.count_nonzero(flags))
    else:
        result["native_filter_state"]["ripflags_captured"] = False
    smoothness_path = cap / "native_vsmoothness.bin"
    if smoothness_path.exists():
        size = provenance["layout"]["vsmoothness_element_size"]
        smoothness = np.fromfile(smoothness_path, dtype={4: "<f4", 8: "<f8"}[size])
        result["native_filter_state"]["vsmoothness_captured"] = True
        result["native_filter_state"]["vsmoothness_nonzero"] = int(np.count_nonzero(smoothness))
    else:
        result["native_filter_state"]["vsmoothness_captured"] = False
    if offset_equal:
        result["target"] = compare(native_target, target, offsets)
        first = result["target"]["first_mismatch"]
        if first is not None:
            vertex = first["vertex"]
            index = first["flat_index"]
            other = int(ids[index])
            reverse = [int(q) for q in range(offsets[other], offsets[other + 1])
                       if ids[q] == vertex]
            result["first_target_pair"] = {
                "vertex": vertex, "neighbor": other,
                "row_position": first["row_position"],
                "python_before_reciprocal_average": float(raw[index]),
                "python_after_reciprocal_average": float(target[index]),
                "native_after_reciprocal_average": float(native_target[index]),
                "reciprocal": [{
                    "row_position": q - int(offsets[other]),
                    "python_before_reciprocal_average": float(raw[q]),
                    "python_after_reciprocal_average": float(target[q]),
                    "native_after_reciprocal_average": float(native_target[q])
                } for q in reverse],
            }
        native_ids_path = cap / "native_neighbor_ids.bin"
        if native_ids_path.exists():
            native_ids = np.fromfile(native_ids_path, dtype="<i4")
            result["neighbor_ids"] = compare(native_ids, ids, offsets)
            if len(native_ids) != len(native_target) or np.any(native_ids < 0) or np.any(native_ids >= len(xyz)):
                raise ValueError("native neighbor IDs are invalid")
            current_ids = native_ids
        else:
            result["neighbor_ids"] = {"captured": False}
            current_ids = ids
        t2 = time.perf_counter()
        predicted_current = current_distances(xyz, offsets, current_ids)
        python_current = (predicted_current if np.array_equal(current_ids, ids)
                          else current_distances(xyz, offsets, ids))
        result["python_current_seconds_including_jit"] = time.perf_counter() - t2
        result["current_on_native_pairs"] = compare(native_current, predicted_current, offsets)
        terms = result["native_log_terms"]
        numerator = np.float32(terms["mris->orig_area"])
        denominator = np.float32(np.float32(terms["mris->total_area"])
                                 - np.float32(terms["mris->neg_area"]))
        scale = math.sqrt(float(np.float32(numerator / denominator)))
        weight = float(np.float32(1e-6))
        result["native_arrays_weighted_distance_sse"] = weighted_distance_sse(
            native_current, native_target, native_offsets, scale, weight)
        result["python_arrays_weighted_distance_sse"] = weighted_distance_sse(
            python_current, target, offsets, scale, weight)
        result["distance_scale"] = scale
        result["distance_weight"] = weight
    result["total_seconds_including_jit"] = time.perf_counter() - t0
    args.report_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "offsets_exact": result["offsets_exact"],
        "target": result.get("target"),
        "neighbor_ids": result.get("neighbor_ids"),
        "current_on_native_pairs": result.get("current_on_native_pairs"),
        "native_arrays_weighted_distance_sse": result.get("native_arrays_weighted_distance_sse"),
        "python_arrays_weighted_distance_sse": result.get("python_arrays_weighted_distance_sse"),
        "total_seconds_including_jit": result["total_seconds_including_jit"]}))


if __name__ == "__main__":
    main()
