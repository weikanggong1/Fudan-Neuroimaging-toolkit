"""同输入准备态及评分交叉诊断；参考只能读入此隔离 benchmark。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import nibabel as nib
import numpy as np
from fnit.recon_all.topology_edge_mri_score import source_mri_edge_scores


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return dict(shape_equal=False, left_shape=list(left.shape), right_shape=list(right.shape))
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return dict(shape_equal=True, different_elements=int(np.count_nonzero(difference)),
                maximum_absolute_error=float(difference.max(initial=0)),
                p99_absolute_error=float(np.percentile(difference, 99)) if difference.size else 0.,
                mean_absolute_error=float(difference.mean()) if difference.size else 0.)


def compare(*, candidate_checkpoint, native_preparation, native_edges, intensity_volume, output_report,
            active_topology_matrix=None):
    """原准备态和候选 checkpoint→诊断 JSON，不写表面或更新算法输入。

    所有路径都是独立冻结真实输入/记录；candidate_checkpoint 的原数据先
    独立计算完成。native_preparation 为本页观察器的完整二进制记录，
    native_edges 为同程序完整 first EDGE.len 表，intensity_volume 为其
    同输入 uint8 MRI。仅诊断将冻结字段逐组交换，分别报告来源，不能
    进入生产链，也不据此修改候选。输出含全部 artifact SHA、各字段
    误差及交叉评分；形状/数据类型/记录字节不符抛 ValueError。
    """
    started = time.perf_counter()
    with open(native_preparation, "rb") as stream:
        nvertices, nfaces, width, height, depth = np.fromfile(stream, "<i4", 5)
        matrix = np.fromfile(stream, "<f4", 16).reshape(4, 4)
        fields = np.fromfile(stream, "<f4", 16 * nvertices).reshape(nvertices, 16)
        faces = np.fromfile(stream, "<i4", 3 * nfaces).reshape(nfaces, 3)
        if stream.read(1):
            raise ValueError("unexpected bytes in source preparation record")
    fresh_matrix = matrix.copy()
    if active_topology_matrix is not None:
        entries = np.fromfile(active_topology_matrix, "<f4")
        if entries.size != 16 or not np.isfinite(entries).all():
            raise ValueError("actual topology matrix requires exactly 16 finite float32 entries")
        matrix = entries.reshape(4, 4)
    volume = np.ascontiguousarray(np.asarray(nib.load(str(intensity_volume)).dataobj))
    if volume.shape != (width, height, depth) or volume.dtype != np.uint8:
        raise ValueError("same uint8 MRI dimensions required")
    own = np.load(candidate_checkpoint)
    expected_fields = dict(original_vertices=own["smoothed_original"], original_normals=own["original_normals"],
                           white_values=own["white_values"], gray_values=own["gray_values"], ras_to_voxel=own["matrix"])
    native_fields = dict(original_vertices=fields[:, :3], original_normals=fields[:, 3:6],
                         white_values=fields[:, 6], gray_values=fields[:, 7], ras_to_voxel=matrix)
    reference_dtype = np.dtype([("a", "<i4"), ("b", "<i4"), ("score", "<f4"), ("used", "<i2"), ("padding", "<i2")])
    reference = np.fromfile(native_edges, reference_dtype)
    own_table = own["candidate_edges"]
    table_equal = np.array_equal(np.column_stack((reference["a"], reference["b"], reference["used"])), own_table)
    if not table_equal:
        raise ValueError("complete edge table/flags changed; cross scoring forbidden")
    arguments = dict(edge_vertices=own["source_edges"], original_edge_present=own_table[:, 2] != 0, volume=volume)
    combinations = {"own_all": expected_fields, "source_all_frozen_inputs": native_fields}
    for name in expected_fields:
        fields_one = dict(expected_fields)
        fields_one[name] = native_fields[name]
        combinations["source_only_" + name] = fields_one
    scores = {}
    score_seconds = {}
    for name, parameters in combinations.items():
        tick = time.perf_counter()
        scores[name] = source_mri_edge_scores(**arguments, **parameters)
        score_seconds[name] = time.perf_counter() - tick
    report = dict(status="complete_isolated_preparation_diagnosis", production_changed=False,
                  full_ga_completed=False, overall_metric_equivalence="not_assessed",
                  scope="same ordered first edge table; source-field swaps are benchmark diagnostics only",
                  artifact_sha256={name: sha(path) for name, path in dict(candidate_checkpoint=candidate_checkpoint,
                      native_preparation=native_preparation, native_edges=native_edges, intensity_volume=intensity_volume).items()},
                  script_sha256=sha(__file__), module_sha256=sha(Path(source_mri_edge_scores.__code__.co_filename)),
                  ordered_faces_equal=bool(np.array_equal(own["faces"], faces)),
                  nvertices=int(nvertices), nfaces=int(nfaces), complete_edge_table_equal=True,
                  source_field_errors={name: error(expected_fields[name], native_fields[name]) for name in expected_fields},
                  marked_defect_vertex_errors=error(own["labels"] != 0, fields[:, 14] != 0),
                  source_ripped_vertices=int(np.count_nonzero(fields[:, 15])),
                  source_matrix=matrix.tolist(), candidate_matrix=own["matrix"].tolist(),
                  fresh_inactive_matrix=fresh_matrix.tolist(),
                  active_matrix_sha256=sha(active_topology_matrix) if active_topology_matrix is not None else None,
                  matrix_scope="actual MATRIX_ALLOCATION" if active_topology_matrix is not None else "inactive fresh surface map; diagnostic only",
                  scores_vs_native={name: error(value, reference["score"]) for name, value in scores.items()},
                  score_seconds=score_seconds, complete_seconds=time.perf_counter() - started)
    output_report = Path(output_report)
    if output_report.exists():
        raise FileExistsError("new diagnosis report required")
    output_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["candidate-checkpoint", "native-preparation", "native-edges", "intensity-volume", "output-report"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--active-topology-matrix", type=Path)
    compare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
