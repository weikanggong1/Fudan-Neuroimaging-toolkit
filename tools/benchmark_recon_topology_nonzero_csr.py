"""自产 compact ORIGINAL 非零相交：完整有序 CSR 清理、自身重复与双复检。"""
from __future__ import annotations

import time
_PROCESS_START = time.perf_counter()
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import struct
import sys

import nibabel.freesurfer.io as fsio
import numpy as np

from benchmark_recon_topology_candidate_state import load_state_checkpoint
from fnit.recon_all.mris_remove_intersection_python import run_remove_intersection
from fnit.recon_all.thread_budget import thread_budget

_IMPORT_SECONDS = time.perf_counter() - _PROCESS_START


def digest(path):
    """返回已有文件全部字节的 SHA-256；缺文件/读失败原样传播。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(values):
    """返回数组原 dtype、原序 C 字节 SHA；不转换坐标或改变数值。"""
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def triangle_layout(path):
    """三角文件→完整字节及坐标/面末位置，不转换几何头文本。

    path必填已存在三角表面；返回(raw,start,end_coordinates,end_faces)。
    counts为源big-endian整数；截断、负数量或magic不符ValueError。
    只用于诊断输入构造与字节合同，不把原nofix坐标/面作为修复输入。
    """
    raw = Path(path).read_bytes()
    with Path(path).open("rb") as stream:
        if stream.read(3) != b"\xff\xff\xfe":
            raise ValueError("triangle surface required")
        stream.readline()
        stream.readline()
        counts = stream.read(8)
        if len(counts) != 8:
            raise ValueError("truncated triangle counts")
        n, nf = struct.unpack(">ii", counts)
        start = stream.tell()
    if min(n, nf) < 0 or start + 12*n + 12*nf > len(raw):
        raise ValueError("truncated triangle arrays")
    return raw, start, start + 12*n, start + 12*n + 12*nf


@contextmanager
def observe_cleanup():
    """透明记录既有完整标记/SOAP 调用，退出时恢复原函数。

    无输入，yield 可变 dict，含每次 source/F64 标记的坐标 SHA、面数、
    顶点数及秒，每次完整 SOAP 的首尾 SHA、100 轮/冻结标记及秒。
    last_vertices 是末次真实有向标记输入的 float32(N,3) 副本，用于严格
    文件 API 拒绝写出后保留私有末态。每次原函数仅调用一次；无额外
    marker、无新随机数、无候选裁剪，不修改算法参数或全局 GPU 策略。
    哈希/副本属于观察开销，计入工具墙钟，单独报告；不当作生产计时。
    任一异常传播，finally 恢复模块函数。本工具不允许并发线程调用。
    """
    marker = importlib.import_module("fnit.recon_all.place_surface_intersection_marking")
    geometry = importlib.import_module("fnit.recon_all.mris_remove_intersection_python")
    soap = importlib.import_module("fnit.recon_all.topology_finish_geometry")
    original_source = marker.mark_source_intersections
    original_geometry = geometry.mark_intersections
    original_soap = soap.soap_bubble_vertex_positions
    record = dict(source_marks=[], geometric_marks=[], soap_calls=[], observation_seconds=0.)

    def source_call(*args, **kwargs):
        values = kwargs["vertices"] if "vertices" in kwargs else args[0]
        tick = time.perf_counter()
        result = original_source(*args, **kwargs)
        elapsed = time.perf_counter() - tick
        tick = time.perf_counter()
        record["source_marks"].append(dict(coordinate_sha256=array_digest(values),
            intersecting_faces=int(result[1]), marked_vertices=int(np.count_nonzero(result[0])),
            seconds=elapsed))
        record["last_vertices"] = np.asarray(values, dtype=np.float32).copy()
        record["last_source_marks"] = np.asarray(result[0]).copy()
        record["observation_seconds"] += time.perf_counter() - tick
        return result

    def geometry_call(*args, **kwargs):
        values = kwargs["vertices"] if "vertices" in kwargs else args[0]
        tick = time.perf_counter()
        result = original_geometry(*args, **kwargs)
        elapsed = time.perf_counter() - tick
        tick = time.perf_counter()
        record["geometric_marks"].append(dict(coordinate_sha256=array_digest(values),
            intersecting_faces=int(result[1]), marked_vertices=int(np.count_nonzero(result[0])),
            seconds=elapsed))
        record["last_geometric_marks"] = np.asarray(result[0]).copy()
        record["observation_seconds"] += time.perf_counter() - tick
        return result

    def soap_call(*args, **kwargs):
        tick = time.perf_counter()
        before = array_digest(kwargs["vertices"])
        record["observation_seconds"] += time.perf_counter() - tick
        tick = time.perf_counter()
        result = original_soap(*args, **kwargs)
        elapsed = time.perf_counter() - tick
        tick = time.perf_counter()
        record["soap_calls"].append(dict(before_coordinate_sha256=before,
            after_coordinate_sha256=array_digest(result["vertices"]),
            iterations=int(kwargs["iterations"]),
            moving_vertices=int(np.count_nonzero(
                (np.asarray(kwargs["vertex_marks"]) != 1) & ~np.asarray(kwargs["vertex_rip"], bool))),
            seconds=elapsed))
        record["observation_seconds"] += time.perf_counter() - tick
        return result

    marker.mark_source_intersections = source_call
    geometry.mark_intersections = geometry_call
    soap.soap_bubble_vertex_positions = soap_call
    try:
        yield record
    finally:
        marker.mark_source_intersections = original_source
        geometry.mark_intersections = original_geometry
        soap.soap_bubble_vertex_positions = original_soap


def run(*, compact_state, input_surface, source_manifest, output, threads=4, repeats=2):
    """完整自产 ORIGINAL 非零输入→两次严格文件 API→全坐标/轨迹重复门。

    compact_state 必填无 pickle NPZ：五坐标槽、logical faces、vnum 与完整
    ordered neighbor_storage；选择 ORIGINAL surface RAS/mm，不改 CURRENT。
    input_surface 必填同源 nofix，仅读取几何头；source_manifest 必填冻结
    源码清单，绑定实际字节；output 必须不存在。threads 默认4正整数，
    repeats 默认2且不得少于2。GPU必须已隐藏；不调用原软件阶段程序。
    每次从相同新写输入运行 source_numba+100 SOAP/周期、停滞/最佳恢复，
    require_zero=True 不变。严格失败保存 JSON 与私有末态，不写替代网格。
    返回 JSON dict，含完整 API/工具/导入/加载/读写/观察秒、峰 RSS KiB、
    源与独立几何残余、逐周期 SHA/面数、所有输入/代码 SHA、全量自身重复。
    坐标重复 max/P99 单位mm；输入→输出位移另列，不当作精度误差。
    初始零交/无100SOAP实算则本非零验证失败；工具最终失败仍写收据。
    非法输入及非残余类异常原样传播。不是完整GA或 recon-all 验收。
    """
    if output.exists():
        raise FileExistsError("fresh independent output directory required")
    if type(threads) is not int or threads < 1 or type(repeats) is not int or repeats < 2:
        raise ValueError("positive threads and at least two complete repeats required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("this CPU benchmark requires CUDA_VISIBLE_DEVICES='' before import")
    started = time.perf_counter()
    state = load_state_checkpoint(compact_state)
    _, _, geometry = fsio.read_geometry(str(input_surface), read_metadata=True)
    vertices = np.asarray(state["original_vertices"], np.float32)
    faces = np.asarray(state["faces"], np.int32)[:state["nfaces"]]
    if np.asarray(state["vertex_rip"]).any() or np.asarray(state["face_rip"])[:len(faces)].any():
        raise ValueError("nonripped compacted state required")
    rows = [np.asarray(row)[:int(top[3])] for row, top in
            zip(state["neighbor_storage"], state["vertex_topology"])]
    offsets = np.r_[0, np.cumsum([len(row) for row in rows])].astype(np.int32)
    neighbors = np.concatenate(rows).astype(np.int32)
    loading_seconds = time.perf_counter() - started
    output.mkdir(parents=True)
    source = output / "own_compact_ORIGINAL.surface"
    tick = time.perf_counter()
    fsio.write_geometry(str(source), vertices, faces,
        create_stamp="FNIT own pre-GA compact ORIGINAL independent stage diagnostic", volume_info=geometry)
    # nibabel生成新数组布局；几何尾部原字节来自同源nofix，避免文本格式化舍入。
    own_raw, _, _, own_end_faces = triangle_layout(source)
    header_raw, _, _, header_end_faces = triangle_layout(input_surface)
    source.write_bytes(own_raw[:own_end_faces] + header_raw[header_end_faces:])
    initial_write_seconds = time.perf_counter() - tick
    source_sha = digest(source)
    generated_xyz, generated_faces, generated_geometry = fsio.read_geometry(str(source), read_metadata=True)
    input_raw, input_start, input_end_coordinates, _ = triangle_layout(source)
    input_noncoordinate_bytes = input_raw[:input_start] + input_raw[input_end_coordinates:]
    generated_header_equal_original = set(generated_geometry) == set(geometry) and all(
        np.array_equal(generated_geometry[k], value) for k, value in geometry.items())
    if not (np.array_equal(np.float32(generated_xyz), vertices) and
            np.array_equal(generated_faces, faces) and generated_header_equal_original):
        raise ValueError("self-generated input coordinates/faces/footer construction failed")
    results = []
    trajectories = []
    final_vertices = []
    with thread_budget(threads=threads):
        for repetition in range(repeats):
            stem = output / ("repeat-" + str(repetition))
            report_path = stem.with_suffix(".private.json")
            file_path = stem.with_suffix(".surface")
            error_class = None
            tick = time.perf_counter()
            with observe_cleanup() as observation:
                try:
                    api_report = run_remove_intersection(input_path=source, output_path=file_path,
                        marking_backend="source_numba", device=None, candidate_grid_cells_per_axis=2,
                        require_zero=True, ordered_neighbor_offsets=offsets,
                        ordered_neighbors=neighbors, report_path=report_path)
                except RuntimeError as error:
                    if not report_path.exists():
                        raise
                    api_report = json.loads(report_path.read_text())
                    if api_report.get("execution_status") != "failed":
                        raise
                    error_class = type(error).__name__
            full_api_seconds = time.perf_counter() - tick
            final = observation.pop("last_vertices")
            source_marks = observation.pop("last_source_marks")
            geometric_marks = observation.pop("last_geometric_marks")
            final_vertices.append(final)
            np.savez_compressed(stem.with_suffix(".terminal.private.npz"), vertices=final,
                faces=faces, source_marks=source_marks, geometric_marks=geometric_marks)
            trace = dict(source_marks=observation["source_marks"],
                         geometric_marks=observation["geometric_marks"], soap_calls=observation["soap_calls"])
            trajectories.append(trace)
            item = dict(repetition=repetition, error_class=error_class,
                full_api_seconds_including_observation=full_api_seconds,
                read_seconds=api_report["read_seconds"], compute_seconds=api_report["compute_seconds"],
                recheck_seconds=api_report["recheck_seconds"], write_seconds=api_report.get("write_seconds"),
                cleanup=api_report["cleanup"],
                source_intersections_after=api_report["source_intersecting_faces_after_independent_recheck"],
                geometric_intersections_after=api_report["geometric_intersecting_faces_after_independent_recheck"],
                output_written=api_report["output_written"], output_sha256=api_report.get("output_sha256"),
                coordinate_sha256=array_digest(final), observations=observation,
                peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if file_path.exists():
                read_xyz, read_faces, read_geometry = fsio.read_geometry(str(file_path), read_metadata=True)
                out_raw, out_start, out_end_coordinates, _ = triangle_layout(file_path)
                item.update(output_coordinates_equal_terminal=array_digest(np.float32(read_xyz)) == array_digest(final),
                    ordered_faces_unchanged=bool(np.array_equal(read_faces, faces)),
                    geometry_header_unchanged=set(read_geometry) == set(generated_geometry) and all(
                        np.array_equal(read_geometry[k], value) for k, value in generated_geometry.items()),
                    all_noncoordinate_surface_bytes_unchanged=
                        (out_raw[:out_start] + out_raw[out_end_coordinates:]) == input_noncoordinate_bytes)
            results.append(item)
            (output / ("progress-" + str(repetition) + ".json")).write_text(json.dumps(item, indent=2, allow_nan=False)+"\n")
            print(json.dumps(dict(repetition=repetition, api_seconds=full_api_seconds,
                source_remaining=item["source_intersections_after"], geometry_remaining=item["geometric_intersections_after"],
                cycles=item["cleanup"]["smoothing_cycles"])), flush=True)
    repeat_coords = all(array_digest(v) == array_digest(final_vertices[0]) for v in final_vertices)
    def numeric_trace(trace):
        return {name: [{k:v for k,v in item.items() if k != "seconds"} for item in items]
                for name, items in trace.items()}
    repeat_trace = all(numeric_trace(trace) == numeric_trace(trajectories[0]) for trace in trajectories)
    difference = np.concatenate([np.abs(v.astype(np.float64)-final_vertices[0]).ravel() for v in final_vertices])
    displacement = np.linalg.norm(final_vertices[0].astype(np.float64)-vertices, axis=1)
    initial_nonzero = all(r["cleanup"]["intersecting_faces_before"] > 0 for r in results)
    soap_full = all(r["cleanup"]["smoothing_cycles"] > 0 and
                    all(call["iterations"] == 100 for call in r["observations"]["soap_calls"]) for r in results)
    zero = all(r["source_intersections_after"] == 0 and r["geometric_intersections_after"] == 0 for r in results)
    structural_contract = all(r.get("output_coordinates_equal_terminal") and
        r.get("ordered_faces_unchanged") and r.get("geometry_header_unchanged") and
        r.get("all_noncoordinate_surface_bytes_unchanged") for r in results)
    passed = initial_nonzero and soap_full and zero and repeat_coords and repeat_trace and structural_contract
    module_names = ("place_surface_final_cleanup", "place_surface_intersection_marking",
        "mris_remove_intersection_python", "topology_finish_geometry", "topology_compact")
    result = dict(status="passed_nonzero_original_CSR_stage" if passed else "failed_nonzero_original_CSR_stage",
        coordinate_slot="ORIGINAL_independent_diagnostic_only", vertices=len(vertices), faces=len(faces),
        ordered_neighbor_entries=len(neighbors), repeats=repeats, threads=threads,
        CPU_affinity=sorted(os.sched_getaffinity(0)), CPU_processor=platform.processor(),
        gpu_used=False, gpu_peak_bytes=None, torch_allocated_bytes=None, torch_reserved_bytes=None,
        initial_nonzero_confirmed=initial_nonzero, actual_full_100_Soap_cycles=soap_full,
        input_original_geometry_footer_preserved=generated_header_equal_original,
        output_structural_file_contract_passed=structural_contract,
        source_and_independent_geometric_residual_zero=zero, same_coordinate_bytes=repeat_coords,
        same_full_numeric_trajectory=repeat_trace,
        same_output_surface_bytes=all(r["output_sha256"] == results[0]["output_sha256"] for r in results)
                                  if all(r["output_written"] for r in results) else None,
        coordinate_repeat_max_mm=float(difference.max(initial=0)), coordinate_repeat_P99_mm=float(np.percentile(difference,99)),
        own_input_to_output_displacement_mean_mm=float(displacement.mean()),
        own_input_to_output_displacement_P99_mm=float(np.percentile(displacement,99)),
        own_input_to_output_displacement_max_mm=float(displacement.max(initial=0)),
        changed_vertices=int(np.count_nonzero(displacement)), input_surface_bytes_unchanged=digest(source)==source_sha,
        results=results, import_seconds=_IMPORT_SECONDS, loading_seconds=loading_seconds,
        initial_surface_write_seconds=initial_write_seconds, tool_wall_seconds=time.perf_counter()-started,
        process_wall_after_first_statement_seconds=time.perf_counter()-_PROCESS_START,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        python_version=sys.version.split()[0], numpy_version=np.__version__,
        input_sha256=dict(compact_state=digest(compact_state), input_surface=digest(input_surface),
                          source_manifest=digest(source_manifest), generated_surface=source_sha),
        ordered_CSR_sha256=dict(offsets=array_digest(offsets), neighbors=array_digest(neighbors)),
        module_sha256={name:digest(importlib.import_module("fnit.recon_all."+name).__file__) for name in module_names},
        script_sha256=digest(__file__), native_program_called=False,
        original_program_reference="not_assessed", full_ga_complete=False,
        completed_25_defects_input=False, full_mesh_quality="not_assessed", overall_metric_equivalence="not_assessed",
        production_original_premesh_preclean=False)
    (output/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (output/("COMPLETE.json" if passed else "FAILED.json")).write_text(json.dumps(dict(status=result["status"],
        report_sha256=digest(output/"report.json")), indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compact-state", "input-surface", "source-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    outcome = run(**vars(parser.parse_args()))
    sys.exit(0 if outcome["status"].startswith("passed_") else 2)
