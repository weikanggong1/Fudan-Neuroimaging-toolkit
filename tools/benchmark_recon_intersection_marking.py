"""真实非零修复的旧/新source标记器完整API ABBA；参考仅事后比较。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from benchmark_recon_topology_candidate_state import load_state_checkpoint
from benchmark_recon_topology_nonzero_csr import array_digest, digest, observe_cleanup, triangle_layout


@contextmanager
def observe_predicate_sequences():
    """每个实际有向源谓词调用只转发一次，绑定完整输入/输出调用序列。

    SHA按实际顺序覆盖shape、所有F32坐标及每个bool判定。额外哈希墙钟
    单列，计入完整API；不额外运行predicate，不改候选或数值阈值。
    """
    from fnit.recon_all import place_surface_collision_torch as collision
    original = collision._source_pairs
    records = dict(calls=0, pairs=0, observation_seconds=0.)
    inputs, outputs = hashlib.sha256(), hashlib.sha256()
    def call(first, second):
        tick = time.perf_counter()
        inputs.update(np.asarray(first.shape, dtype="<i8").tobytes())
        inputs.update(np.ascontiguousarray(first).tobytes())
        inputs.update(np.ascontiguousarray(second).tobytes())
        records["observation_seconds"] += time.perf_counter()-tick
        result = original(first, second)
        tick = time.perf_counter()
        outputs.update(np.asarray(result.shape, dtype="<i8").tobytes())
        outputs.update(np.ascontiguousarray(result).tobytes())
        records["calls"] += 1
        records["pairs"] += len(first)
        records["observation_seconds"] += time.perf_counter()-tick
        return result
    collision._source_pairs = call
    try:
        yield records
    finally:
        collision._source_pairs = original
        records["input_sequence_sha256"] = inputs.hexdigest()
        records["output_sequence_sha256"] = outputs.hexdigest()


def run(*, input_surface: Path, compact_state: Path, baseline_marker: Path,
        source_manifest: Path, output: Path, threads: int = 4) -> dict:
    """同输入/同CSR/同CPU预算完成baseline/candidate/candidate/baseline。

    input_surface为本轮自产非零三角面，compact_state为同一compact的完整
    ORIGINAL/有序CSR。baseline_marker为未改的既有FNIT Python源文件，
    仅诊断加载；不是原软件程序。各调用均完整100SOAP和独立双复检。
    output须新目录；CUDA隐藏，错误不回退。报告不是整例recon-all速度。
    """
    start = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    if threads < 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("positive threads and hidden CUDA are required")
    output.mkdir(mode=0o700, parents=True)
    import nibabel.freesurfer.io as fsio
    import numba
    from fnit.recon_all import place_surface_intersection_marking as marker
    from fnit.recon_all.mris_remove_intersection_python import run_remove_intersection
    numba.set_num_threads(threads)
    tick = time.perf_counter()
    state = load_state_checkpoint(compact_state)
    xyz, faces, header = fsio.read_geometry(input_surface, read_metadata=True)
    xyz, faces = np.float32(xyz), np.int32(faces)
    count = len(xyz)
    if not np.array_equal(xyz, state["original_vertices"]) or not np.array_equal(faces, state["faces"]):
        raise ValueError("surface and compact ORIGINAL/faces differ")
    rows = [np.int32(row)[:int(topology[3])] for row, topology in
            zip(state["neighbor_storage"], state["vertex_topology"])]
    offsets = np.r_[np.int32(0), np.cumsum([len(row) for row in rows], dtype=np.int32)]
    adjacent = np.concatenate(rows).astype(np.int32, copy=False)
    input_bytes, coordinate_start, coordinate_end, face_end = triangle_layout(input_surface)
    loading_seconds = time.perf_counter()-tick
    spec = importlib.util.spec_from_file_location("fnit.recon_all._frozen_source_marker", baseline_marker)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    functions = dict(baseline=baseline.mark_source_intersections, candidate=marker.mark_source_intersections)
    original = marker.mark_source_intersections
    calls = []
    try:
        for index, variant in enumerate(("baseline", "candidate", "candidate", "baseline")):
            marker.mark_source_intersections = functions[variant]
            destination = output/f"{index}-{variant}.surface"
            tick = time.perf_counter()
            with observe_cleanup() as observed, observe_predicate_sequences() as predicates:
                api = run_remove_intersection(input_path=input_surface, output_path=destination,
                    marking_backend="source_numba", require_zero=True,
                    ordered_neighbor_offsets=offsets, ordered_neighbors=adjacent,
                    report_path=output/f"{index}-{variant}.api.private.json")
            api_seconds = time.perf_counter()-tick
            out_xyz, out_faces, out_header = fsio.read_geometry(destination, read_metadata=True)
            out_xyz = np.float32(out_xyz)
            out_bytes, out_start, out_end, out_face_end = triangle_layout(destination)
            structural = dict(ordered_faces_exact=np.array_equal(faces, out_faces),
                header_exact=all(np.array_equal(header[k], out_header[k]) for k in header),
                noncoordinate_bytes_exact=(input_bytes[:coordinate_start]+input_bytes[coordinate_end:] ==
                                           out_bytes[:out_start]+out_bytes[out_end:]))
            trajectory = {key: [{k:v for k,v in event.items() if k != "seconds"} for event in observed[key]]
                          for key in ("source_marks", "geometric_marks", "soap_calls")}
            calls.append(dict(variant=variant, api_seconds=api_seconds,
                read_seconds=api["read_seconds"], compute_seconds=api["compute_seconds"],
                recheck_seconds=api["recheck_seconds"], write_seconds=api["write_seconds"],
                cleanup=api["cleanup"], source_after=api["source_intersecting_faces_after_independent_recheck"],
                geometric_after=api["geometric_intersecting_faces_after_independent_recheck"],
                predicates=predicates, structural=structural, trajectory=trajectory,
                output_sha256=digest(destination), coordinates_sha256=array_digest(out_xyz)))
            np.savez_compressed(output/f"{index}-{variant}.terminal.private.npz", vertices=out_xyz)
            (output/f"progress-{index}.json").write_text(json.dumps(calls[-1], indent=2)+"\n")
    finally:
        marker.mark_source_intersections = original
    reference = calls[0]
    strict = dict(
        ordered_predicate_inputs=all(c["predicates"]["input_sequence_sha256"] ==
                                    reference["predicates"]["input_sequence_sha256"] for c in calls),
        ordered_predicate_outputs=all(c["predicates"]["output_sequence_sha256"] ==
                                     reference["predicates"]["output_sequence_sha256"] for c in calls),
        coordinates=all(c["coordinates_sha256"] == reference["coordinates_sha256"] for c in calls),
        trajectories=all(c["trajectory"] == reference["trajectory"] for c in calls),
        whole_output_file=all(c["output_sha256"] == reference["output_sha256"] for c in calls),
        structural=all(all(c["structural"].values()) for c in calls),
        source_and_geometry_zero=all(c["source_after"] == c["geometric_after"] == 0 for c in calls),
        full_nonzero_cleanup=all(c["cleanup"]["intersecting_faces_before"] > 0 and
                                c["cleanup"]["smoothing_iterations"] == 100 for c in calls),
    )
    medians = {variant: float(np.median([c["api_seconds"] for c in calls if c["variant"] == variant]))
               for variant in functions}
    status = "passed_complete_API_ABBA" if all(strict.values()) else "failed_strict_comparison"
    result = dict(status=status, strict=strict, vertices=count, faces=len(faces),
        ordered_neighbors=len(adjacent), threads=threads, CPU_affinity=sorted(os.sched_getaffinity(0)),
        gpu_used=False, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        input_file_sha256=digest(input_surface), compact_sha256=digest(compact_state),
        baseline_marker_sha256=digest(baseline_marker),
        candidate_marker_sha256=digest(Path(marker.__file__)), source_manifest_sha256=digest(source_manifest),
        tool_sha256=digest(Path(__file__)), loading_seconds=loading_seconds, calls=calls,
        API_medians_seconds=medians, API_speed_ratio=medians["baseline"]/medians["candidate"],
        wall_seconds=time.perf_counter()-start, original_program_reference="not_assessed",
        whole_pipeline_equivalence="not_assessed", speed_scope="complete_intersection_file_API_only")
    (output/"report.private.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (output/("COMPLETE.json" if all(strict.values()) else "FAILED.json")).write_text(
        json.dumps(dict(status=status, strict=strict))+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input_surface", "compact_state", "baseline_marker", "source_manifest", "output"):
        parser.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps(dict(status=result["status"], strict=result["strict"],
                         API_medians_seconds=result["API_medians_seconds"])), flush=True)
    raise SystemExit(0 if result["status"] == "passed_complete_API_ABBA" else 2)


if __name__ == "__main__":
    main()
