"""完整自产GA/orig完成后只读比较固定原生参考，保留两种坐标对应范围。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import nibabel.freesurfer.io as fsio
import numpy as np

from benchmark_recon_topology_genetic_orig import (
    _footer, _geometry_quality, final_quality, sha, verified_premesh,
)
from diagnose_topology_volume_shapes import exact_distances
from fnit.recon_all.compare_subject import _topology


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False) + "\n")


def _summary(values):
    values = np.asarray(values, np.float64)
    return dict(count=len(values), mean_mm=float(values.mean()),
                p99_mm=float(np.quantile(values, .99)), max_mm=float(values.max()),
                over_0_1_mm=int(np.count_nonzero(values > .1)))


def compare_mesh(*, candidate, reference, distance_module, output_checkpoint):
    """两份有完整同空间头的表面→同索引（若成立）及全点双向三角面距离。

    四项必填Path/模块；输入坐标为surface RAS/mm，面为有序整数(F,3)。
    借用现有完整距离函数，不裁剪候选或抽样。N和有序面不同不作同索引
    比较。头的体积/spacing/axes/cras必须完全相同；缺头/非有限坐标失败。
    checkpoint为新的私有NPZ，包含所有源点到目标三角面的float64距离；
    返回仅数值、计数及哈希。该诊断不设新的等效阈值，不修改任何表面。
    """
    if output_checkpoint.exists():
        raise FileExistsError(output_checkpoint)
    cv, cf, cg = fsio.read_geometry(str(candidate), read_metadata=True)
    rv, rf, rg = fsio.read_geometry(str(reference), read_metadata=True)
    if (not _geometry_quality(cg)["valid"] or not _geometry_quality(rg)["valid"] or
            not np.isfinite(cv).all() or not np.isfinite(rv).all()):
        raise ValueError("complete valid geometry and finite coordinates required")
    fields = ("volume", "voxelsize", "xras", "yras", "zras", "cras")
    geometry_equal = {key: bool(np.array_equal(cg[key], rg[key])) for key in fields}
    if not all(geometry_equal.values()):
        raise ValueError("same surface RAS geometry required before distance comparison")
    corresponding = cv.shape == rv.shape and np.array_equal(cf, rf)
    indexed = _summary(np.linalg.norm(cv - rv, axis=1)) if corresponding else None
    forward, forward_proof = exact_distances(cv, rv, rf, distance_module)
    reverse, reverse_proof = exact_distances(rv, cv, cf, distance_module)
    np.savez_compressed(output_checkpoint, candidate_to_reference=forward,
                        reference_to_candidate=reverse)
    return dict(candidate_sha256=sha(candidate), reference_sha256=sha(reference),
        candidate_vertices=len(cv), reference_vertices=len(rv),
        candidate_faces=len(cf), reference_faces=len(rf),
        candidate_topology=_topology(cv, cf), reference_topology=_topology(rv, rf),
        ordered_faces_equal=bool(np.array_equal(cf, rf)),
        vertex_correspondence=bool(corresponding), indexed_distance_mm=indexed,
        coordinate_elements_different=int(np.count_nonzero(cv != rv)) if corresponding else None,
        decoded_coordinates_equal=bool(corresponding and cv.tobytes() == rv.tobytes()),
        geometry_fields_equal=geometry_equal,
        footer_bytes_equal=_footer(candidate) == _footer(reference),
        file_bytes_equal=sha(candidate) == sha(reference),
        candidate_to_reference_triangle=_summary(forward),
        reference_to_candidate_triangle=_summary(reverse),
        forward_proof=forward_proof, reverse_proof=reverse_proof,
        private_distance_checkpoint_sha256=sha(output_checkpoint),
        unit="surface RAS mm", overall_metric_equivalence="not_assessed")


def compare(*, premesh_directory, orig_directory, strengthened_report,
            reference_subject, reference_benchmark, reference_runtime,
            reference_provenance, case_name, hemisphere, surface_distance_module,
            output_directory, threads=4):
    """已完成自有拓扑链→只读原生同输入网格对照；不运行原软件。

    前十一参数必填；目录含完整阶段回执和表面，reference_*仅此诊断读取。
    case_name匹配既有原生provenance；hemisphere为lh/rh。threads当前须为4，
    与现有距离实现固定worker预算一致，外层须同样设置BLAS/亲和性。
    所有自产完成/文件SHA/增强头门先核对，再读取原生历史文件。三个原始
    nofix/qsphere/brain输入SHA须与自产父报告相同；原生程序和报告用历史
    生产者SHA绑定，不以现有程序改标。双向距离完全覆盖所有顶点，不在
    不同拓扑下报告同索引误差，不调整验收阈值。新output保存JSON/COMPLETE
    和私有全距离NPZ；异常保留failure.json并原样抛出。只读参考不会进入
    生产重建。无GPU计算，null不是零显存；官方重复性/整例另外报告。
    """
    started = time.perf_counter()
    if hemisphere not in ("lh", "rh"):
        raise ValueError("hemisphere must be lh or rh")
    if isinstance(threads, bool) or threads != 4:
        raise ValueError("existing complete triangle distance requires a four-thread budget")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    try:
        premesh, parent = verified_premesh(premesh_directory=premesh_directory,
                                         hemisphere=hemisphere)
        orig_directory = Path(orig_directory)
        report_path = orig_directory / "report.json"
        orig_report = json.loads(report_path.read_text())
        complete = json.loads((orig_directory / "COMPLETE.json").read_text())
        orig = orig_directory / f"{hemisphere}.orig"
        if (complete.get("status") != "complete_orig_stage" or
                complete.get("report_sha256") != sha(report_path) or
                orig_report.get("status") != "complete_orig_stage" or
                orig_report.get("native_program_called") is not False or
                orig_report.get("reference_inputs_used") is not False or
                orig_report.get("all_algorithm_inputs_self_generated") is not True or
                orig_report.get("input_sha256", {}).get("premesh") != sha(premesh) or
                orig_report.get("input_sha256", {}).get("premesh_report") !=
                    sha(Path(premesh_directory) / "report.json") or
                orig_report.get("input_sha256", {}).get("premesh_complete") !=
                    sha(Path(premesh_directory) / "COMPLETE.json") or
                orig_report.get("parent_input_sha256") != parent["input_sha256"] or
                orig_report.get("output_sha256", {}).get(orig.name) != sha(orig)):
            raise ValueError("complete own orig and exact parent provenance required")
        quality = final_quality(surface_path=orig, cleanup_report=orig_report["cleanup"])
        stronger = json.loads(Path(strengthened_report).read_text())
        if (quality["status"] != "passed" or
                stronger.get("status") != "passed_strengthened_orig_completion_validation" or
                stronger.get("parent_report_sha256") != sha(report_path) or
                stronger.get("parent_complete_sha256") != sha(orig_directory / "COMPLETE.json") or
                stronger.get("parent_premesh_report_sha256") !=
                    sha(Path(premesh_directory) / "report.json") or
                stronger.get("actual_orig_sha256") != sha(orig) or
                stronger.get("quality", {}).get("geometry", {}).get("valid") is not True or
                _footer(premesh) != _footer(orig)):
            raise ValueError("passed strengthened completion and matching actual orig required")
        own_validation_seconds = time.perf_counter() - started
        # 完整自产候选已落盘、质量通过后，才打开只读原生参考与历史回执。
        provenance = json.loads(Path(reference_provenance).read_text())
        cases = [row for row in provenance["cases"] if row["case"] == case_name]
        if len(cases) != 1:
            raise ValueError("unique recorded reference case required")
        historical = cases[0]
        reference_surfaces = {stage: Path(reference_subject) / "surf" / f"{hemisphere}.{stage}"
                              for stage in ("orig.premesh", "orig")}
        expected_surfaces = historical.get("preflight_reference_surface_sha256")
        if (not isinstance(expected_surfaces, dict) or set(expected_surfaces) != set(reference_surfaces) or
                any(sha(path) != expected_surfaces[stage] for stage, path in reference_surfaces.items())):
            raise ValueError("reference surface SHA must match the frozen preflight observation")
        if (sha(reference_benchmark) != historical["original_benchmark_receipt_sha256"] or
                sha(reference_runtime) != historical["original_runtime_receipt_sha256"]):
            raise ValueError("exact historical reference report SHA required")
        benchmark = json.loads(Path(reference_benchmark).read_text())
        runtime = json.loads(Path(reference_runtime).read_text())
        program_sha = historical["recorded_original_program_sha256"]
        if (benchmark.get("native_program_sha256", {}).get("mris_fix_topology_fnit") != program_sha or
                runtime.get("topology_repair", {}).get("sha256") != program_sha or
                benchmark.get("execution") != "complete"):
            raise ValueError("historical reference topology program and completion required")
        reference_subject = Path(reference_subject)
        reference_inputs = {"nofix": reference_subject / "surf" / f"{hemisphere}.orig.nofix",
                            "qsphere": reference_subject / "surf" / f"{hemisphere}.qsphere.nofix",
                            "brain": reference_subject / "mri/brain.mgz"}
        reference_hashes = {key: sha(path) for key, path in reference_inputs.items()}
        if reference_hashes != {key: parent["input_sha256"][key] for key in reference_inputs}:
            raise ValueError("reference nofix/qsphere/brain must match all actual own GA inputs")
        module_path = Path(surface_distance_module)
        spec = importlib.util.spec_from_file_location("_full_genetic_mesh_distances", module_path)
        distance_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(distance_module)
        rows = {}
        for stage, candidate in (("orig.premesh", premesh), ("orig", orig)):
            tick = time.perf_counter()
            rows[stage] = compare_mesh(candidate=candidate,
                reference=reference_subject / "surf" / f"{hemisphere}.{stage}",
                distance_module=distance_module,
                output_checkpoint=output_directory / f"{stage}.distances.private.npz")
            rows[stage]["complete_read_compare_write_seconds"] = time.perf_counter() - tick
        report = dict(status="complete_readonly_genetic_mesh_comparison", case=case_name,
            hemisphere=hemisphere, own_quality=quality, stages=rows,
            scope="own complete all-defect GA, remesh and cleanup; historical same-input native reference",
            own_premesh_report_sha256=sha(Path(premesh_directory) / "report.json"),
            own_orig_report_sha256=sha(report_path), strengthened_report_sha256=sha(strengthened_report),
            reference_provenance_sha256=sha(reference_provenance),
            reference_benchmark_sha256=sha(reference_benchmark), reference_runtime_sha256=sha(reference_runtime),
            original_source_commit=provenance["frozen_original_source_commit"],
            recorded_reference_program_sha256=program_sha,
            recorded_reference_code_version=historical["recorded_code_version"],
            preflight_reference_surface_sha256=expected_surfaces,
            recorded_reference_thread_budget=benchmark.get("threads"),
            recorded_reference_cpu_affinity=benchmark.get("cpu_affinity"),
            own_complete_ga_threads=parent.get("threads"),
            timing_comparison="not_paired; diagnostic time is excluded from production",
            reference_inputs_sha256=reference_hashes, script_sha256=sha(__file__),
            distance_module_sha256=sha(module_path),
            actual_module_sha256={name: sha(module.__file__) for name, module in sorted(sys.modules.items())
                if (name.startswith("fnit.") or name in ("benchmark_recon_topology_genetic_orig",
                        "diagnose_topology_volume_shapes")) and
                getattr(module, "__file__", None) and str(module.__file__).endswith(".py")},
            threads=threads, cpu_affinity=sorted(os.sched_getaffinity(0)),
            gpu_memory_bytes=None, own_validation_seconds=own_validation_seconds,
            complete_diagnostic_wall_before_report_seconds=time.perf_counter() - started,
            strict_geometry_reproduction="passed" if all(row["decoded_coordinates_equal"] and
                row["ordered_faces_equal"] for row in rows.values()) else "failed",
            reference_full_stage_repeatability="not_assessed", new_reference_program_run=False,
            native_program_called=False, production_reference_inputs_used=False,
            overall_metric_equivalence="not_assessed", whole_recon_all="not_assessed",
            production_default_changed=False)
        _write(output_directory / "report.json", report)
        _write(output_directory / "COMPLETE.json", dict(status=report["status"],
            report_sha256=sha(output_directory / "report.json"),
            total_seconds_including_report=time.perf_counter() - started))
        return report
    except Exception as error:
        _write(output_directory / "failure.json", dict(type=type(error).__name__,
            message=str(error), partial_seconds=time.perf_counter() - started,
            whole_recon_all="not_assessed"))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("premesh-directory", "orig-directory", "strengthened-report", "reference-subject",
                 "reference-benchmark", "reference-runtime", "reference-provenance",
                 "surface-distance-module", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    compare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
