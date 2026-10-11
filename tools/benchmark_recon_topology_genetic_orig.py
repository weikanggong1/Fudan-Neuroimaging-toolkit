"""验证完整自产GA回执后复用标准remesh及最终ORIGINAL清理，不运行原程序。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.compare_subject import _topology
from fnit.recon_all.mris_remesh_python import remesh_surface, _footer_offset
from fnit.recon_all.mris_remove_intersection_python import run_remove_intersection
from fnit.recon_all.thread_budget import thread_budget
from fnit.recon_all.topology_volume_python import _nonmanifold_vertex_links


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")


def _footer(path):
    # Reuse the existing surface parser; face/vertex counts change in remesh.
    with Path(path).open("rb") as stream:
        stream.seek(_footer_offset(path))
        return stream.read()


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def _geometry_quality(geometry):
    """nibabel体积头dict→可用scanner/surface RAS几何诊断，不改头字节。

    valid须为1，volume为正整数(3,)，voxelsize为正有限毫米(3,)；
    xras/yras/zras及cras为有限(3,)，方向矩阵可逆。不要求文本或
    浮点字节与某个官方文件相同；空头/无效头返回failed诊断。
    """
    result = dict(present=bool(geometry), valid=False)
    try:
        valid = np.asarray(geometry["valid"])
        volume = np.asarray(geometry["volume"])
        spacing = np.asarray(geometry["voxelsize"], np.float64)
        center = np.asarray(geometry["cras"], np.float64)
        axes = [np.asarray(geometry[name], np.float64)
                for name in ("xras", "yras", "zras")]
        valid_value = valid.item() if valid.ndim == 0 else None
        # Nibabel preserves source comments, e.g. '1  # volume info valid'.
        if isinstance(valid_value, (str, bytes)):
            valid_value = valid_value.split()[0]
        if (valid.ndim != 0 or float(valid_value) != 1 or
                volume.shape != (3,) or volume.dtype.kind not in "iu" or
                np.any(volume <= 0) or spacing.shape != (3,) or
                not np.isfinite(spacing).all() or np.any(spacing <= 0) or
                center.shape != (3,) or not np.isfinite(center).all() or
                any(axis.shape != (3,) or not np.isfinite(axis).all() for axis in axes)):
            return result
        determinant = float(np.linalg.det(np.column_stack(axes)))
        result["direction_determinant"] = determinant
        result["valid"] = bool(np.isfinite(determinant) and determinant != 0)
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        pass
    return result


def verified_premesh(*, premesh_directory, hemisphere):
    """完整自产premesh目录/侧别→经回执及文件SHA校验的路径和报告。

    两项必填：目录含report.json、COMPLETE.json与H.orig.premesh；侧别
    仅lh/rh。必须完整GA且初次生产输入均自产、无参考、无原版程序调用，
    premesh质量通过；COMPLETE的报告SHA及全部声明输出逐一核对，必须
    包含表面、完整末态及映射文件；三个输入与实际模块SHA不可缺失。
    不要求ORIGINAL premesh零交，不接收官方结果替代。返回Path、dict；
    缺文件抛FileNotFoundError、格式或完成语义错误抛ValueError。
    不改文件、不加载表面、不运行原程序；不是原软件独立CLI步骤。
    """
    if hemisphere not in ("lh", "rh"):
        raise ValueError("explicit lh or rh required")
    directory = Path(premesh_directory)
    report_path = directory / "report.json"
    complete_path = directory / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    if (complete.get("status") != "complete_premesh_stage" or
            complete.get("report_sha256") != sha(report_path)):
        raise ValueError("complete premesh receipt and exact report SHA required")
    report = json.loads(report_path.read_text())
    expected = {"status": "complete_premesh_stage", "execution_status": "complete",
                "complete_ga": True, "all_algorithm_inputs_self_generated": True,
                "reference_inputs_used": False, "native_program_called": False}
    if any(report.get(key) != value or
           isinstance(value, bool) and not isinstance(report.get(key), bool)
           for key, value in expected.items()):
        raise ValueError("complete self-generated genetic premesh report required")
    if report.get("quality", {}).get("status") != "passed":
        raise ValueError("passed premesh structure and CURRENT cleanup required")
    inputs = report.get("input_sha256")
    if (not isinstance(inputs, dict) or
            not {"nofix", "qsphere", "brain"}.issubset(inputs) or
            not all(_is_sha256(value) for value in inputs.values())):
        raise ValueError("complete original input SHA-256 provenance required")
    modules = report.get("actual_module_sha256")
    if (not isinstance(modules, dict) or
            "fnit.recon_all.topology_genetic_repair" not in modules or
            not all(isinstance(name, str) and _is_sha256(value)
                    for name, value in modules.items())):
        raise ValueError("complete actual module SHA-256 provenance required")
    surface = directory / f"{hemisphere}.orig.premesh"
    outputs = report.get("output_sha256")
    required = {surface.name, "premesh_state.private.npz", "mappings.private.npz"}
    if not isinstance(outputs, dict) or not required.issubset(outputs):
        raise ValueError("complete declared premesh outputs required")
    for name, expected_sha in outputs.items():
        if (not isinstance(name, str) or name in ("", ".", "..") or
                Path(name).name != name or not _is_sha256(expected_sha)):
            raise ValueError("relative output filename and SHA-256 required")
        if expected_sha != sha(directory / name):
            raise ValueError("exact completed premesh output surface SHA required: " + name)
    return surface, report


def final_quality(*, surface_path, cleanup_report):
    """实际写出orig及同几何cleanup报告→严格结构/双相交/头检查摘要。

    必填表面Path及刚完成run_remove_intersection的dict；表面SHA必须
    匹配cleanup输出。复用成熟闭合单连通Euler2及非流形vertex-link核，
    补查有向边、有限坐标、零面积；双相交数来自同一SHA几何的已完成
    独立复检，避免再次计算相同全网格判定。无阈值参数，无近似几何。
    返回dict含status、计数与坐标单位；严格结构失败返回failed，报告
    不同版本/缺字段抛ValueError。单位surface RAS/mm；不改文件/算法。
    """
    if (cleanup_report.get("execution_status") != "complete" or
            cleanup_report.get("output_written") is not True or
            cleanup_report.get("require_zero") is not True or
            cleanup_report.get("output_sha256") != sha(surface_path)):
        raise ValueError("exact successful cleanup output and strict report required")
    source_count = cleanup_report["source_intersecting_faces_after_independent_recheck"]
    geometry_count = cleanup_report["geometric_intersecting_faces_after_independent_recheck"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in (source_count, geometry_count)):
        raise ValueError("nonnegative integer independent intersection counts required")
    xyz, faces, geometry = fsio.read_geometry(str(surface_path), read_metadata=True)
    faces = np.ascontiguousarray(faces, np.int32)
    structural = _topology(xyz, faces)
    if "vertices" not in structural:
        return dict(status="failed", topology=structural)
    links = int(_nonmanifold_vertex_links(faces, len(xyz)))
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, group = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    conflicts = int(np.count_nonzero(np.bincount(group, weights=signs)))
    triangles = xyz[faces]
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1)
    zeros = int(np.count_nonzero(area2 == 0))
    finite = bool(np.isfinite(xyz).all() and np.isfinite(area2).all())
    geometry_check = _geometry_quality(geometry)
    passed = bool(structural["valid"] and geometry_check["valid"] and finite and links == conflicts == zeros == 0
                  and source_count == geometry_count == 0)
    return dict(status="passed" if passed else "failed", topology=structural,
                nonmanifold_vertices=links, edge_orientation_conflicts=conflicts,
                zero_area_faces=zeros, finite_coordinates=finite,
                source_intersecting_faces=source_count,
                geometric_intersecting_faces=geometry_count,
                intersection_counts_bound_to_output_sha256=cleanup_report["output_sha256"],
                coordinate_space="surface RAS", coordinate_unit="mm",
                geometry_footer_present=bool(geometry), geometry=geometry_check)


def run(*, premesh_directory, hemisphere, output, threads=4,
        remesh_scalar_storage="numpy"):
    """已完成自产完整GA目录→标准3轮remesh、最终清理和orig质量回执。

    前三项必填Path/侧别；output必须不存在。threads默认4正整数，复用
    thread_budget；remesh_scalar_storage默认numpy，也可python，保持
    同一既有双精度有序算法，不改轮数/停止/精度。surface RAS/mm；
    输出H.orig、完整各阶段报告/源SHA及COMPLETE.json。API秒含校验、
    加载、JIT、计算及读写，不等于原始T1整例；allocated/reserved与
    GPU显存未使用记null。无参考路径、原版程序或可执行目录参数。
    已存在目录报FileExistsError；校验、子函数或质量失败保存failure
    并抛异常，失败不能生成完成回执。输入保持不变。返回JSON dict。
    对应mris_remesh --remesh --iters 3及mris_remove_intersection。
    """
    started = time.perf_counter()
    output = Path(output)
    if output.exists():
        raise FileExistsError("fresh final orig diagnostic directory required")
    output.mkdir(parents=True)
    stages = {}
    try:
        tick = time.perf_counter()
        premesh, parent = verified_premesh(premesh_directory=premesh_directory,
                                          hemisphere=hemisphere)
        stages["validate_premesh_receipt_seconds"] = time.perf_counter() - tick
        orig = output / f"{hemisphere}.orig"
        with thread_budget(threads=threads):
            tick = time.perf_counter()
            remesh_surface(input_path=premesh, output_path=orig, iterations=3,
                           scalar_storage=remesh_scalar_storage)
            stages["standard_remesh_including_io_seconds"] = time.perf_counter() - tick
            remesh_sha = sha(orig)
            tick = time.perf_counter()
            cleanup = run_remove_intersection(input_path=orig, output_path=orig,
                marking_backend="source_numba", device=None, require_zero=True,
                report_path=output / "intersection.json")
            stages["original_cleanup_including_io_seconds"] = time.perf_counter() - tick
            tick = time.perf_counter()
            quality = final_quality(surface_path=orig, cleanup_report=cleanup)
            input_footer, output_footer = _footer(premesh), _footer(orig)
            quality["premesh_geometry_footer_bytes_preserved"] = input_footer == output_footer
            quality["premesh_footer_sha256"] = hashlib.sha256(input_footer).hexdigest()
            quality["orig_footer_sha256"] = hashlib.sha256(output_footer).hexdigest()
            if not quality["premesh_geometry_footer_bytes_preserved"]:
                quality["status"] = "failed"
            stages["final_structural_quality_seconds"] = time.perf_counter() - tick
        if quality["status"] != "passed":
            _write(output / "quality.failed.json", quality)
            raise RuntimeError("final orig failed strict mesh quality")
        report = dict(status="complete_orig_stage", native_program_called=False,
            reference_inputs_used=False, all_algorithm_inputs_self_generated=True,
            input_sha256={"premesh": sha(premesh),
                          "premesh_report": sha(Path(premesh_directory) / "report.json"),
                          "premesh_complete": sha(Path(premesh_directory) / "COMPLETE.json")},
            parent_input_sha256=parent["input_sha256"],
            parent_actual_module_sha256=parent["actual_module_sha256"],
            output_sha256={orig.name: sha(orig)}, before_cleanup_sha256=remesh_sha,
            quality=quality, cleanup=cleanup, stages=stages,
            tool_sha256=sha(__file__), actual_module_sha256={name: sha(module.__file__)
                for name, module in sorted(sys.modules.items()) if name.startswith("fnit.")
                and getattr(module, "__file__", None) and str(module.__file__).endswith(".py")},
            threads=threads, cpu_affinity=sorted(os.sched_getaffinity(0)),
            remesh_iterations=3, remesh_scalar_storage=remesh_scalar_storage,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            gpu_memory_bytes=None, torch_allocated_bytes=None, torch_reserved_bytes=None,
            production_default_changed=False, strict_source_final_reproduction="not_assessed",
            overall_metric_equivalence="not_assessed", whole_recon_all="not_assessed",
            total_seconds_before_final_report=time.perf_counter() - started)
        _write(output / "report.json", report)
        _write(output / "COMPLETE.json", dict(status="complete_orig_stage",
            report_sha256=sha(output / "report.json"),
            total_seconds_including_report=time.perf_counter() - started,
            whole_recon_all="not_assessed", overall_metric_equivalence="not_assessed"))
        return report
    except Exception as error:
        _write(output / "failure.json", dict(type=type(error).__name__, message=str(error),
            stages=stages, partial_seconds=time.perf_counter() - started,
            whole_recon_all="not_assessed"))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--premesh-directory", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--remesh-scalar-storage", choices=("numpy", "python"), default="numpy")
    report = run(**vars(parser.parse_args()))
    print(json.dumps({key: report[key] for key in ("status", "quality", "stages")}, indent=2))


if __name__ == "__main__":
    main()
