"""法向/曲率似然的完整冻结同输入回归；不运行GA，不回写生产目录。"""
from __future__ import annotations

import time
_SCRIPT_START = time.perf_counter()

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np

from benchmark_recon_topology_mri_likelihood import load_statistics_capture
from fnit.recon_all import topology_defect_normal_likelihood as implementation
from fnit.recon_all.thread_budget import thread_budget

_IMPORT_SECONDS = time.perf_counter() - _SCRIPT_START


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    data = np.ascontiguousarray(array)
    return hashlib.sha256(data.tobytes()).hexdigest()


def difference(candidate, reference):
    a, b = np.asarray(candidate), np.asarray(reference)
    if a.shape != b.shape:
        return {"shape_equal": False, "candidate_shape": list(a.shape), "reference_shape": list(b.shape)}
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"shape_equal": True, "different_elements": int(np.count_nonzero(a != b)),
            "max_abs": float(error.max(initial=0)),
            "p99_abs": float(np.percentile(error, 99)) if error.size else 0.0}


def _arguments(fields):
    vertex = fields["vertex_fields"]
    shared = dict(vertex_areas_mm2=vertex[:, 9], patch_vertices=fields["selected_vertices"])
    def histogram(label):
        return {"bins": fields[label + "_bins"], "counts": fields[label + "_counts"],
                "bin_size": fields[label + "_bin_size"][0]}
    normal = dict(**shared, original_vertices=vertex[:, :3], vertex_normals=vertex[:, 6:9],
                  neighbor_offsets=fields["neighbor_offsets"], ordered_neighbors=fields["neighbors"],
                  normal_dot_histogram=histogram("dot"))
    curvature = dict(**shared, principal_curvature_k1_mm_inv=vertex[:, 10],
                     principal_curvature_k2_mm_inv=vertex[:, 11], k1_histogram=histogram("k1"),
                     k2_histogram=histogram("k2"), joint_curvature_pdf=fields["k_pdf"],
                     joint_origin_mm_inv=fields["k_pdf_geometry"][:2],
                     joint_spacing_mm_inv=fields["k_pdf_geometry"][2:])
    return normal, curvature


def _evaluate(normal, curvature):
    tick = time.perf_counter()
    n = implementation.patch_normal_dot_likelihood(**normal)
    normal_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    c = implementation.patch_curvature_likelihood(**curvature)
    return n, c, {"normal_complete_api": normal_seconds,
                  "curvature_complete_api": time.perf_counter() - tick}


def run(*, capture, native_repeat, observer_receipt, output, threads):
    """完整FNITFIT1检查点→4个源字段、2次重复、数组SHA与真实脑图。

    capture/native_repeat为同输入独立参考进程的完整冻结统计；
    observer_receipt为其来源收据。它们只供隔离算子benchmark。
    output须不存在；threads显式正整数并使用现有预算作用域恢复父设置。
    字段由已有完整解析器读取，不重新估计主曲率、不生成原版坐标。
    输出candidate.npz私有逐顶点诊断、report.private.json/aggregate.png；
    不修改参考/生产数据。数组损坏、来源不齐、非法PDF或几何均报错。
    原始运行报告和MRI/NPZ只保留服务器；公开导出仅数值收据/PNG。
    """
    api_start = time.perf_counter()
    if output.exists():
        raise FileExistsError("fresh isolated diagnostic output required")
    if threads < 1:
        raise ValueError("positive threads required")
    fields = load_statistics_capture(capture_path=capture)
    normal, curvature = _arguments(fields)
    if fields["reference_score_sizes"].tolist() != [4] * 8:
        raise ValueError("pinned TP float32 field layout required")
    output.mkdir(parents=True)
    with thread_budget(threads=threads):
        first_normal, first_curvature, first_seconds = _evaluate(normal, curvature)
        second_normal, second_curvature, second_seconds = _evaluate(normal, curvature)
    # Save the entire candidate first; reference scores never enter the algorithms.
    np.savez_compressed(output / "candidate.npz",
        patch_vertices=fields["selected_vertices"],
        normal_per_vertex=first_normal["per_vertex_log_likelihood"],
        curvature_per_vertex=first_curvature["per_vertex_log_likelihood"],
        fields=np.array([first_normal["curv_ll"], first_normal["cll"],
                         first_curvature["qcurv_ll"], first_curvature["qcll"]], np.float32))
    names = ["curv_ll", "cll", "qcurv_ll", "qcll"]
    indices = [2, 6, 3, 7]
    values = [first_normal["curv_ll"], first_normal["cll"], first_curvature["qcurv_ll"], first_curvature["qcll"]]
    second_values = [second_normal["curv_ll"], second_normal["cll"], second_curvature["qcurv_ll"], second_curvature["qcll"]]
    comparisons = {name: difference(value, fields["reference_scores"][index])
                   for name, value, index in zip(names, values, indices)}
    repeated = {name: difference(a, b) for name, a, b in zip(names, values, second_values)}
    repeated["normal_per_vertex"] = difference(first_normal["per_vertex_log_likelihood"], second_normal["per_vertex_log_likelihood"])
    repeated["curvature_per_vertex"] = difference(first_curvature["per_vertex_log_likelihood"], second_curvature["per_vertex_log_likelihood"])
    native_fields = load_statistics_capture(capture_path=native_repeat)
    native_comparisons = {key: difference(value, native_fields[key]) for key, value in fields.items()}
    native_exact = all(value.get("different_elements") == 0 for value in native_comparisons.values())
    candidate_exact = all(value.get("different_elements") == 0 for value in repeated.values())
    score_exact = all(value.get("different_elements") == 0 for value in comparisons.values())
    from benchmark_recon_topology_mri_likelihood import __file__ as parser_file
    source_receipt = json.loads(observer_receipt.read_text())
    report = {"status": "complete_frozen_normal_curvature_likelihood_regression",
        "strict_four_source_fields_passed": score_exact,
        "candidate_repeat_exact": candidate_exact, "native_full_capture_repeat_exact": native_exact,
        "scope": "complete initial surface-statistics operator; not retessellated patch/full fitness/GA/whole reconstruction",
        "native_source_receipt": source_receipt, "native_source_receipt_sha256": sha(observer_receipt),
        "capture_sha256": sha(capture), "native_repeat_capture_sha256": sha(native_repeat),
        "candidate_npz_sha256": sha(output / "candidate.npz"),
        "module_sha256": {"implementation": sha(implementation.__file__), "capture_parser": sha(parser_file),
                          "benchmark": sha(__file__)},
        "input_fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "data_sha256": array_sha(value)}
                         for key, value in fields.items()},
        "candidate_scores": dict(zip(names, values)),
        "reference_scores": {name: float(fields["reference_scores"][index]) for name, index in zip(names, indices)},
        "marginal_qcurv_ll_diagnostic_not_final_fitness": first_curvature["marginal_qcurv_ll"],
        "scores": comparisons, "candidate_repeat": repeated, "native_repeat": native_comparisons,
        "native_per_vertex_likelihood_reference": "not captured; not assessed",
        "vertex_count": len(fields["vertex_fields"]), "selected_vertex_count": len(fields["selected_vertices"]),
        "selected_face_count": len(fields["selected_faces"]), "neighbor_entries": len(fields["neighbors"]),
        "coordinate_space": "surface RAS millimetres; curvature mm^-1; areas mm^2",
        "seconds": {"first_including_numba_jit": first_seconds, "repeat_complete_api": second_seconds,
                    "top_level_imports": _IMPORT_SECONDS, "api_with_load_validation_and_npz_write": time.perf_counter() - api_start},
        "native_component_seconds": None,
        "native_component_timing_limitation": "observer did not separately time these two internal functions; no native speedup ratio",
        "cpu_threads": threads, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "host": platform.node(), "python": platform.python_version(), "numpy": np.__version__,
        "gpu_used": False, "gpu_process_memory_bytes": None, "overall_equivalence": "not_assessed"}
    # The visualization is diagnostic public-derived geometry, not a reconstruction.
    tick = time.perf_counter()
    _plot(fields, first_normal, first_curvature, output / "aggregate.png")
    report["seconds"]["plot_with_import_and_png_write"] = time.perf_counter() - tick
    report["figure_sha256"] = sha(output / "aggregate.png")
    report["seconds"]["script_wall_including_imports_load_jit_io_plot"] = time.perf_counter() - _SCRIPT_START
    (output / "report.private.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _plot(fields, normal, curvature, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = fields["selected_vertices"]
    xyz = fields["vertex_fields"][selected, :3]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, values, title in [(axes[0], normal["per_vertex_log_likelihood"], "Normal-dot mean logP"),
                              (axes[1], curvature["per_vertex_log_likelihood"], "Joint-curvature logP")]:
        limits = np.percentile(values, [.5, 99.5])
        plot = ax.scatter(xyz[:, 1], xyz[:, 2], c=values, s=.3, cmap="viridis", vmin=limits[0], vmax=limits[1], rasterized=True)
        ax.set_aspect("equal"); ax.set_xlabel("surface RAS y (mm)"); ax.set_ylabel("surface RAS z (mm)"); ax.set_title(title)
        fig.colorbar(plot, ax=ax, label="log probability")
    fig.suptitle("Frozen initial surface-statistics input; full GA not evaluated")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["capture", "native-repeat", "observer-receipt", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
