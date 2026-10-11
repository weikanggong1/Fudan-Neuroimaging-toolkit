"""冻结完整候选的零面积诊断回归；不执行GA或修改参考数据。"""
from __future__ import annotations
import time
_START = time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import numpy as np
from benchmark_recon_topology_mri_likelihood import load_statistics_capture
from benchmark_recon_topology_normal_likelihood import _arguments, _evaluate, array_sha, sha
from fnit.recon_all import topology_defect_normal_likelihood as implementation
from fnit.recon_all.thread_budget import thread_budget
_IMPORT = time.perf_counter() - _START


def scalar(value):
    value = np.float32(value)
    return {"value": float(value) if np.isfinite(value) else None,
            "defined": bool(np.isfinite(value)),
            "float32_bits": int(np.asarray(value).view(np.uint32))}


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"shape_equal": False, "bitwise_equal": False}
    same = a.dtype == b.dtype and a.tobytes() == b.tobytes()
    finite = np.isfinite(a) & np.isfinite(b)
    errors = np.abs(a.astype(np.float64)[finite] - b.astype(np.float64)[finite])
    return {"shape_equal": True, "bitwise_equal": same,
            "finite_different_elements": int(np.count_nonzero(a[finite] != b[finite])),
            "nonfinite_elements_a": int(np.count_nonzero(~np.isfinite(a))),
            "nonfinite_elements_b": int(np.count_nonzero(~np.isfinite(b))),
            "max_abs_finite": float(errors.max(initial=0)),
            "p99_abs_finite": float(np.percentile(errors, 99)) if errors.size else None}


def run(*, capture: Path, native_repeat: Path, observer_receipt: Path,
        output: Path, threads: int, snapshot_scope: str):
    """两个FNITFIT1冻结输入→完整两算子回归、有效评分与未定义诊断独立报告。

    参数均显式：capture/native_repeat为同输入参考完整状态，observer_receipt
    为该独立参考构建记录；output须不存在；threads正整数总CPU预算；
    snapshot_scope用于区分初始统计/完整候选。读取不修改影像和原始报告。
    返回写出的报告；私有NPZ保留NaN原位模式，JSON使用null+uint32位模式。
    输入损坏、核心评分非有限或契约失败抛异常，不将诊断NaN替换成0。
    """
    tick = time.perf_counter()
    if output.exists():
        raise FileExistsError("fresh output required")
    if threads < 1:
        raise ValueError("positive thread budget required")
    fields = load_statistics_capture(capture_path=capture)
    other = load_statistics_capture(capture_path=native_repeat)
    if fields["reference_score_sizes"].tolist() != [4] * 8:
        raise ValueError("pinned TP float32 field contract required")
    normal, curvature = _arguments(fields)
    with thread_budget(threads=threads):
        n, c, first_times = _evaluate(normal, curvature)
        rn, rc, repeat_times = _evaluate(normal, curvature)
    names = ["curv_ll", "cll", "qcurv_ll", "qcll"]
    candidates = np.asarray([n["curv_ll"], n["cll"], c["qcurv_ll"], c["qcll"]], np.float32)
    repeats = np.asarray([rn["curv_ll"], rn["cll"], rc["qcurv_ll"], rc["qcll"]], np.float32)
    refs = fields["reference_scores"][[2, 6, 3, 7]].astype(np.float32)
    comparisons = {name: compare(a, b) for name, a, b in zip(names, candidates, refs)}
    repeat_comparisons = {"four_source_fields": compare(candidates, repeats),
                         "normal_per_vertex": compare(n["per_vertex_log_likelihood"], rn["per_vertex_log_likelihood"]),
                         "curvature_per_vertex": compare(c["per_vertex_log_likelihood"], rc["per_vertex_log_likelihood"])}
    native = {key: compare(value, other[key]) for key, value in fields.items()}
    selected = fields["selected_vertices"]
    areas = fields["vertex_fields"][selected, 9]
    area_sum = 0.0
    for area in areas:
        area_sum += float(area)
    output.mkdir(parents=True)
    np.savez_compressed(output / "candidate.private.npz", fields=candidates,
                        normal_per_vertex=n["per_vertex_log_likelihood"],
                        curvature_per_vertex=c["per_vertex_log_likelihood"])
    report = {"status": "complete_frozen_zero_area_likelihood_regression",
        "snapshot_scope": snapshot_scope, "strict_four_source_fields_bits_passed": all(row["bitwise_equal"] for row in comparisons.values()),
        "finite_consumed_core_passed": bool(np.isfinite(candidates[[0, 2]]).all()),
        "undefined_diagnostics_source_bits_passed": all(comparisons[nm]["bitwise_equal"] for nm in ["cll", "qcll"]),
        "candidate_repeat_exact": all(row["bitwise_equal"] for row in repeat_comparisons.values()),
        "native_full_capture_repeat_exact": all(row["bitwise_equal"] for row in native.values()),
        "native_capture_sha256": sha(capture), "native_repeat_capture_sha256": sha(native_repeat),
        "native_source_receipt_sha256": sha(observer_receipt),
        "module_sha256": {"implementation": sha(implementation.__file__), "benchmark": sha(__file__),
            "parser": sha(Path(__import__('benchmark_recon_topology_mri_likelihood').__file__))},
        "candidate_scores": {name: scalar(v) for name, v in zip(names, candidates)},
        "reference_scores": {name: scalar(v) for name, v in zip(names, refs)},
        "scores": comparisons, "candidate_repeat": repeat_comparisons, "native_repeat": native,
        "input_fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "data_sha256": array_sha(value)} for key, value in fields.items()},
        "vertex_count": len(fields["vertex_fields"]), "selected_vertex_count": len(selected),
        "selected_face_count": len(fields["selected_faces"]),
        "selected_vertex_area_mm2": {"min": float(areas.min(initial=np.inf)), "max": float(areas.max(initial=-np.inf)),
            "zero_count": int(np.count_nonzero(areas == 0)), "source_ordered_double_sum": area_sum},
        "native_per_vertex_log_probability": "not captured; no native per-vertex comparison claimed",
        "candidate_npz_sha256": sha(output / "candidate.private.npz"),
        "seconds": {"first_complete_api_with_available_jit_cache": first_times,
            "repeat_complete_api": repeat_times, "imports": _IMPORT,
            "api_load_validation_repeat_io": time.perf_counter() - tick},
        "cpu_threads": threads, "cpu_affinity": sorted(os.sched_getaffinity(0)), "host": platform.node(),
        "python": platform.python_version(), "numpy": np.__version__, "gpu_used": False,
        "gpu_process_memory_bytes": None, "native_component_seconds": None,
        "full_unmri_distance_generation": "not executed", "full_ga": "not executed", "overall_equivalence": "not_assessed"}
    _plot(fields, n, c, output / "aggregate.png", snapshot_scope)
    report["figure_sha256"] = sha(output / "aggregate.png")
    report["seconds"]["tool_wall_including_imports_plot"] = time.perf_counter() - _START
    (output / "report.private.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def _plot(fields, n, c, path, scope):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xyz = fields['vertex_fields'][fields['selected_vertices'], :3]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in [(axes[0], n['per_vertex_log_likelihood'], 'Normal logP'), (axes[1], c['per_vertex_log_likelihood'], 'Curvature joint logP')]:
        obj = ax.scatter(xyz[:, 1], xyz[:, 2], c=data, s=4 if len(xyz) < 1000 else .3, cmap='viridis')
        ax.set_aspect('equal'); ax.set_xlabel('surface RAS y (mm)'); ax.set_ylabel('surface RAS z (mm)'); ax.set_title(title)
        fig.colorbar(obj, ax=ax)
    fig.suptitle(scope + '; operator diagnostic, not full GA')
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['capture', 'native-repeat', 'observer-receipt', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--snapshot-scope', required=True)
    run(**vars(p.parse_args()))
