"""从真实首个完整候选检查点绘图；仅诊断评分，不验收最终网格。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*, capture, parser, comparison, input_type_comparison, output):
    """真实有序表面与标量收据→PNG、坐标/面积诊断和版本 JSON。

    capture 是 FNITFIT1 完整候选二进制，仅用于独立 benchmark。
    parser 是项目已有 load_statistics_capture 所在脚本，动态只读复用。
    comparison/input_type_comparison 为已完成评分回归报告。output 必须
    是新目录。投影坐标为 surface RAS/mm；图中灰色为原始表面上下文，
    红色仅为本次99个候选顶点，不表示最终修复成功。失败时保留已生成
    文件并抛异常；不会加载 MRI、执行原软件或写回候选几何。
    """
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError("fresh figure directory required")
    spec = importlib.util.spec_from_file_location("fnit_fitness_capture_parser", parser)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fields = module.load_statistics_capture(capture_path=capture)
    report = json.loads(comparison.read_text())
    types = json.loads(input_type_comparison.read_text())
    if not report["strict_composition_reproduction"]:
        raise ValueError("completed strict scalar replay required")
    vertex, face = fields["vertex_fields"], fields["face_fields"]
    selected, selected_faces = fields["selected_vertices"], fields["selected_faces"]
    original, current = vertex[:, :3], vertex[:, 3:6]
    if (not np.isfinite(original).all() or not np.isfinite(current[selected]).all()
            or selected.size == 0):
        raise ValueError("finite nonempty real candidate coordinates required")
    output.mkdir(parents=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, projection, title in zip(axes[0], [(1, 2), (0, 2)],
                                     ["LH sagittal projection", "LH coronal projection"]):
        x, y = projection
        ax.scatter(original[:, x], original[:, y], s=.3, c="#a6adb4", alpha=.2,
                   rasterized=True, linewidths=0)
        ax.scatter(current[selected, x], current[selected, y], s=7, c="#cc4034",
                   linewidths=0, label=f"first candidate: {len(selected)} vertices")
        ax.set(xlabel=f"{'XYZ'[x]} surface RAS (mm)", ylabel=f"{'XYZ'[y]} surface RAS (mm)",
               title=title)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8, loc="upper right")
    terms = report["candidate"]["terms"]
    axes[1, 0].barh(["MRI", "unMRI", "joint curvature", "normal dot"],
                   [terms[name] for name in ["l_mri", "l_unmri", "l_qcurv", "l_curv"]],
                   color=["#43699b", "#64a0a8", "#bb9652", "#89719b"])
    axes[1, 0].set(xlabel="weighted log-likelihood", title="Ordered fitness components")
    axes[1, 0].text(.02, .04, f"source = FNIT = {report['candidate']['fitness']:.12f}",
                     transform=axes[1, 0].transAxes, fontsize=8)
    rows = types["rows"]
    errors = [rows["python_float"]["baseline_minus_reference"],
              rows["numpy_float32"]["baseline_minus_reference"],
              rows["numpy_float32"]["candidate_minus_reference"]]
    axes[1, 1].bar(["old Python float", "old NumPy F32", "new NumPy F32"],
                   np.array(errors) * 1e7, color=["#a6adb4", "#cc4034", "#64a0a8"])
    axes[1, 1].axhline(0, c="#444444", linewidth=.7)
    axes[1, 1].set(ylabel="fitness error × 10⁻⁷", title="Same real fields; input-container bug")
    axes[1, 1].tick_params(axis="x", labelsize=8)
    fig.suptitle("Real sub07 LH first complete candidate — diagnostic, not final mesh QC", fontsize=12)
    fig.savefig(output / "real_candidate_fitness.png", dpi=170)
    plt.close(fig)
    selected_vertex_area = vertex[selected, 9]
    selected_face_area = face[selected_faces, 3]
    metadata = {"scope": "real first complete candidate visualization; no final topology/QC claim",
        "coordinates": "surface RAS, mm", "vertices": len(vertex), "faces": len(face),
        "candidate_vertices": len(selected), "candidate_faces": len(selected_faces),
        "selected_vertex_area": {"source_order_double_sum_mm2": float(sum(map(float, selected_vertex_area))),
                                 "zero_count": int(np.count_nonzero(selected_vertex_area == 0))},
        "selected_face_area": {"source_order_double_sum_mm2": float(sum(map(float, selected_face_area))),
                               "zero_count": int(np.count_nonzero(selected_face_area == 0))},
        "files_sha256": {"capture": sha(capture), "parser": sha(parser), "comparison": sha(comparison),
                         "input_type_comparison": sha(input_type_comparison), "script": sha(__file__),
                         "real_candidate_fitness.png": sha(output / "real_candidate_fitness.png")},
        "libraries": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "seconds_including_capture_load_plot_write": time.perf_counter() - started,
        "gpu_used": False}
    (output / "figure_receipt.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["capture", "parser", "comparison", "input-type-comparison", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
