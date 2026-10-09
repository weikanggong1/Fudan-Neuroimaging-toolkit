"""绘制真实四网格 remesh 结果与完整 ABBA 墙钟，诊断图不进入重建流程。

--summary 是 summarize_remesh_scalar 的已完成 JSON，--candidate 是完整候选
结果目录；--output 指新的 PNG。nibabel 读取原有序面及 surface RAS/mm 坐标，
绘制完整三角面，不改变几何。无有效顶点对应时不画同索引误差。
图仅展示 FNIT 新旧实现一致性，不是当前官方、拓扑全质量或整例等效证明。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new figure path required")
    summary = json.loads(args.summary.read_text())
    if summary["status"] != "complete" or summary["strict_same_input_geometry"] != "passed":
        raise ValueError("complete matching geometry required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from nibabel.freesurfer.io import read_geometry

    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1, 1, 1.25))
    for index, row in enumerate(summary["existing_raw589_production_geometry"]):
        mesh = row["mesh"]
        stem = "abba_B2" if mesh == "sub07_lh" else mesh + "_B"
        coordinates, faces = read_geometry(args.candidate / stem / "surface")
        axis = figure.add_subplot(grid[index // 2, index % 2], projection="3d")
        triangles = coordinates[faces]
        normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        length = np.linalg.norm(normal, axis=1)
        brightness = np.abs(normal[:, 0])/np.maximum(length, np.finfo(float).tiny)
        shade = .38 + .40*brightness
        colors = np.column_stack((shade, shade, shade, np.ones(len(shade))))
        axis.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors="none",
                                              linewidth=0, rasterized=True))
        low, high = coordinates.min(axis=0), coordinates.max(axis=0)
        center = (low+high)/2
        radius = (high-low).max()/2
        axis.set_xlim(center[0]-radius, center[0]+radius)
        axis.set_ylim(center[1]-radius, center[1]+radius)
        axis.set_zlim(center[2]-radius, center[2]+radius)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=0, azim=180 if mesh.endswith("lh") else 0)
        axis.set_axis_off()
        axis.set_title(f"{mesh}: {row['candidate_vertices']:,} vertices\nordered geometry: exact; max {row['max_vertex_displacement_mm']:.0f} mm", fontsize=10)
    axis = figure.add_subplot(grid[:, 2])
    abba = [row for row in summary["runs"] if row["name"].startswith("abba_")]
    names = [row["name"].removeprefix("abba_") for row in abba]
    times = [row["outer_seconds_including_child_startup_exit"] for row in abba]
    bars = axis.bar(names, times, color=["#6e7d8f" if row["storage"] == "numpy" else "#27846e" for row in abba])
    for bar, value in zip(bars, times):
        axis.text(bar.get_x()+bar.get_width()/2, value+1, f"{value:.1f}", ha="center", fontsize=10)
    axis.set_ylim(0, max(times)*1.15)
    axis.set_ylabel("Complete child process wall time (seconds)")
    axis.set_title("Same-input sub07 LH; warm JIT cache\nA: NumPy scalar / B: Python double")
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(.02, -.09, "CPU 4-thread budget; shared node\nOriginal GC policy, full I/O and startup\nStage comparison only; no new whole result", transform=axis.transAxes, fontsize=9)
    figure.suptitle("FNIT remesh: public ds000114 self-produced input, full ordered output", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
