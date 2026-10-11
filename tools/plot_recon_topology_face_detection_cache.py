"""已完成同输入候选回归的真实T1叠加及基线/优化坐标差；无算法修改。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel


def run(*, brain, baseline_candidate, cached_candidate, benchmark_report, output):
    """五个具名Path→新PNG及SHA/坐标误差dict，仅用于完成后的诊断。

    brain为实际uint8(X,Y,Z) MGZ；两个NPZ含完整final_original_vertices
    float32(N,3)、final_faces整数(F,3)、final_selected_vertices整数(K,)，
    坐标surface RAS/mm。benchmark_report必须为本完整ABBA成功JSON。
    只有N/全部有序面相同才作同索引；本图不允许数据与严格零差结论
    冲突，冲突即ValueError。output必填新.png；输入不变，无默认阈值。
    返回实际图/数据/报告SHA、切片/覆盖和P99/max；不是最终网格QC。
    """
    output = Path(output)
    if output.exists() or output.suffix.lower() != ".png":
        raise ValueError("fresh PNG required")
    benchmark_report = Path(benchmark_report)
    report = json.loads(benchmark_report.read_text())
    if report["status"] != "passed" or report["order"] != ["control", "candidate", "candidate", "control"]:
        raise ValueError("complete successful stage comparison required")
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if len(report["records"]) != 4 or any(row["input_sha256"]["brain"] != sha(brain)
                                         for row in report["records"]):
        raise ValueError("brain SHA must match all four actual ABBA inputs")
    expected = [benchmark_report.parent / "0-control/candidate/A.final.private.npz",
                benchmark_report.parent / "1-candidate/candidate/A.final.private.npz"]
    if any(Path(actual).resolve() != path.resolve()
           for actual, path in zip((baseline_candidate, cached_candidate), expected)):
        raise ValueError("plot requires the actual baseline and candidate output locations of this ABBA")
    fields = ("final_original_vertices", "final_faces", "final_selected_vertices")
    values = []
    for path in (baseline_candidate, cached_candidate):
        with np.load(path, allow_pickle=False) as data:
            values.append({name: data[name].copy() for name in fields})
    old, new = values
    if any(old[key].shape != new[key].shape or old[key].dtype != new[key].dtype or
           old[key].tobytes() != new[key].tobytes() for key in fields):
        raise ValueError("actual plot data must retain full strictly corresponding stage outputs")
    image = nib.load(brain)
    volume = np.asarray(image.dataobj)
    if volume.ndim != 3 or volume.dtype != np.uint8:
        raise ValueError("same uint8 XYZ brain MRI required")
    vertices = new["final_original_vertices"]
    selected = new["final_selected_vertices"]
    faces = new["final_faces"]
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or
            not np.isfinite(vertices).all() or selected.ndim != 1 or not len(selected) or
            not np.issubdtype(selected.dtype, np.integer) or
            np.any((selected < 0) | (selected >= len(vertices))) or
            faces.ndim != 2 or faces.shape[1] != 3 or
            not np.issubdtype(faces.dtype, np.integer) or
            np.any((faces < 0) | (faces >= len(vertices)))):
        raise ValueError("finite corresponding vertices, valid faces and nonempty selected indices required")
    xyz = vertices[selected]
    matrix = source_topology_ras_to_voxel(volume_header=image.header)
    vox = xyz.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
    kz = int(np.clip(np.rint(np.median(vox[:, 2])), 0, volume.shape[2]-1))
    near = np.abs(vox[:, 2]-kz) <= 2
    difference = np.linalg.norm(vertices.astype(np.float64)-old["final_original_vertices"], axis=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(volume[:, :, kz].T, cmap="gray", origin="lower", vmin=0, vmax=140)
    axes[0].scatter(vox[near, 0], vox[near, 1], s=18, c="orangered")
    axes[0].set_title(f"Cached candidate on brain MRI (Z={kz})")
    axes[0].set_xlabel("voxel X"); axes[0].set_ylabel("voxel Y")
    axes[1].scatter(xyz[:, 1], xyz[:, 2], s=18, c=difference[selected],
                    cmap="viridis", vmin=0, vmax=.001)
    axes[1].set_title("Full indexed difference: max = 0 mm")
    axes[1].set_aspect("equal"); axes[1].set_xlabel("surface RAS Y (mm)")
    axes[1].set_ylabel("surface RAS Z (mm)")
    fig.suptitle("Frozen candidate/finalization stage; not whole GA or final surface QC", fontsize=10)
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)
    return dict(figure_sha256=sha(output), brain_sha256=sha(brain),
        baseline_candidate_sha256=sha(baseline_candidate), cached_candidate_sha256=sha(cached_candidate),
        benchmark_report_sha256=sha(benchmark_report), plot_script_sha256=sha(__file__),
        voxel_z=kz, overlay_vertices=int(near.sum()), selected_vertices=len(selected),
        full_vertices=len(vertices), max_difference_mm=float(difference.max(initial=0)),
        p99_difference_mm=float(np.percentile(difference, 99)),
        scope="same ordered candidate finalization stage; not full GA or raw-T1 whole QC")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("brain", "baseline-candidate", "cached-candidate", "benchmark-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2, allow_nan=False))
