"""已保存候选的公开脑图、局部场及全体素误差；仅用于离线验证。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(*, candidate, report, likelihood_reference, output):
    """具名路径读取已完成自产检查点与事后参考；写PNG及其JSON版本收据。

    candidate是本轮自产NPZ；report是绑定其SHA的完整阶段JSON；
    likelihood_reference是只用于比较的冻结FNITPDF1，不能用作生产输入。
    output为新PNG文件。强度图采用局部网格中央轴位，不改变算法空间。
    checkpoint或reference SHA不符抛ValueError，旧output存在报FileExistsError。
    """
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("new visualization output required")
    receipt = json.loads(report.read_text())
    if (receipt["own_checkpoint_sha256"] != sha(candidate)
            or receipt["reference_sha256"]["likelihood"] != sha(likelihood_reference)):
        raise ValueError("bound saved candidate and after-only reference required")
    with np.load(candidate, allow_pickle=False) as z:
        values = {name: z[name] for name in z.files}
    from benchmark_recon_topology_volume_prepare_fields import load_volume_preparation_capture
    ref = load_volume_preparation_capture(capture_path=likelihood_reference)
    intensity = values["defect_intensity_volume"]
    werror = np.abs(values["white_log_probability"].astype(np.float64)-ref["white_log_pdf"])
    gerror = np.abs(values["gray_log_probability"].astype(np.float64)-ref["gray_log_pdf"])
    matrix = values["source_matrix"].astype(np.float64)
    reference_matrix = np.asarray(ref["ras_to_voxel"], np.float64)
    matrix_bits = np.ascontiguousarray(matrix).view(np.uint64)
    reference_bits = np.ascontiguousarray(reference_matrix).view(np.uint64)
    coords = values["current_corrected_original"]
    edge_vertices = np.unique(values["corrected_edges"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes[0, 0].scatter(coords[::5, 1], coords[::5, 2], s=.3, color=".65", rasterized=True)
    axes[0, 0].scatter(coords[edge_vertices, 1], coords[edge_vertices, 2], s=2,
                       color="#d95035", rasterized=True)
    axes[0, 0].set(title="Public LH / first-defect candidate vertices",
                    xlabel="surface RAS Y (mm)", ylabel="Z (mm)")
    axes[0, 0].axis("equal")
    level = intensity.shape[2]//2
    image = axes[0, 1].imshow(intensity[:, :, level].T, origin="lower", cmap="gray")
    axes[0, 1].set(title="Own local MRI / central slice", xlabel="local X voxel", ylabel="Y voxel")
    fig.colorbar(image, ax=axes[0, 1], shrink=.7)
    axes[0, 2].plot(values["bins_calibration"], values["white_smooth_calibration"], label="white")
    axes[0, 2].plot(values["bins_calibration"], values["gray_smooth_calibration"], label="gray")
    axes[0, 2].set(title="Own 472-vertex calibration", xlabel="MRI intensity bin", ylabel="PDF")
    axes[0, 2].legend()
    for ax, error, name in [(axes[1, 0], werror, "White logP"), (axes[1, 1], gerror, "Gray logP")]:
        image = ax.imshow(error[:, :, level].T, origin="lower", vmin=0,
                          vmax=max(float(error.max()), 1e-12), cmap="magma")
        ax.set(title=name+" |own-reference| / full max "+format(float(error.max()), ".3g"),
               xlabel="local X voxel", ylabel="Y voxel")
        fig.colorbar(image, ax=ax, shrink=.7)
    groups = receipt["comparisons"]
    axes[1, 2].plot(np.arange(len(groups)), [v["max_abs"] for v in groups.values()], "o")
    axes[1, 2].set(title="All 24 consumed input groups", xlabel="comparison group index", ylabel="max absolute error")
    axes[1, 2].set_ylim(-1e-6, 1e-6)
    fig.suptitle("ds000114 sub-07 ses-test LH / same-input first defect; complete GA not assessed")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    metadata = dict(schema_version=1, source_report_sha256=sha(report),
        candidate_sha256=sha(candidate), likelihood_reference_sha256=sha(likelihood_reference),
        plot_script_sha256=sha(__file__), figure_sha256=sha(output),
        full_volume_white_log_probability_max_error=float(werror.max()),
        full_volume_gray_log_probability_max_error=float(gerror.max()),
        internal_matrix_different_float64_bits=int(np.count_nonzero(matrix_bits != reference_bits)),
        internal_matrix_different_values=int(np.count_nonzero(matrix != reference_matrix)),
        surface_display_stride=5, comparison_volume_shape=list(intensity.shape),
        central_z_voxel=int(level), coordinate_space="surface RAS millimetres; local voxel indices",
        reference_after_candidate_only=True, complete_GA=False)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "report", "likelihood-reference", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    plot(**vars(parser.parse_args()))
