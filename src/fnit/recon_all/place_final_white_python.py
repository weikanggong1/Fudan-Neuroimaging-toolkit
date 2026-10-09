"""最终white实验接口：复用完整white四轮优化器，独立接注释rip状态。"""
from __future__ import annotations

from pathlib import Path

from .place_white_preaparc_python import _place_white_preaparc


def place_final_white(
    subject_dir: str | Path, hemi: str, output: str | Path,
    *, max_steps: int = 400, output_volume: str | Path | None = None,
    regularization_backend: str = "cpu", sampling_backend: str = "cpu",
    candidate_backend: str = "tree", candidate_grid_cells_per_axis: int = 2,
    retained_mht_backend: str = "tree", cleanup_marking_backend: str = "legacy",
    cleanup_candidate_grid_cells_per_axis: int = 2,
    collision_profile: bool = False, device: str | None = None, trace_callback=None,
) -> dict:
    """从preaparc、cortex与aparc执行最终white四轮；尚未生产接线。

    subject_dir需surf/H.white.preaparc、autodet.gw.stats.H.dat，
    label/H.cortex.label、H.aparc.annot以及mri/brain.finalsurfs、wm、
    aseg.presurf.mgz。MRI为相同conform网格；表面surface RAS/mm，
    label/annot对应完全相同顶点顺序。output为独立表面路径；可选
    output_volume写uint8预处理MRI并保留几何。max_steps默认400，
    其他显式后端/设备/trace参数沿用完整place_white_preaparc。
    不平滑输入；--rip-label关闭midline，仅在独立原preaparc上执行
    注释BG过滤和247标记。不套用preaparc冻结规则，不调用外部程序
    或读取原生输出。
    返回完整四轮与最终清理、路径/网格大小/轨迹/分项秒的dict。
    缺失文件、输入空间/参数错误、四轮未完成或残余相交均抛异常；
    output不得覆盖任何输入。对应mris_place_surface --white --nsmooth 0
    --rip-label H.cortex.label --rip-surf H.white.preaparc --aparc H.aparc.annot。
    该实验接口必须通过完整同输入原生回归后才能用于生产。
    """
    subject = Path(subject_dir)
    inputs = {*(subject/f"surf/{hemi}.{name}" for name in ("white.preaparc",)),
        subject/f"surf/autodet.gw.stats.{hemi}.dat",
        *(subject/f"label/{hemi}.{name}" for name in ("cortex.label", "aparc.annot")),
        *(subject/f"mri/{name}.mgz" for name in ("brain.finalsurfs", "wm", "aseg.presurf"))}
    inputs = {path.resolve() for path in inputs}
    outputs = [Path(output).resolve()]
    if output_volume is not None:
        outputs.append(Path(output_volume).resolve())
    if any(path in inputs for path in outputs) or len(set(outputs)) != len(outputs):
        raise ValueError("experimental final white outputs cannot overwrite inputs or each other")
    return _place_white_preaparc(subject_dir=subject, hemi=hemi, output=output,
        steps=max_steps, complete=True, final_white=True, output_volume=output_volume,
        regularization_backend=regularization_backend, sampling_backend=sampling_backend,
        candidate_backend=candidate_backend, candidate_grid_cells_per_axis=candidate_grid_cells_per_axis,
        retained_mht_backend=retained_mht_backend, cleanup_marking_backend=cleanup_marking_backend,
        cleanup_candidate_grid_cells_per_axis=cleanup_candidate_grid_cells_per_axis,
        collision_profile=collision_profile, device=device, trace_callback=trace_callback)
