"""生产表面放置适配器：复用完整 Python/Numba 优化器，不启动原生程序。"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import torch


def placement_options(*, device: str) -> dict:
    """明确设备→完整候选/清理后端；保留已验证的 CPU 有序优化与采样。

    device 为 cpu 或带索引的 cuda:N；返回优化器具名参数字典。CPU 使用
    snapshot/compiled/source_numba，CUDA 使用 torch_snapshot/source_torch。
    两者均保留完整候选扩展、实时碰撞接受和四轮停止规则；不设置全局
    精度、线程或分配器。非法/不带索引的设备抛 ValueError。
    """
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"} or (target.type == "cuda" and target.index is None):
        raise ValueError("placement device must be cpu or explicit cuda:N")
    cuda = target.type == "cuda"
    return dict(regularization_backend="cpu", sampling_backend="cpu",
                candidate_backend="torch_snapshot" if cuda else "snapshot",
                candidate_grid_cells_per_axis=3 if cuda else 2,
                retained_mht_backend="compiled",
                cleanup_marking_backend="source_torch" if cuda else "source_numba",
                cleanup_candidate_grid_cells_per_axis=3 if cuda else 2,
                device=str(target))


def _publish_report(subject: Path, hemi: str, stage: str, report: dict) -> dict:
    path = subject / "scripts" / f"{hemi}.{stage}.python.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report.update(implementation="fnit-python-numba-torch", native_programs=[],
                  log=str(path), hemisphere=hemi)
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    temporary.replace(path)
    return report


def _check_ordered_mesh(source: Path, output: Path) -> tuple[int, int]:
    before, faces = fsio.read_geometry(str(source))
    after, out_faces = fsio.read_geometry(str(output))
    if after.shape != before.shape or not np.isfinite(after).all() or not np.array_equal(faces, out_faces):
        raise RuntimeError(f"Python placement changed the ordered input mesh: {output}")
    return len(after), len(out_faces)


def run_preaparc_python(*, subject_dir: str | Path, hemi: str, device: str) -> dict:
    """自产 orig/premesh/MRI→white.preaparc、阈值表及 mrisps.wpa.mgz。

    subject_dir 使用标准 MRI/surf 结构；hemi=lh/rh。三张 MRI 必须同一
    conform 网格；网格为 surface RAS/mm。写出完整四轮结果和 uint8
    诊断体积，返回路径、四轮轨迹及含阈值计算/读写的秒数。输入、设备、
    网格或收敛失败抛异常，不回退。对应 mris_place_surface --white
    --nsmooth 5 --rip-bg-no-annot --rip-bg；完整格式与示例见功能专页。
    """
    from .autodet_gwstats_python import write_autodet_stats
    from .place_white_preaparc_python import place_white_preaparc
    tick = perf_counter()
    subject = Path(subject_dir).resolve()
    options = placement_options(device=device)
    if hemi not in {"lh", "rh"}:
        raise ValueError("hemi must be lh or rh")
    surf, mri = subject / "surf", subject / "mri"
    stats = surf / f"autodet.gw.stats.{hemi}.dat"
    write_autodet_stats(mri / "brain.finalsurfs.mgz", mri / "wm.mgz",
                       surf / f"{hemi}.orig.premesh", stats, hemi)
    stats_seconds = perf_counter() - tick
    report = place_white_preaparc(subject_dir=subject, hemi=hemi,
        output=surf / f"{hemi}.white.preaparc", output_volume=mri / "mrisps.wpa.mgz",
        max_steps=400, **options)
    _check_ordered_mesh(surf / f"{hemi}.orig", Path(report["output"]))
    report.update(stats=str(stats), outvol=str(mri / "mrisps.wpa.mgz"),
                  stats_seconds=stats_seconds, place_seconds=report["seconds"],
                  seconds=perf_counter() - tick)
    return _publish_report(subject, hemi, "white.preaparc", report)


def run_final_white_python(*, subject_dir: str | Path, hemi: str, device: str) -> dict:
    """自产 preaparc、cortex/aparc、MRI→最终 white 与 mrisps.white.mgz。

    输入/输出沿用完整 place_final_white；surface RAS/mm、有序面不变。
    返回完整轨迹、路径、原生程序空清单及包含网格检查/读写的秒数。
    对应 --white --nsmooth 0 --rip-label --rip-surf --aparc；失败抛异常。
    """
    from .place_final_white_python import place_final_white
    tick = perf_counter()
    subject = Path(subject_dir).resolve()
    report = place_final_white(subject_dir=subject, hemi=hemi,
        output=subject / "surf" / f"{hemi}.white",
        output_volume=subject / "mri/mrisps.white.mgz", max_steps=400,
        **placement_options(device=device))
    _check_ordered_mesh(subject / "surf" / f"{hemi}.white.preaparc", Path(report["output"]))
    report.update(outvol=str(subject / "mri/mrisps.white.mgz"), seconds=perf_counter() - tick)
    return _publish_report(subject, hemi, "white", report)


def run_pial_python(*, subject_dir: str | Path, hemi: str, device: str, profile: bool = False) -> dict:
    """自产 white、cortex 标签与 MRI→完整四轮 pial.T1；不写近似表面。

    MRI 同 conform 网格，输入标签与 white 顶点一一对应，输出 surface
    RAS/mm。profile 默认 False；True 同步目标 CUDA 并保留分项计时。
    返回路径、轮界、清理和含输出检查/读写的秒数；对应 --pial
    --rip-label --pin-medial-wall --repulse-surf --white-surf。失败抛异常。
    """
    from .place_pial_python import place_pial_t1
    tick = perf_counter()
    subject = Path(subject_dir).resolve()
    report = place_pial_t1(subject=subject, hemisphere=hemi,
        output=subject / "surf" / f"{hemi}.pial.T1", max_steps=400,
        profile=profile, **placement_options(device=device))
    vertices, faces = _check_ordered_mesh(subject / "surf" / f"{hemi}.white", Path(report["output"]))
    report.update(vertices=vertices, faces=faces, seconds=perf_counter() - tick)
    return _publish_report(subject, hemi, "pial", report)
