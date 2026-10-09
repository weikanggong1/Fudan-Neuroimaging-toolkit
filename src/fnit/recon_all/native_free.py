"""单幅 T1 的 FNIT 皮层重建。必要表面阶段使用固定的完整调用链。"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np
import torch


def _native_binary(native_bin_dir: str | Path, name: str) -> tuple[Path, str]:
    binary = Path(native_bin_dir).resolve() / name
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"executable {name} not found: {binary}")
    digest = hashlib.sha256()
    with binary.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return binary, digest.hexdigest()


def _native_bin_directory(value: str | Path | None) -> Path:
    """默认从当前 Conda 环境发现 FNIT 自编译原生程序。"""
    directory = Path(value).resolve() if value is not None else Path(
        os.environ.get("FNIT_RECON_ALL_BIN_DIR", Path(sys.prefix) / "bin")).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"FNIT recon-all native bin directory not found: {directory}")
    return directory


def _native_em_register_binary(native_bin_dir: str | Path) -> tuple[Path, str]:
    return _native_binary(native_bin_dir, "mri_em_register")


def _native_topology_binary(native_bin_dir: str | Path) -> tuple[Path, str]:
    return _native_binary(native_bin_dir, "mris_fix_topology_fnit")


def _native_surface_metrics_binary(native_bin_dir: str | Path) -> tuple[Path, str]:
    return _native_binary(native_bin_dir, "mris_place_surface")


def _native_inflate_binary(native_bin_dir: str | Path) -> tuple[Path, str]:
    return _native_binary(native_bin_dir, "mris_inflate")


def _validate_standard_python_dependencies() -> None:
    """在输入目录检查后、输出创建前导入标准球面配准的完整模块。

    tifffile 已属于标准 recon-all 安装依赖。这里提前检查导入，避免完成
    体积和表面阶段后才因缺包退出；缺失或导入失败仍传播原始异常。
    不执行配准、Numba JIT 或读取影像、模板，不修改环境或调用外部软件。
    """
    importlib.import_module("tifffile")
    registration = importlib.import_module("fnit.recon_all.mris_register_run")
    if not callable(registration.run_register_sphere):
        raise TypeError("standard sphere registration entry point is not callable")


def _run_native_wm_segment(binary: Path, mri: Path, assets: Path) -> None:
    env = dict(os.environ, FREESURFER_HOME=str(assets))
    subprocess.run([str(binary), "-wsizemm", "13", "-mprage",
                    "antsdn.brain.mgz", "wm.seg.mgz"],
                   cwd=mri, env=env, check=True)


def _run_torch_wm_segment(mri: Path, *, device: str, optimized: bool = False,
                          execution: str = "in-process", threads: int = 4,
                          profile_stages: bool = False) -> dict:
    """复用完整WM分割，读取自产强度并写出同网格uint8 WM。

    mri为被试MRI目录；device为显式CUDA设备；optimized=False保留旧
    混合实现，True使用已验证的直方图和缓存平面几何。execution默认
    in-process保留父缓存策略；isolated须optimized=True，复用新exec
    worker在子进程启用缓存，父CUDA/TF32状态不改。threads默认4，
    profile_stages默认False，仅传给隔离worker。不省略有序CPU反馈。
    读取antsdn.brain.mgz，写wm.seg.mgz；隔离模式另写scripts/wm-isolated.json。
    返回既有阶段字典或带完整exec墙钟/源码哈希的worker报告；参数非法、
    IO和子失败传播，不调用外部神经影像软件或自动回退。空间/单位与
    官方mri_segment配方、真实验证见WM_PLANAR_TORCH.md。
    """
    if execution not in {"in-process", "isolated"}:
        raise ValueError("WM execution must be in-process or isolated")
    if execution == "isolated":
        if not optimized:
            raise ValueError("isolated WM requires torch-optimized backend")
        from .wm_torch_worker import run_isolated_segmentation
        return run_isolated_segmentation(
            source_path=mri / "antsdn.brain.mgz", output_path=mri / "wm.seg.mgz",
            report_path=mri.parent / "scripts/wm-isolated.json", device=device,
            threads=threads, histogram_batch_size=2048, planar_batch_size=256,
            profile_stages=profile_stages)
    from .mri_segment import segment_white_matter_mgz

    options = ({"histogram_backend": "torch", "histogram_batch_size": 2048,
                "planar_backend": "cached", "planar_batch_size": 256} if optimized else {})
    return segment_white_matter_mgz(
        mri / "antsdn.brain.mgz", mri / "wm.seg.mgz", device=device, **options)


def _run_native_wm_edit(binary: Path, mri: Path, assets: Path) -> None:
    env = dict(os.environ, FREESURFER_HOME=str(assets))
    subprocess.run([str(binary), "-keep-in", "-fix-ento-wm", "entowm.mgz",
                    "3", "255", "255", "-fix-acj", "aseg.presurf.mgz",
                    "255", "255", "-fill-seg-wm", "-fix-scm-ha", "1",
                    "wm.seg.mgz", "brain.mgz", "aseg.presurf.mgz",
                    "wm.asegedit.mgz"], cwd=mri, env=env, check=True)


def _run_torch_wm_edit(mri: Path, *, device: str) -> dict:
    """自产同网格WM/brain/aseg/EntoWM→wm.asegedit，保留有序Numba反馈。

    device为明确CUDA；固定keep-in/fill/SCM-HA/EntoWM/ACJ配方。
    返回完整读写时间和CPU/GPU分段。原byte-path几何证明不成立、原存储
    不受支持或CUDA错误直接抛出，不能回退原生或读取参考来补齐。
    """
    from .edit_wm_aseg_torch import write_wm_asegedit_hybrid_diagnostic
    return write_wm_asegedit_hybrid_diagnostic(
        wm_file=mri / "wm.seg.mgz", brain_file=mri / "brain.mgz",
        aseg_file=mri / "aseg.presurf.mgz", entowm_file=mri / "entowm.mgz",
        output_file=mri / "wm.asegedit.mgz", device=device, fill_seg_wm=True)


def _folding_atlas(assets: Path, hemi: str) -> Path:
    atlas = assets / "average" / (
        f"{hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif")
    if not atlas.is_file():
        raise FileNotFoundError(f"folding atlas not found: {atlas}")
    return atlas


def _run_native_em_register(binary: Path, mri: Path, atlas: Path,
                            assets: Path, *, backend: str = "original",
                            binary_sha256: str | None = None, threads: int = 4,
                            device: str = "cpu", inverse_backend: str = "cpu",
                            candidate_chunk: int = 64,
                            execution: str = "in-process") -> dict | None:
    """运行完整原生注册；缓存后端只改变已验证的搜索密度复用。

    binary为独立Conda程序；mri含自产conformed nu/brainmask和transforms。
    atlas/assets为已校验GCA和资产路径。backend默认original；cpu_cached
    验证binary_sha256及能力2，threads默认4控制新子进程，保持父环境。
    输出talairach.lta包含4×4 voxel空间变换与两侧几何，返回None。
    inverse_backend默认cpu、candidate_chunk默认64；仅Torch评分使用这些
    参数。torch求逆复用同余子式，不改变候选网格、EM和LTA几何。
    execution=in-process保留父allocator；isolated仅用于Torch，新exec阶段
    内开启CUDA缓存，退出释放资源，不改变已初始化父CUDA/精度/缓存设置。
    原生执行/输入/输出/能力失败抛异常；Torch返回分段报告，原生返回None。
    """
    if backend == "torch":
        # The expensive translation/linear candidate search is implemented by
        # FNIT's chunked Torch scorer.  EM refinement and LTA serialization stay
        # in the existing Python implementation, so this is an explicit
        # hybrid backend rather than a claim of a complete GPU GCA rewrite.
        from .mri_em_register_python import register_t1

        if not str(device).startswith("cuda"):
            raise ValueError("GCA torch backend requires an explicit CUDA device")
        if execution == "isolated":
            from .gca_torch_worker import run_isolated_registration
            result = run_isolated_registration(
                nu_path=mri / "nu.mgz", mask_path=mri / "brainmask.mgz",
                atlas_path=atlas, output_path=mri / "transforms/talairach.lta",
                report_path=mri.parent / "scripts/gca-isolated.json", device=device,
                threads=threads, candidate_chunk=candidate_chunk,
                inverse_backend=inverse_backend)
        elif execution == "in-process":
            result = register_t1(
                mri / "nu.mgz", atlas, mri / "brainmask.mgz",
                mri / "transforms/talairach.lta", device=device,
                search_backend="torch", candidate_chunk=candidate_chunk, sample_chunk=8192,
                reduce_on_device=True, inverse_backend=inverse_backend)
        else:
            raise ValueError("invalid GCA execution")
        if not (mri / "transforms/talairach.lta").is_file():
            raise FileNotFoundError(mri / "transforms/talairach.lta")
        return result
    if execution != "in-process":
        raise ValueError("isolated execution requires the Torch GCA backend")
    if backend == "cpu_cached":
        from .mri_em_register_cached_conda import run_cached_em_register
        run_cached_em_register(binary=binary, mri=mri, atlas=atlas, assets=assets,
                               binary_sha256=binary_sha256 or hashlib.sha256(binary.read_bytes()).hexdigest(),
                               threads=threads)
        return
    if backend != "original":
        raise ValueError("invalid native GCA backend")
    from .thread_budget import native_thread_environment
    env, _ = native_thread_environment(threads=threads)
    env["FREESURFER_HOME"] = str(assets)
    env.pop("FNIT_GCA_QUERY_CAPABILITIES", None)
    env["FNIT_GCA_SCORER"] = "original"
    subprocess.run([str(binary), "-uns", "3", "-mask", "brainmask.mgz",
                    "nu.mgz", str(atlas), "transforms/talairach.lta"],
                   cwd=mri, env=env, check=True)
    if not (mri / "transforms/talairach.lta").is_file():
        raise FileNotFoundError(mri / "transforms/talairach.lta")


def _run_native_topology(binary: Path, subject: Path, hemi: str,
                         assets: Path) -> None:
    from .topology_conda_ga import run_topology_ga_conda

    result = run_topology_ga_conda(subject, hemi, binary, assets)
    output = Path(result["output"])
    if not output.is_file():
        raise FileNotFoundError(f"topology GA produced no surface: {output}")


def _run_native_sphere_step(binary: Path, subject: Path, assets: Path,
                            arguments: list[str], outputs: tuple[Path, ...]) -> float:
    scripts = subject / "scripts"
    scripts.mkdir(exist_ok=True)
    env = dict(os.environ, SUBJECTS_DIR=str(subject.parent),
               FREESURFER_HOME=str(assets))
    tick = time.perf_counter()
    subprocess.run([str(binary), *arguments], cwd=scripts, env=env, check=True)
    for output in outputs:
        if not output.is_file():
            raise FileNotFoundError(f"{binary.name} produced no output: {output}")
    return time.perf_counter() - tick


def _prepare_native_topology(binary: Path, subject: Path, hemi: str,
                             assets: Path, device: str,
                             native_inflate_binary: Path,
                             intersection_binary: Path,
                             ) -> tuple[float, float, dict[str, float], float, float]:
    from .mris_remesh_python import remesh_surface
    from .mris_remove_intersection_python import remove_intersection_surface
    from .smooth_surface_python import smooth_surface
    from .sphere_quick_python import write_quick_sphere

    surf = subject / "surf"
    tick = time.perf_counter()
    smooth_nofix = surf / f"{hemi}.smoothwm.nofix"
    inflated_nofix = surf / f"{hemi}.inflated.nofix"
    qsphere_nofix = surf / f"{hemi}.qsphere.nofix"
    smooth_surface(surf / f"{hemi}.orig.nofix", smooth_nofix, device=device)
    python_seconds = time.perf_counter() - tick
    sphere_seconds = {}
    sphere_seconds["inflate_nofix"] = _run_native_sphere_step(
        native_inflate_binary, subject, assets,
        ["-no-save-sulc", str(smooth_nofix), str(inflated_nofix)],
        (inflated_nofix,))
    tick = time.perf_counter()
    write_quick_sphere(inflated_nofix, qsphere_nofix)
    quick_seconds = time.perf_counter() - tick
    sphere_seconds["qsphere_nofix_python"] = quick_seconds
    native_started = time.perf_counter()
    _run_native_topology(binary, subject, hemi, assets)
    native_seconds = time.perf_counter() - native_started
    tick = time.perf_counter()
    remesh_surface(surf / f"{hemi}.orig.premesh", surf / f"{hemi}.orig", iterations=3)
    remesh_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    orig = surf / f"{hemi}.orig"
    remove_intersection_surface(orig, orig, binary=intersection_binary,
                                assets_dir=assets)
    intersection_seconds = time.perf_counter() - tick
    return (python_seconds + remesh_seconds + intersection_seconds, native_seconds,
            sphere_seconds, remesh_seconds, intersection_seconds)


def _run_accurate_sphere_pair(inflate_binary: Path, subject: Path,
                              hemi: str, assets: Path, *, device: str = "cpu",
                              normals_backend: str = "numba",
                              inflate_backend: str = "native") -> tuple[dict, dict]:
    """同序smoothwm→标准inflated/sulc→sphere，返回阶段秒数与球面报告。

    subject/hemi提供surface RAS mm网格；assets和inflate_binary供native
    默认路径。inflate_backend可显式torch，复用完整GPU算法与真实sulc；
    device为其明确CUDA设备，缓存由外层fresh worker选择。球面默认Numba
    法向/CPU收尾保持；normals_backend仅选择既有法向。报告含inflate实际
    后端与完整读写秒数；异常向上传播，不复制参考或回退。nofix不属此接口。
    """
    from .sphere_standard_run import run_standard_sphere

    surf = subject / "surf"
    inflated, sulc = surf / f"{hemi}.inflated", surf / f"{hemi}.sulc"
    if inflate_backend == "torch":
        from .inflate_standard_run import run_standard_inflate
        inflate_report = run_standard_inflate(
            input_surface=surf / f"{hemi}.smoothwm", inflated_output=inflated,
            sulc_output=sulc, backend="torch", device=device, profile=False)
        inflate_seconds = inflate_report["total_seconds_including_io"]
    elif inflate_backend == "native":
        inflate_seconds = _run_native_sphere_step(
            inflate_binary, subject, assets,
            [str(surf / f"{hemi}.smoothwm"), str(inflated)], (inflated, sulc))
        inflate_report = {"backend": "native", "total_seconds_including_io": inflate_seconds}
    else:
        raise ValueError("inflate_backend must be native or torch")
    sphere_report = run_standard_sphere(
        inflated, surf / f"{hemi}.smoothwm", surf / f"{hemi}.sphere",
        finish_device="cpu", averaging_device=device,
        **({"normals_device": device} if normals_backend == "torch" else {}))
    sphere_report["inflate_runtime"] = inflate_report
    return ({"inflate": inflate_seconds,
             "sphere": sphere_report["total_seconds_including_io"]}, sphere_report)


def _run_surface_metrics(binary: Path, subject: Path, hemi: str,
                         assets: Path, *, device: str) -> dict[str, float]:
    """从对应的 white/pial 计算厚度、双侧面积和平均曲率顶点图。

    subject 为被试目录，hemi 为 lh/rh；输入表面为 surface RAS（mm），
    顶点与有序面须对应。CUDA 使用已有 PyTorch 函数，CPU 使用 binary
    指定的 Conda mris_place_surface；assets 是原生程序的 FNIT 资产目录。
    写出 H.thickness（mm）、H.area/H.area.pial（mm²）、
    H.curv/H.curv.pial（mm⁻¹），返回包含读写与 GPU 同步的逐图秒数。
    参数固定为 20 跳、5 mm 最大厚度、2 阶邻域及 10 次曲率平滑，
    对应官方 --thickness/--area-map/--curv-map；失败抛异常。
    同输入精度、速度及完整示例见 docs/recon_all/SURFACE_METRICS.md。
    """
    surf = subject / "surf"
    white, pial = surf / f"{hemi}.white", surf / f"{hemi}.pial"
    if torch.device(device).type == "cuda":
        from .surface_area_gpu import area_map
        from .surface_curvature_gpu import curvature_map
        from .surface_thickness_gpu import thickness_map

        timings = {}
        for name, function, kwargs in (
            ("thickness", thickness_map, {"white_file": white, "pial_file": pial,
                                         "output_file": surf / f"{hemi}.thickness"}),
            ("area", area_map, {"surface": white, "output": surf / f"{hemi}.area"}),
            ("area.pial", area_map, {"surface": pial, "output": surf / f"{hemi}.area.pial"}),
            ("curv", curvature_map, {"surface": white, "output": surf / f"{hemi}.curv"}),
            ("curv.pial", curvature_map, {"surface": pial, "output": surf / f"{hemi}.curv.pial"}),
        ):
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            function(**kwargs, device=device)
            torch.cuda.synchronize(device)
            timings[name] = time.perf_counter() - tick
        return timings
    return _run_native_surface_metrics(binary, subject, hemi, assets)


def _run_native_surface_metrics(binary: Path, subject: Path, hemi: str,
                                assets: Path) -> dict[str, float]:
    surf = subject / "surf"
    white, pial = surf / f"{hemi}.white", surf / f"{hemi}.pial"
    commands = (
        ("thickness", ["--thickness", str(white), str(pial), "20", "5",
                       str(surf / f"{hemi}.thickness")]),
        ("area", ["--area-map", str(white), str(surf / f"{hemi}.area")]),
        ("area.pial", ["--area-map", str(pial), str(surf / f"{hemi}.area.pial")]),
        ("curv", ["--curv-map", str(white), "2", "10",
                  str(surf / f"{hemi}.curv")]),
        ("curv.pial", ["--curv-map", str(pial), "2", "10",
                       str(surf / f"{hemi}.curv.pial")]),
    )
    scripts = subject / "scripts"
    scripts.mkdir(exist_ok=True)
    env = dict(os.environ, SUBJECTS_DIR=str(subject.parent),
               FREESURFER_HOME=str(assets))
    timings = {}
    for name, args in commands:
        tick = time.perf_counter()
        subprocess.run([str(binary), *args], cwd=scripts, env=env, check=True)
        output = surf / f"{hemi}.{name}"
        if not output.is_file():
            raise FileNotFoundError(f"mris_place_surface produced no morph: {output}")
        timings[name] = time.perf_counter() - tick
    return timings


def _run_avg_curv(binary: Path, subject: Path, hemi: str,
                  atlas: Path, assets: Path) -> None:
    """执行固定的 mrisp_paint 第 6 幅图谱映射。"""
    output = subject / "surf" / f"{hemi}.avg_curv"
    env = dict(os.environ, SUBJECTS_DIR=str(subject.parent),
               FREESURFER_HOME=str(assets))
    subprocess.run([str(binary), "-a", "5", f"{atlas}#6",
                    str(subject / "surf" / f"{hemi}.sphere.reg"), str(output)],
                   cwd=subject / "scripts", env=env, check=True)
    if not output.is_file():
        raise FileNotFoundError(output)


def _run_curvature_stats(binary: Path, subject: Path, hemi: str,
                         assets: Path) -> None:
    """生成固定的 curv.stats 及 smoothwm 曲率附图。"""
    output = subject / "stats" / f"{hemi}.curv.stats"
    env = dict(os.environ, SUBJECTS_DIR=str(subject.parent),
               FREESURFER_HOME=str(assets))
    subprocess.run([str(binary), "-m", "--writeCurvatureFiles", "-G",
                    "-o", str(output), "-F", "smoothwm", subject.name,
                    hemi, "curv", "sulc"], cwd=subject / "scripts",
                   env=env, check=True)
    for path in (output, *(subject / "surf" / f"{hemi}.smoothwm.{name}.crv"
                           for name in ("BE", "C", "FI", "S"))):
        if not path.is_file():
            raise FileNotFoundError(path)


def _run_defects_volume(binary: Path | None, subject: Path, hemi: str,
                        assets: Path, *, backend: str = "native",
                        device: str = "cpu") -> dict | None:
    """按固定 recon-all --defects 命令将拓扑缺陷投射到 conform 网格。"""
    mri, surf, labels = (subject / name for name in ("mri", "surf", "label"))
    output = mri / "surface.defects.mgz"
    template = mri / "orig.mgz" if hemi == "lh" else output
    defect_labels = surf / f"{hemi}.defect_labels"
    cortex = labels / f"{hemi}.nofix.cortex.label"
    for required in (template, defect_labels, cortex):
        if not required.is_file():
            raise FileNotFoundError(required)
    if backend == "torch":
        from .defects_label_volume_torch import defects_to_volume
        return defects_to_volume(
            surface_file=surf / f"{hemi}.orig.nofix", defect_file=defect_labels,
            template_file=template, output_file=output,
            offset=1000 if hemi == "lh" else 2000, merge=hemi == "rh",
            cortex_file=cortex, device=device)
    if backend != "native":
        raise ValueError("defects backend must be native or torch")
    if binary is None:
        raise ValueError("native defects backend requires mri_label2vol")
    env = dict(os.environ, SUBJECTS_DIR=str(subject.parent),
               FREESURFER_HOME=str(assets))
    subprocess.run([str(binary), "--defects", str(surf / f"{hemi}.orig.nofix"),
                    str(defect_labels), str(template),
                    "1000" if hemi == "lh" else "2000",
                    "0" if hemi == "lh" else "1", str(output), str(cortex)],
                   cwd=subject / "scripts", env=env, check=True)
    if not output.is_file():
        raise FileNotFoundError(output)


def _run_white_mri_chain(subject: Path, weights: Path, assets: Path,
                         threads: int, warp_binaries: tuple[Path, Path, Path],
                         stage, *, device: str) -> dict:
    """生成 MNI 辅助图、非线性变换和 finalsurfs；显式传递主设备。

    subject 提供自产 conform MRI；weights、assets 为已校验资源，threads
    为线程数，warp_binaries 按转换/求逆/重采样排列。stage 是记录耗时和
    失败的回调，device 为 CPU 或 CUDA。输出写入 subject，返回辅助网络实际
    前向记录字典；原生程序或计算失败抛异常。全部网络使用 device，
    辅助网络卷积在局部作用域采用经同输入验证的FP32，matmul TF32不变；
    CUDA 的 MNI warp 转换、完整求逆和检查图采用自有 GPU 后处理，CPU
    使用原有 Conda 程序；finalsurfs 后处理使用 CPU。空间、命令与实测
    见 MNI_NONLINEAR_CHAIN.md。后处理失败抛出异常，不静默切换后端。
    """
    from .finalsurfs_python import run_finalsurfs
    from .mni_aux_chain import run_mni_aux_chain
    from .mni_nonlinear_chain import run_mni_nonlinear_chain

    with torch.backends.cudnn.flags(allow_tf32=False):
        auxiliary = stage("mni_aux", run_mni_aux_chain, subject, weights, assets,
              device=device, threads=threads)
    stage("mni_nonlinear", run_mni_nonlinear_chain, subject, weights, assets,
          warp_convert=warp_binaries[0], ca_register=warp_binaries[1],
          mri_convert=warp_binaries[2], device=device, threads=threads,
          postprocess_backend="gpu" if torch.device(device).type == "cuda" else "conda")
    # ``finalsurfs_python`` has a validated CUDA implementation for the
    # volume masks and entorhinal/ACJ edits. Keep its FP32 semantics while
    # avoiding an unnecessary device round-trip on GPU recon-all runs.
    stage("brain_finalsurfs", run_finalsurfs, subject, device=device)
    return auxiliary["runtime"]


def _place_preaparc_and_smooth(subject: Path, hemi: str, binary: Path,
                               assets: Path, threads: int) -> dict:
    from .smooth_surface_python import smooth_surface
    from .white_preaparc_conda import run_white_preaparc

    report = run_white_preaparc(subject, hemi, binary, assets, threads=threads)
    tick = time.perf_counter()
    surf = subject / "surf"
    smooth_surface(surf / f"{hemi}.white.preaparc", surf / f"{hemi}.smoothwm",
                   iterations=3, device="cpu")
    report["smoothwm_seconds"] = time.perf_counter() - tick
    return report


def _run_native_pial(binary: Path, subject: Path, hemi: str,
                     assets: Path, threads: int) -> dict:
    """用 Conda 源码构建程序完成四轮 pial 放置，并检查输出网格。"""
    surf, mri, labels = (subject / name for name in ("surf", "mri", "label"))
    white = surf / f"{hemi}.white"
    output = surf / f"{hemi}.pial.T1"
    command = [str(binary), "--adgws-in",
               str(surf / f"autodet.gw.stats.{hemi}.dat"),
               "--seg", str(mri / "aseg.presurf.mgz"),
               "--threads", str(threads), "--wm", str(mri / "wm.mgz"),
               "--invol", str(mri / "brain.finalsurfs.mgz"),
               f"--{hemi}", "--i", str(white), "--o", str(output),
               "--pial", "--nsmooth", "0", "--rip-label",
               str(labels / f"{hemi}.cortex+hipamyg.label"),
               "--pin-medial-wall", str(labels / f"{hemi}.cortex.label"),
               "--aparc", str(labels / f"{hemi}.aparc.annot"),
               "--repulse-surf", str(white), "--white-surf", str(white),
               "--restore-255"]
    env = dict(os.environ, SUBJECTS_DIR=str(subject.parent),
               FREESURFER_HOME=str(assets))
    log = subject / "scripts" / f"{hemi}.pial.log"
    tick = time.perf_counter()
    with log.open("w") as stream:
        subprocess.run(command, cwd=subject / "scripts", env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    if not output.is_file():
        raise FileNotFoundError(output)
    vertices, faces = fs.read_geometry(str(output))
    white_vertices, white_faces = fs.read_geometry(str(white))
    if not np.isfinite(vertices).all() or len(vertices) != len(white_vertices) \
            or not np.array_equal(faces, white_faces):
        raise RuntimeError(f"pial output changed the ordered white mesh: {output}")
    return {"implementation": "conda-source-cpp", "output": str(output),
            "log": str(log), "vertices": len(vertices), "faces": len(faces),
            "seconds": time.perf_counter() - tick}


def _write_principal_curvature_maps(surf: Path, hemi: str,
                                    device: str) -> None:
    from .surface_roi_curvature_gpu import principal_curvatures

    for surface, prefix in (("white.preaparc", "white.preaparc"),
                            ("inflated", "inflated")):
        xyz, mesh = fs.read_geometry(str(surf / f"{hemi}.{surface}"))
        k1, k2 = principal_curvatures(xyz, mesh, device=device)
        fs.write_morph_data(str(surf / f"{hemi}.{prefix}.H"),
                            ((k1 + k2) / 2).astype(np.float32))
        fs.write_morph_data(str(surf / f"{hemi}.{prefix}.K"),
                            (k1 * k2).astype(np.float32))
    xyz, mesh = fs.read_geometry(str(surf / f"{hemi}.smoothwm"))
    k1, k2 = principal_curvatures(xyz, mesh, device=device)
    for name, values in (("H", (k1 + k2) / 2), ("K", k1 * k2),
                         ("K1", k1), ("K2", k2)):
        fs.write_morph_data(str(surf / f"{hemi}.smoothwm.{name}.crv"),
                            np.asarray(values, np.float32))


def _surface_pair(subject: Path, hemi: str, filled: Path, norm: Path,
                  *, device: str, threads: int, topology_binary: Path,
                  inflate_binary: Path, intersection_binary: Path,
                  place_binary: Path, defect_binary: Path | None,
                  assets: Path, defer_defects: bool = False,
                  defects_backend: str = "native",
                  sphere_normals_backend: str = "numba",
                  inflate_backend: str = "native") -> dict:
    """从 filled 生成已修复 orig、预白质表面及标准球面。"""
    from .extract_main_component_python import extract_main_component
    from .label_cortex_fix_ga_python import label_cortex_fix_ga
    from .label_cortex_python import label_cortex
    from .pretess_python import pretess_mgh
    from .tessellate_gpu import tessellate_mgh, write_quad_surface

    surf, mri, labels = (subject / name for name in ("surf", "mri", "label"))
    code = 255 if hemi == "lh" else 127
    pretess = mri / f"filled-pretess{code}.mgz"
    pretess_mgh(filled, code, norm, pretess)
    raw = surf / f"{hemi}.orig.raw.quad"
    vertices, quads = tessellate_mgh(pretess, code, device=device)
    write_quad_surface(raw, vertices, quads, nib.load(str(pretess)),
                       Path("../mri") / pretess.name)
    components = extract_main_component(raw, surf / f"{hemi}.orig.nofix")
    label_cortex(surf / f"{hemi}.orig.nofix", mri / "aseg.presurf.mgz",
                 labels / f"{hemi}.nofix.cortex.label")
    topology_python_seconds, topology_native_seconds, pre_sphere_seconds, remesh_seconds, intersection_seconds = (
        _prepare_native_topology(topology_binary, subject, hemi, assets,
                                 device, inflate_binary, intersection_binary))
    if not defer_defects:
        _run_defects_volume(defect_binary, subject, hemi, assets,
                            backend=defects_backend, device=device)
    preaparc = _place_preaparc_and_smooth(subject, hemi, place_binary,
                                         assets, threads)
    base, ga = label_cortex_fix_ga(
        surf / f"{hemi}.white.preaparc", mri / "aseg.presurf.mgz",
        mri / "entowm.mgz", hemi, labels / f"{hemi}.cortex.label")
    label_cortex(surf / f"{hemi}.white.preaparc", mri / "aseg.presurf.mgz",
                 labels / f"{hemi}.cortex+hipamyg.label", keep_hip_amyg=True)
    sphere_timings, sphere_report = _run_accurate_sphere_pair(
        inflate_binary, subject, hemi, assets, device=device,
        normals_backend=sphere_normals_backend, inflate_backend=inflate_backend)
    _write_principal_curvature_maps(surf, hemi, device)
    white, faces = fs.read_geometry(str(surf / f"{hemi}.smoothwm"))
    return {"hemisphere": hemi, "vertices": len(white), "faces": len(faces),
            "raw_components": components, "cortex_vertices": len(base) + len(ga),
            "topology_python_seconds": topology_python_seconds,
            "topology_native_seconds": topology_native_seconds,
            "topology_remesh_seconds": remesh_seconds,
            "topology_intersection_seconds": intersection_seconds,
            "sphere_seconds": {**pre_sphere_seconds, **sphere_timings},
            "standard_sphere_report": sphere_report,
            "white_preaparc_report": preaparc,
            "placement_pending": True}


def _finish_cortical_surface(subject: Path, hemi: str, binary: Path,
                             assets: Path, *, device: str, threads: int,
                             defer_metrics: bool = False,
                             white_binary: Path | None = None) -> dict:
    """球面配准和注释完成后，依次放置最终white、pial并计算顶点图。

    subject为自产被试目录，hemi为lh/rh；binary是pial及CPU顶点图程序，
    white_binary可单独指定white专用程序，默认复用binary。device/threads
    控制已有GPU顶点图和原生线程，defer_metrics默认False，供并行屏障。
    写出surface RAS/mm的white、pial.T1/pial和所需顶点图，返回各阶段路径/
    秒数/待办状态。输入、网格或执行失败传播异常；不读取官方结果。
    """
    from .final_white_conda import run_final_white
    surf = subject / "surf"
    white_report = run_final_white(subject, hemi, white_binary or binary, assets, threads=threads)
    pial_report = _run_native_pial(binary, subject, hemi, assets, threads)
    shutil.copyfile(surf / f"{hemi}.pial.T1", surf / f"{hemi}.pial")
    result = {"final_white_report": white_report, "pial_report": pial_report,
              "placement_pending": False, "metrics_pending": defer_metrics}
    if not defer_metrics:
        result.update(_finish_cortical_metrics(subject, hemi, binary, assets, device=device))
    return result


def _finish_cortical_metrics(subject: Path, hemi: str, binary: Path,
                            assets: Path, *, device: str) -> dict:
    """完成已放置white/pial的指标；真实短配对显示GPU指标串行更快。"""
    from .surface_area_gpu import mid_area_map
    from .surface_roi_gpu import vertex_volume_map
    surf, labels = subject / 'surf', subject / 'label'
    metric_seconds = _run_surface_metrics(binary, subject, hemi, assets, device=device)
    mid_area_map(surf / f"{hemi}.area", surf / f"{hemi}.area.pial",
                 surf / f"{hemi}.area.mid", device=device)
    vertex_volume_map(surf / f"{hemi}.white", surf / f"{hemi}.pial",
                      labels / f"{hemi}.cortex.label",
                      surf / f"{hemi}.volume", device=device)
    return {"metric_seconds": metric_seconds,
            "mean_thickness_mm": float(np.mean(fs.read_morph_data(
                str(surf / f"{hemi}.thickness")))), "metrics_pending": False}


def _project_parcels(subject: Path) -> None:
    """Run the validated bilateral cortex-volume mapping for three atlases."""
    from .surf2volseg_cortex_python import label_cortex_volume

    mri, surf, labels = (subject / name for name in ("mri", "surf", "label"))
    for atlas, output in (("aparc", "aparc+aseg.mgz"),
                          ("aparc.a2009s", "aparc.a2009s+aseg.mgz"),
                          ("aparc.DKTatlas", "aparc.DKTatlas+aseg.mgz")):
        label_cortex_volume(mri / "aseg.mgz", surf, labels, mri / output,
                            atlas=atlas)


def _project_exvivo_annotations(subject: Path, assets: Path) -> None:
    """将固定 fsaverage 标签投射到被试，并生成 BA/VPNL 注释。"""
    from .label2annot_python import write_label_annotation
    from .label2label_surface_python import SurfaceLabelMapper

    surf, labels = subject / "surf", subject / "label"
    ba_names = ("BA1", "BA2", "BA3a", "BA3b", "BA4a", "BA4p", "BA6",
                "BA44", "BA45", "V1", "V2", "MT", "perirhinal", "entorhinal")
    vpnl_names = ("FG1", "FG2", "FG3", "FG4", "hOc1", "hOc2", "hOc3v", "hOc4v")
    for hemi in ("lh", "rh"):
        mapper = SurfaceLabelMapper(
            assets / f"subjects/fsaverage/surf/{hemi}.sphere.reg",
            surf / f"{hemi}.sphere.reg", surf / f"{hemi}.white", subject.name)
        for group, names, color_table in (
            ("BA_exvivo", [f"{name}_exvivo" for name in ba_names],
             "colortable_BA.txt"),
            ("BA_exvivo.thresh", [f"{name}_exvivo.thresh" for name in ba_names],
             "colortable_BA_thresh.txt"),
            ("mpm.vpnl", [f"{name}.mpm.vpnl" for name in vpnl_names],
             "colortable_vpnl.txt"),
        ):
            targets = []
            for name in names:
                source = assets / f"subjects/fsaverage/label/{hemi}.{name}.label"
                target = labels / source.name
                mapper.map_label(source, target)
                targets.append(target)
            write_label_annotation(surf / f"{hemi}.orig",
                                   assets / "average" / color_table,
                                   targets, labels / f"{hemi}.{group}.annot")


def _project_wmparc(subject: Path) -> None:
    """Run the validated white-matter volume mapping from aparc+aseg."""
    from .surf2volseg_wm_python import label_wm_volume

    mri, surf, labels = (subject / name for name in ("mri", "surf", "label"))
    label_wm_volume(mri / "aparc+aseg.mgz", surf, labels, mri / "wmparc.mgz")


def _segment_callosum(mri: Path) -> dict[str, float | int]:
    from .mri_cc_python import run_mri_cc

    source = mri / "aseg.auto_noCCseg.mgz"
    output = mri / "aseg.auto.mgz"
    shutil.copyfile(mri / "synthseg.rca.mgz", source)
    info = run_mri_cc(source, mri / "norm.mgz", output,
                      mri / "transforms/cc_up.lta")
    for name in ("aseg.presurf.mgz", "aseg.mgz"):
        shutil.copyfile(output, mri / name)
    return info


def _validate_meshes(subject: Path) -> dict:
    """检查双侧网格闭合、球面拓扑和最终 white/pial 自相交。"""
    from .mris_remove_intersection_python import mark_intersections

    result = {}
    for hemi in ("lh", "rh"):
        surf = subject / "surf"
        meshes = {name: fs.read_geometry(str(surf / f"{hemi}.{name}"))
                  for name in ("orig", "white", "pial", "sphere.reg")}
        orig_vertices, orig_faces = meshes["orig"]
        ordered_faces = all(np.array_equal(orig_faces, faces)
                            for _, faces in meshes.values())
        finite = all(np.isfinite(vertices).all()
                     for vertices, _ in meshes.values())
        edges = np.sort(np.vstack((orig_faces[:, (0, 1)],
                                   orig_faces[:, (1, 2)],
                                   orig_faces[:, (2, 0)])), axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        euler = len(orig_vertices) - len(unique_edges) + len(orig_faces)
        unclosed_edges = int(np.count_nonzero(counts != 2))
        intersections = {name: mark_intersections(*meshes[name])[1]
                         for name in ("white", "pial")}
        passed = ordered_faces and finite and euler == 2 and unclosed_edges == 0 \
            and all(count == 0 for count in intersections.values())
        result[hemi] = {"vertices": len(orig_vertices),
                        "faces": len(orig_faces), "euler": int(euler),
                        "unclosed_edges": unclosed_edges,
                        "ordered_faces_preserved": ordered_faces,
                        "finite_coordinates": finite,
                        "intersecting_faces": intersections,
                        "status": "passed" if passed else "failed"}
    result["status"] = ("passed" if all(result[hemi]["status"] == "passed"
                                       for hemi in ("lh", "rh")) else "failed")
    return result


def _write_hemisphere_stats(subject: Path, hemi: str, volumes: dict, cache,
                            stage, device: str) -> None:
    """同一半球六套图谱共享统计缓存，white 与 pial 按文件版本分别保留。

    subject 为完整被试目录；hemi 为 lh/rh；volumes 单位 mm³；cache 为
    SurfaceStatsCache；stage 为计时调度函数；device 是 cache 的逻辑设备。
    写出六套标准十列 .stats，无返回值；IO/网格/设备错误向上传递。
    脑区体积沿用 -no-th3 定义，不读取 TH3 顶点 volume 图。
    """
    from .anatomical_stats_file import write_anatomical_stats

    stats = subject / "stats"
    for atlas, surface in (("aparc", "white"), ("aparc.a2009s", "white"),
                           ("aparc.DKTatlas", "white"), ("aparc", "pial"),
                           ("BA_exvivo", "white"), ("BA_exvivo.thresh", "white")):
        suffix = "aparc.pial" if surface == "pial" else atlas
        stage(f"stats_{hemi}_{suffix}", write_anatomical_stats,
              subject, hemi, atlas, surface, volumes, stats / f"{hemi}.{suffix}.stats",
              device=device, cache=cache)



def _hemisphere_operation(subject, hemi, device, threads, operation, *, assets,
                          binaries=None, registration_atlases=None,
                          sphere_normals_backend="numba", inflate_backend="native"):
    """可 exec 的半球阶段入口；维持现有子函数接口与依赖屏障。"""
    from .profiling import StageProfiler, configure_cuda_allocator
    subject, assets = Path(subject), Path(assets)
    binaries = {name: Path(value) for name, value in (binaries or {}).items()}
    profiler = StageProfiler(device=device, synchronize=False,
                             allocator=configure_cuda_allocator(device))
    steps = []
    def step(name, function, *args, **kwargs):
        try:
            return profiler.run(name, function, *args, **kwargs)
        finally:
            steps.append(dict(profiler.last_row))
    surf, mri, labels = (subject / name for name in ('surf', 'mri', 'label'))
    if operation == 'surface':
        value = step(f'surface_{hemi}', _surface_pair, subject, hemi,
                     mri / 'filled.mgz', mri / 'norm.mgz', device=device,
                     threads=threads, topology_binary=binaries['topology'],
                     inflate_binary=binaries['inflate'], intersection_binary=binaries['intersection'],
                     place_binary=binaries.get('white', binaries['metrics']), defect_binary=binaries.get('defect'),
                     assets=assets, defer_defects=True,
                     sphere_normals_backend=sphere_normals_backend, inflate_backend=inflate_backend)
    elif operation == 'register':
        from .mris_register_run import run_register_sphere
        value = step(f'register_{hemi}', run_register_sphere,
                     surf / f'{hemi}.sphere', surf / f'{hemi}.smoothwm',
                     surf / f'{hemi}.sulc', Path(registration_atlases[hemi]),
                     surf / f'{hemi}.sphere.reg', overlap_device='cpu', averaging_device=device)
        step(f'avg_curv_{hemi}', _run_avg_curv, binaries['paint'], subject, hemi,
             Path(registration_atlases[hemi]), assets)
    elif operation == 'annotation':
        from .gcsa_label_python import GCSAFeatureCache, label_surface
        value = {}
        cache = step(f'annot_{hemi}_prepare', GCSAFeatureCache,
                     subject, hemi, device=device)
        for atlas, prefix in (('aparc', 'DKaparc'), ('aparc.a2009s', 'CDaparc'),
                              ('aparc.DKTatlas', 'DKTaparc')):
            atlas_file = assets / 'average' / f'{hemi}.{prefix}.atlas.acfb40.noaparc.i12.2016-08-02.gcs'
            value[atlas] = step(f'annot_{hemi}_{atlas}', label_surface, subject, hemi,
                  atlas_file, assets / 'lib/bem/ic4.tri', assets / 'lib/bem/ic7.tri',
                  labels / f'{hemi}.{atlas}.annot', device=device, prepared=cache)
    elif operation == 'finish_surface':
        value = step(f'finish_surface_{hemi}', _finish_cortical_surface, subject,
                     hemi, binaries['metrics'], assets, device=device, threads=threads, defer_metrics=True,
                     white_binary=binaries.get('white'))
    else:
        raise ValueError(f'unknown hemisphere operation: {operation}')
    return {'result': value, 'stages': steps}


def _normalization_controls_options(backend: str, device: str) -> dict:
    """校验控制点邻域后端，返回两轮归一化共用的具名选项。

    backend 默认由入口传入 cpu，保留原 CPU 邻域；torch 复用同规则的
    GPU 邻域，device 必须为 cuda:N。只校验名称，不初始化 CUDA 或创建
    输出。返回空 dict 或 controls_neighbor_backend 字符串参数；非法值
    抛 ValueError。不改变强度、坐标、TF32 或其余归一化步骤。
    """
    if backend not in {"cpu", "torch"}:
        raise ValueError("normalization_controls_backend must be cpu or torch")
    if backend == "torch":
        target = torch.device(device)
        if target.type != "cuda" or target.index is None:
            raise ValueError("torch normalization controls require an explicit cuda:N device")
        return {"controls_neighbor_backend": "torch"}
    return {}


def _normalization_initial_bias_options(backend: str, device: str) -> dict:
    """校验第二轮初始偏置后端，返回aseg归一化的具名选项。

    backend为cpu/torch，device为入口设备；cpu返回空dict保留原路径。
    torch须cuda:N，复用已有传播/平滑，返回initial_bias_backend参数；
    其余有序控制点规则、float64除后乘及float32输出保持。不创建输出
    或初始化CUDA；非法后端抛ValueError，设备解析异常沿PyTorch传播。
    没有独立官方命令，空间/单位与第二轮归一化输入相同。
    """
    if backend not in {"cpu", "torch"}:
        raise ValueError("normalization_initial_bias_backend must be cpu or torch")
    if backend == "torch":
        _normalization_controls_options("torch", device)
        return {"initial_bias_backend": "torch"}
    return {}


def _validate_inflate_backend(backend: str, device: str, hemisphere_workers: int) -> None:
    """校验生产标准inflate后端；不初始化CUDA或创建输出。

    native保留默认流程；torch须cuda:N与两个隔离半球worker，确保只在
    fresh子进程启用已验证缓存，父CUDA和注册策略保持。非法参数抛异常。
    不改变nofix/qsphere或标准sphere步骤，真实输出/速度见inflate专页。
    """
    if backend not in {"native", "torch"}:
        raise ValueError("inflate_backend must be native or torch")
    if backend == "torch":
        selected = torch.device(device)
        if selected.type != "cuda" or selected.index is None:
            raise ValueError("torch inflate_backend requires explicit cuda:N")
        if hemisphere_workers != 2:
            raise ValueError("torch inflate_backend requires two cached hemisphere workers")


def _run_recon_all_python(t1: str | Path, subject_dir: str | Path,
                         weights_dir: str | Path, assets_dir: str | Path,
                         *, device: str = "cuda:0", threads: int = 4,
                         native_bin_dir: str | Path | None = None,
                         profile_stages: bool = False,
                         cuda_allocator_cache: str = "auto",
                         hemisphere_workers: int = 1,
                         native_optimizations: str = "auto",
                         n4_backend: str = "native",
                         n4_execution: str = "in-process",
                         normalization_controls_backend: str = "cpu",
                         normalization_initial_bias_backend: str = "cpu",
                         wm_backend: str = "native",
                         wm_execution: str = "in-process",
                         wm_edit_backend: str = "native",
                         defects_backend: str = "native",
                         sphere_normals_backend: str = "numba",
                         inflate_backend: str = "native",
                         gca_inverse_backend: str = "cpu",
                         gca_candidate_chunk: int = 64,
                         gca_execution: str = "in-process",
                         fill_backend: str = "python",
                         backend: str = "native") -> dict:
    """从单幅原始 T1 连续生成 conform 体积、双侧表面和脑区统计。

    t1、subject_dir、weights_dir、assets_dir 是输入影像、空输出目录、
    已校验权重和资产的路径；native_bin_dir=None 时使用当前 Conda bin。
    device 默认 cuda:0，threads 默认 4；不自动使用 FP16/BF16。
    n4_backend默认native保留Conda ITK；torch复用完整200轮PyTorch N4。
    n4_execution默认in-process；isolated仅Torch/cuda:N，完整缓存exec
    保留父CUDA/精度，输入输出SHA、实际迭代和子显存另记，不读取参考。
    normalization_controls_backend默认cpu；torch复用两轮归一化的GPU邻域，
    须显式cuda:N。控制点规则、有序离群清理及其余偏置步骤保持原算法。
    normalization_initial_bias_backend默认cpu；torch只在第二轮复用GPU
    初始偏置传播/平滑，保留CPU距离/排序、原除乘顺序和float32输出。
    inflate_backend默认native；torch复用完整标准inflated/sulc GPU算法，
    仅支持显式cuda:N和hemisphere_workers=2，在surface子exec局部启用
    缓存。nofix、球面优化/配准和父进程CUDA策略保持，失败不回退。
    native_optimizations=auto在CUDA使用已有FNIT Torch评分+Python EM；CPU
    查询独立产物能力，4线程可使用验证过的GCA缓存。white使用专用快速
    程序，pial保留原程序；original固定原生GCA用于控制。
    native_optimizations=torch 将 GCA 的候选平移/线性搜索切换到 FNIT
    的分块 PyTorch CUDA scorer；EM 精修仍为 Python FP32，并在报告中明确标注。
    defects_backend=native 使用原生投射；torch 使用完整 PyTorch 投射，仍按
    左侧清零、右侧合并执行。标签值/空间保持，随机调色板改为固定颜色。
    wm_edit_backend 默认 native；torch-hybrid 将完整 WM/aseg 编辑的静态
    子步骤交给 CUDA，有序反馈仍复用 Numba CPU，几何/存储 guard 失败即报错。
    sphere_normals_backend 默认 numba；torch 只替换标准 sphere 的有序面法向，
    完整目标函数、迭代决策及 CPU finish 保持。两个实验后端均要求 CUDA。
    wm_backend=torch-optimized复用Torch直方图与缓存平面几何，有序反馈为Numba。
    wm_execution=in-process保留父缓存；isolated只允许torch-optimized，
    子exec启用缓存并返回完整阶段报告，父CUDA状态保持。
    gca_inverse_backend默认cpu，torch按同公式批量求逆；gca_candidate_chunk
    默认64，只改变完整候选的分块，非默认设置须使用CUDA的Torch GCA后端。
    gca_execution=in-process保留父缓存策略；isolated在独立exec内启用缓存，
    子进程结束释放GPU资源，不修改父CUDA状态；仅CUDA Torch GCA可用。
    fill_backend默认python；numba复用有序CPU堆，torch-numba另用CUDA初始边界。
    三者均保留切割、距离场与标签规则；torch-numba要求CUDA，失败不回退。
    hemisphere_workers 默认1保持串行，2用独立 exec 半球进程；总 threads
    在双侧之间平分（奇数向下取整），父进程保留依赖屏障和共享发布。
    profile_stages=False 不增加阶段 CUDA 同步；True 分别记录前同步、函数、
    后同步和父子 CPU 秒数。cuda_allocator_cache=auto 延续首次 CUDA 调用
    关闭缓存的策略，并保留已初始化 API 的 allocator；enabled/disabled
    仅可在初始化前显式选择。total_seconds 包含校验、加载、传输及输出读写。
    成功返回与 fnit-native-free-run.json 相同的字典，含输出路径、耗时、
    网格检查及程序来源。失败抛异常，已开始的阶段另保存失败报告。
    体积为 1 mm conform 网格，表面使用 surface RAS（mm）；完整参数、
    输出结构、限制、官方命令和真实数据见 docs/recon_all/README.md。
    """
    normalization_options = _normalization_controls_options(normalization_controls_backend, device)
    initial_bias_options = _normalization_initial_bias_options(normalization_initial_bias_backend, device)
    _validate_inflate_backend(inflate_backend, device, hemisphere_workers)
    from .input_n4_chain import validate_n4_execution
    validate_n4_execution(n4_backend=n4_backend, n4_execution=n4_execution, device=device)
    if backend not in {"native", "python-gpu"}:
        raise ValueError("backend must be native or python-gpu")
    if wm_backend not in {"native", "torch", "torch-optimized"}:
        raise ValueError("wm_backend must be native, torch or torch-optimized")
    if wm_backend != "native" and not str(device).startswith("cuda"):
        raise ValueError("Torch WM backends require an explicit CUDA device")
    if wm_execution not in {"in-process", "isolated"}:
        raise ValueError("wm_execution must be in-process or isolated")
    if wm_execution == "isolated" and wm_backend != "torch-optimized":
        raise ValueError("isolated WM requires torch-optimized backend")
    if gca_inverse_backend not in {"cpu", "torch"}:
        raise ValueError("gca_inverse_backend must be cpu or torch")
    if isinstance(gca_candidate_chunk, bool) or not isinstance(gca_candidate_chunk, int) or gca_candidate_chunk < 1:
        raise ValueError("gca_candidate_chunk must be a positive integer")
    if gca_execution not in {"in-process", "isolated"}:
        raise ValueError("gca_execution must be in-process or isolated")
    if (gca_inverse_backend != "cpu" or gca_candidate_chunk != 64 or gca_execution != "in-process") and (
        native_optimizations == "original" or not str(device).startswith("cuda")
    ):
        raise ValueError("GCA candidate options require the CUDA Torch GCA backend")
    if fill_backend not in {"python", "numba", "torch-numba"}:
        raise ValueError("fill_backend must be python, numba or torch-numba")
    if fill_backend == "torch-numba" and not str(device).startswith("cuda"):
        raise ValueError("fill_backend='torch-numba' requires an explicit CUDA device")
    if wm_edit_backend not in {"native", "torch-hybrid"}:
        raise ValueError("wm_edit_backend must be native or torch-hybrid")
    if wm_edit_backend == "torch-hybrid" and not str(device).startswith("cuda"):
        raise ValueError("wm_edit_backend='torch-hybrid' requires an explicit CUDA device")
    if defects_backend not in {"native", "torch"}:
        raise ValueError("defects_backend must be native or torch")
    if sphere_normals_backend not in {"numba", "torch"}:
        raise ValueError("sphere_normals_backend must be numba or torch")
    if sphere_normals_backend == "torch" and not str(device).startswith("cuda"):
        raise ValueError("sphere_normals_backend='torch' requires an explicit CUDA device")
    if backend == "python-gpu":
        # Fail before validating/creating subject output and, critically,
        # before resolving any Conda native binary.  The capability report is
        # machine-readable so each blocked stage can be migrated and tested
        # independently rather than silently falling back to C++.
        from .python_gpu_profile import require_complete
        require_complete(device=device)
    started = time.perf_counter()
    from .hemisphere_parallel import validate_hemisphere_workers
    validate_hemisphere_workers(hemisphere_workers, threads)
    from .profiling import StageProfiler, configure_cuda_allocator, autocast_state
    from fnit.synthseg_parc import SynthSeg
    from .brain_volume_stats_python import compute_brain_volume_stats
    from .ca_normalize_python import run_ca_normalize
    from .gcsa_label_python import GCSAFeatureCache, label_surface
    from .input_talairach_chain import run_input_talairach_chain
    from .mri_mask_gpu import mask_volume
    from .n4_wrapper import make_nu
    from .normalization import normalize_t1
    from .segstats_aseg_python import write_aseg_stats
    from .segstats_wmparc_python import write_wmparc_stats

    if threads < 1:
        raise ValueError("threads must be positive")
    if hemisphere_workers == 2 and any(autocast_state(kind)["enabled"] for kind in ("cpu", "cuda")):
        raise ValueError("parallel hemispheres require caller autocast disabled; FP16/BF16 is not authorized")
    allocator = configure_cuda_allocator(device, cuda_allocator_cache)
    t1, subject = Path(t1).resolve(), Path(subject_dir).resolve()
    weights, assets = Path(weights_dir).resolve(), Path(assets_dir).resolve()
    if not t1.is_file() or not weights.is_dir() or not assets.is_dir():
        raise FileNotFoundError("T1, weights, and assets must exist")
    if subject.exists() and any(subject.iterdir()):
        raise ValueError("subject_dir must be empty")
    _validate_standard_python_dependencies()
    from .mni_aux_chain import validate_mni_aux_assets
    from .assets import validate_core_assets

    validate_mni_aux_assets(weights, assets)
    from fnit.weights import WEIGHT_FILES, verify_file
    deform_weight = weights / "synthmorph.deform.3.h5"
    _, deform_size, deform_sha256 = WEIGHT_FILES[deform_weight.name]
    if not verify_file(deform_weight, deform_size, deform_sha256):
        raise ValueError(f"Missing or invalid nonlinear registration weight: {deform_weight}")
    validate_core_assets(assets)
    native_bin_dir = _native_bin_directory(native_bin_dir)
    native_em = _native_em_register_binary(native_bin_dir)
    n4_binary = (_native_binary(native_bin_dir, "fnit_n4_itk")
                 if n4_backend == "native" else None)
    topology_binary = _native_topology_binary(native_bin_dir)
    metrics_binary = _native_surface_metrics_binary(native_bin_dir)
    from .native_runtime_selection import select_native_optimizations
    native_selection = select_native_optimizations(native_bin_dir, threads=threads,
                                                   mode=native_optimizations, device=device)
    white_binary = _native_binary(native_bin_dir, Path(native_selection["white_binary"]).name)
    inflate_binary = _native_inflate_binary(native_bin_dir)
    intersection_binary = _native_binary(native_bin_dir, "mris_remove_intersection")
    paint_binary = _native_binary(native_bin_dir, "mrisp_paint")
    curvature_stats_binary = _native_binary(native_bin_dir, "mris_curvature_stats")
    defect_binary = (_native_binary(native_bin_dir, "mri_label2vol")
                     if defects_backend == "native" else None)
    warp_binaries = tuple(_native_binary(native_bin_dir, name) for name in
                          ("mri_warp_convert", "mri_ca_register", "mri_convert"))
    wm_segment_binary = (_native_binary(native_bin_dir, "mri_segment")
                         if wm_backend == "native" else None)
    wm_edit_binary = (_native_binary(native_bin_dir, "mri_edit_wm_with_aseg")
                      if wm_edit_backend == "native" else None)
    registration_atlases = {hemi: _folding_atlas(assets, hemi)
                            for hemi in ("lh", "rh")}
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    validation_seconds = time.perf_counter() - started
    pipeline_started = time.perf_counter()
    profile = "single-t1-standard"
    caller_autocast = {kind: autocast_state(kind) for kind in ("cpu", "cuda")}
    report: dict = {"profile": profile, "input": str(t1),
                    "subject_dir": str(subject), "device": device,
                    "backend": backend,
                    "n4_binary": ({"binary": str(n4_binary[0]),
                                   "sha256": n4_binary[1]} if n4_binary else None), "threads": threads,
                    "precision": {"matmul_tf32_default": True,
                                  "cudnn_tf32_default": True,
                                  "cuda_policy_applied": torch.device(device).type == "cuda",
                                  "fp16_or_bf16_requested_by_fnit": False,
                                  "caller_autocast": caller_autocast,
                                  "fp16_or_bf16_enabled": any(
                                      value["enabled"] for value in caller_autocast.values()),
                                  "fp32_exceptions": ["SynthStrip", "SynthSeg",
                                                      "Talairach affine",
                                                      "MNI nonlinear CUDA",
                                                      "MNI152 affine matmul",
                                                      "EntoWM/MCA-dura/vsinus cuDNN"],
                                  "auxiliary_convolution_policy": "scoped cuDNN FP32; same-input GPU/CPU label regression; matmul TF32 preserved"},
                    "gpu_memory_mode": allocator["effective"], "cuda_allocator": allocator,
                    "timing": {"profile_stages": profile_stages,
                               "total_scope": "API entry through validation, loading, transfers and output writes",
                               "validation_seconds": validation_seconds},
                    "stages": [], "status": "running"}
    report["hemisphere_scheduling"] = {"workers": hemisphere_workers, "total_thread_budget": threads, "groups": [], "mode": "serial" if hemisphere_workers == 1 else "independent-exec-private-subjects"}
    report["n4_configuration"] = {
        "backend": n4_backend, "execution": n4_execution,
        "implementation": "Conda ITK" if n4_backend == "native" else "FNIT complete PyTorch N4",
        "device": "cpu" if n4_backend == "native" else device,
        "production_default_changed": False,
        "known_strict_intensity_differences": n4_backend == "torch"}
    report["normalization_configuration"] = {
        "controls_neighbor_backend": normalization_controls_backend,
        "initial_bias_backend": normalization_initial_bias_backend,
        "requested_device": device,
        "stages": ["T1_normalize", "brain_second_normalize"],
        "implementation": ("FNIT resident PyTorch/Triton neighbor buffers"
                           if normalization_controls_backend == "torch" else "FNIT SciPy CPU neighbors"),
        "ordered_control_selection": "CPU; original update and rejection rules",
        "remaining_bias_steps": "existing implementation; device unchanged",
        "production_default_changed": False}
    gca_backend = native_selection["em_backend"]
    report["gca_registration"] = {
        "implementation": ("FNIT PyTorch candidate scorer + Python EM"
                           if gca_backend == "torch" else "native-c++"),
        "binary": str(native_em[0]), "sha256": native_em[1],
        "backend": gca_backend,
        "inverse_backend": gca_inverse_backend if gca_backend == "torch" else None,
        "candidate_chunk": gca_candidate_chunk if gca_backend == "torch" else None,
        "execution": gca_execution if gca_backend == "torch" else "native-subprocess",
        "device": device if gca_backend == "torch" else "cpu",
        "strict_native_em": gca_backend != "torch"}
    report["native_optimizations"] = native_selection
    report["topology_repair"] = {
        "implementation": "native-c++", "binary": str(topology_binary[0]),
        "sha256": topology_binary[1],
        "intersection_binary": str(intersection_binary[0]),
        "intersection_sha256": intersection_binary[1],
        "upstream": "intensity-derived WM and aseg-guided fill"}
    report["white_matter_chain"] = {
        "implementation": ("FNIT PyTorch hybrid segmentation + Conda C++ edit"
                           if wm_backend != "native" else "Python + Conda C++"),
        "mri_segment_sha256": wm_segment_binary[1] if wm_segment_binary else None,
        "segmentation_backend": wm_backend,
        "segmentation_execution": wm_execution if wm_backend != "native" else "native-subprocess",
        "edit_backend": wm_edit_backend,
        "edit_implementation": "Conda C++" if wm_edit_binary else "ordered Numba + PyTorch CUDA",
        "mri_edit_wm_with_aseg_sha256": wm_edit_binary[1] if wm_edit_binary else None}
    report["white_preaparc"] = {
        "implementation": "Python MNI/aux/finalsurfs + Conda C++ placement",
        "binary": str(white_binary[0]), "sha256": white_binary[1],
        "final_smoothwm": "Python 3 passes on CPU",
        "final_white_pial": "Conda source-built C++ white and pial after annotation"}
    report["surface_metrics"] = {
        "implementation": ("PyTorch CUDA" if torch.device(device).type == "cuda"
                           else "Conda source-built C++"),
        "binary": str(metrics_binary[0]),
        "sha256": metrics_binary[1], "upstream": "placed final white/pial"}
    report["sphere_generation"] = {
        "implementation": ("FNIT Torch standard inflate + Conda nofix inflate + Python quick/standard sphere"
                           if inflate_backend == "torch" else
                           "Conda mris_inflate + Python quick/standard sphere"),
        "mris_inflate": {"binary": str(inflate_binary[0]), "sha256": inflate_binary[1]},
        "normals_backend": sphere_normals_backend,
        "standard_inflate_backend": inflate_backend,
        "nofix_inflate_backend": "native",
        "surface_worker_cache": "enabled" if inflate_backend == "torch" else "inherit",
        "normals_device": device if sphere_normals_backend == "torch" else "cpu",
        "finish_device": "cpu", "upstream": "repaired topology"}
    registration_device = torch.device(device)
    if registration_device.type == "cuda" and registration_device.index is None:
        registration_device = torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_initialized() else 0)
    report["sphere_registration"] = {
        "implementation": "Python/Numba + ordered CUDA averaging" if registration_device.type == "cuda" else "Python/Numba",
        "averaging_device": str(registration_device), "overlap_device": "cpu",
        "hemisphere_seconds": {}, "reports": {}}
    report["extra_curvature"] = {
        "mrisp_paint_sha256": paint_binary[1],
        "mris_curvature_stats_sha256": curvature_stats_binary[1]}
    report["defects_volume"] = ({"backend": "native", "binary": str(defect_binary[0]),
                                 "sha256": defect_binary[1]} if defect_binary else
                                {"backend": "torch", "device": device,
                                 "palette": "deterministic colors; original label names and values"})
    if defects_backend == "torch":
        from . import defects_label_volume_torch
        report["defects_volume"]["source_sha256"] = hashlib.sha256(
            Path(defects_label_volume_torch.__file__).read_bytes()).hexdigest()
    report["mni_nonlinear"] = {
        "implementation": ("PyTorch deform + FNIT GPU warp conversion/inversion/check"
                           if torch.device(device).type == "cuda" else
                           "PyTorch deform + Conda source-built warp conversion"),
        "postprocess_backend": "gpu" if torch.device(device).type == "cuda" else "conda",
        "native_programs_used": torch.device(device).type != "cuda",
        "device": device,
        "precision": "FP32 CUDA exception" if torch.device(device).type == "cuda" else "FP32 CPU",
        "native_sha256": {name: binary[1] for name, binary in zip(
            ("mri_warp_convert", "mri_ca_register", "mri_convert"), warp_binaries)}}

    profiler = StageProfiler(device=device, synchronize=profile_stages, allocator=allocator)

    def stage(name, function, *args, **kwargs):
        """记录读写墙钟及原函数内部秒数；内部异步计时不追加GPU同步或累加。"""
        try:
            value = profiler.run(name, function, *args, **kwargs)
        except Exception as error:
            report["stages"].append(profiler.last_row)
            report.update(status="failed", failed_stage=name, error=repr(error),
                          total_seconds=time.perf_counter() - started)
            subject.mkdir(parents=True, exist_ok=True)
            try:
                (subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))
            except Exception as metadata_error:
                error.add_note(f"Stage failure metadata could not be saved: {metadata_error!r}")
            raise
        row = profiler.last_row
        from .stage_metadata import extract_algorithm_seconds
        algorithm_seconds = extract_algorithm_seconds(value)
        if algorithm_seconds:
            row["algorithm_substep_seconds"] = algorithm_seconds
            row["algorithm_substep_scope"] = (
                "original function timers, possibly asynchronous and nested; "
                "not additive with the synchronized stage wall")
        if isinstance(value, dict) and value.get("actual_forwards"):
            row["actual_forwards"] = value["actual_forwards"]
        if name == "n4" and isinstance(value, dict):
            row["n4_runtime"] = value
        if name == "mni_nonlinear" and isinstance(value, dict):
            row["precision"] = value.get("precision")
            row["postprocess_backend"] = value.get("postprocess_backend")
            row["postprocess"] = value.get("postprocess")
        if isinstance(value, dict) and "timings_seconds" in value:
            row["timings_seconds"] = value["timings_seconds"]
        if isinstance(value, dict) and isinstance(value.get("timing"), dict):
            row["substep_seconds"] = value["timing"]
        if isinstance(value, dict) and isinstance(value.get("seconds"), dict):
            row["substep_seconds"] = value["seconds"]
        if isinstance(value, dict) and value.get("talairach_child_gpu"):
            row["talairach_child_gpu"] = value["talairach_child_gpu"]
            for key in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
                report[key] = max(report.get(key, 0), value["talairach_child_gpu"][key])
        for key in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
            if key in row:
                report[key] = max(report.get(key, 0), row[key])
        report["stages"].append(row)
        (subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))
        return value

    initial = stage("input_talairach", run_input_talairach_chain,
                    t1, subject, weights, assets, device=device, threads=threads)
    mri, surf, labels, stats = (subject / name for name in ("mri", "surf", "label", "stats"))
    for folder in (surf, labels, stats, mri / "tmp", subject / "scripts"):
        folder.mkdir(parents=True, exist_ok=True)
    nu0 = mri / "tmp/nu0.mgz"
    from .input_n4_chain import run_n4_stage
    n4_profile = subject / "scripts/n4.profile.json"
    report["n4_runtime"] = stage(
        "n4", run_n4_stage, input_path=mri / "orig.mgz", output_path=nu0,
        n4_backend=n4_backend, n4_execution=n4_execution,
        native_binary=n4_binary[0] if n4_binary else None,
        device=device, threads=threads, profile=profile_stages,
        report_path=subject / "scripts/n4-isolated.json" if n4_execution == "isolated"
        else n4_profile if n4_backend == "native" else None)
    stage("nu", make_nu, mri / "orig.mgz", nu0,
          initial["talairach_xfm"], mri / "nu.mgz")
    stage("T1_normalize", normalize_t1, mri / "nu.mgz",
          initial["talairach_xfm"], mri / "T1.mgz", device=device, **normalization_options)
    stage("brainmask", mask_volume, mri / "T1.mgz",
          initial["synthstrip"], mri / "brainmask.mgz", device=device)
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    def run_synthseg_and_write():
        """构造后应用卷积精度策略；阶段计时包括分割图和体积 CSV 写出。"""
        model = SynthSeg(weights=weights, device=device, threads=threads, cudnn_tf32=False)
        try:
            result = model(mri / "orig.mgz", keep_geometry=True,
                           color_lut=assets / "FreeSurferColorLUT.txt")
        finally:
            report["precision"]["SynthSeg_actual_forward"] = model.segmenter.precision
        result.segmentation.save(str(mri / "synthseg.rca.mgz"))
        result.write_volumes_csv(mri / "orig.mgz", stats / "synthseg.vol.csv")
        return result

    try:
        result = stage("SynthSeg", run_synthseg_and_write)
    finally:
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    stiv_mm3 = result.total_intracranial_mm3
    report["precision"]["SynthSeg_actual_forward"] = result.precision
    lta = mri / "transforms/talairach.lta"
    gca = assets / "average/RB_all_2020-01-02.gca"
    report["gca_registration"]["stage_result"] = stage(
          "mri_em_register", _run_native_em_register, native_em[0], mri, gca, assets,
          backend=native_selection["em_backend"], binary_sha256=native_em[1],
          threads=threads, device=device, inverse_backend=gca_inverse_backend,
          candidate_chunk=gca_candidate_chunk, execution=gca_execution)
    stage("mri_ca_normalize", run_ca_normalize, mri / "nu.mgz",
          mri / "brainmask.mgz", gca,
          lta, mri / "norm.mgz", mri / "ctrl_pts.mgz")
    report["corpus_callosum"] = stage("mri_cc", _segment_callosum, mri)
    from .ants_denoise_python import denoise_volume
    from .fill_cutting_plane_python import fill_mgz
    from .normalization.aseg_pipeline import normalize_t1_aseg
    from .pretess_python import pretess_mgh
    from .sclimbic import mri_entowm_seg
    from .wm_edits_python import fix_ento_wm
    wm_edit_function = fix_ento_wm
    wm_edit_device = {}
    if torch.device(device).type == "cuda":
        from .wm_edits_gpu import fix_ento_wm_gpu
        wm_edit_function = fix_ento_wm_gpu
        wm_edit_device = {"device": device}
    report["white_matter_chain"]["post_edit_implementation"] = (
        "FNIT PyTorch CUDA" if wm_edit_device else "FNIT Python/Numba CPU")

    stage("brain_second_normalize", normalize_t1_aseg,
          mri / "norm.mgz", mri / "aseg.presurf.mgz",
          mri / "brainmask.mgz", mri / "brain.mgz", device=device,
          **normalization_options, **initial_bias_options)
    auxiliary_forwards = []
    with torch.backends.cudnn.flags(allow_tf32=False):
        stage("entowm", mri_entowm_seg, mri / "nu.mgz", mri / "entowm.mgz",
              weights, device=device, stats_path=stats / "entowm.stats",
              talairach_lta=mri / "transforms/talairach.xfm.lta",
              precision_report=auxiliary_forwards)
    report["precision"]["EntoWM_actual_forwards"] = auxiliary_forwards
    stage("ants_denoise", denoise_volume, mri / "brain.mgz",
          mri / "antsdn.brain.mgz")
    if wm_backend != "native":
        report["white_matter_chain"]["segmentation_runtime"] = stage(
              "mri_segment", _run_torch_wm_segment, mri, device=device,
              optimized=wm_backend == "torch-optimized", execution=wm_execution,
              threads=threads, profile_stages=profile_stages)
    else:
        stage("mri_segment", _run_native_wm_segment,
              wm_segment_binary[0], mri, assets)
    if wm_edit_backend == "torch-hybrid":
        report["white_matter_chain"]["edit_runtime"] = stage(
            "mri_edit_wm_with_aseg", _run_torch_wm_edit, mri, device=device)
    else:
        stage("mri_edit_wm_with_aseg", _run_native_wm_edit,
              wm_edit_binary[0], mri, assets)
    stage("wm_pretess", pretess_mgh, mri / "wm.asegedit.mgz", "wm",
          mri / "norm.mgz", mri / "wm.mgz")
    stage("wm_fix_ento", wm_edit_function, mri / "wm.mgz",
          mri / "entowm.mgz", mri / "wm.mgz", level=3,
          left_value=255, right_value=255, **wm_edit_device)
    stage("wm_fix_acj", wm_edit_function, mri / "wm.mgz",
          mri / "aseg.presurf.mgz", mri / "wm.mgz", level=3,
          left_value=255, right_value=255, acj=True, **wm_edit_device)
    fill_options = {}
    if fill_backend != "python":
        fill_options["cc_marching_backend"] = "numba"
    if fill_backend == "torch-numba":
        fill_options.update(cc_boundary_backend="torch", device=device)
    report["fill"] = {"backend": fill_backend,
                      "marching_device": "cpu", "ordered_feedback_preserved": True,
                      "initial_boundary_device": device if fill_backend == "torch-numba" else "cpu"}
    stage("mri_fill", fill_mgz, mri / "wm.mgz",
          mri / "aseg.presurf.mgz", lta,
          assets / "SubCorticalMassLUT.txt", mri / "filled.mgz",
          subject / "scripts/ponscc.cut.log", **fill_options)
    # 固定 recon-all 脚本此处执行 cp filled.mgz filled.auto.mgz。
    stage("filled_auto_checkpoint", shutil.copyfile,
          mri / "filled.mgz", mri / "filled.auto.mgz")
    try:
        report["Synth_auxiliary_runtime"] = _run_white_mri_chain(subject, weights, assets, threads,
                             tuple(binary[0] for binary in warp_binaries), stage,
                             device=device)
    except Exception as error:
        if report["status"] != "failed":
            report.update(status="failed", failed_stage="white_mri_chain",
                          error=f"{type(error).__name__}: {error}")
            (subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))
        raise
    if hemisphere_workers == 2:
        from .hemisphere_parallel import run_hemisphere_group, HemisphereGroupError
        common = {'assets': str(assets),
                  'sphere_normals_backend': sphere_normals_backend,
                  'inflate_backend': inflate_backend,
                  'binaries': {name: str(value) for name, value in (
                      ('topology', topology_binary[0]), ('inflate', inflate_binary[0]),
                      ('intersection', intersection_binary[0]), ('metrics', metrics_binary[0]),
                      ('white', white_binary[0]),
                      *((('defect', defect_binary[0]),) if defect_binary else ()),
                      ('paint', paint_binary[0]))},
                  'registration_atlases': {hemi: str(path) for hemi, path in registration_atlases.items()}}
        for operation in ('surface', 'register', 'annotation', 'finish_surface'):
            try:
                group = stage(f'{operation}_hemisphere_group', run_hemisphere_group,
                              subject, operation, device=str(registration_device), threads=threads,
                              workers=hemisphere_workers, profile_stages=profile_stages, kwargs=common,
                              cuda_allocator_cache=("enabled" if operation == "surface" and
                                                    inflate_backend == "torch" else "inherit"))
            except HemisphereGroupError as error:
                report['hemisphere_scheduling']['groups'].append(error.report)
                try:
                    (subject / 'fnit-native-free-run.json').write_text(json.dumps(report, indent=2))
                except Exception as metadata_error:
                    error.add_note(f"Group failure metadata could not be saved: {metadata_error!r}")
                raise
            report['hemisphere_scheduling']['groups'].append(group)
            for hemi in ('lh', 'rh'):
                value = group['values'][hemi]
                # Worker 步骤用于诊断，不加到父组墙钟形成重复计时。
                result = value['result']
                if operation == 'surface':
                    report.setdefault('surfaces', {})[hemi] = result
                elif operation == 'register':
                    report['sphere_registration']['hemisphere_seconds'][hemi] = result['total_seconds_including_io']
                    report['sphere_registration']['reports'][hemi] = result
                elif operation == 'finish_surface':
                    report['surfaces'][hemi].update(result)
            if operation == 'finish_surface':
                # 实测GPU顶点指标独立exec并行更慢；放置并行、指标串行。
                for hemi in ('lh', 'rh'):
                    metrics_result = stage(f'finish_metrics_{hemi}', _finish_cortical_metrics,
                          subject, hemi, metrics_binary[0], assets, device=device)
                    report['surfaces'][hemi].update(metrics_result)
            if operation == 'surface':
                # 此诊断无下游计算依赖，但左右累计有先后依赖。
                for hemi in ('lh', 'rh'):
                    stage(f'defects_{hemi}', _run_defects_volume,
                          defect_binary[0] if defect_binary else None, subject, hemi, assets,
                          backend=defects_backend, device=device)
            (subject / 'fnit-native-free-run.json').write_text(json.dumps(report, indent=2))
        surf = subject / 'surf'
    else:
        for hemi in ("lh", "rh"):
            result = stage(f"surface_{hemi}", _surface_pair, subject, hemi,
                           mri / "filled.mgz", mri / "norm.mgz",
                           device=device, threads=threads,
                           topology_binary=topology_binary[0],
                           inflate_binary=inflate_binary[0],
                           intersection_binary=intersection_binary[0],
                           place_binary=white_binary[0],
                           defect_binary=defect_binary[0] if defect_binary else None,
                           assets=assets, defects_backend=defects_backend,
                           sphere_normals_backend=sphere_normals_backend, inflate_backend=inflate_backend)
            report.setdefault("surfaces", {})[hemi] = result
            (subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))

        from .mris_register_run import run_register_sphere
        for hemi in ("lh", "rh"):
            surf = subject / "surf"
            result = stage(f"register_{hemi}", run_register_sphere,
                           surf / f"{hemi}.sphere", surf / f"{hemi}.smoothwm",
                           surf / f"{hemi}.sulc", registration_atlases[hemi],
                           surf / f"{hemi}.sphere.reg", overlap_device="cpu",
                           averaging_device=str(registration_device))
            report["sphere_registration"]["hemisphere_seconds"][hemi] = result[
                "total_seconds_including_io"]
            report["sphere_registration"]["reports"][hemi] = result
            stage(f"avg_curv_{hemi}", _run_avg_curv, paint_binary[0], subject,
                  hemi, registration_atlases[hemi], assets)

        for hemi in ("lh", "rh"):
            cache = stage(f"annot_{hemi}_prepare", GCSAFeatureCache,
                          subject, hemi, device=device)
            for atlas, prefix in (("aparc", "DKaparc"),
                                  ("aparc.a2009s", "CDaparc"),
                                  ("aparc.DKTatlas", "DKTaparc")):
                atlas_file = assets / "average" / (
                    f"{hemi}.{prefix}.atlas.acfb40.noaparc.i12.2016-08-02.gcs")
                stage(f"annot_{hemi}_{atlas}", label_surface, subject, hemi,
                      atlas_file, assets / "lib/bem/ic4.tri",
                      assets / "lib/bem/ic7.tri",
                      labels / f"{hemi}.{atlas}.annot", device=device,
                      prepared=cache)
        for hemi in ("lh", "rh"):
            result = stage(f"finish_surface_{hemi}", _finish_cortical_surface,
                           subject, hemi, metrics_binary[0], assets,
                           device=device, threads=threads, white_binary=white_binary[0])
            report["surfaces"][hemi].update(result)
            (subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))
    stage("exvivo_annotations", _project_exvivo_annotations, subject, assets)
    from .surface_jacobian_gpu import jacobian_map
    from .vol2surf_contrast_python import write_contrast_percentage

    for hemi in ("lh", "rh"):
        # The GPU kernel remains available for explicit stage experiments, but
        # the complete 114k-vertex map is launch/I/O bound on the reference
        # H100 (1.155 s CUDA vs 0.079 s CPU in the paired real-subject test).
        # Keep the measured faster CPU path in the production critical path.
        stage(f"jacobian_{hemi}", jacobian_map,
              surf / f"{hemi}.white.preaparc", surf / f"{hemi}.sphere.reg",
              surf / f"{hemi}.jacobian_white", device="cpu")
        stage(f"contrast_{hemi}", write_contrast_percentage,
              subject, hemi, surf / f"{hemi}.w-g.pct.mgh", device="cpu")
    from .relabel_hypointensities_python import relabel_volume
    from .surf2volseg_fix_python import fix_presurf_volume
    from .volmask_python import write_ribbon

    stage("ribbon", write_ribbon, mri / "aseg.presurf.mgz", surf,
          mri, assets / "FreeSurferColorLUT.txt")
    stage("relabel_hypointensities", relabel_volume,
          mri / "aseg.presurf.mgz", surf, mri / "aseg.presurf.hypos.mgz")
    stage("aseg_ribbon_fix", fix_presurf_volume,
          mri / "aseg.presurf.hypos.mgz", mri / "ribbon.mgz",
          surf, labels, mri / "aseg.mgz")
    stage("project_aparc_volumes", _project_parcels, subject)
    stage("project_wmparc", _project_wmparc, subject)

    def compute_and_write_brain_volumes():
        """计算全脑体积并写出 brainvol.stats/tiv；体积单位为 mm³。"""
        values = compute_brain_volume_stats(subject, assets / "ASegStatsLUT.txt")
        (stats / "brainvol.stats").write_text("".join(
            f"# Measure {name}, {name}, {name}, {value:.6f}, mm^3\n"
            for name, value in values.items()))
        (stats / "synthseg.tiv.dat").write_text(f"{stiv_mm3:.6f}\n")
        return values

    volumes = stage("brain_volume_stats", compute_and_write_brain_volumes)
    stage("aseg_stats", write_aseg_stats, subject, assets / "ASegStatsLUT.txt",
          stats / "aseg.stats")
    stage("wmparc_stats", write_wmparc_stats, subject, assets / "WMParcStatsLUT.txt",
          stats / "wmparc.stats")
    from .surface_stats_cache import SurfaceStatsCache

    for hemi in ("lh", "rh"):
        with SurfaceStatsCache(device=device) as stats_cache:
            _write_hemisphere_stats(subject, hemi, volumes, stats_cache, stage, device)
            report.setdefault("surface_stats_cache", {})[hemi] = dict(stats_cache.counters)
        from .segstats_surface_snr_python import write_surface_snr_stats
        stage(f"stats_{hemi}_w-g.pct", write_surface_snr_stats,
              subject, hemi, stats / f"{hemi}.w-g.pct.stats")
        stage(f"stats_{hemi}_curv", _run_curvature_stats,
              curvature_stats_binary[0], subject, hemi, assets)
    from .expected_outputs import paths as expected_paths

    expected = expected_paths()
    present = {name: (subject / name).is_file() for name in expected}
    missing = [name for name, exists in present.items() if not exists]
    report["output_validation"] = {
        "profile": "single-t1-138", "expected": len(expected),
        "present": len(expected) - len(missing), "missing": missing,
        "status": "passed" if not missing else "failed"}
    report["mesh_validation"] = stage("mesh_validation", _validate_meshes, subject)
    report["numeric_validation"] = {"status": "not_run",
                                    "reason": "reference_subject_not_provided"}
    report["outputs"] = {name: str(subject / name) for name in expected if present[name]}
    valid = not missing and report["mesh_validation"]["status"] == "passed"
    report.update(status="complete" if valid else "incomplete",
                  total_seconds=time.perf_counter() - started)
    report["timing"]["pipeline_seconds"] = time.perf_counter() - pipeline_started
    (subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))
    if not valid:
        raise RuntimeError(f"recon-all output or mesh validation failed; "
                           f"see {subject / 'fnit-native-free-run.json'}")
    return report


def run_recon_all_python(t1: str | Path, subject_dir: str | Path,
                         weights_dir: str | Path, assets_dir: str | Path,
                         *, device: str = "cuda:0", threads: int = 4,
                         native_bin_dir: str | Path | None = None,
                         profile_stages: bool = False,
                         cuda_allocator_cache: str = "auto",
                         hemisphere_workers: int = 1,
                         native_optimizations: str = "auto",
                         n4_backend: str = "native",
                         n4_execution: str = "in-process",
                         normalization_controls_backend: str = "cpu",
                         normalization_initial_bias_backend: str = "cpu",
                         wm_backend: str = "native",
                         wm_execution: str = "in-process",
                         wm_edit_backend: str = "native",
                         defects_backend: str = "native",
                         sphere_normals_backend: str = "numba",
                         inflate_backend: str = "native",
                         gca_inverse_backend: str = "cpu",
                         gca_candidate_chunk: int = 64,
                         gca_execution: str = "in-process",
                         fill_backend: str = "python",
                         backend: str = "native") -> dict:
    """从原始单 T1 连续重建；输入、输出及坐标定义见 recon-all 中文说明。

    t1 为原始影像；subject_dir 须为空；weights_dir/assets_dir 为已校验资源；
    native_bin_dir=None 使用当前 Conda bin。device 默认 cuda:0，threads=4
    约束 Torch intraop 和调用线程的 Numba 掩码，退出时恢复调用方设置。
    hemisphere_workers=1保持串行，2启用私有被试目录的双侧 exec 进程，
    总 threads 平分，须至少2；返回 hemisphere_scheduling 组墙钟/worker报告，
    所有共享发布失败均传播到主报告。并行要求 caller autocast 关闭。
    backend=native 使用当前已验证的混合 FNIT/Conda 实现；backend=python-gpu
    是严格纯 Python/CUDA profile，若仍有未完成阶段会在创建输出前抛出结构化
    PurePythonGpuUnavailable，不回退到原生程序。native_optimizations=auto在CUDA使用已有Torch GCA评分+Python EM，CPU
    按原生产物能力选择缓存；white选择专用快速程序，pial保留原生。
    original固定原始原生实现。Torch GCA与原生仍有既有数值差异，单列报告。
    n4_backend默认native，torch使用完整N4；n4_execution默认in-process，
    isolated只允许Torch/cuda:N，复用局部缓存worker并记录完整阶段/子显存。
    非法组合在创建被试目录前报错，既有N4系统强度差异单列，不判整体等效。
    normalization_controls_backend=cpu保留原邻域；torch在两轮归一化复用
    同规则GPU邻域，须显式cuda:N，其余步骤/设备不变。实际选择写入报告。
    normalization_initial_bias_backend=cpu保留第二轮初始偏置；torch仅
    复用该轮既有GPU传播/平滑，要求cuda:N，除乘/输出精度与原规则相同。
    inflate_backend=native保留标准inflation；torch复用完整GPU算法生成
    同序inflated/sulc，须cuda:N与hemisphere_workers=2。仅surface组的
    fresh worker局部启用缓存；nofix、sphere及注册算法保持，父策略不变。
    非法组合在CUDA初始化和创建输出前抛ValueError，执行失败不回退。
    defects_backend=native 使用原生缺陷投射，torch 用本项目完整投射规则
    并保持双侧累积顺序；仅颜色表换成固定颜色，当前默认为 native。
    wm_edit_backend=native 调用独立构建编辑程序；torch-hybrid 使用 CUDA
    静态编辑和 Numba 有序核心，不是纯 GPU；输入证明不成立时抛异常。
    wm_backend=torch-optimized复用已有Torch直方图和缓存平面几何；有序反馈仍CPU。
    wm_execution=in-process保留父策略，isolated只允许torch-optimized，
    使用完整缓存worker，不改变父CUDA/TF32，记录完整exec墙钟。
    gca_inverse_backend=cpu、gca_candidate_chunk=64保留旧GCA；torch求逆和
    非默认分块要求CUDA Torch后端。gca_execution=in-process保留父策略，
    isolated使用新exec局部缓存，不修改父精度或已初始化allocator。
    fill_backend=python保留旧fill，numba复用
    有序CPU堆，torch-numba另用CUDA初始边界；所有实验均失败报错而不回退。
    sphere_normals_backend=numba 使用已有法向；torch 使用同设备有序
    Torch 法向，其余标准 sphere 阶段保持原计算，要求显式 CUDA 设备。
    profile_stages=False 不插入阶段同步；True 分列 CUDA 等待与父子 CPU 秒数。
    球面配准在 device 上执行完整有序 float32 梯度平均，其余目标函数、
    步长决策及末尾清理保持 CPU；不自动启用半精度，计时包含往返传输。
    cuda_allocator_cache=auto 保留已初始化 API 的 allocator，首次 CUDA
    默认关闭缓存；enabled/disabled 必须在初始化前选择。无自动半精度。
    返回完整路径、精度、线程、耗时和执行/完整性/网格检查字典；同时写 JSON。
    total_seconds 覆盖校验、线程设置/恢复、模型加载、传输、计算及数据读写；
    最终报告写出/CLI 启动仍由外层命令计时。体积为 conform 网格；表面为
    surface RAS/mm。Numba 请求超过初始线程容量、输入非法或阶段失败抛异常。
    不支持同一进程内多个线程并发更改全局 Torch 线程预算。
    阶段失败时补记本次线程恢复与公开 API 耗时；恢复失败将已有 complete
    改为 failed。阶段和恢复同时失败时保留阶段异常，附记恢复错误。
    失败报告读取/写入错误不遮盖原异常，也不改写先前运行的未变报告。
    thread_setup_and_restore_seconds 是公开/内部计时差，包含线程设置/恢复
    及内部末尾报告写出/返回开销，不能视为纯线程操作时间。
    阶段间异常可能没有完成内部 total_seconds；此时仍记录完整公开耗时，
    差值为 None，并在 scope 中注明内部计时尚未完成，不推算缺失时间。
    """
    tick = time.perf_counter()
    _normalization_controls_options(normalization_controls_backend, device)
    _normalization_initial_bias_options(normalization_initial_bias_backend, device)
    _validate_inflate_backend(inflate_backend, device, hemisphere_workers)
    from .thread_budget import thread_budget

    report_path = Path(subject_dir) / "fnit-native-free-run.json"

    def report_version():
        """返回现有报告的 stat 版本；不存在或不可访问时不读取其内容。"""
        try:
            stat = report_path.stat()
            return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        except OSError:
            return None

    entry_version = report_version()
    report = budget = pipeline_error = None

    def record_public_timing(value: dict, wall: float) -> None:
        """更新本次公开入口的秒数/线程记录，最终元数据写出不计入 wall。"""
        if budget is not None:
            value["thread_budget"] = budget
        timing = value.setdefault("timing", {})
        timing["total_scope"] = (
            "API entry through validation, thread restoration, loading, transfers "
            "and output writes; excludes final public metadata write")
        internal_seconds = value.get("total_seconds")
        if internal_seconds is None:
            timing["thread_setup_and_restore_seconds"] = None
            timing["thread_setup_and_restore_scope"] = (
                "unavailable: internal total_seconds was not finalized; "
                "public API wall includes thread setup/restoration")
        else:
            timing["thread_setup_and_restore_seconds"] = wall - internal_seconds
            timing["thread_setup_and_restore_scope"] = (
                "public-wrapper residual including thread setup/restoration and "
                "internal final report write/return overhead")
        value["total_seconds"] = wall

    try:
        with thread_budget(threads=threads) as budget:
            try:
                report = _run_recon_all_python(
                    t1=t1, subject_dir=subject_dir, weights_dir=weights_dir, assets_dir=assets_dir,
                    device=device, threads=threads, native_bin_dir=native_bin_dir,
                    profile_stages=profile_stages, cuda_allocator_cache=cuda_allocator_cache,
                    **({"hemisphere_workers": hemisphere_workers} if hemisphere_workers != 1 else {}),
                    **({"native_optimizations": native_optimizations} if native_optimizations != "auto" else {}),
                    **({"n4_backend": n4_backend} if n4_backend != "native" else {}),
                    **({"n4_execution": n4_execution} if n4_execution != "in-process" else {}),
                    **({"normalization_controls_backend": normalization_controls_backend}
                       if normalization_controls_backend != "cpu" else {}),
                    **({"normalization_initial_bias_backend": normalization_initial_bias_backend}
                       if normalization_initial_bias_backend != "cpu" else {}),
                    **({"wm_backend": wm_backend} if wm_backend != "native" else {}),
                    **({"wm_execution": wm_execution} if wm_execution != "in-process" else {}),
                    **({"wm_edit_backend": wm_edit_backend} if wm_edit_backend != "native" else {}),
                    **({"defects_backend": defects_backend} if defects_backend != "native" else {}),
                    **({"sphere_normals_backend": sphere_normals_backend}
                       if sphere_normals_backend != "numba" else {}),
                    **({"inflate_backend": inflate_backend} if inflate_backend != "native" else {}),
                    **({"gca_inverse_backend": gca_inverse_backend} if gca_inverse_backend != "cpu" else {}),
                    **({"gca_candidate_chunk": gca_candidate_chunk} if gca_candidate_chunk != 64 else {}),
                    **({"gca_execution": gca_execution} if gca_execution != "in-process" else {}),
                    **({"fill_backend": fill_backend} if fill_backend != "python" else {}),
                    backend=backend)
            except Exception as error:
                pipeline_error = error
                raise
    except Exception as error:
        wall = time.perf_counter() - tick
        original_error = pipeline_error if pipeline_error is not None else error
        restoration_error = error if pipeline_error is not None and error is not pipeline_error else None
        try:
            # 输入校验可能拒绝已有被试目录；不得把以前的报告改成本次失败。
            if report is None and report_version() != entry_version:
                report = json.loads(report_path.read_text())
            if isinstance(report, dict):
                record_public_timing(report, wall)
                report.update(status="failed", error=repr(original_error))
                if pipeline_error is None:
                    report["failed_stage"] = "thread_budget_restore"
                else:
                    report.setdefault("failed_stage", "pipeline")
                if restoration_error is not None:
                    report["thread_budget_restoration_error"] = repr(restoration_error)
                report_path.write_text(json.dumps(report, indent=2))
        except Exception as metadata_error:
            if hasattr(original_error, "add_note"):
                original_error.add_note(f"Public failure metadata could not be saved: {metadata_error!r}")
        if original_error is not error:
            raise original_error.with_traceback(original_error.__traceback__) from error
        raise
    wall = time.perf_counter() - tick
    record_public_timing(report, wall)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("t1", type=Path)
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n4-backend", choices=("native", "torch"), default="native",
                        help="native Conda ITK or existing complete PyTorch N4; default unchanged")
    parser.add_argument("--n4-execution", choices=("in-process", "isolated"), default="in-process",
                        help="isolated only for Torch cuda:N; cached child preserves parent allocator")
    parser.add_argument("--normalization-controls-backend", choices=("cpu", "torch"), default="cpu",
                        help="same-rule control-point neighbors; torch requires cuda:N, remaining solver unchanged")
    parser.add_argument("--normalization-initial-bias-backend", choices=("cpu", "torch"), default="cpu",
                        help="second normalization initial bias; torch reuses existing propagation/smoothing with original arithmetic")
    parser.add_argument("--inflate-backend", choices=("native", "torch"), default="native",
                        help="standard smoothwm inflation; torch requires cuda:N and two cached hemisphere workers; nofix unchanged")
    parser.add_argument("--native-bin-dir", type=Path)
    parser.add_argument("--hemisphere-workers", type=int, choices=(1, 2), default=1,
                        help="independent hemisphere processes; total threads split across two workers")
    parser.add_argument("--native-optimizations", choices=("auto", "original", "torch"), default="auto",
                        help="auto uses Torch GCA on CUDA, validated Conda selection on CPU; original fixes Conda; torch requests FNIT scorer")
    parser.add_argument("--wm-backend", choices=("native", "torch", "torch-optimized"), default="native",
                        help="native, existing Torch hybrid, or Torch histograms/cached geometry with ordered CPU feedback")
    parser.add_argument("--wm-execution", choices=("in-process", "isolated"), default="in-process",
                        help="isolated torch-optimized WM enables caching in a fresh child, preserving parent CUDA")
    parser.add_argument("--gca-inverse-backend", choices=("cpu", "torch"), default="cpu")
    parser.add_argument("--gca-candidate-chunk", type=int, default=64)
    parser.add_argument("--gca-execution", choices=("in-process", "isolated"), default="in-process",
                        help="isolated Torch GCA enables caching in a fresh child, preserving parent allocator")
    parser.add_argument("--fill-backend", choices=("python", "numba", "torch-numba"), default="python",
                        help="ordered fill; numba CPU heap, optionally CUDA initial boundary")
    parser.add_argument("--wm-edit-backend", choices=("native", "torch-hybrid"), default="native",
                        help="native edit or ordered NumBa + Torch GPU edit, rejects unsupported byte-access paths")
    parser.add_argument("--defects-backend", choices=("native", "torch"), default="native",
                        help="native mri_label2vol or FNIT PyTorch exact voxel projection; colors differ")
    parser.add_argument("--sphere-normals-backend", choices=("numba", "torch"), default="numba",
                        help="ordered sphere normals on CPU or explicit CUDA; remaining solver unchanged")
    parser.add_argument("--backend", choices=("native", "python-gpu"), default="native",
                        help="native: current validated mixed pipeline; python-gpu: strict profile, fails before output while required ports are incomplete")
    parser.add_argument("--profile-stages", action="store_true",
                        help="record CUDA synchronization and parent/child CPU time")
    parser.add_argument("--cuda-allocator-cache", choices=("auto", "enabled", "disabled"),
                        default="auto")
    args = parser.parse_args(argv)
    report = run_recon_all_python(args.t1, args.subject_dir, args.weights_dir,
                                  args.assets_dir, device=args.device,
                                  threads=args.threads,
                                  native_bin_dir=args.native_bin_dir,
                                  profile_stages=args.profile_stages,
                                  cuda_allocator_cache=args.cuda_allocator_cache,
                                  hemisphere_workers=args.hemisphere_workers,
                                  native_optimizations=args.native_optimizations,
                                  n4_backend=args.n4_backend,
                                  n4_execution=args.n4_execution,
                                  normalization_controls_backend=args.normalization_controls_backend,
                                  normalization_initial_bias_backend=args.normalization_initial_bias_backend,
                                  wm_backend=args.wm_backend,
                                  wm_execution=args.wm_execution,
                                  wm_edit_backend=args.wm_edit_backend,
                                  defects_backend=args.defects_backend,
                                  sphere_normals_backend=args.sphere_normals_backend,
                                  inflate_backend=args.inflate_backend,
                                  gca_inverse_backend=args.gca_inverse_backend,
                                  gca_candidate_chunk=args.gca_candidate_chunk,
                                  gca_execution=args.gca_execution,
                                  fill_backend=args.fill_backend,
                                  backend=args.backend)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
