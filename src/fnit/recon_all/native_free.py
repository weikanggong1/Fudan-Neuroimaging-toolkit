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


def _validate_topology_finish_backend(value: str, *, topology_backend: str,
                                      backend: str) -> None:
    """校验 remesh/相交后端；Conda 路径仅用于显式混合性能配方。

    value 默认 python，保持有序 remesh 和严格自有清理；conda-cpp 要求
    native GA 与非纯 Python profile。非法组合在创建被试输出前失败。
    """
    if value not in {"python", "conda-cpp"}:
        raise ValueError("topology_finish_backend must be python or conda-cpp")
    if value == "conda-cpp" and (topology_backend != "native" or backend != "native"):
        raise ValueError("Conda C++ topology finish requires the native GA and mixed native profile")


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
    返回完整读写时间和CPU/GPU分段。路径搜索保留源typed/byte差异；
    几何证明只用于安全跳过不可达搜索，证明不成立时执行完整有序搜索。
    原存储不受支持、源内循环状态不再变化或CUDA错误直接抛出。
    两例完整同输入与独立参考解码0差；十例未触发active分支，见
    WM_REMOVE_PATHS.md；不回退原程序或读取参考补齐。
    """
    from .edit_wm_aseg_torch import write_wm_asegedit_hybrid_diagnostic
    return write_wm_asegedit_hybrid_diagnostic(
        wm_file=mri / "wm.seg.mgz", brain_file=mri / "brain.mgz",
        aseg_file=mri / "aseg.presurf.mgz", entowm_file=mri / "entowm.mgz",
        output_file=mri / "wm.asegedit.mgz", device=device, fill_seg_wm=True)


def _run_python_wm_edit(mri: Path) -> dict:
    """四张自产conform MRI→完整wm.asegedit，不执行原版编辑程序。

    mri包含wm.seg、brain、aseg.presurf、entowm.mgz；网格/原存储类型须
    满足源字节访问的4字节存储和安全网格约束。复用标准Python/Numba CPU配方，保留
    keep-in、fill-seg-wm、SCM/HA、EntoWM和ACJ顺序，输出同网格uint8。
    几何不可达证明失败时仍执行完整路径搜索，保留typed/byte语义。
    返回阶段报告及路径/秒数；输入不支持或源内循环不再推进时抛异常。
    两例完整同输入解码0差，active分支尚无真实触发，见WM_REMOVE_PATHS.md。
    对应mri_edit_wm_with_aseg固定调用，详见WM_ASEG_TORCH.md。
    """
    from .edit_wm_aseg_core_python import write_wm_asegedit_profile
    return write_wm_asegedit_profile(wm_file=mri / "wm.seg.mgz",
        brain_file=mri / "brain.mgz", aseg_file=mri / "aseg.presurf.mgz",
        entowm_file=mri / "entowm.mgz", output_file=mri / "wm.asegedit.mgz",
        fill_seg_wm=True)


def _folding_atlas(assets: Path, hemi: str) -> Path:
    atlas = assets / "average" / (
        f"{hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif")
    if not atlas.is_file():
        raise FileNotFoundError(f"folding atlas not found: {atlas}")
    return atlas


def _run_native_em_register(binary: Path | None, mri: Path, atlas: Path,
                            assets: Path, *, backend: str = "original",
                            binary_sha256: str | None = None, threads: int = 4,
                            device: str = "cpu", inverse_backend: str = "cpu",
                            candidate_chunk: int = 64,
                            execution: str = "in-process",
                            reuse_atlas_context: bool = False) -> dict | None:
    """运行完整原生注册；缓存后端只改变已验证的搜索密度复用。

    binary为独立Conda程序，仅原生后端消费；torch允许None且不查找程序。
    mri含自产conformed nu/brainmask和transforms。
    atlas/assets为已校验GCA和资产路径。backend默认original；cpu_cached
    验证binary_sha256及能力2，threads默认4控制新子进程，保持父环境。
    输出talairach.lta包含4×4 voxel空间变换与两侧几何，返回None。
    inverse_backend默认cpu、candidate_chunk默认64；仅Torch评分使用这些
    参数。torch求逆复用同余子式，不改变候选网格、EM和LTA几何。
    execution=in-process保留父allocator；isolated仅用于Torch，新exec阶段
    内开启CUDA缓存，退出释放资源，不改变已初始化父CUDA/精度/缓存设置。
    reuse_atlas_context默认False；True只在同次完整Torch注册中复用不可变
    协方差/分类器，不改变优化器，调用者验证CUDA后端。
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
                inverse_backend=inverse_backend,
                **({"reuse_atlas_context": True} if reuse_atlas_context else {}))
        elif execution == "in-process":
            result = register_t1(
                mri / "nu.mgz", atlas, mri / "brainmask.mgz",
                mri / "transforms/talairach.lta", device=device,
                search_backend="torch", candidate_chunk=candidate_chunk, sample_chunk=8192,
                reduce_on_device=True, inverse_backend=inverse_backend,
                **({"reuse_atlas_context": True} if reuse_atlas_context else {}))
        else:
            raise ValueError("invalid GCA execution")
        if not (mri / "transforms/talairach.lta").is_file():
            raise FileNotFoundError(mri / "transforms/talairach.lta")
        return result
    if execution != "in-process":
        raise ValueError("isolated execution requires the Torch GCA backend")
    if binary is None:
        raise ValueError("native GCA backend requires an explicit program")
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


def _same_surface_volume_info(first: dict | None, second: dict | None) -> bool:
    """比较两张 surface RAS 表面的 MRI 几何头，不依赖字典数组真值转换。"""
    if first is None or second is None:
        return first is None and second is None
    return first.keys() == second.keys() and all(
        np.array_equal(first[key], second[key]) for key in first)


def _bounded_geometric_intersections(vertices: np.ndarray, faces: np.ndarray,
                                     *, large_face_radius_mm: float = 8.0) -> tuple[int, dict]:
    """逐面有界候选复用既有三角形谓词，返回相交面数和半径诊断。

    输入为 surface RAS 毫米三角面；小面以各自半径查询小面 KD 树，
    大面逐面查询全体面，不生成全局 query_pairs 数组。两类的并集覆盖
    原半径和 AABB 候选；共享顶点排除及 1e-5 边界保持。输出几何面数，
    无写出；非法半径或坐标直接报错。供原生清理后的独立零相交门。
    """
    from scipy.spatial import cKDTree
    from .mris_remove_intersection_python import _triangles_intersect

    if not np.isfinite(large_face_radius_mm) or large_face_radius_mm <= 0:
        raise ValueError("large_face_radius_mm must be finite and positive")
    xyz = np.asarray(vertices, dtype=np.float64)
    mesh = np.asarray(faces, dtype=np.int32)
    triangles = xyz[mesh]
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    marked = np.zeros(len(mesh), dtype=bool)
    small = np.flatnonzero(radii <= large_face_radius_mm)
    large = np.flatnonzero(radii > large_face_radius_mm)
    pairs_tested = 0

    def check(first: int, second: int) -> None:
        nonlocal pairs_tested
        if (np.sum((centers[first] - centers[second]) ** 2)
                > (radii[first] + radii[second] + 1e-5) ** 2):
            return
        if (np.any(low[first] > high[second] + 1e-5)
                or np.any(low[second] > high[first] + 1e-5)
                or np.any(mesh[first, :, None] == mesh[second, None, :])):
            return
        pairs_tested += 1
        if _triangles_intersect(triangles[first], triangles[second]):
            marked[first] = True
            marked[second] = True

    if len(small):
        small_tree = cKDTree(centers[small])
        maximum_small = float(radii[small].max())
        for face in small:
            nearby = small_tree.query_ball_point(
                centers[face], float(radii[face]) + maximum_small + 1e-5)
            for neighbor in nearby:
                other = int(small[neighbor])
                if other > face:
                    check(int(face), other)
    if len(large):
        all_tree = cKDTree(centers)
        maximum = float(radii.max())
        is_large = np.zeros(len(mesh), dtype=bool)
        is_large[large] = True
        for face in large:
            nearby = all_tree.query_ball_point(
                centers[face], float(radii[face]) + maximum + 1e-5)
            for other in nearby:
                if other != face and (not is_large[other] or other > face):
                    check(int(face), int(other))
    return int(marked.sum()), {"max_face_radius_mm": float(radii.max(initial=0.0)),
                               "large_face_radius_mm": float(large_face_radius_mm),
                               "large_faces": int(len(large)),
                               "predicate_pairs_tested": int(pairs_tested),
                               "candidate_strategy": "bounded per-face KD tree; exact existing triangle predicate"}


def _read_native_surface(path: Path) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """读取 surface RAS 毫米三角面并验证非空、有限坐标和有界索引。"""
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"native surface is missing or empty: {path}")
    vertices, faces, volume_info = fs.read_geometry(str(path), read_metadata=True)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.isfinite(vertices).all() or faces.min() < 0
            or faces.max() >= len(vertices)):
        raise RuntimeError(f"native surface has invalid vertices or faces: {path}")
    return vertices, faces, volume_info


def _native_final_mesh_structure(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """复用闭合单连通 genus-0 门，并检查顶点流形、方向与零面积。"""
    from .compare_subject import _topology
    from .topology_volume_python import _nonmanifold_vertex_links

    topology = _topology(vertices, faces)
    topology["valid"] = bool(topology["valid"])
    if not topology["valid"]:
        return {"status": "failed", "topology": topology,
                "nonmanifold_vertices": None, "edge_orientation_conflicts": None,
                "zero_area_faces": None}
    links = int(_nonmanifold_vertex_links(np.ascontiguousarray(faces, np.int32), len(vertices)))
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, groups = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    conflicts = int(np.count_nonzero(np.bincount(groups, weights=signs)))
    triangles = vertices[faces].astype(np.float64)
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1)
    zeros = int(np.count_nonzero(area2 == 0))
    passed = bool(topology["valid"] and links == conflicts == zeros == 0
                  and np.isfinite(area2).all())
    return {"status": "passed" if passed else "failed", "topology": topology,
            "nonmanifold_vertices": links, "edge_orientation_conflicts": conflicts,
            "zero_area_faces": zeros}


def _run_native_remesh(binary: Path, subject: Path, hemi: str, assets: Path,
                       *, input_path: Path) -> float:
    """Conda 自编译 mris_remesh 三轮，输出只留私有候选供末次清理。

    input_path 必须是已清理的 premesh 同序网格；检查其有序面和 MRI 几何头
    与自产 orig.premesh 相同。输出 surface RAS 毫米候选可改变拓扑，但须
    非空、有限且保留 MRI 几何头。返回包含输入校验、CLI 和文件检查的秒数；
    失败保留候选与 JSON，绝不发布 orig。原命令带 -i/-o/--remesh/--iters。
    """
    tick = time.perf_counter()
    surf = subject / "surf"
    premesh = surf / f"{hemi}.orig.premesh"
    candidate = surf / f"{hemi}.orig.native-remesh.candidate"
    receipt = subject / "scripts" / f"{hemi}.native-remesh.json"
    if candidate.exists() or receipt.exists():
        raise FileExistsError(f"preserve previous native remesh evidence: {candidate}")
    report = {"backend": "conda-cpp", "binary": str(binary),
              "input": str(input_path), "premesh": str(premesh),
              "candidate": str(candidate), "output": str(candidate),
              "coordinate_space": "surface_RAS_mm", "iterations": 3,
              "published_to_orig": False, "status": "running"}
    env = dict(os.environ, SUBJECTS_DIR=str(subject.parent), FREESURFER_HOME=str(assets))
    try:
        premesh_vertices, premesh_faces, premesh_info = _read_native_surface(premesh)
        source_vertices, source_faces, source_info = _read_native_surface(input_path)
        report.update(ordered_input_faces_preserved=bool(len(source_vertices) == len(premesh_vertices)
                      and np.array_equal(source_faces, premesh_faces)),
                      input_volume_info_preserved=_same_surface_volume_info(premesh_info, source_info),
                      binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest())
        if not report["ordered_input_faces_preserved"] or not report["input_volume_info_preserved"]:
            raise RuntimeError("native remesh input changed the ordered premesh faces or MRI geometry")
        subprocess.run([str(binary), "-i", str(input_path), "-o", str(candidate),
                        "--remesh", "--iters", "3"],
                       cwd=subject / "scripts", env=env, check=True)
        vertices, faces, output_info = _read_native_surface(candidate)
        report.update(vertices=len(vertices), faces=len(faces),
                      volume_info_preserved=_same_surface_volume_info(source_info, output_info),
                      candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
        if not report["volume_info_preserved"]:
            raise RuntimeError(f"native remesh changed MRI geometry: {candidate}")
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds_before_receipt_write"] = time.perf_counter() - tick
        receipt.write_text(json.dumps(report, indent=2) + "\n")
    return time.perf_counter() - tick


def _run_native_intersection(binary: Path, subject: Path, hemi: str,
                             assets: Path, *, phase: str) -> float:
    """原生 preclean/postclean 写私有候选；末次清理还须通过双谓词和结构门。

    preclean 输入自产 premesh、保持有序面和 MRI 头；postclean 输入 remesh
    候选，并检查闭合连通 genus-0、顶点流形、方向、零面积和双谓词零相交。
    两阶段均不覆盖输入或 orig。失败保留候选/JSON；返回含校验及读写秒数。
    对应 mris_remove_intersection INPUT OUTPUT；phase 仅取这两个值。
    """
    if phase not in {"preclean", "postclean"}:
        raise ValueError("native intersection phase must be preclean or postclean")
    tick = time.perf_counter()
    surf, scripts = subject / "surf", subject / "scripts"
    source = (surf / f"{hemi}.orig.premesh" if phase == "preclean" else
              surf / f"{hemi}.orig.native-remesh.candidate")
    candidate = (surf / f"{hemi}.orig.native-preclean.candidate" if phase == "preclean" else
                 surf / f"{hemi}.orig.native-intersection.candidate")
    receipt = scripts / f"{hemi}.native-{phase if phase == 'preclean' else 'intersection'}.json"
    if candidate.exists() or receipt.exists():
        raise FileExistsError(f"preserve previous native intersection evidence: {candidate}")
    report = {"backend": "conda-cpp", "phase": phase, "binary": str(binary),
              "input": str(source), "candidate": str(candidate), "output": str(candidate),
              "coordinate_space": "surface_RAS_mm", "published_to_orig": False,
              "status": "running"}
    env = dict(os.environ, SUBJECTS_DIR=str(subject.parent), FREESURFER_HOME=str(assets))
    try:
        original_vertices, original_faces, original_info = _read_native_surface(source)
        report.update(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        subprocess.run([str(binary), str(source), str(candidate)], cwd=scripts,
                       env=env, check=True)
        vertices, faces, output_info = _read_native_surface(candidate)
        ordered_faces = np.array_equal(faces, original_faces)
        finite = len(vertices) == len(original_vertices) and np.isfinite(vertices).all()
        info_preserved = _same_surface_volume_info(original_info, output_info)
        report.update(vertices=len(vertices), faces=len(faces),
                      ordered_faces_preserved=ordered_faces,
                      finite_coordinates=bool(finite),
                      volume_info_preserved=info_preserved,
                      candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
        if not ordered_faces or not finite or not info_preserved:
            raise RuntimeError("native intersection changed the ordered mesh, MRI geometry or produced nonfinite coordinates")
        if phase == "postclean":
            report["mesh_structure"] = _native_final_mesh_structure(vertices, faces)
            if report["mesh_structure"]["status"] != "passed":
                raise RuntimeError("native intersection output failed the final structural mesh gate")
            from .place_surface_intersection_marking import mark_source_intersections
            geometric_faces, geometry_diagnostics = _bounded_geometric_intersections(vertices, faces)
            report["geometric_recheck"] = geometry_diagnostics
            _, source_faces = mark_source_intersections(vertices, faces, predicate_backend="numba")
            report.update(source_intersecting_faces_after_independent_recheck=int(source_faces),
                          geometric_intersecting_faces_after_independent_recheck=int(geometric_faces))
            if source_faces or geometric_faces:
                raise RuntimeError(f"native intersection left {source_faces} source / "
                                   f"{geometric_faces} geometric intersecting faces")
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds_before_receipt_write"] = time.perf_counter() - tick
        receipt.write_text(json.dumps(report, indent=2) + "\n")
    return time.perf_counter() - tick


def _finish_native_topology(*, subject: Path, hemi: str, assets: Path,
                             remesh_binary: Path,
                             intersection_binary: Path) -> tuple[float, float]:
    """原生预清理→三轮 remesh→末次清理，全部通过后原子发布 orig。

    subject/hemi 定位自产 orig.premesh（surface RAS/mm、有序三角面）；
    assets 为声明资产目录，两个 binary 为固定源码 Conda 独立编译程序。
    三份私有候选和各阶段 JSON 永久保留，原始 premesh 不覆盖。末次清理
    校验头几何、闭合单连通 genus-0、流形、方向、零面积及双谓词零相交。
    返回 remesh 秒数及两次清理/发布总秒数，包含校验、CLI 和全部读写。
    失败抛原异常、保留现场，orig 不创建或不改变；对应官方三个命令见
    NATIVE_TOPOLOGY_FINISH.md，真实阶段回归及整例结果由版本绑定报告记录。
    """
    surf, scripts = subject / "surf", subject / "scripts"
    premesh = surf / f"{hemi}.orig.premesh"
    preclean = surf / f"{hemi}.orig.native-preclean.candidate"
    final_candidate = surf / f"{hemi}.orig.native-intersection.candidate"
    publication_candidate = surf / f"{hemi}.orig.native-publish.candidate"
    orig = surf / f"{hemi}.orig"
    receipt = scripts / f"{hemi}.native-topology-finish.json"
    if receipt.exists() or publication_candidate.exists():
        raise FileExistsError(f"preserve previous native topology publication evidence: {receipt}")
    report = {"backend": "conda-cpp", "input": str(premesh), "output": str(orig),
              "candidate": str(final_candidate), "coordinate_space": "surface_RAS_mm",
              "steps": ["preclean", "remesh3", "postclean", "atomic_publication"],
              "published_to_orig": False, "status": "running"}
    tick = time.perf_counter()
    try:
        premesh_sha = hashlib.sha256(premesh.read_bytes()).hexdigest()
        report["input_sha256"] = premesh_sha
        preclean_seconds = _run_native_intersection(
            intersection_binary, subject, hemi, assets, phase="preclean")
        remesh_seconds = _run_native_remesh(
            remesh_binary, subject, hemi, assets, input_path=preclean)
        postclean_seconds = _run_native_intersection(
            intersection_binary, subject, hemi, assets, phase="postclean")
        report["stage_seconds"] = {"preclean": preclean_seconds,
                                   "remesh": remesh_seconds,
                                   "postclean": postclean_seconds}
        if hashlib.sha256(premesh.read_bytes()).hexdigest() != premesh_sha:
            raise RuntimeError("native topology chain changed the original premesh")
        final_receipt = json.loads((scripts / f"{hemi}.native-intersection.json").read_text())
        final_sha = hashlib.sha256(final_candidate.read_bytes()).hexdigest()
        if (final_receipt.get("status") != "complete"
                or final_receipt.get("candidate_sha256") != final_sha
                or final_receipt.get("mesh_structure", {}).get("status") != "passed"
                or final_receipt.get("source_intersecting_faces_after_independent_recheck") != 0
                or final_receipt.get("geometric_intersecting_faces_after_independent_recheck") != 0):
            raise RuntimeError("native topology final candidate is not bound to a complete zero-intersection receipt")
        publication_started = time.perf_counter()
        shutil.copyfile(final_candidate, publication_candidate)
        if hashlib.sha256(publication_candidate.read_bytes()).hexdigest() != final_sha:
            raise RuntimeError("native topology publication SHA-256 changed")
        # This is the only operation that changes the public orig filename.
        publication_candidate.replace(orig)
        publication_seconds = time.perf_counter() - publication_started
        report["stage_seconds"]["publication"] = publication_seconds
        report.update(status="complete", published_to_orig=True,
                      output_sha256=final_sha,
                      final_mesh_structure=final_receipt["mesh_structure"])
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds_before_receipt_write"] = time.perf_counter() - tick
        receipt.write_text(json.dumps(report, indent=2) + "\n")
    return remesh_seconds, time.perf_counter() - tick - remesh_seconds


def _run_and_publish_genetic_topology(*, subject: Path, hemi: str,
                                      threads: int, remesh_scalar_storage: str,
                                      distance_backend: str = "serial",
                                      distance_device: str | None = None,
                                      state_backend: str = "legacy") -> dict:
    """自产nofix/qsphere/brain/WM→完整自有GA及orig门→发布真实表面。

    subject为FNIT被试目录，hemi为lh/rh，threads为当前半球CPU预算；
    remesh_scalar_storage为已校验的numpy/python双精度存储策略。
    distance_backend默认serial，parallel/GPU仅执行同公式完整距离；GPU
    distance_device为显式cuda:N。state_backend默认legacy，packed每候选
    完整装配，不持久缓存变化几何。选项/实测见ROUND2_INTEGRATION.md。
    独立新目录scripts/H.topology.genetic保存完整阶段产物、原域标签及报告；
    显式复用每缺陷不可变保留面索引，并以Numba执行完整有序阻挡扫描；
    不缓存动态几何、不缩减候选域或改变GA控制。独立库保留Python参考。
    完成remesh3、source_numba清理和最终网格门后，校验报告/三个文件SHA，
    再发布surf/H.orig.premesh、H.defect_labels、H.orig。表面surface RAS/mm；
    原域标签不采用压缩后编号，WM头和brain采样矩阵由文件接口区分。
    返回该接口的JSON字典；耗时由调用者覆盖计算、验证和全部发布读写。
    不启动原程序、不读取参考。缺失/篡改文件、未完成或失败网格抛异常；
    不把未验收中间态写成最终orig。真实范围与参数见TOPOLOGY_GENETIC_FILES.md。
    """
    from .topology_genetic_files import run_genetic_topology_files
    import tempfile

    subject = Path(subject).resolve()
    output = subject / "scripts" / f"{hemi}.topology.genetic"
    result = run_genetic_topology_files(
        input_surface=subject / "surf" / f"{hemi}.orig.nofix",
        input_qsphere=subject / "surf" / f"{hemi}.qsphere.nofix",
        brain=subject / "mri" / "brain.mgz", wm=subject / "mri" / "wm.mgz",
        hemisphere=hemi, output_directory=output, threads=threads,
        seed=1234, population_size=10, max_unchanged=10, niters=-1,
        initial_selection=True, vertex_eliminate=True, use_overlap=False,
        normal_status="surface", soap_iterations=5, final_orig=True,
        face_detection_cache=True,
        blocker_scan_backend="numba",
        remesh_scalar_storage=remesh_scalar_storage,
        **({"distance_backend": distance_backend, "distance_device": distance_device}
           if distance_backend != "serial" else {}),
        **({"state_backend": state_backend} if state_backend != "legacy" else {}))
    if result.get("status") != "complete" or result.get("final_orig") is not True:
        raise RuntimeError("complete genetic topology and final orig receipt required")
    if result.get("quality", {}).get("status") != "passed":
        raise RuntimeError("genetic topology final orig quality gate failed")
    if result.get("parameters", {}).get("face_detection_cache") is not True:
        raise RuntimeError("genetic topology did not apply the selected face cache")
    if result.get("parameters", {}).get("blocker_scan_backend") != "numba":
        raise RuntimeError("genetic topology did not apply the selected ordered blocker scan")
    output = output.resolve()
    for field in ("report_path", "completion_path"):
        path = Path(result[field]).resolve()
        if not path.is_relative_to(output) or not path.is_file():
            raise RuntimeError("genetic topology receipt is missing or outside its run")
    report_bytes = Path(result["report_path"]).read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != result["report_sha256"]:
        raise RuntimeError("genetic topology report SHA-256 changed")
    recorded = json.loads(report_bytes)
    if (recorded.get("status") != "complete" or recorded.get("final_orig") is not True
            or recorded.get("quality") != result["quality"]
            or recorded.get("parameters") != result["parameters"]
            or recorded.get("output_sha256") != result["output_sha256"]):
        raise RuntimeError("genetic topology report does not bind this complete final orig")
    completion = json.loads(Path(result["completion_path"]).read_text())
    if (completion.get("status") != "complete" or completion.get("final_orig") is not True
            or completion.get("report_sha256") != result["report_sha256"]
            or completion.get("output_sha256") != result["output_sha256"]):
        raise RuntimeError("genetic topology completion receipt is not complete")
    sources = {}
    for name, suffix in (("premesh", "orig.premesh"), ("defect_labels", "defect_labels"), ("orig", "orig")):
        path = Path(result["outputs"][name]).resolve()
        if not path.is_relative_to(output) or not path.is_file():
            raise RuntimeError("genetic topology artifact is missing or outside its run")
        if hashlib.sha256(path.read_bytes()).hexdigest() != result["output_sha256"][name]:
            raise RuntimeError(f"genetic topology {name} SHA-256 changed")
        sources[name] = (path, subject / "surf" / f"{hemi}.{suffix}")
    # All artifacts pass before any final filename is published. Copies retain
    # the actual algorithm output; temporary files are verified before replace.
    with tempfile.TemporaryDirectory(prefix=".fnit-ga-publish-", dir=subject / "surf") as temporary:
        pending = []
        for name, (source, destination) in sources.items():
            candidate = Path(temporary) / destination.name
            shutil.copyfile(source, candidate)
            if hashlib.sha256(candidate.read_bytes()).hexdigest() != result["output_sha256"][name]:
                raise RuntimeError(f"genetic topology {name} publication SHA-256 changed")
            pending.append((candidate, destination))
        for candidate, destination in pending:
            candidate.replace(destination)
    publication = {
        "status": "complete", "final_quality": result["quality"],
        "source_report": str(Path(result["report_path"]).relative_to(subject)),
        "source_report_sha256": result["report_sha256"],
        "outputs": {name: str(destination.relative_to(subject))
                    for name, (_, destination) in sources.items()},
        "output_sha256": result["output_sha256"],
        "paths_relative_to": "subject_dir", "worker_threads": threads,
        "face_detection_cache": True,
        "blocker_scan_backend": "numba",
        "strict_reproduction": "reported_separately",
        "overall_metric_equivalence": "not_assessed",
    }
    (subject / "scripts" / f"{hemi}.topology.genetic.publish.json").write_text(
        json.dumps(publication, indent=2))
    return result


def _prepare_native_topology(binary: Path | None, subject: Path, hemi: str,
                             assets: Path, device: str,
                             native_inflate_binary: Path | None,
                             intersection_binary: Path | None,
                             *, remesh_binary: Path | None = None,
                             topology_finish_backend: str = "python",
                             remesh_scalar_storage: str = "numpy",
                             python_auxiliary: bool = False,
                             topology_backend: str = "native",
                             topology_distance_backend: str = "serial",
                             topology_state_backend: str = "legacy",
                             pretess_file: Path | None = None,
                             threads: int = 4,
                             ) -> tuple[float, float, dict[str, float], float, float]:
    """完成nofix/拓扑/remesh/清理；remesh存储候选不改任何几何规则。

    路径、半球、设备和声明程序沿用表面主接口；新增remesh_scalar_storage
    默认numpy，python复用同算法的CPU双精度标量存储。返回Python总秒、
    原生拓扑秒、nofix子时间、remesh秒、intersection秒；均含读写。
    表面为surface RAS/mm，orig的有序面由完整算法生成；失败传播。
    官方步骤/完整参数及实测见CONDA_CPP_STAGES和REMESH_SCALAR_STORAGE。
    python_auxiliary默认False；True使用完整标准默认10次nofix inflation
    并关闭sulc文件写出，不改变优化次数。topology_backend默认native；
    python必须提供pretess_file（自产uint8 conform体积），执行完整数字
    拓扑不同算法及真实删除区缺陷映射，不启动GA程序；binary/两个兼容
    辅助程序此时可为None。数字拓扑全质量失败即抛异常，输出/空间见
    PYTHON_SURFACE_PIPELINE.md；整体指标等效另测，不复制orig充当修复。
    topology_finish_backend默认python；conda-cpp仅与native GA搭配，使用
    Conda自编译预清理、三轮remesh及末次相交清理；完整链通过结构与双谓词零残余门后才原子发布。
    python-ga使用完整自有有序GA；threads默认4，由表面worker显式传入
    本半球预算，不扩大总线程数。文件接口在独立目录一次完成标准remesh、
    最终清理和严格orig门，再校验SHA发布；不重复执行下面的公共remesh。
    topology_distance_backend默认serial，topology_state_backend默认legacy，
    仅向完整python-ga传递；GPU距离消费device，packed不改变完整搜索。
    """
    _validate_remesh_scalar_storage(remesh_scalar_storage)
    _validate_topology_finish_backend(topology_finish_backend,
                                      topology_backend=topology_backend,
                                      backend="native" if topology_backend == "native" else "python")
    if topology_backend == "python-ga" and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads < 1
    ):
        raise ValueError("genetic topology requires an explicit positive worker thread budget")
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
    if python_auxiliary:
        from .inflate_standard_run import run_standard_inflate
        nofix_report = run_standard_inflate(
            input_surface=smooth_nofix, inflated_output=inflated_nofix,
            sulc_output=None, save_sulc=False,
            backend="torch" if torch.device(device).type == "cuda" else "numpy",
            device=device, profile=False)
        sphere_seconds["inflate_nofix"] = nofix_report["total_seconds_including_io"]
    else:
        if native_inflate_binary is None:
            raise ValueError("native nofix inflation requires a declared program")
        sphere_seconds["inflate_nofix"] = _run_native_sphere_step(
            native_inflate_binary, subject, assets,
            ["-no-save-sulc", str(smooth_nofix), str(inflated_nofix)],
            (inflated_nofix,))
    tick = time.perf_counter()
    write_quick_sphere(inflated_nofix, qsphere_nofix)
    quick_seconds = time.perf_counter() - tick
    sphere_seconds["qsphere_nofix_python"] = quick_seconds
    native_started = time.perf_counter()
    if topology_backend == "python":
        if pretess_file is None:
            raise ValueError("Python topology requires an own pretess volume")
        from .topology_volume_python import run_volume_topology
        run_volume_topology(
            input_volume=pretess_file, value=255 if hemi == "lh" else 127,
            output_volume=subject / "mri" / f"filled-topology-{hemi}.mgz",
            output_surface=surf / f"{hemi}.orig.premesh", device=device,
            source_surface=surf / f"{hemi}.orig.nofix",
            output_defect_labels=surf / f"{hemi}.defect_labels",
            report_path=subject / "scripts" / f"{hemi}.topology.python.json")
        python_seconds += time.perf_counter() - native_started
        native_seconds = 0.0
    elif topology_backend == "native":
        if binary is None:
            raise ValueError("native topology requires a declared program")
        _run_native_topology(binary, subject, hemi, assets)
        native_seconds = time.perf_counter() - native_started
    elif topology_backend == "python-ga":
        result = _run_and_publish_genetic_topology(
            subject=subject, hemi=hemi, threads=threads,
            remesh_scalar_storage=remesh_scalar_storage,
            distance_backend=topology_distance_backend,
            distance_device=device if topology_distance_backend == "gpu_defect_signed_distance" else None,
            state_backend=topology_state_backend)
        python_seconds += time.perf_counter() - native_started
        return (python_seconds, 0.0, sphere_seconds,
                result["seconds"]["remesh"],
                result["seconds"]["original_intersection_cleanup"])
    else:
        raise ValueError("topology_backend must be native, python or python-ga")
    if topology_finish_backend == "conda-cpp":
        if remesh_binary is None or intersection_binary is None:
            raise ValueError("native topology finish requires declared Conda remesh and intersection programs")
        remesh_seconds, intersection_seconds = _finish_native_topology(
            subject=subject, hemi=hemi, assets=assets,
            remesh_binary=remesh_binary, intersection_binary=intersection_binary)
    else:
        tick = time.perf_counter()
        orig = surf / f"{hemi}.orig"
        remesh_surface(surf / f"{hemi}.orig.premesh", orig, iterations=3,
                       **({"scalar_storage": remesh_scalar_storage} if remesh_scalar_storage != "numpy" else {}))
        remesh_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        remove_intersection_surface(orig, orig, binary=intersection_binary,
                                    assets_dir=assets)
        intersection_seconds = time.perf_counter() - tick
    python_total = (python_seconds if topology_finish_backend == "conda-cpp" else
                    python_seconds + remesh_seconds + intersection_seconds)
    return (python_total, native_seconds,
            sphere_seconds, remesh_seconds, intersection_seconds)


def _run_accurate_sphere_pair(inflate_binary: Path | None, subject: Path,
                              hemi: str, assets: Path, *, device: str = "cpu",
                              normals_backend: str = "numba",
                              inflate_backend: str = "native",
                              sphere_finish_backend: str = "cpu",
                              require_unfolded: bool = False) -> tuple[dict, dict]:
    """同序smoothwm→标准inflated/sulc→sphere，返回阶段秒数与球面报告。

    subject/hemi提供surface RAS mm网格；assets和inflate_binary供native
    默认路径。inflate_backend可显式torch，复用完整GPU算法与真实sulc；
    device为其明确CUDA设备，缓存由外层fresh worker选择。球面默认Numba
    法向/CPU收尾保持；sphere_finish_backend=torch复用既有完整GPU收尾，
    保留负面标记、SOAP更新、全顶点投影和停止规则；normals_backend仅选择既有法向。报告含inflate实际
    后端与完整读写秒数；异常向上传播，不复制参考或回退。nofix不属此接口。
    require_unfolded 默认 False；Python 完整配方显式 True，在球面
    残余翻折时写出前失败，并保留实际停止规则和质量诊断。
    """
    from .sphere_standard_run import run_standard_sphere
    if sphere_finish_backend not in {"cpu", "torch"}:
        raise ValueError("sphere_finish_backend must be cpu or torch")
    if sphere_finish_backend == "torch":
        target = torch.device(device)
        if target.type != "cuda" or target.index is None:
            raise ValueError("torch sphere finish requires explicit cuda:N")

    surf = subject / "surf"
    inflated, sulc = surf / f"{hemi}.inflated", surf / f"{hemi}.sulc"
    if inflate_backend == "torch":
        from .inflate_standard_run import run_standard_inflate
        inflate_report = run_standard_inflate(
            input_surface=surf / f"{hemi}.smoothwm", inflated_output=inflated,
            sulc_output=sulc, backend="torch", device=device, profile=False)
        inflate_seconds = inflate_report["total_seconds_including_io"]
    elif inflate_backend == "native":
        if inflate_binary is None:
            raise ValueError("native standard inflation requires a declared program")
        inflate_seconds = _run_native_sphere_step(
            inflate_binary, subject, assets,
            [str(surf / f"{hemi}.smoothwm"), str(inflated)], (inflated, sulc))
        inflate_report = {"backend": "native", "total_seconds_including_io": inflate_seconds}
    else:
        raise ValueError("inflate_backend must be native or torch")
    sphere_report = run_standard_sphere(
        inflated, surf / f"{hemi}.smoothwm", surf / f"{hemi}.sphere",
        finish_device=device if sphere_finish_backend == "torch" else "cpu", averaging_device=device,
        **({"require_unfolded": True} if require_unfolded else {}),
        **({"normals_device": device} if normals_backend == "torch" else {}))
    sphere_report["inflate_runtime"] = inflate_report
    return ({"inflate": inflate_seconds,
             "sphere": sphere_report["total_seconds_including_io"]}, sphere_report)


def _run_surface_metrics(binary: Path | None, subject: Path, hemi: str,
                         assets: Path, *, device: str, python_only: bool = False) -> dict[str, float]:
    """从对应的 white/pial 计算厚度、双侧面积和平均曲率顶点图。

    subject 为被试目录，hemi 为 lh/rh；输入表面为 surface RAS（mm），
    顶点与有序面须对应。CUDA 或 python_only=True 使用已有自有
    PyTorch 函数；仅参考 CPU 路径使用 binary 指定的 Conda 程序。
    assets 是声明的资产目录；Python 路径不需要原生程序。
    写出 H.thickness（mm）、H.area/H.area.pial（mm²）、
    H.curv/H.curv.pial（mm⁻¹），返回包含读写与 GPU 同步的逐图秒数。
    参数固定为 20 跳、5 mm 最大厚度、2 阶邻域及 10 次曲率平滑，
    对应官方 --thickness/--area-map/--curv-map；失败抛异常。
    同输入精度、速度及完整示例见 docs/recon_all/SURFACE_METRICS.md。
    """
    surf = subject / "surf"
    white, pial = surf / f"{hemi}.white", surf / f"{hemi}.pial"
    if torch.device(device).type == "cuda" or python_only:
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
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            function(**kwargs, device=device)
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            timings[name] = time.perf_counter() - tick
        return timings
    if binary is None:
        raise ValueError("native CPU metrics require a declared binary")
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


def _run_avg_curv(binary: Path | None, subject: Path, hemi: str,
                  atlas: Path, assets: Path, *, python_only: bool = False) -> dict | None:
    """sphere.reg与第6幅TIFF图谱→(N,)avg_curv，五轮邻域平均。

    网格为surface RAS/mm，有序顶点不变；subject/hemi指定输出，atlas为
    已声明图谱。python_only默认False；True使用已验证的Numba CPU采样
    和自有平均，不读取binary/assets程序，返回路径/哈希/含IO秒数字典。
    报告落到scripts/H.avg_curv.python.json。原生返回None，失败抛异常。
    对应mrisp_paint -a 5 atlas#6 sphere.reg avg_curv，见自有表面说明。
    """
    output = subject / "surf" / f"{hemi}.avg_curv"
    if python_only:
        from .mrisp_paint_python import run_paint_atlas
        return run_paint_atlas(atlas_path=atlas,
            sphere_path=subject / "surf" / f"{hemi}.sphere.reg",
            output_path=output, frame=6, averages=5, device="cpu",
            report_path=subject / "scripts" / f"{hemi}.avg_curv.python.json")
    if binary is None:
        raise ValueError("native atlas paint requires a declared program")
    env = dict(os.environ, SUBJECTS_DIR=str(subject.parent),
               FREESURFER_HOME=str(assets))
    subprocess.run([str(binary), "-a", "5", f"{atlas}#6",
                    str(subject / "surf" / f"{hemi}.sphere.reg"), str(output)],
                   cwd=subject / "scripts", env=env, check=True)
    if not output.is_file():
        raise FileNotFoundError(output)


def _run_curvature_stats(binary: Path | None, subject: Path, hemi: str,
                         assets: Path, *, device: str = "cpu",
                         python_only: bool = False) -> dict | None:
    """同序smoothwm/curv/sulc→完整curv.stats与八张离散曲率图。

    subject/hemi为自产目录与lh/rh；图以原顶点顺序、surface RAS/mm解释。
    python_only默认False保留参考程序，True复用自有Torch/Numba完整统计，
    device默认cpu且不回退。生产固定零平滑和source_libm源浮点规则，
    overwrite仅允许覆盖本阶段标准输出。返回完整路径/统计/读写秒数字典，
    原生返回None；输入和写出失败抛异常，不复制参考。官方固定命令见下。
    """
    if python_only:
        from .curvature_stats_python import write_curvature_stats
        return write_curvature_stats(subject_dir=subject, hemisphere=hemi,
            device=device, angle_backend="source_libm", averages=0, overwrite=True)
    if binary is None:
        raise ValueError("native curvature statistics require a declared binary")
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
                         threads: int, warp_binaries: tuple[Path | None, Path | None, Path | None],
                         stage, *, device: str, defer_mni_nonlinear: bool = False) -> dict:
    """生成 MNI 辅助图、非线性变换和 finalsurfs；显式传递主设备。

    subject 提供自产 conform MRI；weights、assets 为已校验资源，threads
    为线程数，warp_binaries 是按转换/求逆/重采样排列的兼容参数，允许
    三个None且不读取或执行。stage 是记录耗时和失败的回调，device 必须为
    CUDA。输出写入 subject，返回辅助网络实际前向记录字典；计算失败抛异常。
    全部网络使用 device，
    辅助网络卷积在局部作用域采用经同输入验证的FP32，matmul TF32不变；
    MNI warp 转换、完整求逆和检查图采用自有 GPU 后处理；非CUDA在任何
    辅助图读写和前向前报错。finalsurfs复用所选device的既有后处理。空间、命令与实测
    见 MNI_NONLINEAR_CHAIN.md。后处理失败抛出异常，不静默切换后端。
    defer_mni_nonlinear默认False；True仅延后非线性，MNI affine/crop和
    finalsurfs照常生成。调用方必须在输出检查前运行完整非线性并join；
    延后本身没有近似/占位变换，也不生成标准warp文件。
    """
    _validate_mni_execution("in-process", device, threads)
    from .finalsurfs_python import run_finalsurfs
    from .mni_aux_chain import run_mni_aux_chain
    from .mni_nonlinear_chain import run_mni_nonlinear_chain

    with torch.backends.cudnn.flags(allow_tf32=False):
        auxiliary = stage("mni_aux", run_mni_aux_chain, subject, weights, assets,
              device=device, threads=threads)
    if not defer_mni_nonlinear:
        stage("mni_nonlinear", run_mni_nonlinear_chain, subject, weights, assets,
              warp_convert=warp_binaries[0], ca_register=warp_binaries[1],
              mri_convert=warp_binaries[2], device=device, threads=threads,
              postprocess_backend="gpu")
    # ``finalsurfs_python`` has a validated CUDA implementation for the
    # volume masks and entorhinal/ACJ edits. Keep its FP32 semantics while
    # avoiding an unnecessary device round-trip on GPU recon-all runs.
    stage("brain_finalsurfs", run_finalsurfs, subject, device=device)
    return auxiliary["runtime"]


def _place_preaparc_and_smooth(subject: Path, hemi: str, binary: Path | None,
                               assets: Path, threads: int, *, device: str = "cpu",
                               placement_backend: str = "native") -> dict:
    from .smooth_surface_python import smooth_surface
    _validate_surface_placement_backend(placement_backend)
    if placement_backend == "python":
        from .surface_placement_python import run_preaparc_python
        report = run_preaparc_python(subject_dir=subject, hemi=hemi, device=device)
    else:
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
                  *, device: str, threads: int, topology_binary: Path | None,
                  inflate_binary: Path | None, intersection_binary: Path | None,
                  remesh_binary: Path | None = None,
                  place_binary: Path, defect_binary: Path | None,
                  assets: Path, defer_defects: bool = False,
                  defects_backend: str = "native",
                  sphere_normals_backend: str = "numba",
                  inflate_backend: str = "native",
                  sphere_finish_backend: str = "cpu",
                  remesh_scalar_storage: str = "numpy",
                  surface_placement_backend: str = "native",
                  topology_backend: str = "native",
                  topology_finish_backend: str = "python",
                  topology_distance_backend: str = "serial",
                  topology_state_backend: str = "legacy") -> dict:
    """从自产filled生成单半球orig、预白质和标准球面，返回完整阶段报告。

    subject/hemi指被试目录和lh/rh；filled/norm为conform体积，网格为
    surface RAS/mm。device/threads指定预算；各binary和assets为已声明
    独立构建产物/资源。defer_defects默认False；并行worker设True由父
    顺序发布共享缺陷。defects_backend=native/torch、sphere_normals_backend
    =numba/torch、inflate_backend=native/torch沿用各成熟函数。新
    sphere_finish_backend=cpu/torch只选择完整dense收尾设备，默认cpu。
    remesh_scalar_storage默认numpy，python复用已验CPU双精度存储；
    堆顺序、拆缩边/接受规则和完整Numba顺序平滑均保持。
    返回顶点/面数、组件、分阶段时间、预白质/球面实际报告及placement_pending；
    写出同序标准表面/标签，后续最终放置另行执行。输入、网格或计算失败
    传播；不读取参考，不占位。原命令/真实数据见recon-all各功能页。
    topology_backend=python-ga显式使用完整自有GA文件接口，threads按本
    半球传入；其质量/SHA门和发布回执另见topology_genetic_publication。
    topology_finish_backend默认python；显式conda-cpp只用于native GA后续
    三轮remesh与相交清理，两个独立程序均需在已声明的Conda bin中。
    topology_distance_backend=serial与topology_state_backend=legacy保持默认；
    非默认仅完整GA，逐项意义/限制见ROUND2_INTEGRATION.md。
    """
    _validate_remesh_scalar_storage(remesh_scalar_storage)
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
                                 device, inflate_binary, intersection_binary,
                                 remesh_binary=remesh_binary,
                                 topology_finish_backend=topology_finish_backend,
                                 remesh_scalar_storage=remesh_scalar_storage,
                                 threads=threads,
                                 **({"topology_distance_backend": topology_distance_backend}
                                    if topology_distance_backend != "serial" else {}),
                                 **({"topology_state_backend": topology_state_backend}
                                    if topology_state_backend != "legacy" else {}),
                                 **({"python_auxiliary": True} if surface_placement_backend == "python" else {}),
                                 **({"topology_backend": topology_backend, "pretess_file": pretess}
                                    if topology_backend != "native" else {})))
    if not defer_defects:
        _run_defects_volume(defect_binary, subject, hemi, assets,
                            backend=defects_backend, device=device)
    preaparc = _place_preaparc_and_smooth(subject, hemi, place_binary,
                                        assets, threads, device=device,
                                        placement_backend=surface_placement_backend)
    base, ga = label_cortex_fix_ga(
        surf / f"{hemi}.white.preaparc", mri / "aseg.presurf.mgz",
        mri / "entowm.mgz", hemi, labels / f"{hemi}.cortex.label")
    label_cortex(surf / f"{hemi}.white.preaparc", mri / "aseg.presurf.mgz",
                 labels / f"{hemi}.cortex+hipamyg.label", keep_hip_amyg=True)
    sphere_timings, sphere_report = _run_accurate_sphere_pair(
        inflate_binary, subject, hemi, assets, device=device,
        normals_backend=sphere_normals_backend, inflate_backend=inflate_backend,
        sphere_finish_backend=sphere_finish_backend,
        require_unfolded=surface_placement_backend == "python")
    _write_principal_curvature_maps(surf, hemi, device)
    white, faces = fs.read_geometry(str(surf / f"{hemi}.smoothwm"))
    return {"hemisphere": hemi, "vertices": len(white), "faces": len(faces),
            "raw_components": components, "cortex_vertices": len(base) + len(ga),
            "topology_python_seconds": topology_python_seconds,
            "topology_native_seconds": topology_native_seconds,
            "topology_backend": topology_backend,
            "topology_finish_backend": topology_finish_backend,
            **({"topology_genetic_publication": json.loads(
                (subject / "scripts" / f"{hemi}.topology.genetic.publish.json").read_text())}
               if topology_backend == "python-ga" else {}),
            "topology_remesh_seconds": remesh_seconds,
            "remesh_scalar_storage": remesh_scalar_storage,
            "topology_intersection_seconds": intersection_seconds,
            "sphere_seconds": {**pre_sphere_seconds, **sphere_timings},
            "standard_sphere_report": sphere_report,
            "white_preaparc_report": preaparc,
            "placement_pending": True}


def _finish_python_final_white(subject: Path, hemi: str, *, device: str) -> dict:
    """自产MRI/注释/preaparc→完整white，供独立缓存worker。

    subject为标准被试目录；hemi=lh/rh；device为cpu或显式cuda:N。
    复用run_final_white_python完整四轮与网格检查，写surface RAS/mm
    white及conform网格mrisps.white.mgz。返回final_white_report和待完成
    pial/指标标记；设备、输入、优化或写出失败传播，不启动原版程序。
    同输入缓存回归见SURFACE_CUDA_ALLOCATOR，不代表整例提速。
    """
    from .surface_placement_python import run_final_white_python
    report = run_final_white_python(subject_dir=subject, hemi=hemi, device=device)
    return {"final_white_report": report, "placement_pending": True,
            "metrics_pending": True}


def _finish_python_pial(subject: Path, hemi: str, *, device: str) -> dict:
    """自产完整white/MRI/标签→完整pial，保持独立低显存worker策略。

    subject/hemi/device同white入口；复用run_pial_python完整四轮和清理，
    写surface RAS/mm的pial.T1，并按官方语义复制为pial。返回pial_report、
    放置完成及指标待办标记。失败传播，不执行指标或原版阶段程序；
    指标在双侧pial发布屏障之后串行计算，不更改父CUDA缓存策略。
    """
    from .surface_placement_python import run_pial_python
    report = run_pial_python(subject_dir=subject, hemi=hemi, device=device)
    surf = subject / "surf"
    shutil.copyfile(surf / f"{hemi}.pial.T1", surf / f"{hemi}.pial")
    return {"pial_report": report, "placement_pending": False,
            "metrics_pending": True}


def _finish_cortical_surface(subject: Path, hemi: str, binary: Path | None,
                             assets: Path, *, device: str, threads: int,
                             defer_metrics: bool = False,
                             white_binary: Path | None = None,
                             surface_placement_backend: str = "native") -> dict:
    """球面配准和注释完成后，依次放置最终white、pial并计算顶点图。

    subject为自产被试目录，hemi为lh/rh；binary是pial及CPU顶点图程序，
    white_binary可单独指定white专用程序，默认复用binary；python放置时
    两者可为None且不启动程序。surface_placement_backend默认native，python
    复用完整四轮优化器，返回路径、轨迹和清理报告。device/threads
    控制已有GPU顶点图和原生线程，defer_metrics默认False，供并行屏障。
    写出surface RAS/mm的white、pial.T1/pial和所需顶点图，返回各阶段路径/
    秒数/待办状态。输入、网格或执行失败传播异常；不读取官方结果。
    """
    _validate_surface_placement_backend(surface_placement_backend)
    surf = subject / "surf"
    if surface_placement_backend == "python":
        result = {**_finish_python_final_white(subject, hemi, device=device),
                  **_finish_python_pial(subject, hemi, device=device)}
    else:
        from .final_white_conda import run_final_white
        white_report = run_final_white(subject, hemi, white_binary or binary, assets, threads=threads)
        pial_report = _run_native_pial(binary, subject, hemi, assets, threads)
        shutil.copyfile(surf / f"{hemi}.pial.T1", surf / f"{hemi}.pial")
        result = {"final_white_report": white_report, "pial_report": pial_report,
                  "placement_pending": False}
    result["metrics_pending"] = defer_metrics
    if not defer_metrics:
        result.update(_finish_cortical_metrics(subject, hemi, binary, assets, device=device,
                                             python_only=surface_placement_backend == "python"))
    return result


def _finish_cortical_metrics(subject: Path, hemi: str, binary: Path,
                            assets: Path, *, device: str, python_only: bool = False) -> dict:
    """完成已放置white/pial的指标；真实短配对显示GPU指标串行更快。"""
    from .surface_area_gpu import mid_area_map
    from .surface_roi_gpu import vertex_volume_map
    surf, labels = subject / 'surf', subject / 'label'
    metric_seconds = _run_surface_metrics(binary, subject, hemi, assets, device=device, python_only=python_only)
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
                          sphere_normals_backend="numba", inflate_backend="native",
                          sphere_finish_backend="cpu", annotation_gibbs_backend="python",
                          remesh_scalar_storage="numpy", surface_placement_backend="native",
                          topology_backend="native", topology_finish_backend="python",
                          profile_stages=False,
                          topology_distance_backend="serial", topology_state_backend="legacy"):
    """可exec的半球阶段入口，返回原结果与逐步读写/设备时间列表。

    subject为私有worker被试目录，hemi=lh/rh，device/threads为设备与
    该worker预算；assets/binaries/registration_atlases为声明资源和程序。
    operation=surface/register/annotation/finish_surface选择原完整阶段；
    Python配方将最终放置拆成final_white/pial_surface，复用相同完整函数，
    仅white新exec启用缓存，pial维持继承策略；共享发布与指标屏障不变。
    sphere_normals_backend默认numba，inflate_backend默认native，
    sphere_finish_backend默认cpu；仅surface消费这些选项，其余阶段算法保持。
    annotation_gibbs_backend默认python；numba复用已有完整有序Gibbs实现，
    仅annotation消费，GPU几何/随机序列/标签语义和缓存失效规则保持。
    remesh_scalar_storage默认numpy，仅surface消费python双精度存储候选，
    不改算法、GPU精度或线程预算。
    topology_finish_backend默认python；conda-cpp只消费surface阶段的
    remesh/intersection程序，其余半球组仍按原有依赖屏障执行。
    topology_distance_backend默认serial、topology_state_backend默认legacy；
    仅surface消费，按完整GA文件接口传递，不改变其他操作。
    profile_stages默认False；True在各子步骤前后同步明确device，保留外层
    worker显存峰值。嵌套秒数包含在worker/组墙钟内，不重复累加。
    返回result/stages；体积为conform网格，表面为surface RAS/mm，发布
    和共享输出屏障由父调度负责。非法阶段或计算失败抛异常，不读取参考。
    原软件命令与真实阶段结果见各功能页；本内部门口无独立官方CLI。
    """
    from .profiling import StageProfiler, configure_cuda_allocator
    _validate_annotation_gibbs_backend(annotation_gibbs_backend)
    _validate_remesh_scalar_storage(remesh_scalar_storage)
    subject, assets = Path(subject), Path(assets)
    binaries = {name: Path(value) for name, value in (binaries or {}).items()}
    profiler = StageProfiler(device=device, synchronize=profile_stages,
                             allocator=configure_cuda_allocator(device),
                             reset_peak_memory=False)
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
                     threads=threads, topology_binary=binaries.get('topology'),
                     inflate_binary=binaries.get('inflate'), intersection_binary=binaries.get('intersection'),
                     remesh_binary=binaries.get('remesh'),
                     place_binary=binaries.get('white', binaries.get('metrics')), defect_binary=binaries.get('defect'),
                     assets=assets, defer_defects=True,
                     sphere_normals_backend=sphere_normals_backend, inflate_backend=inflate_backend,
                     sphere_finish_backend=sphere_finish_backend,
                     remesh_scalar_storage=remesh_scalar_storage,
                     surface_placement_backend=surface_placement_backend,
                     topology_backend=topology_backend,
                     topology_finish_backend=topology_finish_backend,
                     **({"topology_distance_backend": topology_distance_backend}
                        if topology_distance_backend != "serial" else {}),
                     **({"topology_state_backend": topology_state_backend}
                        if topology_state_backend != "legacy" else {}))
    elif operation == 'register':
        from .mris_register_run import run_register_sphere
        value = step(f'register_{hemi}', run_register_sphere,
                     surf / f'{hemi}.sphere', surf / f'{hemi}.smoothwm',
                     surf / f'{hemi}.sulc', Path(registration_atlases[hemi]),
                     surf / f'{hemi}.sphere.reg', overlap_device='cpu', averaging_device=device,
                     require_unfolded=surface_placement_backend == "python")
        step(f'avg_curv_{hemi}', _run_avg_curv, binaries.get('paint'), subject, hemi,
             Path(registration_atlases[hemi]), assets,
             python_only=surface_placement_backend == "python")
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
                  labels / f'{hemi}.{atlas}.annot', device=device, prepared=cache,
                  gibbs_backend=annotation_gibbs_backend)
    elif operation == 'finish_surface':
        value = step(f'finish_surface_{hemi}', _finish_cortical_surface, subject,
                     hemi, binaries.get('metrics'), assets, device=device, threads=threads, defer_metrics=True,
                     white_binary=binaries.get('white'), surface_placement_backend=surface_placement_backend)
    elif operation in ('final_white', 'pial_surface'):
        if surface_placement_backend != 'python':
            raise ValueError('split placement operations require the Python backend')
        function = _finish_python_final_white if operation == 'final_white' else _finish_python_pial
        value = step(f'{operation}_{hemi}', function, subject, hemi, device=device)
    else:
        raise ValueError(f'unknown hemisphere operation: {operation}')
    return {'result': value, 'stages': steps}


def _validate_surface_placement_backend(backend: str) -> None:
    """python完整自有放置/native固定源码参考；非法值在输出创建前报错。"""
    if backend not in {"native", "python"}:
        raise ValueError("surface_placement_backend must be native or python")


def _validate_remesh_scalar_storage(storage: str) -> None:
    """只校验已有remesh双精度存储选择；无设备/文件/线程副作用。

    storage为numpy默认或python候选，返回None；其他值抛ValueError。
    两者都是CPU双精度，不使用半精度、Jacobi更新或近似重网格。
    属于mris_remesh的内部执行方式，没有独立等价官方CLI。
    """
    if storage not in {"numpy", "python"}:
        raise ValueError("remesh_scalar_storage must be numpy or python")


def _validate_annotation_gibbs_backend(backend: str) -> None:
    """在读写/设备设置前校验有序注释后端；python默认，numba为已验证候选。

    backend为字符串python/numba，返回None；非法名称抛ValueError。
    不读取网格、改变标签、初始化CUDA或调整线程。两种后端共用现有
    label_surface几何/随机序列及有序反馈，仅完整Gibbs循环执行器不同。
    它属于mris_ca_label内部步骤，没有独立官方命令。
    """
    if backend not in {"python", "numba"}:
        raise ValueError("annotation_gibbs_backend must be python or numba")


def _validate_hemisphere_execution(execution: str, *, backend: str, workers: int) -> None:
    """校验半球生命周期，不初始化CUDA或写文件。

    execution为groups（默认）或persistent；后者仅支持完整Python候选
    和两个隔离worker，调用者仍约束总线程并关闭autocast。不改变MRI、
    坐标/标签或优化器。非法组合抛ValueError；返回None，无独立官方CLI。
    """
    if not isinstance(execution, str) or execution not in {"groups", "persistent"}:
        raise ValueError("hemisphere_execution must be groups or persistent")
    if execution == "persistent" and (backend != "python" or workers != 2):
        raise ValueError("persistent hemispheres require backend='python' and two workers")


def _validate_round2_options(*, topology_backend: str,
                             topology_distance_backend: str,
                             topology_state_backend: str,
                             gca_reuse_atlas_context: bool,
                             device: str, native_optimizations: str) -> None:
    """校验显式候选后端；不初始化设备、不读取数据、不写出目录。

    topology_backend须为python-ga才可选择非默认距离或状态装配；距离
    serial/parallel为同公式CPU，gpu_defect_signed_distance须显式cuda:N。
    状态legacy/packed均完整装配，不缓存变化几何或改变随机序列。
    gca_reuse_atlas_context是布尔值，True仅用于CUDA Torch GCA的同次调用。
    默认serial/legacy/False保持既有执行；非法组合抛ValueError，无返回值。
    属原拓扑/注册命令内部步骤，参数及实测见ROUND2_INTEGRATION.md。
    """
    if not isinstance(topology_distance_backend, str) or topology_distance_backend not in {
            "serial", "parallel", "gpu_defect_signed_distance"}:
        raise ValueError("invalid topology distance backend")
    if not isinstance(topology_state_backend, str) or topology_state_backend not in {"legacy", "packed"}:
        raise ValueError("invalid topology state backend")
    if type(gca_reuse_atlas_context) is not bool:
        raise ValueError("gca_reuse_atlas_context must be bool")
    if topology_backend != "python-ga" and (
            topology_distance_backend != "serial" or topology_state_backend != "legacy"):
        raise ValueError("topology candidate options require complete python-ga")
    if topology_distance_backend == "gpu_defect_signed_distance" or gca_reuse_atlas_context:
        target = torch.device(device)
        if target.type != "cuda" or target.index is None:
            raise ValueError("candidate GPU options require explicit cuda:N")
    if gca_reuse_atlas_context and native_optimizations == "original":
        raise ValueError("atlas context reuse requires the Torch GCA implementation")


def _consume_hemisphere_result(report: dict, operation: str, hemi: str, value: dict) -> None:
    """将已原子发布的半球阶段结果写入本次报告，不重复累计嵌套耗时。"""
    result = value["result"]
    if operation == "surface":
        report.setdefault("surfaces", {})[hemi] = result
    elif operation == "register":
        report["sphere_registration"]["hemisphere_seconds"][hemi] = result["total_seconds_including_io"]
        report["sphere_registration"]["reports"][hemi] = result
    elif operation in ("finish_surface", "final_white", "pial_surface"):
        report["surfaces"][hemi].update(result)


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


def _validate_mni_execution(execution: str, device: str, threads: int) -> None:
    """校验完整MNI调度方式；不初始化CUDA或创建目录。

    完整MNI只支持CUDA自有后处理，非CUDA在校验阶段报错，不回退原生
    程序。in-process保留默认；parallel-late须显式cuda:N且总线程至少2，
    完成全部半球写出后才与CPU网格检查并行，caller autocast须关闭；非法
    组合抛ValueError，fresh exec不继承autocast而TF32策略保持。
    """
    if execution not in {"in-process", "parallel-late"}:
        raise ValueError("mni_execution must be in-process or parallel-late")
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("Complete MNI requires CUDA; choose an explicit cuda:N device. CPU reconstruction is unsupported without a validated Python MNI backend")
    if execution == "parallel-late":
        if target.type != "cuda" or target.index is None:
            raise ValueError("parallel-late MNI requires explicit cuda:N")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 2:
            raise ValueError("parallel-late MNI requires an integer thread budget >=2")
        from .profiling import autocast_state
        if any(autocast_state(kind)["enabled"] for kind in ("cpu", "cuda")):
            raise ValueError("parallel-late MNI requires caller autocast disabled; FP16/BF16 is not authorized")


def _validate_sphere_finish_backend(backend: str, device: str, hemisphere_workers: int) -> None:
    """校验既有标准sphere收尾后端，不初始化CUDA或创建输出。

    cpu保持默认；torch须显式cuda:N与两个fresh表面worker，以局部缓存
    复用完整dense GPU收尾。非法值抛ValueError，未选实验marked算子。
    """
    if backend not in {"cpu", "torch"}:
        raise ValueError("sphere_finish_backend must be cpu or torch")
    if backend == "torch":
        target = torch.device(device)
        if target.type != "cuda" or target.index is None:
            raise ValueError("torch sphere finish requires explicit cuda:N")
        if hemisphere_workers != 2:
            raise ValueError("torch sphere finish requires two cached hemisphere workers")


def _validate_input_execution(execution: str, *, n4_backend: str,
                              n4_execution: str, device: str, threads: int) -> None:
    """创建输出前检查N4/SynthSeg调度；串行默认，无影像或设备初始化。

    parallel仅支持完整SimpleITK N4、in-process算子、显式cuda:N及总4线程；
    内部以fresh exec执行N4=1、SynthSeg=3，拟合固定1。调用方autocast
    必须关闭，保留既有FP32例外；非法参数抛ValueError，无输出或物理单位。
    """
    if execution not in {"serial", "parallel"}:
        raise ValueError("input_execution must be serial or parallel")
    if execution == "parallel":
        target = torch.device(device)
        if n4_backend != "simpleitk" or n4_execution != "in-process":
            raise ValueError("parallel input requires simpleitk N4 with in-process execution")
        if target.type != "cuda" or target.index is None:
            raise ValueError("parallel input requires explicit cuda:N")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads != 4:
            raise ValueError("parallel input requires total threads=4")
        from .profiling import autocast_state
        if any(autocast_state(kind)["enabled"] for kind in ("cpu", "cuda")):
            raise ValueError("parallel input requires caller autocast disabled")


def _complete_output_validation(subject: Path, stage, *, late_mni_options: dict | None = None) -> dict:
    """join完整MNI/网格检查后列出138输出，返回原始分项报告和路径。

    subject为自产目录；stage是原计时/异常回调。late_mni_options默认
    None保留串行网格检查，否则为既有run_mni_and_validate具名输入字典，
    不读取官方结果，设备/线程须由入口预校验。返回output_validation、
    mesh_validation、outputs和可选mni_mesh_parallel完整报告。表面RAS/mm，
    warp为原scanner RAS毫米；本函数不变换数据或判定指标等效。
    任一阶段失败传播；不会在子任务join前报告输出缺失或完成。
    """
    result = {}
    if late_mni_options is not None:
        from .mni_mesh_parallel import run_mni_and_validate
        group = stage("mni_mesh_parallel", run_mni_and_validate, **late_mni_options)
        result.update(mni_mesh_parallel=group, mesh_validation=group["mesh_validation"])
    else:
        result["mesh_validation"] = stage("mesh_validation", _validate_meshes, subject)
    from .expected_outputs import paths as expected_paths
    expected = expected_paths()
    present = {name: (subject / name).is_file() for name in expected}
    missing = [name for name, exists in present.items() if not exists]
    result["output_validation"] = {"profile": "single-t1-138", "expected": len(expected),
        "present": len(expected) - len(missing), "missing": missing,
        "status": "passed" if not missing else "failed"}
    result["outputs"] = {name: str(subject / name) for name in expected if present[name]}
    return result


def _run_recon_all_python(t1: str | Path, subject_dir: str | Path,
                         weights_dir: str | Path, assets_dir: str | Path,
                         *, device: str = "cuda:0", threads: int = 4,
                         native_bin_dir: str | Path | None = None,
                         profile_stages: bool = False,
                         cuda_allocator_cache: str = "auto",
                         hemisphere_workers: int = 1,
                         hemisphere_execution: str = "groups",
                         native_optimizations: str = "auto",
                         n4_backend: str = "native",
                         n4_execution: str = "in-process",
                         input_execution: str = "serial",
                         normalization_controls_backend: str = "cpu",
                         normalization_initial_bias_backend: str = "cpu",
                         wm_backend: str = "native",
                         wm_execution: str = "in-process",
                         wm_edit_backend: str = "native",
                         defects_backend: str = "native",
                         sphere_normals_backend: str = "numba",
                         inflate_backend: str = "native",
                         sphere_finish_backend: str = "cpu",
                         surface_placement_backend: str = "native",
                         topology_backend: str = "native",
                         topology_finish_backend: str = "python",
                         topology_distance_backend: str = "serial",
                         topology_state_backend: str = "legacy",
                         annotation_gibbs_backend: str = "python",
                         remesh_scalar_storage: str = "numpy",
                         mni_execution: str = "in-process",
                         gca_inverse_backend: str = "cpu",
                         gca_candidate_chunk: int = 64,
                         gca_execution: str = "in-process",
                         gca_reuse_atlas_context: bool = False,
                         fill_backend: str = "python",
                         backend: str = "native") -> dict:
    """从单幅原始 T1 连续生成 conform 体积、双侧表面和脑区统计。

    t1、subject_dir、weights_dir、assets_dir 是输入影像、空输出目录、
    已校验权重和资产的路径；native_bin_dir=None 时使用当前 Conda bin。
    device 默认 cuda:0，threads 默认 4；不自动使用 FP16/BF16。
    n4_backend默认native保留Conda ITK；torch复用完整200轮PyTorch N4。
    n4_execution默认in-process；isolated仅Torch/cuda:N，完整缓存exec
    保留父CUDA/精度，输入输出SHA、实际迭代和子显存另记，不读取参考。
    input_execution默认serial；parallel以1+3线程并行完整SimpleITK N4和
    SynthSeg GPU，须总threads=4；复用同orig及完整标签/CSV，后续不重算。
    normalization_controls_backend默认cpu；torch复用两轮归一化的GPU邻域，
    须显式cuda:N。控制点规则、有序离群清理及其余偏置步骤保持原算法。
    normalization_initial_bias_backend默认cpu；torch只在第二轮复用GPU
    初始偏置传播/平滑，保留CPU距离/排序、原除乘顺序和float32输出。
    inflate_backend默认native；torch复用完整标准inflated/sulc GPU算法，
    仅支持显式cuda:N和hemisphere_workers=2，在surface子exec局部启用
    缓存。nofix、球面优化/配准和父进程CUDA策略保持，失败不回退。
    surface_placement_backend默认native；python复用完整四轮preaparc、
    最终white与pial，不执行放置程序或white能力探测；CPU有序计算保留，
    CUDA只迁移完整候选与相交标记，默认400步保护且失败不回退。
    接入/输入输出/真实回归见PYTHON_SURFACE_PIPELINE.md。
    topology_backend默认native；backend=python将缺省解析为数字拓扑，
    显式python-ga保留完整自有GA选择。GA CPU有序核使用半球实际threads，
    必须通过完整标准remesh、相交清理和最终orig门后才继续放置；不调用
    原拓扑程序。GA逐阶段回归与原始T1整例分别记录。
    sphere_finish_backend默认cpu；torch仅将标准sphere完整dense收尾迁往
    主CUDA，须cuda:N和两个fresh缓存worker，投影/停止和其余球面算法保持。
    annotation_gibbs_backend默认python；numba复用完整有序Gibbs核，
    三图谱共享现有版本化几何缓存，保留随机序列/同分规则/逐顶点反馈；
    首次JIT及打包成本计入阶段，GPU几何设备与总CPU预算不变。
    remesh_scalar_storage默认numpy；python复用同序拆缩/双精度平滑，
    仅减少CPU标量调度开销，不改拓扑/阈值/停止规则，读写计入秒数。
    topology_finish_backend默认python保留现有收尾；conda-cpp仅与native
    GA及混合native profile搭配，程序退出后仍独立检查零相交。
    mni_execution默认in-process；parallel-late将完整非线性延后到全部
    半球写出之后，与CPU网格检查按同一threads预算并行，join后检查输出。
    要求cuda:N及显式整数threads>=2；错误传播，不生成占位文件。
    native_optimizations=auto在CUDA使用已有FNIT Torch评分+Python EM；CPU
    查询独立产物能力，4线程可使用验证过的GCA缓存。white使用专用快速
    程序，pial保留原程序；original固定原生GCA用于控制。
    native_optimizations=torch 将 GCA 的候选平移/线性搜索切换到 FNIT
    的分块 PyTorch CUDA scorer；EM 精修仍为 Python FP32，并在报告中明确标注。
    defects_backend=native 使用原生投射；torch 使用完整 PyTorch 投射，仍按
    左侧清零、右侧合并执行。标签值/空间保持，随机调色板改为固定颜色。
    wm_edit_backend 默认 native；torch-hybrid 将完整 WM/aseg 编辑的静态
    子步骤交给 CUDA，有序反馈仍复用 Numba CPU。python使用完整CPU配方。
    路径几何证明失败时执行完整源序搜索；不支持的存储/网格仍报错。
    sphere_normals_backend 默认 numba；torch 只替换标准 sphere 的有序面法向，
    完整目标函数、迭代决策及 CPU finish 保持。两个实验后端均要求 CUDA。
    wm_backend=torch-optimized复用Torch直方图与缓存平面几何，有序反馈为Numba。
    wm_execution=in-process保留父缓存；isolated只允许torch-optimized，
    子exec启用缓存并返回完整阶段报告，父CUDA状态保持。
    gca_inverse_backend默认cpu，torch按同公式批量求逆；gca_candidate_chunk
    默认64，只改变完整候选的分块，非默认设置须使用CUDA的Torch GCA后端。
    gca_execution=in-process保留父缓存策略；isolated在独立exec内启用缓存，
    子进程结束释放GPU资源，不修改父CUDA状态；仅CUDA Torch GCA可用。
    新增gca_reuse_atlas_context默认False（同次GCA复用）；
    topology_distance_backend默认serial，topology_state_backend默认legacy；
    非默认拓扑选项仅完整python-ga，GPU须cuda:N；返回报告记录实际选择，
    全参数、阶段返回和真实范围见ROUND2_INTEGRATION.md。
    fill_backend默认python；numba复用有序CPU堆，torch-numba另用CUDA初始边界。
    三者均保留切割、距离场与标签规则；torch-numba要求CUDA，失败不回退。
    hemisphere_workers 默认1保持串行，2用独立 exec 半球进程；总 threads
    在双侧之间平分（奇数向下取整），父进程保留依赖屏障和共享发布。
    CUDA首次科学JIT前核验并选择当前环境ptxas；显式编译器设置优先。
    triton_runtime记录版本/SHA和缓存状态，不改变TF32或影像算法。
    同进程已缓存其他编译器时要求新解释器，不清第三方JIT缓存。
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
    _validate_hemisphere_execution(hemisphere_execution, backend=backend, workers=hemisphere_workers)
    initial_bias_options = _normalization_initial_bias_options(normalization_initial_bias_backend, device)
    _validate_inflate_backend(inflate_backend, device, hemisphere_workers)
    _validate_sphere_finish_backend(sphere_finish_backend, device, hemisphere_workers)
    _validate_annotation_gibbs_backend(annotation_gibbs_backend)
    _validate_remesh_scalar_storage(remesh_scalar_storage)
    _validate_mni_execution(mni_execution, device, threads)
    from .input_n4_chain import validate_n4_execution
    validate_n4_execution(n4_backend=n4_backend, n4_execution=n4_execution, device=device)
    _validate_input_execution(input_execution, n4_backend=n4_backend,
                              n4_execution=n4_execution, device=device, threads=threads)
    if backend not in {"native", "python", "python-gpu"}:
        raise ValueError("backend must be native, python or python-gpu")
    if topology_backend not in {"native", "python", "python-ga"}:
        raise ValueError("topology_backend must be native, python or python-ga")
    _validate_topology_finish_backend(topology_finish_backend,
                                      topology_backend=topology_backend, backend=backend)
    _validate_round2_options(topology_backend=topology_backend,
        topology_distance_backend=topology_distance_backend, topology_state_backend=topology_state_backend,
        gca_reuse_atlas_context=gca_reuse_atlas_context, device=device,
        native_optimizations=native_optimizations)
    if backend == "python":
        from .python_profile import python_profile_options
        expected = python_profile_options(device=device, threads=threads,
                                          n4_backend=n4_backend, n4_execution=n4_execution,
                                          topology_backend=topology_backend, input_execution=input_execution,
                                          hemisphere_execution=hemisphere_execution,
                                          gca_reuse_atlas_context=gca_reuse_atlas_context,
                                          gca_candidate_chunk=gca_candidate_chunk,
                                          topology_distance_backend=topology_distance_backend,
                                          topology_state_backend=topology_state_backend)
        actual = {"native_optimizations": native_optimizations, "n4_backend": n4_backend,
                  "n4_execution": n4_execution, "wm_backend": wm_backend, "wm_execution": wm_execution,
                  "wm_edit_backend": wm_edit_backend, "defects_backend": defects_backend,
                  "inflate_backend": inflate_backend, "surface_placement_backend": surface_placement_backend,
                  "topology_backend": topology_backend, "annotation_gibbs_backend": annotation_gibbs_backend,
                  "remesh_scalar_storage": remesh_scalar_storage, "fill_backend": fill_backend,
                  "gca_execution": gca_execution, "gca_inverse_backend": gca_inverse_backend,
                  "gca_candidate_chunk": gca_candidate_chunk,
                  "normalization_controls_backend": normalization_controls_backend,
                  "normalization_initial_bias_backend": normalization_initial_bias_backend,
                  "sphere_finish_backend": sphere_finish_backend, "mni_execution": mni_execution,
                  "hemisphere_workers": hemisphere_workers, "input_execution": input_execution,
                  "hemisphere_execution": hemisphere_execution,
                  "gca_reuse_atlas_context": gca_reuse_atlas_context,
                  "topology_distance_backend": topology_distance_backend,
                  "topology_state_backend": topology_state_backend}
        if actual != expected:
            raise ValueError("Python profile requires its complete own implementation recipe")
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
    if wm_edit_backend not in {"native", "python", "torch-hybrid"}:
        raise ValueError("wm_edit_backend must be native, python or torch-hybrid")
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
    from .input_n4_chain import require_n4_runtime_dependencies
    n4_dependencies = require_n4_runtime_dependencies(n4_backend=n4_backend)
    from .triton_runtime import configure_triton_compiler
    triton_runtime = configure_triton_compiler(device=str(device))
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
    if (hemisphere_workers == 2 or mni_execution == "parallel-late") and any(
            autocast_state(kind)["enabled"] for kind in ("cpu", "cuda")):
        raise ValueError("parallel stages require caller autocast disabled; FP16/BF16 is not authorized")
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
    _validate_surface_placement_backend(surface_placement_backend)
    # Independent Python execution must not inspect an installed upstream program directory.
    native_bin_dir = (Path(sys.prefix) / "bin" if backend == "python"
                      else _native_bin_directory(native_bin_dir))
    n4_binary = (_native_binary(native_bin_dir, "fnit_n4_itk")
                 if n4_backend == "native" else None)
    topology_binary = (_native_topology_binary(native_bin_dir)
                       if topology_backend == "native" else (None, None))
    remesh_binary = (_native_binary(native_bin_dir, "mris_remesh")
                     if topology_finish_backend == "conda-cpp" else (None, None))
    metrics_binary = (_native_surface_metrics_binary(native_bin_dir)
                      if surface_placement_backend == "native" else (None, None))
    from .native_runtime_selection import select_native_optimizations
    if backend == "python":
        from .execution_recipe import python_runtime_selection
        native_selection = python_runtime_selection()
    else:
        native_selection = select_native_optimizations(native_bin_dir, threads=threads,
                                                       mode=native_optimizations, device=device,
                                                       include_white=surface_placement_backend == "native")
    native_em = (_native_em_register_binary(native_bin_dir)
                 if native_selection["em_backend"] != "torch" else (None, None))
    white_binary = (_native_binary(native_bin_dir, Path(native_selection["white_binary"]).name)
                    if surface_placement_backend == "native" else (None, None))
    inflate_binary = (_native_inflate_binary(native_bin_dir)
                      if surface_placement_backend == "native" or inflate_backend == "native" else (None, None))
    intersection_binary = (_native_binary(native_bin_dir, "mris_remove_intersection")
                           if topology_finish_backend == "conda-cpp" else (None, None))
    paint_binary = (_native_binary(native_bin_dir, "mrisp_paint")
                    if surface_placement_backend == "native" else (None, None))
    curvature_stats_binary = (_native_binary(native_bin_dir, "mris_curvature_stats")
                              if surface_placement_backend == "native" else (None, None))
    defect_binary = (_native_binary(native_bin_dir, "mri_label2vol")
                     if defects_backend == "native" else None)
    # Complete MNI uses the own GPU implementation in both reconstruction profiles.
    warp_binaries = ((None, None),) * 3
    wm_segment_binary = (_native_binary(native_bin_dir, "mri_segment")
                         if wm_backend == "native" else None)
    wm_edit_binary = (_native_binary(native_bin_dir, "mri_edit_wm_with_aseg")
                      if wm_edit_backend == "native" else None)
    from .execution_recipe import validate_program_boundary, execution_dag
    validate_program_boundary(backend=backend, programs={
        "n4": n4_binary[0] if n4_binary else None,
        "topology": topology_binary[0], "metrics": metrics_binary[0],
        "em": native_em[0], "white": white_binary[0],
        "inflate": inflate_binary[0], "remesh": remesh_binary[0],
        "intersection": intersection_binary[0],
        "paint": paint_binary[0], "curvature_stats": curvature_stats_binary[0],
        "defects": defect_binary[0] if defect_binary else None,
        "wm_segment": wm_segment_binary[0] if wm_segment_binary else None,
        "wm_edit": wm_edit_binary[0] if wm_edit_binary else None,
        **{f"warp_{i}": value[0] for i, value in enumerate(warp_binaries)}})
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
                    "n4_runtime_dependencies": n4_dependencies,
                    "triton_runtime": triton_runtime,
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
    report["execution_dag"] = execution_dag(threads=threads, input_execution=input_execution,
        mni_execution=mni_execution, hemisphere_workers=hemisphere_workers, backend=backend)
    if backend == "python":
        report["implementation_profile"] = {"name": "python", "declared_original_stage_programs": [],
            "options": expected, "cpu_python_kernels_allowed": True,
            "generic_python_libraries_allowed": True,
            "topology_algorithm": ("FNIT complete ordered genetic topology" if topology_backend == "python-ga"
                                   else "ordered digital topology; different from GA"),
            "strict_reproduction": "reported_separately", "overall_metric_equivalence": "not_assessed"}
    report["n4_configuration"] = {
        "backend": n4_backend, "execution": n4_execution,
        "implementation": ("Conda ITK" if n4_backend == "native" else
                           "SimpleITK Python API; generic ITK library, no stage executable" if n4_backend == "simpleitk"
                           else "FNIT complete PyTorch N4"),
        "device": device if n4_backend == "torch" else "cpu",
        "production_default_changed": False,
        "known_strict_intensity_differences": n4_backend in {"torch", "simpleitk"}}
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
        "binary": str(native_em[0]) if native_em[0] is not None else None, "sha256": native_em[1],
        "backend": gca_backend,
        "inverse_backend": gca_inverse_backend if gca_backend == "torch" else None,
        "candidate_chunk": gca_candidate_chunk if gca_backend == "torch" else None,
        "execution": gca_execution if gca_backend == "torch" else "native-subprocess",
        "reuse_atlas_context": gca_reuse_atlas_context if gca_backend == "torch" else False,
        "device": device if gca_backend == "torch" else "cpu",
        "strict_native_em": gca_backend != "torch"}
    report["native_optimizations"] = native_selection
    report["topology_repair"] = {
        "implementation": ("FNIT complete ordered genetic topology; Python/Numba CPU" if topology_backend == "python-ga"
                           else "FNIT ordered digital topology; different complete algorithm" if topology_backend == "python"
                           else "native-c++"),
        "backend": topology_backend,
        "distance_backend": topology_distance_backend if topology_backend == "python-ga" else None,
        "state_backend": topology_state_backend if topology_backend == "python-ga" else None,
        "strict_GA_reproduction": topology_backend == "native",
        "metric_equivalence": "not_assessed" if topology_backend != "native" else "reported_separately",
        "strict_numerical_validation": "reported_separately",
        "binary": str(topology_binary[0]) if topology_binary[0] is not None else None,
        "sha256": topology_binary[1],
        "finish_backend": topology_finish_backend,
        "intersection_implementation": ("Conda source-built C++ with independent double zero gate"
                                         if topology_finish_backend == "conda-cpp" else
                                         "FNIT Python/Numba complete SOAP; residual intersections fail"),
        "intersection_binary": str(intersection_binary[0]) if intersection_binary[0] else None,
        "intersection_sha256": intersection_binary[1],
        "upstream": "intensity-derived WM and aseg-guided fill"}
    report["white_matter_chain"] = {
        "implementation": (
            ("FNIT PyTorch hybrid segmentation" if wm_backend != "native"
             else "Conda C++ segmentation") + " + " +
            ("Conda C++ edit" if wm_edit_binary else
             "Python/Numba CPU edit" if wm_edit_backend == "python" else
             "ordered Numba + PyTorch CUDA edit")),
        "mri_segment_sha256": wm_segment_binary[1] if wm_segment_binary else None,
        "segmentation_backend": wm_backend,
        "segmentation_execution": wm_execution if wm_backend != "native" else "native-subprocess",
        "edit_backend": wm_edit_backend,
        "edit_implementation": ("Conda C++" if wm_edit_binary else
                                "Python/Numba CPU" if wm_edit_backend == "python" else "ordered Numba + PyTorch CUDA"),
        "mri_edit_wm_with_aseg_sha256": wm_edit_binary[1] if wm_edit_binary else None}
    report["white_preaparc"] = {
        "implementation": ("FNIT Python/Numba + Torch candidates/cleanup" if surface_placement_backend == "python"
                           else "Python MNI/aux/finalsurfs + Conda C++ placement"),
        "backend": surface_placement_backend,
        "binary": str(white_binary[0]) if white_binary[0] is not None else None, "sha256": white_binary[1],
        "final_smoothwm": "Python 3 passes on CPU",
        "final_white_pial": ("FNIT complete Python/Numba white and pial after annotation"
                             if surface_placement_backend == "python" else "Conda source-built C++ white and pial after annotation")}
    report["surface_metrics"] = {
        "implementation": ("FNIT PyTorch " + str(device) if torch.device(device).type == "cuda" or surface_placement_backend == "python"
                           else "Conda source-built C++"),
        "binary": str(metrics_binary[0]) if metrics_binary[0] is not None else None,
        "sha256": metrics_binary[1], "upstream": "placed final white/pial"}
    report["annotation"] = {"gibbs_backend": annotation_gibbs_backend,
        "geometry_device": device, "ordered_feedback_device": "cpu",
        "implementation": "FNIT Torch geometry + ordered " + annotation_gibbs_backend,
        "random_sequence": "existing VnlRandom/permutation; no seed or order change"}
    report["remesh"] = {"implementation": ("Conda source-built C++" if topology_finish_backend == "conda-cpp"
                                            else "FNIT ordered Python/Numba CPU"),
        "backend": topology_finish_backend,
        "binary": str(remesh_binary[0]) if remesh_binary[0] else None,
        "sha256": remesh_binary[1],
        "scalar_storage": remesh_scalar_storage if topology_finish_backend == "python" else None,
        "precision": "float64; no precision reduction" if topology_finish_backend == "python" else "pinned Conda program",
        "iterations": 3,
        "topology_and_update_order": "FNIT ordered" if topology_finish_backend == "python" else "pinned Conda source"}
    report["sphere_generation"] = {
        "implementation": "FNIT quick/standard sphere; explicit nofix and standard inflation backends",
        "mris_inflate": {"binary": str(inflate_binary[0]) if inflate_binary[0] is not None else None,
                         "sha256": inflate_binary[1]},
        "normals_backend": sphere_normals_backend,
        "standard_inflate_backend": inflate_backend,
        "nofix_inflate_backend": ("torch" if torch.device(device).type == "cuda" else "numpy")
                                if surface_placement_backend == "python" else "native",
        "surface_worker_cache": "enabled" if inflate_backend == "torch" or sphere_finish_backend == "torch" else "inherit",
        "normals_device": device if sphere_normals_backend == "torch" else "cpu",
        "finish_backend": sphere_finish_backend,
        "finish_device": device if sphere_finish_backend == "torch" else "cpu", "upstream": "repaired topology"}
    registration_device = torch.device(device)
    if registration_device.type == "cuda" and registration_device.index is None:
        registration_device = torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_initialized() else 0)
    report["sphere_registration"] = {
        "implementation": "Python/Numba + ordered CUDA averaging" if registration_device.type == "cuda" else "Python/Numba",
        "averaging_device": str(registration_device), "overlap_device": "cpu",
        "hemisphere_seconds": {}, "reports": {}}
    report["extra_curvature"] = {
        "mrisp_paint_sha256": paint_binary[1],
        "paint_backend": "python-numba-cpu" if surface_placement_backend == "python" else "native",
        "mris_curvature_stats_sha256": curvature_stats_binary[1],
        "statistics_backend": "python" if surface_placement_backend == "python" else "native",
        "reports": {}}
    report["defects_volume"] = ({"backend": "native", "binary": str(defect_binary[0]),
                                 "sha256": defect_binary[1]} if defect_binary else
                                {"backend": "torch", "device": device,
                                 "palette": "deterministic colors; original label names and values"})
    if defects_backend == "torch":
        from . import defects_label_volume_torch
        report["defects_volume"]["source_sha256"] = hashlib.sha256(
            Path(defects_label_volume_torch.__file__).read_bytes()).hexdigest()
    report["mni_nonlinear"] = {
        "implementation": "PyTorch deform + FNIT GPU warp conversion/inversion/check",
        "postprocess_backend": "gpu",
        "native_programs_used": False,
        "execution": mni_execution,
        "device": device,
        "precision": "FP32 CUDA exception",
        "native_sha256": {}}

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
        if name == "mni_mesh_parallel" and isinstance(value, dict):
            row["mni_runtime"] = value.get("mni_nonlinear")
            row["parallel_intervals"] = value.get("intervals")
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
    precomputed_synthseg = None
    report["input_execution"] = input_execution
    if input_execution == "parallel":
        from .input_stage_parallel import run_input_stage_group
        report["input_parallel"] = stage(
            "n4_synthseg_parallel", run_input_stage_group,
            orig_path=mri / "orig.mgz", n4_output=nu0,
            segmentation_output=mri / "synthseg.rca.mgz",
            volumes_output=stats / "synthseg.vol.csv", weights_dir=weights,
            color_lut=assets / "FreeSurferColorLUT.txt",
            report_path=subject / "scripts/n4-synthseg-parallel.json",
            device=device, threads=threads, execution="parallel",
            profile_stages=profile_stages)
        report["n4_runtime"] = report["input_parallel"]["workers"]["n4"]["value"]
        precomputed_synthseg = report["input_parallel"]["workers"]["synthseg"]["value"]
    else:
        report["n4_runtime"] = stage(
            "n4", run_n4_stage, input_path=mri / "orig.mgz", output_path=nu0,
            n4_backend=n4_backend, n4_execution=n4_execution,
            native_binary=n4_binary[0] if n4_binary else None,
            device=device, threads=threads, profile=profile_stages,
            report_path=subject / "scripts/n4-isolated.json" if n4_execution == "isolated"
            else n4_profile if n4_backend in {"native", "simpleitk"} else None)
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

    if precomputed_synthseg is None:
        try:
            result = stage("SynthSeg", run_synthseg_and_write)
        finally:
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
        stiv_mm3 = result.total_intracranial_mm3
        report["precision"]["SynthSeg_actual_forward"] = result.precision
    else:
        # 完整分割和CSV已由组worker写出；这里只消费元数据，不读参考或重复模型。
        stiv_mm3 = precomputed_synthseg["total_intracranial_mm3"]
        report["precision"]["SynthSeg_actual_forward"] = precomputed_synthseg["precision"]
    lta = mri / "transforms/talairach.lta"
    gca = assets / "average/RB_all_2020-01-02.gca"
    report["gca_registration"]["stage_result"] = stage(
          "mri_em_register", _run_native_em_register, native_em[0], mri, gca, assets,
          backend=native_selection["em_backend"], binary_sha256=native_em[1],
          threads=threads, device=device, inverse_backend=gca_inverse_backend,
          candidate_chunk=gca_candidate_chunk, execution=gca_execution,
          **({"reuse_atlas_context": True} if gca_reuse_atlas_context else {}))
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
    elif wm_edit_backend == "python":
        report["white_matter_chain"]["edit_runtime"] = stage(
            "mri_edit_wm_with_aseg", _run_python_wm_edit, mri)
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
                             device=device,
                             **({"defer_mni_nonlinear": True}
                                if mni_execution == "parallel-late" else {}))
    except Exception as error:
        if report["status"] != "failed":
            report.update(status="failed", failed_stage="white_mri_chain",
                          error=f"{type(error).__name__}: {error}")
            (subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))
        raise
    if hemisphere_workers == 2:
        from .hemisphere_parallel import run_hemisphere_group, HemisphereGroupError
        common = {'assets': str(assets),
                  'profile_stages': profile_stages,
                  'sphere_normals_backend': sphere_normals_backend,
                  'inflate_backend': inflate_backend,
                  'sphere_finish_backend': sphere_finish_backend,
                  'surface_placement_backend': surface_placement_backend,
                  'topology_backend': topology_backend,
                  'topology_finish_backend': topology_finish_backend,
                  **({'topology_distance_backend': topology_distance_backend}
                     if topology_distance_backend != 'serial' else {}),
                  **({'topology_state_backend': topology_state_backend}
                     if topology_state_backend != 'legacy' else {}),
                  'annotation_gibbs_backend': annotation_gibbs_backend,
                  'remesh_scalar_storage': remesh_scalar_storage,
                  'binaries': {name: str(value) for name, value in (
                      *((('topology', topology_binary[0]),) if topology_binary[0] is not None else ()),
                      *((('remesh', remesh_binary[0]),) if remesh_binary[0] is not None else ()),
                      *((('intersection', intersection_binary[0]),)
                        if intersection_binary[0] is not None else ()),
                      *((('inflate', inflate_binary[0]),) if inflate_binary[0] is not None else ()),
                      *((('metrics', metrics_binary[0]), ('white', white_binary[0]))
                        if surface_placement_backend == "native" else ()),
                      *((('defect', defect_binary[0]),) if defect_binary else ()),
                      *((('paint', paint_binary[0]),) if paint_binary[0] is not None else ()) )},
                  'registration_atlases': {hemi: str(path) for hemi, path in registration_atlases.items()}}
        placement_operations = (('final_white', 'pial_surface')
                                if surface_placement_backend == 'python'
                                else ('finish_surface',))
        report['hemisphere_scheduling']['execution'] = hemisphere_execution
        if hemisphere_execution == "persistent":
            from .hemisphere_integration import run_persistent_cortical_chain
            run_persistent_cortical_chain(
                subject=subject, report=report, stage=stage, common_kwargs=common,
                device=device, registration_device=registration_device, threads=threads,
                profile_stages=profile_stages, assets=assets,
                defects_backend=defects_backend)
        else:
            for operation in ('surface', 'register', 'annotation', *placement_operations):
                try:
                    group = stage(f'{operation}_hemisphere_group', run_hemisphere_group,
                                  subject, operation, device=str(registration_device), threads=threads,
                                  workers=hemisphere_workers, profile_stages=profile_stages, kwargs=common,
                                  cuda_allocator_cache=("enabled" if operation == "final_white" or
                                                        (operation == "surface" and
                                                         (inflate_backend == "torch" or
                                                          sphere_finish_backend == "torch")) else "inherit"))
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
                    elif operation in ('finish_surface', 'final_white', 'pial_surface'):
                        report['surfaces'][hemi].update(result)
                if operation in ('finish_surface', 'pial_surface'):
                    # 实测GPU顶点指标独立exec并行更慢；放置并行、指标串行。
                    for hemi in ('lh', 'rh'):
                        metrics_result = stage(f'finish_metrics_{hemi}', _finish_cortical_metrics,
                              subject, hemi, metrics_binary[0], assets, device=device,
                              python_only=surface_placement_backend == "python")
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
                           remesh_binary=remesh_binary[0],
                           place_binary=white_binary[0],
                           defect_binary=defect_binary[0] if defect_binary else None,
                           assets=assets, defects_backend=defects_backend,
                           sphere_normals_backend=sphere_normals_backend, inflate_backend=inflate_backend,
                           sphere_finish_backend=sphere_finish_backend,
                           remesh_scalar_storage=remesh_scalar_storage,
                           surface_placement_backend=surface_placement_backend,
                           topology_backend=topology_backend,
                           topology_finish_backend=topology_finish_backend,
                           **({"topology_distance_backend": topology_distance_backend}
                              if topology_distance_backend != "serial" else {}),
                           **({"topology_state_backend": topology_state_backend}
                              if topology_state_backend != "legacy" else {}))
            report.setdefault("surfaces", {})[hemi] = result
            (subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))

        from .mris_register_run import run_register_sphere
        for hemi in ("lh", "rh"):
            surf = subject / "surf"
            result = stage(f"register_{hemi}", run_register_sphere,
                           surf / f"{hemi}.sphere", surf / f"{hemi}.smoothwm",
                           surf / f"{hemi}.sulc", registration_atlases[hemi],
                           surf / f"{hemi}.sphere.reg", overlap_device="cpu",
                           averaging_device=str(registration_device),
                           require_unfolded=surface_placement_backend == "python")
            report["sphere_registration"]["hemisphere_seconds"][hemi] = result[
                "total_seconds_including_io"]
            report["sphere_registration"]["reports"][hemi] = result
            stage(f"avg_curv_{hemi}", _run_avg_curv, paint_binary[0], subject,
                  hemi, registration_atlases[hemi], assets,
                  python_only=surface_placement_backend == "python")

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
                      prepared=cache, gibbs_backend=annotation_gibbs_backend)
        for hemi in ("lh", "rh"):
            result = stage(f"finish_surface_{hemi}", _finish_cortical_surface,
                           subject, hemi, metrics_binary[0], assets,
                           device=device, threads=threads, white_binary=white_binary[0],
                           surface_placement_backend=surface_placement_backend)
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
        report["extra_curvature"]["reports"][hemi] = stage(f"stats_{hemi}_curv", _run_curvature_stats,
              curvature_stats_binary[0], subject, hemi, assets, device=device,
              python_only=surface_placement_backend == "python")
    late_mni_options = None
    if mni_execution == "parallel-late":
        late_mni_options = {"subject": subject, "weights": weights, "assets": assets,
            "warp_binaries": tuple(binary[0] for binary in warp_binaries),
            "report_path": subject / "scripts/mni-mesh-parallel.json", "device": device,
            "threads": threads, "execution": "parallel", "profile_stages": profile_stages}
    report.update(_complete_output_validation(subject, stage, late_mni_options=late_mni_options))
    if late_mni_options is not None:
        report["mni_nonlinear"]["runtime"] = report["mni_mesh_parallel"]["mni_nonlinear"]
    report["numeric_validation"] = {"status": "not_run",
                                    "reason": "reference_subject_not_provided"}
    valid = report["output_validation"]["status"] == "passed" and report["mesh_validation"]["status"] == "passed"
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
                         hemisphere_execution: str = "groups",
                         native_optimizations: str = "auto",
                         n4_backend: str = "native",
                         n4_execution: str = "in-process",
                         input_execution: str = "serial",
                         normalization_controls_backend: str = "cpu",
                         normalization_initial_bias_backend: str = "cpu",
                         wm_backend: str = "native",
                         wm_execution: str = "in-process",
                         wm_edit_backend: str = "native",
                         defects_backend: str = "native",
                         sphere_normals_backend: str = "numba",
                         inflate_backend: str = "native",
                         sphere_finish_backend: str = "cpu",
                         surface_placement_backend: str = "native",
                         topology_backend: str = "native",
                         topology_finish_backend: str = "python",
                         topology_distance_backend: str = "serial",
                         topology_state_backend: str = "legacy",
                         annotation_gibbs_backend: str = "python",
                         remesh_scalar_storage: str = "numpy",
                         mni_execution: str = "in-process",
                         gca_inverse_backend: str = "cpu",
                         gca_candidate_chunk: int = 64,
                         gca_execution: str = "in-process",
                         gca_reuse_atlas_context: bool = False,
                         fill_backend: str = "python",
                         backend: str = "native") -> dict:
    """从原始单 T1 连续重建；输入、输出及坐标定义见 recon-all 中文说明。

    t1 为原始影像；subject_dir 须为空；weights_dir/assets_dir 为已校验资源；
    native_bin_dir=None 使用当前 Conda bin。device 默认 cuda:0，threads=4
    约束 Torch intraop 和调用线程的 Numba 掩码，退出时恢复调用方设置。
    hemisphere_workers=1保持串行，2启用私有被试目录的双侧 exec 进程，
    总 threads 平分，须至少2；返回 hemisphere_scheduling 组墙钟/worker报告，
    所有共享发布失败均传播到主报告。并行要求 caller autocast 关闭。
    新增gca_reuse_atlas_context默认False（同次GCA复用）；
    topology_distance_backend默认serial，topology_state_backend默认legacy；
    非默认拓扑选项仅完整python-ga，GPU须cuda:N；返回报告记录实际选择，
    全参数、阶段返回和真实范围见ROUND2_INTEGRATION.md。
    hemisphere_execution默认groups保留阶段屏障；persistent仅完整Python
    配方，在各自半球依赖满足后连续推进，一侧一个fresh worker/私有目录。
    共享defects仍LH→RH累计，终态指标仍串行，失败保留部分回执但不验收。
    backend=python选择完整独立Python候选：SimpleITK N4、自产WM/编辑、
    完整数字拓扑、white/pial、inflation、图谱与统计；允许有序CPU核。
    此配方固定双侧worker=2及其后端，忽略这些逐阶段后端的混合默认；
    不读取native_bin_dir或执行原程序。topology_backend缺省解析为数字
    python；显式python-ga使用完整自有GA及最终orig门，按半球线程预算
    执行，保留原域缺陷标签及WM写出头，不重复remesh。数字拓扑不同于FS GA，严格复现、
    整体指标等效和整例墙钟须独立评估，不能仅凭无外程认定验收。
    backend=native 使用当前已验证的混合 FNIT/Conda 对照；backend=python-gpu
    是严格纯 Python/CUDA profile，若仍有未完成阶段会在创建输出前抛出结构化
    PurePythonGpuUnavailable，不回退到原生程序。native_optimizations=auto在CUDA使用已有Torch GCA评分+Python EM，CPU
    按原生产物能力选择缓存；white选择专用快速程序，pial保留原生。
    original固定原始原生实现。Torch GCA与原生仍有既有数值差异，单列报告。
    n4_backend默认native，torch使用完整N4，simpleitk使用通用Python库；n4_execution默认in-process，
    isolated只允许Torch/cuda:N，复用局部缓存worker并记录完整阶段/子显存。
    非法组合在创建被试目录前报错，既有N4系统强度差异单列，不判整体等效。
    input_execution默认serial，parallel只改变完整N4/SynthSeg调度：总4线程，
    N4固定拟合1，SynthSeg GPU保留原图/翻转及FP32例外。二者只依赖同orig，
    join后继续nu/归一化；分割不重复，父CUDA保持，失败整体传播。
    normalization_controls_backend=cpu保留原邻域；torch在两轮归一化复用
    同规则GPU邻域，须显式cuda:N，其余步骤/设备不变。实际选择写入报告。
    normalization_initial_bias_backend=cpu保留第二轮初始偏置；torch仅
    复用该轮既有GPU传播/平滑，要求cuda:N，除乘/输出精度与原规则相同。
    inflate_backend=native保留标准inflation；torch复用完整GPU算法生成
    同序inflated/sulc，须cuda:N与hemisphere_workers=2。仅surface组的
    fresh worker局部启用缓存；nofix、sphere及注册算法保持，父策略不变。
    surface_placement_backend=native保留原放置；python接入已有完整四轮
    preaparc/white/pial，并使用自有CPU顶点指标；CPU有序核允许保留，
    CUDA候选/清理不改顺序、不回退或开启半精度，完整400步保护保持。
    其他尚未替代阶段另记录，不能据此称整例脱离原版程序。
    sphere_finish_backend=cpu保留收尾，torch复用既有完整dense GPU收尾，
    要求cuda:N与两个隔离worker；局部缓存不改变父allocator/精度。
    非法组合在CUDA初始化和创建输出前抛ValueError，执行失败不回退。
    annotation_gibbs_backend=python保持原循环；numba复用已有完整有序
    Gibbs核，三图谱共享现有版本化几何缓存，不改随机序列/标签语义。
    GPU几何设备/线程预算保持，编译/打包/读写计入实际阶段报告。
    remesh_scalar_storage默认numpy；python为完整有序remesh的CPU双精度
    存储候选，算法/阈值/停止及总线程保持；不称GPU重网格。
    topology_finish_backend默认python；conda-cpp显式运行声明Conda目录内
    独立源码构建的remesh与相交程序，输出经双谓词零残余和几何门。
    mni_execution=in-process保留默认；parallel-late在全部统计/半球写出后
    运行完整MNI GPU子exec，与CPU网格检查共享总threads预算（至少2）。
    要求cuda:N；返回mni_mesh_parallel完整报告及原mni_nonlinear.runtime，
    join后才检查138输出；同一原始scanner RAS毫米变换，失败传播。
    defects_backend=native 使用原生缺陷投射，torch 用本项目完整投射规则
    并保持双侧累积顺序；仅颜色表换成固定颜色，当前默认为 native。
    wm_edit_backend=native 调用独立构建编辑程序；python使用完整Numba
    CPU编辑；torch-hybrid 使用 CUDA 静态编辑和 Numba 有序核心，不是纯GPU。
    路径几何证明失败时执行完整有序搜索，不支持的4字节存储/网格或
    源内循环不再推进时报错。两例同输入解码0差、真实active未触发见WM_REMOVE_PATHS.md。
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
    CUDA首次科学JIT前选择当前环境编译器，尊重显式TRITON_PTXAS_PATH，
    triton_runtime返回其版本/SHA和缓存核验；不重置已初始化CUDA或改精度。
    本进程缓存其他编译器时抛RuntimeError要求新解释器，详见TRITON_RUNTIME.md。
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
    if backend == "python":
        from .python_profile import python_profile_options
        # 历史API缺省native在独立配方中解析为SimpleITK；显式Torch保持。
        options = python_profile_options(device=device, threads=threads,
            n4_backend="simpleitk" if n4_backend == "native" else n4_backend,
            n4_execution=n4_execution,
            topology_backend="python" if topology_backend == "native" else topology_backend,
            input_execution=input_execution, hemisphere_execution=hemisphere_execution,
            gca_reuse_atlas_context=gca_reuse_atlas_context,
            gca_candidate_chunk=1024 if gca_candidate_chunk == 64 else gca_candidate_chunk,
            topology_distance_backend=topology_distance_backend, topology_state_backend=topology_state_backend)
        native_optimizations = options["native_optimizations"]
        n4_backend, n4_execution = options["n4_backend"], options["n4_execution"]
        wm_backend, wm_execution = options["wm_backend"], options["wm_execution"]
        wm_edit_backend, defects_backend = options["wm_edit_backend"], options["defects_backend"]
        inflate_backend = options["inflate_backend"]
        surface_placement_backend, topology_backend = options["surface_placement_backend"], options["topology_backend"]
        annotation_gibbs_backend, remesh_scalar_storage = options["annotation_gibbs_backend"], options["remesh_scalar_storage"]
        gca_execution, gca_inverse_backend = options["gca_execution"], options["gca_inverse_backend"]
        gca_candidate_chunk = options["gca_candidate_chunk"]
        normalization_controls_backend = options["normalization_controls_backend"]
        normalization_initial_bias_backend = options["normalization_initial_bias_backend"]
        sphere_finish_backend, mni_execution = options["sphere_finish_backend"], options["mni_execution"]
        fill_backend, hemisphere_workers = options["fill_backend"], options["hemisphere_workers"]
    _validate_hemisphere_execution(hemisphere_execution, backend=backend, workers=hemisphere_workers)
    _validate_round2_options(topology_backend=topology_backend,
        topology_distance_backend=topology_distance_backend, topology_state_backend=topology_state_backend,
        gca_reuse_atlas_context=gca_reuse_atlas_context, device=device,
        native_optimizations=native_optimizations)
    _normalization_controls_options(normalization_controls_backend, device)
    _normalization_initial_bias_options(normalization_initial_bias_backend, device)
    _validate_inflate_backend(inflate_backend, device, hemisphere_workers)
    _validate_sphere_finish_backend(sphere_finish_backend, device, hemisphere_workers)
    _validate_annotation_gibbs_backend(annotation_gibbs_backend)
    _validate_remesh_scalar_storage(remesh_scalar_storage)
    _validate_mni_execution(mni_execution, device, threads)
    _validate_input_execution(input_execution, n4_backend=n4_backend,
                              n4_execution=n4_execution, device=device, threads=threads)
    from .thread_budget import thread_budget
    from .execution_recipe import original_stage_guard

    report_path = Path(subject_dir) / "fnit-native-free-run.json"

    def report_version():
        """返回现有报告的 stat 版本；不存在或不可访问时不读取其内容。"""
        try:
            stat = report_path.stat()
            return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        except OSError:
            return None

    entry_version = report_version()
    report = budget = pipeline_error = program_guard = None

    def record_public_timing(value: dict, wall: float) -> None:
        """更新本次公开入口的秒数/线程记录，最终元数据写出不计入 wall。"""
        if budget is not None:
            value["thread_budget"] = budget
        if program_guard is not None:
            value['stage_program_boundary'] = program_guard
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
        with original_stage_guard(enabled=backend == "python") as program_guard, \
                thread_budget(threads=threads) as budget:
            try:
                report = _run_recon_all_python(
                    t1=t1, subject_dir=subject_dir, weights_dir=weights_dir, assets_dir=assets_dir,
                    device=device, threads=threads, native_bin_dir=native_bin_dir,
                    profile_stages=profile_stages, cuda_allocator_cache=cuda_allocator_cache,
                    **({"hemisphere_workers": hemisphere_workers} if hemisphere_workers != 1 else {}),
                    **({"hemisphere_execution": hemisphere_execution}
                       if hemisphere_execution != "groups" or backend == "python" else {}),
                    **({"native_optimizations": native_optimizations} if native_optimizations != "auto" else {}),
                    **({"n4_backend": n4_backend} if n4_backend != "native" else {}),
                    **({"n4_execution": n4_execution} if n4_execution != "in-process" else {}),
                    **({"input_execution": input_execution} if input_execution != "serial" or backend == "python" else {}),
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
                    **({"sphere_finish_backend": sphere_finish_backend} if sphere_finish_backend != "cpu" else {}),
                    **({"surface_placement_backend": surface_placement_backend} if surface_placement_backend != "native" else {}),
                    **({"topology_backend": topology_backend} if topology_backend != "native" else {}),
                    **({"topology_finish_backend": topology_finish_backend}
                       if topology_finish_backend != "python" else {}),
                    **({"topology_distance_backend": topology_distance_backend}
                       if topology_distance_backend != "serial" or backend == "python" else {}),
                    **({"topology_state_backend": topology_state_backend}
                       if topology_state_backend != "legacy" or backend == "python" else {}),
                    **({"annotation_gibbs_backend": annotation_gibbs_backend}
                       if annotation_gibbs_backend != "python" else {}),
                    **({"remesh_scalar_storage": remesh_scalar_storage}
                       if remesh_scalar_storage != "numpy" else {}),
                    **({"mni_execution": mni_execution} if mni_execution != "in-process" else {}),
                    **({"gca_inverse_backend": gca_inverse_backend} if gca_inverse_backend != "cpu" else {}),
                    **({"gca_candidate_chunk": gca_candidate_chunk} if gca_candidate_chunk != 64 else {}),
                    **({"gca_execution": gca_execution} if gca_execution != "in-process" else {}),
                    **({"gca_reuse_atlas_context": gca_reuse_atlas_context}
                       if gca_reuse_atlas_context or backend == "python" else {}),
                    **({"fill_backend": fill_backend} if fill_backend != "python" else {}),
                    backend=backend)
                if program_guard is not None:
                    report['stage_program_boundary'] = program_guard
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
                quality = getattr(original_error, "diagnostics", None)
                if isinstance(quality, dict):
                    report["quality_failure"] = quality
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
    parser.add_argument("--hemisphere-execution", choices=("groups", "persistent"), default="groups",
                        help="persistent: complete Python only, one private workspace and fresh worker per hemisphere")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n4-backend", choices=("native", "torch", "simpleitk"), default="native",
                        help="native benchmark wrapper, complete Torch N4, or SimpleITK Python API without stage executable")
    parser.add_argument("--n4-execution", choices=("in-process", "isolated"), default="in-process",
                        help="isolated only for Torch cuda:N; cached child preserves parent allocator")
    parser.add_argument("--input-execution", choices=("serial", "parallel"), default="serial",
                        help="parallel: complete SimpleITK N4 + SynthSeg CUDA, four threads split 1+3")
    parser.add_argument("--normalization-controls-backend", choices=("cpu", "torch"), default="cpu",
                        help="same-rule control-point neighbors; torch requires cuda:N, remaining solver unchanged")
    parser.add_argument("--normalization-initial-bias-backend", choices=("cpu", "torch"), default="cpu",
                        help="second normalization initial bias; torch reuses existing propagation/smoothing with original arithmetic")
    parser.add_argument("--inflate-backend", choices=("native", "torch"), default="native",
                        help="standard smoothwm inflation; torch requires cuda:N and two cached hemisphere workers; nofix unchanged")
    parser.add_argument("--mni-execution", choices=("in-process", "parallel-late"), default="in-process",
                        help="complete MNI in a late exec overlapping CPU mesh validation; explicit cuda:N and >=2 total threads")
    parser.add_argument("--annotation-gibbs-backend", choices=("python", "numba"), default="python",
                        help="ordered GCSA reclassification; numba reuses the exact compiled feedback loop")
    parser.add_argument("--remesh-scalar-storage", choices=("numpy", "python"), default="numpy",
                        help="ordered CPU double-precision remesh storage; algorithm and update order preserved")
    parser.add_argument("--surface-placement-backend", choices=("native", "python"), default="native",
                        help="complete own Python/Numba white.preaparc, final white and pial; no original placement executable")
    parser.add_argument("--topology-backend", choices=("native", "python", "python-ga"), default="native",
                        help="python: 数字拓扑；python-ga: 完整自有有序GA，无原版程序，最终网格仍须验收")
    parser.add_argument("--topology-finish-backend", choices=("python", "conda-cpp"), default="python",
                        help="explicit Conda source-built remesh3 and intersection with independent zero-residual gate")
    parser.add_argument("--sphere-finish-backend", choices=("cpu", "torch"), default="cpu",
                        help="existing full dense standard-sphere overlap cleanup; torch requires cuda:N and two cached surface workers")
    parser.add_argument("--native-bin-dir", type=Path)
    parser.add_argument("--hemisphere-workers", type=int, choices=(1, 2), default=1,
                        help="independent hemisphere processes; total threads split across two workers")
    parser.add_argument("--native-optimizations", choices=("auto", "original", "torch"), default="auto",
                        help="auto uses Torch GCA on CUDA, validated Conda selection on CPU; original fixes Conda; torch requests FNIT scorer")
    parser.add_argument("--wm-backend", choices=("native", "torch", "torch-optimized"), default="native",
                        help="native, existing Torch hybrid, or Torch histograms/cached geometry with ordered CPU feedback")
    parser.add_argument("--wm-execution", choices=("in-process", "isolated"), default="in-process",
                        help="isolated torch-optimized WM enables caching in a fresh child, preserving parent CUDA")
    parser.add_argument("--gca-reuse-atlas-context", action="store_true",
                        help="reuse immutable covariance/classifier context within complete Torch GCA call")
    parser.add_argument("--topology-distance-backend", choices=("serial", "parallel", "gpu_defect_signed_distance"), default="serial")
    parser.add_argument("--topology-state-backend", choices=("legacy", "packed"), default="legacy")
    parser.add_argument("--gca-inverse-backend", choices=("cpu", "torch"), default="cpu")
    parser.add_argument("--gca-candidate-chunk", type=int, default=64)
    parser.add_argument("--gca-execution", choices=("in-process", "isolated"), default="in-process",
                        help="isolated Torch GCA enables caching in a fresh child, preserving parent allocator")
    parser.add_argument("--fill-backend", choices=("python", "numba", "torch-numba"), default="python",
                        help="ordered fill; numba CPU heap, optionally CUDA initial boundary")
    parser.add_argument("--wm-edit-backend", choices=("native", "python", "torch-hybrid"), default="native",
                        help="native edit or ordered NumBa + Torch GPU edit, rejects unsupported byte-access paths")
    parser.add_argument("--defects-backend", choices=("native", "torch"), default="native",
                        help="native mri_label2vol or FNIT PyTorch exact voxel projection; colors differ")
    parser.add_argument("--sphere-normals-backend", choices=("numba", "torch"), default="numba",
                        help="ordered sphere normals on CPU or explicit CUDA; remaining solver unchanged")
    parser.add_argument("--backend", choices=("native", "python", "python-gpu"), default="native",
                        help="python: complete own Python candidate including digital topology, CPU kernels allowed; native: mixed benchmark control; python-gpu: legacy strict CUDA capability diagnostic")
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
                                  hemisphere_execution=args.hemisphere_execution,
                                  native_optimizations=args.native_optimizations,
                                  n4_backend=args.n4_backend,
                                  n4_execution=args.n4_execution,
                                  input_execution=args.input_execution,
                                  normalization_controls_backend=args.normalization_controls_backend,
                                  normalization_initial_bias_backend=args.normalization_initial_bias_backend,
                                  wm_backend=args.wm_backend,
                                  wm_execution=args.wm_execution,
                                  wm_edit_backend=args.wm_edit_backend,
                                  defects_backend=args.defects_backend,
                                  sphere_normals_backend=args.sphere_normals_backend,
                                  inflate_backend=args.inflate_backend,
                                  sphere_finish_backend=args.sphere_finish_backend,
                                  surface_placement_backend=args.surface_placement_backend,
                                  topology_backend=args.topology_backend,
                                  topology_finish_backend=args.topology_finish_backend,
                                  topology_distance_backend=args.topology_distance_backend,
                                  topology_state_backend=args.topology_state_backend,
                                  annotation_gibbs_backend=args.annotation_gibbs_backend,
                                  remesh_scalar_storage=args.remesh_scalar_storage,
                                  mni_execution=args.mni_execution,
                                  gca_inverse_backend=args.gca_inverse_backend,
                                  gca_candidate_chunk=args.gca_candidate_chunk,
                                  gca_execution=args.gca_execution,
                                  gca_reuse_atlas_context=args.gca_reuse_atlas_context,
                                  fill_backend=args.fill_backend,
                                  backend=args.backend)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
