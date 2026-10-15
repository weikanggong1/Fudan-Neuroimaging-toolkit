"""自产 nofix/qsphere/MRI 文件到完整自有 GA premesh 和可选严格 orig。"""
from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Integral
from pathlib import Path
import struct
import sys
import time

import nibabel as nib
from nibabel.freesurfer import io as fsio
import numpy as np

from .thread_budget import thread_budget
from .topology_edge_mri_score import source_topology_ras_to_voxel
from .topology_genetic_repair import repair_genetic_topology
from .mris_remesh_python import remesh_surface, _footer_offset
from .mris_remove_intersection_python import run_remove_intersection


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    """诊断非有限标量写 null；影像、坐标及核心分数的有效性另行检查。"""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_json_value(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n")


def _loaded_module_hashes():
    hashes = {name: _sha256(module.__file__) for name, module in sorted(sys.modules.items())
            if name.startswith("fnit.recon_all.") and
            isinstance(getattr(module, "__file__", None), str) and module.__file__.endswith(".py")}
    hashes["fnit.recon_all.topology_genetic_files"] = _sha256(__file__)
    return hashes


def _geometry_valid(geometry):
    """复用 orig 文件门的体积头条件；不改变坐标或头字段。"""
    try:
        valid = np.asarray(geometry["valid"])
        flag = valid.item() if valid.ndim == 0 else None
        if isinstance(flag, (str, bytes)):
            flag = flag.split()[0]
        dimensions = np.asarray(geometry["volume"])
        spacing = np.asarray(geometry["voxelsize"], np.float64)
        center = np.asarray(geometry["cras"], np.float64)
        axes = [np.asarray(geometry[key], np.float64) for key in ("xras", "yras", "zras")]
        if (valid.ndim != 0 or float(flag) != 1 or dimensions.shape != (3,) or
                dimensions.dtype.kind not in "iu" or np.any(dimensions <= 0) or
                spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0) or
                center.shape != (3,) or not np.isfinite(center).all() or
                any(axis.shape != (3,) or not np.isfinite(axis).all() for axis in axes)):
            return False
        determinant = float(np.linalg.det(np.column_stack(axes)))
        return bool(np.isfinite(determinant) and determinant != 0)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return False


def _wm_geometry(image, path):
    """MRIScopyVolGeomFromMRI 对应的 WM 头，不能使用 brain 或 nofix 替代。"""
    if not isinstance(image, nib.MGHImage) or len(image.shape) != 3:
        raise ValueError("wm must be a three-dimensional MGH/MGZ image")
    header = image.header
    geometry = dict(head=np.array([2, 0, 20], np.int32),
        valid=str(int(header["goodRASFlag"])), filename=str(path),
        volume=np.array(header["dims"][:3], np.int32),
        voxelsize=np.array(header["delta"], np.float32),
        xras=np.array(header["Mdc"][0], np.float32),
        yras=np.array(header["Mdc"][1], np.float32),
        zras=np.array(header["Mdc"][2], np.float32),
        cras=np.array(header["Pxyz_c"], np.float32))
    if not _geometry_valid(geometry):
        raise ValueError("valid finite WM volume geometry required")
    return geometry


def _same_geometry(actual, expected):
    if not _geometry_valid(actual):
        return False
    if not np.array_equal(actual["volume"], expected["volume"]):
        return False
    # The surface text footer uses decimal serialization of original F32 MRI fields.
    return all(np.asarray(actual[key], np.float32).tobytes() ==
               np.asarray(expected[key], np.float32).tobytes()
               for key in ("voxelsize", "xras", "yras", "zras", "cras"))


def _read_surface(path):
    vertices, faces = fsio.read_geometry(str(path))
    xyz = np.asarray(vertices, np.float32)
    triangles = np.asarray(faces, np.int32)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all() or
            triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0 or
            np.any(triangles < 0) or np.any(triangles >= len(xyz))):
        raise ValueError("finite nonempty surface coordinates and valid ordered triangular faces required")
    return xyz, triangles


def _checked_core_result(result, raw_vertices, raw_faces):
    if (result.get("execution_status") != "complete" or result.get("complete_ga") is not True or
            result.get("native_program_called") is not False or
            result.get("quality", {}).get("status") != "passed"):
        raise RuntimeError("complete own GA and passed premesh structure/current quality required")
    state = result["surface_state"]
    original = np.asarray(state["original_vertices"])
    current = np.asarray(state["current_vertices"])
    nfaces = state["nfaces"]
    if isinstance(nfaces, (bool, np.bool_)) or not isinstance(nfaces, Integral):
        raise ValueError("logical compact face count must be an integer")
    face_storage = np.asarray(state["faces"])
    if not 0 < nfaces <= len(face_storage) or face_storage.dtype.kind not in "iu":
        raise ValueError("valid logical compact faces required")
    faces = np.asarray(face_storage[:nfaces], np.int32)
    if (original.dtype != np.float32 or original.ndim != 2 or original.shape[1] != 3 or
            len(original) == 0 or current.dtype != np.float32 or current.shape != original.shape or
            not np.isfinite(original).all() or current.tobytes() != original.tobytes() or
            faces.shape != (nfaces, 3) or np.any(faces < 0) or np.any(faces >= len(original))):
        raise ValueError("compact restored ORIGINAL coordinates and their logical ordered faces required")
    count = len(raw_vertices)
    raw_to_base = np.asarray(result["raw_to_base_vertex_translation"])
    base_to_compact = np.asarray(result["base_to_compact_vertex_translation"])
    labels = np.asarray(result["original_domain_defect_labels"])
    if (raw_to_base.shape != (count,) or raw_to_base.dtype.kind not in "iu" or
            not np.array_equal(np.sort(raw_to_base), np.arange(count)) or
            base_to_compact.shape != (count,) or base_to_compact.dtype.kind not in "iu" or
            np.any(base_to_compact < -1) or
            not np.array_equal(np.sort(base_to_compact[base_to_compact >= 0]), np.arange(len(original)))):
        raise ValueError("raw-to-base permutation and base-to-compact deletion mapping required")
    if (labels.shape != (count,) or labels.dtype.kind not in "iu" or np.any(labels < 0) or
            np.any(labels > len(result["per_defect"])) or
            not np.array_equal(labels.astype(np.float32).astype(np.int64), labels)):
        raise ValueError("complete raw-domain integer defect labels exactly representable as F32 required")
    cleanup_vertices = np.asarray(result["current_cleanup_vertices"])
    if cleanup_vertices.shape != original.shape or not np.isfinite(cleanup_vertices).all():
        raise ValueError("compact CURRENT cleanup coordinates required for diagnostics")
    return original, faces, raw_to_base, base_to_compact, labels, cleanup_vertices


def _final_quality(*, surface_path, cleanup_report, expected_geometry):
    """复用结构/vertex-link门及同 SHA 的两个实际相交复检，非参考输出。"""
    from .compare_subject import _topology
    from .topology_volume_python import _nonmanifold_vertex_links
    if (cleanup_report.get("execution_status") != "complete" or
            cleanup_report.get("output_written") is not True or
            cleanup_report.get("require_zero") is not True or
            cleanup_report.get("output_sha256") != _sha256(surface_path)):
        raise ValueError("exact successful strict cleanup report bound to written orig required")
    counts = [cleanup_report[key] for key in
        ("source_intersecting_faces_after_independent_recheck",
         "geometric_intersecting_faces_after_independent_recheck")]
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0 for value in counts):
        raise ValueError("nonnegative integer independent intersection counts required")
    xyz, faces, geometry = fsio.read_geometry(str(surface_path), read_metadata=True)
    faces = np.ascontiguousarray(faces, np.int32)
    structural = _topology(xyz, faces)
    if not structural.get("valid"):
        return dict(status="failed", topology=structural)
    links = int(_nonmanifold_vertex_links(faces, len(xyz)))
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, groups = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    conflicts = int(np.count_nonzero(np.bincount(groups, weights=signs)))
    triangles = xyz[faces]
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    zeros = int(np.count_nonzero(area2 == 0))
    finite = bool(np.isfinite(xyz).all() and np.isfinite(area2).all())
    header_matches_wm = _same_geometry(geometry, expected_geometry)
    passed = bool(finite and links == conflicts == zeros == 0 and counts == [0, 0] and header_matches_wm)
    return dict(status="passed" if passed else "failed", topology=structural,
        nonmanifold_vertices=links, edge_orientation_conflicts=conflicts, zero_area_faces=zeros,
        finite_coordinates=finite, source_intersecting_faces=int(counts[0]),
        geometric_intersecting_faces=int(counts[1]), wm_geometry_matches=header_matches_wm,
        intersection_counts_bound_to_output_sha256=cleanup_report["output_sha256"],
        coordinate_space="surface_RAS", coordinate_unit="mm")


def _footer_sha256(path):
    with Path(path).open("rb") as stream:
        stream.seek(_footer_offset(path))
        return hashlib.sha256(stream.read()).hexdigest()


def run_genetic_topology_files(*, input_surface, input_qsphere, brain, wm, hemisphere,
        output_directory, threads=4, seed=1234, population_size=10, max_unchanged=10,
        niters=-1, initial_selection=True, vertex_eliminate=True, use_overlap=False,
        normal_status="surface", soap_iterations=5, final_orig=False,
        remesh_scalar_storage="numpy", face_detection_cache=False, blocker_scan_backend="python",
        distance_backend="serial", distance_device=None, state_backend="legacy", random_audit_callback=None, defect_observer=None,
        failure_state_callback=None):
    """四个自产文件→完整自有 GA、原域标签和可选严格最终 orig。

    input_surface/input_qsphere 是同 N、同有序面的 nofix/qsphere 三角表面；
    brain 是 uint8 三维 MGZ，wm 是有有效体积头的三维 MGZ。二者不交换用途：
    brain 头决定 tkregister RAS→MRI 采样矩阵，wm 头决定写出表面 footer。
    hemisphere='lh'|'rh'；output_directory 必须不存在，创建权限 700。
    threads 默认4并传播完整 GA/remesh/cleanup，恢复调用方 Torch/Numba 状态。
    seed/population_size/max_unchanged/niters 默认1234/10/10/−1，两个选择开关
    默认True，use_overlap默认False，normal_status='surface'，SOAP默认5。
    face_detection_cache默认False；仅接受bool/np.bool_并规范为Python bool，
    True复用完整控制器的每缺陷面身份缓存，不缓存动态几何/邻接判定。
    blocker_scan_backend默认python，可显式numba；同一成熟谓词与完整顺序，
    无自动回退。非法选择在创建输出目录前拒绝，报告记录core实际返回值。
    state_backend默认legacy，显式packed需任务5完整状态API，搜索、最终评分与
    几何发布均传同一选择；缺实现明确失败，不退回legacy，不启用旧缓存。
    random_audit_callback/defect_observer默认None，仅Python接口；分别具名观测
    (index,low,high,value)和(defect_number,result)，用于完整流/全域状态验证。
    返回值忽略，异常传播；观察者不得修改result。默认不新增审计读写。
    distance_device默认None；GPU后端gpu_defect_signed_distance须显式cuda:N，
    验证可用设备，无CPU回退；CPU路径不接受设备。
    distance_backend默认serial，可显式parallel；只选择现有有序F32距离
    核，在threads总预算内按不重叠x slab执行，不并行候选。参数先校验，
    core返回选择必须一致；报告分开记录requested/returned选择及每缺陷
    最终评分的实际执行/skip。Numba mask/slab不是采样到的OS线程数。
    final_orig=False 只完成 premesh 中间域；True 再执行 remesh3、source_numba
    require_zero 清理及实际网格/WM 几何门。remesh_scalar_storage 默认numpy，
    可显式python，均使用CPU双精度标量，不改变算法。参数原义由成熟 core 检查。

    返回严格JSON可序列化dict：status/final_orig、outputs及同键output_sha256、
    report_path/report_sha256/completion_path、maps/per_defect路径、具名seconds、
    premesh_quality及最终quality。标签始终rawN/fnum rawF；premesh为压缩M域的
    ORIGINAL，不写TMP清理坐标；remesh后编号另变，无伪造remesh映射。
    无原版程序/源码/参考输出/tools运行依赖、无全局猴补，无模型操作；仅显式GPU距离选择使用CUDA。
    非法数据/映射/已存在目录ValueError，计算或质量失败传播异常并留failure.json；
    失败不写COMPLETE.json，已产生的中间文件仅供诊断，不能按其存在判定成功。
    failure_state_callback默认None；仅完整GA最终质量门失败时由核心传递
    compact坐标、面、rip、有序CSR和双判据诊断的独立副本。接口不自动
    写出失败网格；显式调用者负责私有目录和数据权限，回调异常原样传播。
    四文件上游由调用者声明，不依据文件名推定自产；哈希/格式/编号只校验
    本次输入。internal_ga_inputs_self_generated只指本阶段内部GA状态；
    reference_inputs_used=False仅指阶段未额外读取独立参考，不证明上游来源。
    """
    started = time.perf_counter()
    from .topology_ga_backend_policy import validate_state_backend
    validate_state_backend(state_backend)
    for name, callback in (("random_audit_callback", random_audit_callback), ("defect_observer", defect_observer),
                           ("failure_state_callback", failure_state_callback)):
        if callback is not None and not callable(callback):
            raise ValueError(name + " must be callable or None")
    if hemisphere not in ("lh", "rh"):
        raise ValueError("hemisphere must be lh or rh")
    if isinstance(threads, (bool, np.bool_)) or not isinstance(threads, Integral) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if not isinstance(final_orig, (bool, np.bool_)):
        raise ValueError("final_orig must be an explicit bool")
    if not isinstance(face_detection_cache, (bool, np.bool_)):
        raise ValueError("face_detection_cache must be an explicit bool")
    face_detection_cache = bool(face_detection_cache)
    if not isinstance(blocker_scan_backend, str) or blocker_scan_backend not in ("python", "numba"):
        raise ValueError("blocker_scan_backend must be python or numba")
    from .topology_distance_policy import validate_distance_policy
    distance_device = validate_distance_policy(distance_backend=distance_backend, distance_device=distance_device)
    if remesh_scalar_storage not in ("numpy", "python"):
        raise ValueError("remesh_scalar_storage must be numpy or python")
    directory = Path(output_directory).resolve()
    if directory.exists():
        raise ValueError("output_directory must be a new directory")
    directory.mkdir(parents=True, mode=0o700)
    paths = {"nofix": Path(input_surface).resolve(), "qsphere": Path(input_qsphere).resolve(),
             "brain": Path(brain).resolve(), "wm": Path(wm).resolve()}
    seconds = {key: None for key in ("read_inputs", "genetic_api", "write_premesh", "remesh",
        "original_intersection_cleanup", "final_mesh_quality", "complete_file_api")}
    try:
        source_before = _loaded_module_hashes()
        input_hashes = {key: _sha256(path) for key, path in paths.items()}
        vertices, faces = _read_surface(paths["nofix"])
        qsphere, sphere_faces = _read_surface(paths["qsphere"])
        if vertices.shape != qsphere.shape or not np.array_equal(faces, sphere_faces):
            raise ValueError("nofix and qsphere must share N and the complete ordered faces")
        brain_image = nib.load(paths["brain"])
        if not isinstance(brain_image, nib.MGHImage) or len(brain_image.shape) != 3:
            raise ValueError("brain must be a three-dimensional MGH/MGZ image")
        volume = np.asanyarray(brain_image.dataobj)
        if volume.dtype != np.uint8 or volume.ndim != 3 or min(volume.shape) <= 0:
            raise ValueError("brain must contain uint8 MRI samples without quantization")
        matrix = source_topology_ras_to_voxel(volume_header=brain_image.header)
        wm_geometry = _wm_geometry(nib.load(paths["wm"]), paths["wm"])
        metadata = dict(use_real_ras=0, hemisphere=0 if hemisphere == "lh" else 1,
                        status=0, origxyz_status=0, volume_geometry=wm_geometry)
        seconds["read_inputs"] = time.perf_counter() - started
        progress_path = directory / f"{hemisphere}.genetic.progress.jsonl"
        progress_path.touch()

        def progress(*, defect_number, result):
            search = result["search"]
            row = dict(defect_number=defect_number, best_fitness=search["best_fitness"],
                actual_final_fitness=result["finalization"]["fitness"], counts=search["counts"],
                legacy_last_generation_index=search["generations"], termination=search["termination"])
            with progress_path.open("a") as stream:
                stream.write(json.dumps(_json_value(row), ensure_ascii=False, allow_nan=False) + "\n")
            if defect_observer is not None:
                defect_observer(defect_number=defect_number, result=result)

        with thread_budget(threads=int(threads)) as budget_report:
            tick = time.perf_counter()
            result = repair_genetic_topology(vertices=vertices, faces=faces, qsphere_vertices=qsphere,
                volume=volume, ras_to_voxel=matrix, source_metadata=metadata, threads=int(threads),
                seed=seed, population_size=population_size, max_unchanged=max_unchanged, niters=niters,
                initial_selection=initial_selection, vertex_eliminate=vertex_eliminate, use_overlap=use_overlap,
                normal_status=normal_status, soap_iterations=soap_iterations, progress_callback=progress,
                face_detection_cache=face_detection_cache, blocker_scan_backend=blocker_scan_backend,
                distance_backend=distance_backend, distance_device=distance_device, state_backend=state_backend,
                random_audit_callback=random_audit_callback,
                failure_state_callback=failure_state_callback)
            if result.get("blocker_scan_backend") != blocker_scan_backend:
                raise RuntimeError("core blocker_scan_backend differs from the requested backend")
            if result.get("distance_backend") != distance_backend:
                raise RuntimeError("core distance_backend differs from the requested backend")
            core_distance_device = result.get("distance_device")
            core_state_backend = result.get("state_backend", "legacy")
            if core_distance_device != distance_device:
                raise RuntimeError("core distance_device differs from the requested device")
            if core_state_backend != state_backend:
                raise RuntimeError("core state_backend differs from the requested backend")
            seconds["genetic_api"] = time.perf_counter() - tick
            original, compact_faces, raw_to_base, base_to_compact, labels, cleaned_current = \
                _checked_core_result(result, vertices, faces)
            tick = time.perf_counter()
            premesh = directory / f"{hemisphere}.orig.premesh"
            defect_labels = directory / f"{hemisphere}.defect_labels"
            fsio.write_geometry(str(premesh), original, compact_faces, volume_info=wm_geometry)
            fsio.write_morph_data(str(defect_labels), labels.astype(np.float32), fnum=len(faces))
            written_xyz, written_faces, written_geometry = fsio.read_geometry(str(premesh), read_metadata=True)
            if (np.asarray(written_xyz, np.float32).tobytes() != original.tobytes() or
                    not np.array_equal(written_faces, compact_faces) or not _same_geometry(written_geometry, wm_geometry)):
                raise RuntimeError("written compact ORIGINAL geometry or WM footer differs")
            with defect_labels.open("rb") as stream:
                magic = stream.read(3)
                morph_counts = struct.unpack(">iii", stream.read(12))
            if (magic != b"\xff\xff\xff" or morph_counts != (len(vertices), len(faces), 1) or
                    not np.array_equal(fsio.read_morph_data(str(defect_labels)), labels.astype(np.float32))):
                raise RuntimeError("written defect labels do not preserve the raw vertex/face domain")
            maps_path = directory / f"{hemisphere}.genetic.maps.private.npz"
            np.savez(maps_path, raw_to_base_vertex_translation=raw_to_base,
                base_to_compact_vertex_translation=base_to_compact,
                raw_to_compact_vertex_translation=base_to_compact[raw_to_base],
                original_domain_defect_labels=labels, raw_vertex_count=np.int64(len(vertices)),
                raw_face_count=np.int64(len(faces)), compact_vertex_count=np.int64(len(original)),
                compact_face_count=np.int64(len(compact_faces)), current_cleanup_vertices=cleaned_current)
            per_defect_path = directory / f"{hemisphere}.genetic.defects.json"
            _write_json(per_defect_path, result["per_defect"])
            seconds["write_premesh"] = time.perf_counter() - tick
            orig = None
            cleanup_report = None
            final_quality = dict(status="not_assessed", scope="final_orig",
                required_next_steps=list(result["required_next_steps"]))
            if final_orig:
                orig = directory / f"{hemisphere}.orig"
                tick = time.perf_counter()
                remesh_surface(input_path=premesh, output_path=orig, iterations=3,
                               scalar_storage=remesh_scalar_storage)
                seconds["remesh"] = time.perf_counter() - tick
                tick = time.perf_counter()
                cleanup_report = run_remove_intersection(input_path=orig, output_path=orig,
                    marking_backend="source_numba", device=None, require_zero=True,
                    report_path=directory / f"{hemisphere}.genetic.orig-intersection.json")
                seconds["original_intersection_cleanup"] = time.perf_counter() - tick
                tick = time.perf_counter()
                final_quality = _final_quality(surface_path=orig, cleanup_report=cleanup_report,
                                               expected_geometry=wm_geometry)
                final_quality["footer_bytes_preserved"] = _footer_sha256(premesh) == _footer_sha256(orig)
                if final_quality["status"] != "passed" or not final_quality["footer_bytes_preserved"]:
                    raise RuntimeError("final orig failed strict mesh or WM footer quality: " + repr(final_quality))
                seconds["final_mesh_quality"] = time.perf_counter() - tick
        after_hashes = {key: _sha256(path) for key, path in paths.items()}
        if after_hashes != input_hashes:
            raise RuntimeError("input files changed during complete genetic file API")
        outputs = dict(premesh=str(premesh), defect_labels=str(defect_labels), orig=str(orig) if orig else None)
        output_hashes = {key: _sha256(path) if path else None for key, path in outputs.items()}
        artifacts = [premesh, defect_labels, maps_path, per_defect_path, progress_path]
        if orig is not None:
            artifacts.extend([orig, directory / f"{hemisphere}.genetic.orig-intersection.json"])
        source_hashes = _loaded_module_hashes()
        if any(source_hashes.get(name) != digest for name, digest in source_before.items()):
            raise RuntimeError("loaded module source files changed during genetic file API")
        seconds["complete_file_api"] = time.perf_counter() - started
        report_path = directory / "report.json"
        completion_path = directory / "COMPLETE.json"
        report = _json_value(dict(status="complete", execution_status="complete", final_orig=bool(final_orig),
            scope="genetic_topology_file_stage_not_whole_recon_all", hemisphere=hemisphere,
            outputs={key: Path(path).name if path else None for key, path in outputs.items()},
            output_sha256=output_hashes, report_path=report_path.name,
            completion_path=completion_path.name, maps_path=maps_path.name, per_defect_path=per_defect_path.name,
            input_paths=paths, input_sha256=input_hashes, input_sha256_after=after_hashes,
            loaded_module_sha256=source_hashes, loaded_module_sha256_before=source_before,
            artifacts_sha256={path.name: _sha256(path) for path in artifacts},
            parameters=dict(threads=int(threads), seed=seed, population_size=population_size,
                max_unchanged=max_unchanged, niters=niters, initial_selection=initial_selection,
                vertex_eliminate=vertex_eliminate, use_overlap=use_overlap, normal_status=normal_status,
                face_detection_cache=face_detection_cache,
                blocker_scan_backend=result["blocker_scan_backend"],
                distance_backend=result["distance_backend"], distance_device=core_distance_device, state_backend=core_state_backend,
                soap_iterations=soap_iterations, remesh_iterations=3 if final_orig else None,
                remesh_scalar_storage=remesh_scalar_storage), thread_budget=budget_report,
            distance_backend_policy=dict(requested=distance_backend,returned=result["distance_backend"],
                requested_device=distance_device, returned_device=core_distance_device,
                per_defect_final_evaluation=[row.get("final_distance_policy") for row in result["per_defect"]],
                worker_count_observed=False),
            state_backend_policy=dict(requested=state_backend, returned=core_state_backend),
            sampling_matrix=matrix, sampling_matrix_source="brain_MGHHeader_tkregister_RAS_to_voxel",
            written_geometry=wm_geometry, written_geometry_source="wm_MGHHeader",
            domains=dict(raw_vertices=len(vertices), raw_faces=len(faces), compact_vertices=len(original),
                compact_faces=len(compact_faces), premesh_coordinates="compact_ORIGINAL_surface_RAS_mm",
                defect_labels="raw_nofix_vertices_F32_with_raw_face_count",
                final_orig_vertices="remeshed_domain" if final_orig else "not_produced"),
            seconds=seconds, report_timing_boundary="complete_file_api_until_final_receipt_serialization",
            core_seconds=result["seconds"], per_defect=result["per_defect"],
            random_draws=result["random_draws"], premesh_quality=result["quality"], quality=final_quality,
            original_intersection_cleanup=cleanup_report, standard_policy=result["standard_policy"],
            complete_ga=True, native_program_called=False, reference_inputs_used=False,
            reference_inputs_used_scope="no_additional_reference_inputs_read_by_this_stage",
            reference_comparison_performed=False, internal_ga_inputs_self_generated=True,
            input_file_upstream_provenance="caller_supplied_not_verified",
            input_file_upstream_self_generated=None, declared_inputs_only=True, gpu_memory_bytes=None,
            strict_source_final_reproduction="not_assessed", overall_metric_equivalence="not_assessed",
            whole_recon_all="not_assessed", diagnostic_nonfinite_json="null"))
        _write_json(report_path, report)
        report_sha256 = _sha256(report_path)
        _write_json(completion_path, dict(status="complete", final_orig=bool(final_orig),
            report_sha256=report_sha256, output_sha256=output_hashes))
        returned_seconds = dict(seconds, complete_file_api=time.perf_counter() - started)
        return dict(report, outputs=outputs, report_path=str(report_path),
                    completion_path=str(completion_path), maps_path=str(maps_path),
                    per_defect_path=str(per_defect_path), seconds=returned_seconds,
                    report_timing_boundary="complete_file_api_including_final_receipts",
                    report_sha256=report_sha256)
    except Exception as error:
        _write_json(directory / "failure.json", dict(status="failed", final_orig=bool(final_orig),
            error_type=type(error).__name__, error=str(error), seconds=seconds,
            parameters_requested=dict(distance_backend=distance_backend, distance_device=distance_device, state_backend=state_backend),
            actual_core_parameters_confirmed=False,
            complete_file_api_seconds=time.perf_counter() - started,
            native_program_called=False, whole_recon_all="not_assessed"))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="FNIT 自产文件的完整自有 GA，独立目录写出")
    for name in ("input_surface", "input_qsphere", "brain", "wm", "output_directory"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--max-unchanged", type=int, default=10)
    parser.add_argument("--niters", type=int, default=-1)
    parser.add_argument("--initial-selection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vertex-eliminate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--face-detection-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--blocker-scan-backend", choices=("python", "numba"), default="python")
    parser.add_argument("--distance-backend", choices=("serial", "parallel", "gpu_defect_signed_distance"), default="serial")
    parser.add_argument("--distance-device", default=None)
    parser.add_argument("--state-backend", choices=("legacy", "packed"), default="legacy")
    parser.add_argument("--normal-status", choices=("surface",), default="surface")
    parser.add_argument("--soap-iterations", type=int, default=5)
    parser.add_argument("--final-orig", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--remesh-scalar-storage", choices=("numpy", "python"), default="numpy")
    result = run_genetic_topology_files(**vars(parser.parse_args(argv)))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
