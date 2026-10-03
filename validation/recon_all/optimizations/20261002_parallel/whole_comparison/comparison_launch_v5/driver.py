"""复用现有 FNIT 比较器，汇总两例原始 T1 整例的三方诊断。

--config 指定 Python、比较源码/脚本、LUT 和每例 baseline/candidate 执行配置、
official 目录；--output 必须为新目录；--lock 为整例共用的本地 flock。
--wait 在锁外等所有 FNIT completion；--plan 只检查配置结构并输出调用计划。
所有数值工作在锁内串行运行，CPU 总线程4，CUDA不可见；不运行生产流程。
138项原门槛、局部精确差异、质量覆盖和相对官方的变化分列；整体等效未判定。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path: Path) -> dict:
    config = read(path)
    for key in ("python", "code_root", "scripts_dir", "label_table", "cases"):
        if key not in config:
            raise ValueError("missing config field: " + key)
    if config.get("threads", 4) != 4:
        raise ValueError("whole comparison uses the frozen four-thread budget")
    if not config["cases"] or len({row["id"] for row in config["cases"]}) != len(config["cases"]):
        raise ValueError("nonempty, unique case IDs required")
    for case in config["cases"]:
        if Path(case["id"]).name != case["id"] or case["id"] in (".", ".."):
            raise ValueError("case ID must be one path component")
        for key in ("baseline_config", "candidate_config", "official", "official_code_version"):
            if key not in case:
                raise ValueError("missing case field: " + key)
    return config


def ready(case: dict) -> bool:
    for kind in ("baseline", "candidate"):
        config = read(Path(case[kind + "_config"]))
        path = Path(config["diagnostic_root"]) / "completion.json"
        if not path.exists():
            return False
        completion = read(path)
        if completion.get("execution_status") != "complete" or completion.get("exit_code") != 0:
            raise RuntimeError(f"{case['id']} {kind} whole execution failed: {path}")
    return True


def validate_paths(config: dict) -> None:
    if Path(sys.executable).resolve() != Path(config["python"]).resolve():
        raise ValueError("start the driver with the declared homepage Conda Python")
    paths = [Path(config["label_table"]), Path(config["code_root"]) / "src/fnit/recon_all/compare_subject.py"]
    paths += [Path(config["scripts_dir"]) / (name + ".py") for name in
              ("compare_complete_subject", "compare_surface_chain", "compare_region_stats",
               "compare_parcellation_dice", "benchmark_surface_quality_extended", "plot_recon_all_comparison")]
    paths += [Path(config[name]) for name in ("data_sources", "official_provenance") if config.get(name)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("comparison inputs/scripts unavailable: " + ", ".join(missing))


@contextmanager
def common_lock(path: Path):
    # 使用与flock命令相同的独占锁；调用方不用再包一层同文件flock。
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def tool(name: str, scripts: Path):
    path = scripts / (name + ".py")
    spec = importlib.util.spec_from_file_location("_whole_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExactDistanceCache:
    """单例三方比较的精确数组缓存；不按路径/近似坐标复用距离。"""
    def __init__(self):
        self.entries = {}
        self.requests = self.hits = 0
        self.hash_seconds = self.search_seconds = 0.0

    def evaluate(self, operation, source, target, faces):
        self.requests += 1
        tick = time.perf_counter()
        key = tuple((tuple(array.shape), array.dtype.str,
                     hashlib.sha256(array.tobytes(order="C")).hexdigest())
                    for array in (source, target, faces))
        self.hash_seconds += time.perf_counter() - tick
        if key in self.entries:
            self.hits += 1
            return self.entries[key]
        tick = time.perf_counter()
        distances = operation(source, target, faces)
        self.search_seconds += time.perf_counter() - tick
        distances.setflags(write=False)
        self.entries[key] = distances
        return distances

    def bind(self, operation):
        return lambda source, target, faces: self.evaluate(operation, source, target, faces)

    def report(self):
        return {"requests": self.requests, "cache_hits": self.hits, "computed_searches": self.requests-self.hits,
                "array_hash_seconds": self.hash_seconds, "uncached_mesh_search_seconds": self.search_seconds,
                "cached_result_count": len(self.entries), "cached_outputs_readonly": True,
                "key": "source,target,faces each: shape, dtype.str, SHA256(full C-order bytes)",
                "scope": "single case; formula unchanged; comparison time includes hashing/search/cache hits; no whole-run timing reuse"}

    def clear(self):
        self.entries.clear()


def execute(command: list[str], log: Path, commands: list) -> None:
    tick = time.perf_counter()
    with log.open("a") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    commands.append({"command": command, "seconds": time.perf_counter() - tick,
                     "exit_code": result.returncode})
    write(log.parent / "commands.json", commands)
    if result.returncode:
        raise RuntimeError("comparison command failed; see " + str(log))


def receipts(case: dict) -> dict:
    result = {}
    configs = {kind: read(Path(case[kind + "_config"])) for kind in ("baseline", "candidate")}
    if configs["baseline"]["input"] != configs["candidate"]["input"]:
        raise ValueError("paired whole runs must declare the same original T1")
    for kind, config in configs.items():
        root = Path(config["output"])
        completion_path = Path(config["diagnostic_root"]) / "completion.json"
        launch_path = Path(config["diagnostic_root"]) / "launch.json"
        completion, launch, run = read(completion_path), read(launch_path), read(root / "fnit-native-free-run.json")
        if (completion.get("code_commit") != config["code_commit"] or
                completion.get("source_archive_sha256") != config["source_archive_sha256"] or
                completion.get("execution_status") != "complete" or completion.get("exit_code") != 0 or
                launch.get("code_commit") != config["code_commit"] or
                launch.get("source_archive_sha256") != config["source_archive_sha256"] or
                launch.get("config_sha256") != digest(Path(case[kind + "_config"])) or
                run.get("status") != "complete" or run.get("threads") != 4):
            raise ValueError(kind + " completion/config/source binding differs")
        validation = run.get("output_validation", {})
        if validation.get("expected") != 138 or validation.get("present") != 138 or validation.get("missing"):
            raise ValueError(kind + " whole output manifest is incomplete")
        source = Path(config["code_root"]) / "src/fnit/recon_all"
        if launch.get("candidate_native_free_sha256") != digest(source / "native_free.py"):
            raise ValueError(kind + " declared native_free source differs from launch")
        result[kind] = {"subject": str(root), "code_commit": config["code_commit"],
                        "source_archive_sha256": config["source_archive_sha256"],
                        "config_sha256": digest(Path(case[kind + "_config"])),
                        "completion_sha256": digest(completion_path), "launch_sha256": digest(launch_path),
                        "run_sha256": digest(root / "fnit-native-free-run.json"),
                        "original_t1_sha256": digest(Path(config["input"])),
                        "execution_status": completion["execution_status"], "output_validation": validation,
                        "pipeline_seconds": completion["pipeline_total_seconds"],
                        "entry_seconds": completion["command_seconds"], "mesh_validation": run.get("mesh_validation"),
                        "stages": run.get("stages"), "stage_timing_scope": run.get("timing"),
                        "generator_source_sha256": {name: digest(Path(config["code_root"]) / "src/fnit/recon_all" / name)
                                                    for name in ("surface_stats_cache.py", "anatomical_stats_file.py")}}
    result["official"] = {"subject": case["official"], "code_version": case["official_code_version"],
                          "timing_scope": "archived official run; not newly paired"}
    return result


def geometry(reference: Path, candidate: Path, surface_tool) -> dict:
    """复用拓扑检查，明确相同TkRAS框架；不重采样或修补网格。"""
    import nibabel as nib
    import nibabel.freesurfer.io as fs
    import numpy as np
    from fnit.recon_all.compare_subject import _topology

    images = [nib.load(str(root / "mri/orig.mgz")) for root in (reference, candidate)]
    conform = (images[0].shape == images[1].shape and
               np.allclose(images[0].affine, images[1].affine, atol=1e-6, rtol=0) and
               np.array_equal(images[0].header.get_vox2ras_tkr(), images[1].header.get_vox2ras_tkr()))
    # MGH 头的 shape 元素可能是 NumPy int32；报告保留整数值并转为 JSON 原生类型。
    result = {"conform_shape": [[int(size) for size in image.shape] for image in images],
              "conform_affine_max_abs_difference": float(np.max(np.abs(images[0].affine - images[1].affine))),
              "conform_scanner_affine_atol": 1e-6, "common_conform_and_tkras": bool(conform), "surfaces": {}}
    for hemi in ("lh", "rh"):
        whites = [fs.read_geometry(str(root / "surf" / f"{hemi}.white")) for root in (reference, candidate)]
        for stage in sorted(set(surface_tool.STAGES) | {"smoothwm", "inflated"}):
            key = f"{hemi}.{stage}"
            paths = [root / "surf" / key for root in (reference, candidate)]
            if not all(path.exists() for path in paths):
                result["surfaces"][key] = {"status": "missing", "vertex_correspondence": False}
                continue
            pairs = [fs.read_geometry(str(path), read_metadata=True) for path in paths]
            metadata = [pair[2] for pair in pairs]
            fields = set(metadata[0]) | set(metadata[1])
            footer_equal = all(np.array_equal(metadata[0].get(field), metadata[1].get(field))
                               for field in fields if field != "filename")
            valid = all(x.ndim == 2 and x.shape[1] == 3 and len(x) and np.isfinite(x).all()
                        and f.ndim == 2 and f.shape[1] == 3 and len(f) and
                        np.issubdtype(f.dtype, np.integer) and np.all((f >= 0) & (f < len(x)))
                        for x, f, _ in pairs)
            topologies = [_topology(x, f) for x, f, _ in pairs] if valid else [{"valid": False}] * 2
            for topology in topologies:
                # 拓扑判断中的 np.all 可能留下 np.bool_；仅规范报告类型。
                topology["valid"] = bool(topology["valid"])
            ordered = valid and pairs[0][0].shape == pairs[1][0].shape and np.array_equal(pairs[0][1], pairs[1][1])
            matches_white = valid and all(x.shape == w.shape and np.array_equal(f, wf)
                                         for (x, f, _), (w, wf) in zip(pairs, whites))
            frame = bool(conform and footer_equal)
            result["surfaces"][key] = {"status": "available" if valid else "invalid_arrays",
                "surface_ras_mm_comparable": frame, "geometry_footer_data_equal": bool(footer_equal),
                "geometry_footer_present": [bool(value) for value in metadata],
                "frame_basis": "conform scanner affine and vox2ras_tkr plus available surface volume_info; filename excluded",
                "ordered_faces_and_vertex_counts_equal": bool(ordered),
                "each_output_ordered_topology_matches_own_white": bool(matches_white),
                "topology": topologies, "vertex_correspondence": bool(frame and ordered and matches_white and all(t['valid'] for t in topologies))}
            if valid and stage in ("white", "pial"):
                import torch
                from fnit.recon_all.mris_register_nonlinear import face_area_normals
                result["surfaces"][key]["zero_area_face_counts"] = [int((face_area_normals(
                    torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(f.astype(np.int64)))[0] == 0).sum())
                    for x, f, _ in pairs]
    return result


def local_differences(reference: Path, candidate: Path, gate: dict) -> dict:
    import nibabel.freesurfer.io as fs
    import numpy as np
    from fnit.recon_all.compare_subject import _annotation, _numeric
    from fnit.recon_all.expected_outputs import ANNOTS, MORPHS

    report = {"purpose": "all-element exact differences; not additional acceptance tolerances",
              "maps": {}, "annotations": {}}
    anchors = {"thickness": ("white", "pial"), "area": ("white",), "area.pial": ("pial",),
               "area.mid": ("white", "pial"), "volume": ("white", "pial"), "curv": ("white",),
               "curv.pial": ("pial",), "sulc": ("inflated",), "avg_curv": ("sphere.reg",),
               "jacobian_white": ("white", "sphere.reg")}
    for hemi in ("lh", "rh"):
        for name in MORPHS:
            anchor = anchors.get(name, ("smoothwm" if name.startswith("smoothwm.") else
                                      "white.preaparc" if name.startswith("white.preaparc.") else "inflated",))
            key = f"surf/{hemi}.{name}"
            if not all(gate["surfaces"].get(f"{hemi}.{a}", {}).get("vertex_correspondence") for a in anchor):
                report["maps"][key] = {"status": "not_assessed_vertex_correspondence", "anchors": anchor}
                continue
            count = gate["surfaces"][f"{hemi}.{anchor[0]}"]["topology"][0]["vertices"]
            arrays = [fs.read_morph_data(str(root / key)) for root in (reference, candidate)]
            if any(array.shape != (count,) for array in arrays):
                report["maps"][key] = {"status": "invalid_vertex_count", "anchors": anchor}
                continue
            row = _numeric(*arrays, {"atol": 0.0, "rtol": 0.0}, 100)
            row.update(anchors=anchor, exact_differences_only=True,
                       volume_definition="TH3 vertex map; not no-th3 ROI GrayVol" if name == "volume" else None)
            report["maps"][key] = row
        for name in ANNOTS:
            key = f"label/{hemi}.{name}.annot"
            white = gate["surfaces"].get(f"{hemi}.white", {})
            if not white.get("vertex_correspondence"):
                report["annotations"][key] = {"status": "not_assessed_vertex_correspondence"}
            else:
                report["annotations"][key] = _annotation(reference / key, candidate / key,
                    white["topology"][0]["vertices"], 1.0, 100)
    return report


def no_th3_subject(subject: Path) -> dict:
    from fnit.recon_all.surface_stats_cache import SurfaceStatsCache
    result = {"definition": "existing FNIT no-th3 on each output's own white/pial/thickness/annotation",
              "device": "cpu", "unit": "mm3", "atlases": {}, "input_sha256": {}}
    with SurfaceStatsCache(device="cpu") as cache:
        for atlas in ("aparc", "aparc.DKTatlas", "aparc.a2009s"):
            rows = {}
            for hemi in ("lh", "rh"):
                paths = {name: subject / "surf" / f"{hemi}.{name}" for name in ("white", "pial", "thickness")}
                paths["annotation"] = subject / "label" / f"{hemi}.{atlas}.annot"
                values = cache.roi_volumes(paths["white"], paths["pial"], paths["thickness"], paths["annotation"])
                rows.update({f"{hemi}/{name}": {"GrayVolNoTH3": value} for name, value in values.items()})
                result["input_sha256"].update({str(path.relative_to(subject)): digest(path) for path in paths.values()})
            result["atlases"][atlas] = rows
    return result


def stats_modes(subject: Path) -> dict:
    from fnit.recon_all.compare_subject import _read_stats
    result = {}
    for hemi in ("lh", "rh"):
        for atlas in ("aparc", "aparc.DKTatlas", "aparc.a2009s", "aparc.pial"):
            path = subject / "stats" / f"{hemi}.{atlas}.stats"
            command = _read_stats(path)[3]
            modes = [flag for flag in shlex.split(command or "") if flag in ("-th3", "-no-th3")]
            result[path.name] = {"command": command, "explicit_volume_mode": modes[-1] if modes else "unspecified",
                                 "file_sha256": digest(path)}
    return result


def compare_pair(reference: Path, candidate: Path, label: str, commit: str,
                 config: dict, folder: Path, commands: list, no_th3: dict, distance_cache: ExactDistanceCache) -> dict:
    import nibabel.freesurfer.io as fs
    import numpy as np
    scripts = Path(config["scripts_dir"])
    strict_tool, surface_tool, region_tool = (tool(name, scripts) for name in
        ("compare_complete_subject", "compare_surface_chain", "compare_region_stats"))
    surface_tool._point_to_mesh = distance_cache.bind(surface_tool._point_to_mesh)
    strict = strict_tool.compare(reference, candidate)
    if strict["checked"] != 138:
        raise ValueError("strict comparator no longer checks the frozen 138 items")
    strict["interpretation"] = "existing file diagnostics; native morph/annotation indices require separate geometry gate"
    write(folder / f"strict_{label}.json", strict)
    gate = geometry(reference, candidate, surface_tool)
    write(folder / f"geometry_{label}.json", gate)
    for script, prefix in (("compare_region_stats", "region"), ("compare_parcellation_dice", "dice")):
        command = [config["python"], str(scripts / (script + ".py")), "--reference", str(reference),
                   "--candidate", str(candidate), "--code-commit", commit,
                   "--output", str(folder / f"{prefix}_{label}.json")]
        if prefix == "dice":
            command += ["--label-table", config["label_table"]]
        execute(command, folder / "commands.log", commands)
    available = [value for value in gate["surfaces"].values() if value["status"] != "missing"]
    if not all(value.get("surface_ras_mm_comparable") and value["status"] == "available" for value in available):
        surfaces = {"status": "not_assessed_invalid_geometry_or_space", "geometry_report": f"geometry_{label}.json"}
    else:
        surfaces = surface_tool.compare(reference, candidate)
        # 即便ordered faces相同，也给最终white/pial的双向顶点到三角面距离。
        for hemi in ("lh", "rh"):
            for stage in ("white", "pial"):
                pairs = [fs.read_geometry(str(root / "surf" / f"{hemi}.{stage}")) for root in (reference, candidate)]
                (a, fa), (b, fb) = pairs
                row = surfaces["stages"][hemi][stage]
                identical = gate["surfaces"][f"{hemi}.{stage}"]["vertex_correspondence"] and np.array_equal(a, b)
                if identical:
                    zero = {"mean_mm": 0.0, "p99_mm": 0.0, "max_mm": 0.0, "over_0_1_mm": 0}
                    row.update(candidate_to_reference_triangle=dict(zero), reference_to_candidate_triangle=dict(zero),
                               triangle_distance_basis="identical coordinates/ordered faces; closed connected topology; every vertex lies on mesh")
                elif "candidate_to_reference_triangle" not in row:
                    row["candidate_to_reference_triangle"] = surface_tool._summary(surface_tool._point_to_mesh(b, a, fa))
                    row["reference_to_candidate_triangle"] = surface_tool._summary(surface_tool._point_to_mesh(a, b, fb))
                    row["triangle_distance_basis"] = "all source vertices to full target triangle mesh in both directions"
        surfaces["comparator_sha256"] = digest(scripts / "compare_surface_chain.py")
        surfaces["scope"] = "vertex-sampled distances, not continuous Hausdorff; indexed values require geometry gate"
    write(folder / f"surface_{label}.json", surfaces)
    write(folder / f"local_{label}.json", local_differences(reference, candidate, gate))
    volumes = {atlas: region_tool._metric(no_th3[str(reference)]["atlases"][atlas],
               no_th3[str(candidate)]["atlases"][atlas], "GrayVolNoTH3") for atlas in no_th3[str(reference)]["atlases"]}
    write(folder / f"no_th3_{label}.json", {"unit": "mm3", "atlases": volumes,
           "reference_stats_modes": stats_modes(reference), "candidate_stats_modes": stats_modes(candidate),
           "scope": "common no-th3 algorithm on own output meshes; raw stats GrayVol is separately retained"})
    return {"checked": strict["checked"], "passed": strict["passed"], "failed_files": [k for k, v in strict["files"].items() if not v["pass"]]}


def observations(folder: Path, strict: dict) -> dict:
    """只汇总已产生的原始报告，不新增等效门槛或把变化自动判为退化。"""
    inherited = set(strict["baseline_vs_official"]["failed_files"])
    current = set(strict["candidate_vs_official"]["failed_files"])
    result = {"assessment": "not_assessed; no unified prospective degradation gates",
              "candidate_vs_baseline_failed_files": strict["candidate_vs_baseline"]["failed_files"],
              "inherited_strict_official_failures": sorted(inherited & current),
              "new_strict_official_failures": sorted(current - inherited),
              "resolved_strict_official_failures": sorted(inherited - current),
              "regional_absolute_error_changes": [], "dice_changes": [], "quality_count_changes": []}
    before, after = [read(folder / f"region_{label}_vs_official.json") for label in ("baseline", "candidate")]
    for group in ("aparc_68", "aseg", "wmparc"):
        fields = before[group] if group == "aparc_68" else {"Volume_mm3": before[group]}
        for metric, data in fields.items():
            other = after[group][metric] if group == "aparc_68" else after[group]
            for name in sorted(data["per_region"].keys() & other["per_region"].keys()):
                a, b = data["per_region"][name]["absolute_error"], other["per_region"][name]["absolute_error"]
                result["regional_absolute_error_changes"].append({"group": group, "metric": metric, "region": name,
                    "baseline_error": a, "candidate_error": b, "error_change": b-a, "difference_nonzero": b != a})
    before, after = [read(folder / f"no_th3_{label}_vs_official.json") for label in ("baseline", "candidate")]
    for atlas in before["atlases"].keys() & after["atlases"].keys():
        first, second = before["atlases"][atlas]["per_region"], after["atlases"][atlas]["per_region"]
        for name in sorted(first.keys() & second.keys()):
            a, b = first[name]["absolute_error"], second[name]["absolute_error"]
            result["regional_absolute_error_changes"].append({"group": atlas, "metric": "GrayVolNoTH3", "region": name,
                "baseline_error": a, "candidate_error": b, "error_change": b-a, "difference_nonzero": b != a})
    before, after = [read(folder / f"dice_{label}_vs_official.json") for label in ("baseline", "candidate")]
    for name in before["files"].keys() & after["files"].keys():
        a, b = before["files"][name]["per_label"], after["files"][name]["per_label"]
        for label in sorted(a.keys() | b.keys()):
            first, second = a.get(label, {}).get("dice"), b.get(label, {}).get("dice")
            result["dice_changes"].append({"file": name, "label": label, "baseline_dice": first, "candidate_dice": second,
                "candidate_minus_baseline": second-first if first is not None and second is not None else None})
    qualities = {kind: read(folder / f"quality_{kind}" / "report.json") for kind in ("baseline", "candidate")}
    binding = read(folder / "execution_binding.json")
    for hemi in ("lh", "rh"):
        pair_geometry = read(folder / "geometry_candidate_vs_baseline.json")
        for stage in ("white", "pial"):
            zero_area = pair_geometry["surfaces"].get(f"{hemi}.{stage}", {}).get("zero_area_face_counts")
            result["quality_count_changes"].append({"hemisphere": hemi, "section": "zero_area_faces", "metric": stage,
                "baseline": zero_area[0] if zero_area else None, "candidate": zero_area[1] if zero_area else None,
                "difference": zero_area[1]-zero_area[0] if zero_area else None, "coverage_complete": zero_area is not None})
            counts = [binding[kind].get("mesh_validation", {}).get(hemi, {}).get("intersecting_faces", {}).get(stage)
                      for kind in ("baseline", "candidate")]
            result["quality_count_changes"].append({"hemisphere": hemi, "section": "pipeline_self_intersections", "metric": stage,
                "baseline": counts[0], "candidate": counts[1], "difference": counts[1]-counts[0] if all(v is not None for v in counts) else None,
                "coverage_complete": all(v is not None for v in counts), "basis": "bound whole pipeline mesh_validation"})
        for section, fields in (("topology", ("boundary_edges", "nonmanifold_edges", "degenerate_index_faces", "duplicate_faces")),
                                ("vertex_links", ("noncycle_links", "isolated_vertices")),
                                ("white_pial_crossings", ("proper_transverse_pairs",))):
            a, b = (qualities[kind]["hemispheres"][hemi][section] for kind in ("baseline", "candidate"))
            for field in fields:
                first, second = a.get(field), b.get(field)
                complete = section != "white_pial_crossings" or (a.get("status") == b.get("status") == "complete")
                result["quality_count_changes"].append({"hemisphere": hemi, "section": section, "metric": field,
                    "baseline": first, "candidate": second, "difference": second-first if complete and first is not None and second is not None else None,
                    "coverage_complete": complete})
        for stage in ("sphere", "sphere.reg"):
            for metric in ("negative_faces", "zero_area_faces", "nonfinite_areas"):
                first, second = [qualities[kind]["hemispheres"][hemi]["sphere_orientation"][stage][metric] for kind in ("baseline", "candidate")]
                result["quality_count_changes"].append({"hemisphere": hemi, "section": stage, "metric": metric,
                                                       "baseline": first, "candidate": second, "difference": second-first, "coverage_complete": True})
    return result


@lru_cache(maxsize=1)
def configure_runtime() -> None:
    # interop只能在本进程开始数值工作前配置一次，第二例复用同一进程。
    import numba
    import torch
    numba.set_num_threads(4)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)


def compare_case(case: dict, config: dict, folder: Path) -> dict:
    configure_runtime()
    info = receipts(case)
    if config.get("official_provenance"):
        provenance_path = Path(config["official_provenance"])
        official = read(provenance_path)["subjects"][case["id"]]
        if official["subject"] != case["official"]:
            raise ValueError("archived official provenance subject differs")
        info["official"].update(archived_wall_seconds=official["whole_wall_seconds"],
            archived_metadata=official,
            provenance_sha256=digest(provenance_path), log_sha256=official["log_sha256"],
            done_sha256=official["done_sha256"], archived_stage_rows=official["command_wall_rows"],
            timing_restriction="archived reference, not current paired resources/speedup; sub02 nodecw10 differs from FNIT gpucw1; official bilateral openmp 4 may total 8 versus FNIT total 4; nested/concurrent command times are not additive")
    write(folder / "execution_binding.json", info)
    subjects = {kind: Path(info[kind]["subject"]) for kind in ("baseline", "candidate", "official")}
    derived = {str(subject): no_th3_subject(subject) for subject in subjects.values()}
    write(folder / "no_th3_inputs.json", derived)
    strict, commands = {}, []
    distance_cache = ExactDistanceCache()
    for label, reference, candidate in (("candidate_vs_baseline", "baseline", "candidate"),
                                       ("candidate_vs_official", "official", "candidate"),
                                       ("baseline_vs_official", "official", "baseline")):
        strict[label] = compare_pair(subjects[reference], subjects[candidate], label, info[candidate]["code_commit"],
                                     config, folder, commands, derived, distance_cache)
        write(folder / "commands.json", commands)
    write(folder / "distance_cache.json", {**distance_cache.report(), "released_after_pair_comparisons": True})
    distance_cache.clear()
    for kind, subject in subjects.items():
        execute([config["python"], str(Path(config["scripts_dir"]) / "benchmark_surface_quality_extended.py"),
            "--subject", str(subject), "--output", str(folder / f"quality_{kind}"),
            "--code-version", info[kind].get("code_commit", info[kind].get("code_version")),
            "--source-kind", "official" if kind == "official" else "fnit", "--threads", "4",
            "--cross-timeout-seconds", "180", "--max-bbox-pairs", "20000000"], folder / "commands.log", commands)
        write(folder / "commands.json", commands)
    changes = observations(folder, strict)
    write(folder / "changes.json", changes)
    execute([config["python"], str(Path(config["scripts_dir"]) / "plot_recon_all_comparison.py"),
        "--reference", str(subjects["official"]), "--candidate", str(subjects["candidate"]),
        "--region-report", str(folder / "region_candidate_vs_official.json"),
        "--dice-report", str(folder / "dice_candidate_vs_official.json"),
        "--output-dir", str(folder / "figures"), "--code-commit", info["candidate"]["code_commit"]],
        folder / "commands.log", commands)
    write(folder / "commands.json", commands)
    figures = [folder / "figures" / name for name in
               ("t1_surface_overlay.png", "region_errors.png", "local_region_boundary.png")]
    if not all(path.is_file() and path.stat().st_size for path in figures):
        raise RuntimeError("plot command did not produce all three required brain figures")
    write(folder / "figure_manifest.json", {"scope": "brain-only derivatives of public defaced CC0 examples; coordinator verified source license",
        "original_t1_sha256": info["candidate"]["original_t1_sha256"],
        "data_sources_sha256": digest(Path(config["data_sources"])) if config.get("data_sources") else None,
        "script_sha256": digest(Path(config["scripts_dir"]) / "plot_recon_all_comparison.py"),
        "figure_sha256": {path.name: digest(path) for path in (folder / "figures").glob("*.png")},
        "timing_scope": "comparison/plot time only; excluded from whole pipeline and entry time"})
    return {"case": case["id"], "execution_status": "complete", "strict_138": strict,
            "new_degradation": changes["assessment"], "overall_metric_equivalence": "not_assessed",
            "self_intersection_scope": "FNIT pipeline mesh_validation reused and bound by run SHA; official not assessed when absent",
            "quality_status": {kind: read(folder / f"quality_{kind}/report.json")["status"] for kind in subjects},
            "timings": {kind: {key: info[kind][key] for key in ("pipeline_seconds", "entry_seconds")} for kind in ("baseline", "candidate")},
            "report_sha256": {str(path.relative_to(folder)): digest(path) for path in folder.rglob("*.json")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.plan:
        print(json.dumps({"config": config, "output": str(args.output), "lock": str(args.lock),
                          "numeric_work": "not_run", "threads": 4, "overall_metric_equivalence": "not_assessed"}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    progress = {"status": "waiting", "started_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": digest(args.config), "driver_sha256": digest(Path(__file__)),
                "completed": [], "overall_metric_equivalence": "not_assessed"}
    try:
        validate_paths(config)
        while not all(ready(case) for case in config["cases"]):
            write(args.output / "progress.json", progress)
            if not args.wait:
                raise RuntimeError("whole runs incomplete; use --wait or run after all completions")
            time.sleep(15)
        env = {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(Path(config["code_root"]) / "src")}
        env.update({name: "4" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        env["NUMBA_CACHE_DIR"] = str(args.output / "numba_cache")
        os.environ.update(env)
        sys.path.insert(0, str(Path(config["code_root"]) / "src"))
        progress["environment"] = env
        progress["comparator_sha256"] = {path.name: digest(path) for path in Path(config["scripts_dir"]).glob("*.py")}
        for case in config["cases"]:
            progress["status"] = "waiting_for_common_lock_" + case["id"]
            write(args.output / "progress.json", progress)
            with common_lock(args.lock):
                if "fnit_comparator_source_sha256" not in progress:
                    source = Path(config["code_root"]) / "src/fnit/recon_all"
                    progress["fnit_comparator_source_sha256"] = {str(path.relative_to(source)): digest(path)
                        for path in source.rglob("*.py")}
                progress["status"] = "comparing_" + case["id"]
                write(args.output / "progress.json", progress)
                folder = args.output / case["id"]
                folder.mkdir()
                tick = time.perf_counter()
                summary = compare_case(case, config, folder)
                summary["comparison_seconds_excluding_lock_wait"] = time.perf_counter() - tick
                write(folder / "summary.json", summary)
                progress["completed"].append(case["id"])
        progress["status"] = "complete"
    except BaseException as error:
        progress.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        progress["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.output / "progress.json", progress)


if __name__ == "__main__":
    main()
