"""复用现有整例比较器，评价一次已完成的原始T1运行；不运行生产。

候选benchmark、冻结代码、既有官方参考清单、比较脚本目录和新输出目录
均须显式指定。严格138诊断、几何对应、Dice、逐脑区与no-th3统计、
双向点到三角面、质量覆盖和脑图分别保存；整体等效不设新阈值。
官方参考输入绑定必须与候选原始T1 SHA一致。对照跨环境/历史时间单列。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

from recon_evaluation_checks import (all_partition_stats, independent_self_intersections,
                                     validate_reference_manifest)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_candidate_output_files(*, subject: Path, runtime: dict) -> dict:
    """现场核验固定138个自产输出的普通文件、路径与完整字节摘要。

    subject 是同次 benchmark 的被试根目录；runtime 是生产 JSON 报告。
    输出按相对路径给出字节数与 SHA-256，空间仍由各文件本身定义；不读取
    官方参考，也不改变原报告。缺失、符号链接、报告路径错配或读时改变
    抛 ValueError。原软件对应 recon-all 完成后的输出核查，无独立命令。
    """
    from fnit.recon_all.expected_outputs import paths as expected_paths

    expected = expected_paths()
    if len(expected) != 138 or len(set(expected)) != len(expected):
        raise ValueError("candidate fixed output list is no longer unique 138")
    if subject.is_symlink() or not subject.is_dir():
        raise ValueError("candidate subject must be a real directory")
    root = subject.resolve(strict=True)
    recorded = runtime.get("outputs")
    if not isinstance(recorded, dict) or set(recorded) != set(expected):
        raise ValueError("candidate runtime output paths do not match fixed 138 manifest")
    files = {}
    for relative in expected:
        path = subject
        for component in Path(relative).parts:
            path = path / component
            if path.is_symlink():
                raise ValueError("candidate output contains symbolic link: " + relative)
        if not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("candidate output missing or outside subject: " + relative)
        declared = recorded[relative]
        if not isinstance(declared, str) or Path(declared).resolve() != path.resolve():
            raise ValueError("candidate runtime output path differs: " + relative)
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("candidate output is not a regular file: " + relative)
        digest = sha(path)
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("candidate output changed during validation: " + relative)
        files[relative] = {"bytes": after.st_size, "sha256": digest}
    return {"status": "verified", "expected": 138, "verified": len(files),
            "files": files, "symbolic_links_allowed": False,
            "scope": "current complete candidate outputs, independently rehashed before comparison"}


def validate_candidate_cli_paths(*, report: dict, runtime: dict,
                                 benchmark_path: Path) -> None:
    """将 benchmark 的实际子命令原始T1/输出根与自产运行报告绑定。

    benchmark_path 是整例报告位置；report/runtime 来自本次整例 JSON。
    不修改影像或坐标；命令结构、输入路径和被试目录不一致时抛 ValueError。
    这是 FNIT benchmark 包装层检查，没有独立原软件命令。
    """
    command = report.get("cli_command")
    if (not isinstance(command, list) or len(command) < 8 or
            command[1:6] != ["-X", "faulthandler", "-u", "-m", "fnit.recon_all.native_free"] or
            not all(isinstance(value, str) for value in command[:8])):
        raise ValueError("candidate benchmark lacks the actual FNIT CLI command")
    subject = benchmark_path.parent / "subject"
    if (not isinstance(runtime.get("input"), str) or
            Path(command[6]).resolve(strict=True) != Path(runtime["input"]).resolve(strict=True)):
        raise ValueError("candidate CLI original T1 differs from runtime input")
    if (not isinstance(runtime.get("subject_dir"), str) or
            Path(command[7]).resolve() != subject.resolve() or
            Path(runtime["subject_dir"]).resolve() != subject.resolve()):
        raise ValueError("candidate CLI subject differs from benchmark/runtime subject")


def validate_mixed_topology_program(*, report: dict, runtime: dict,
                                    expected_sha256: str) -> dict:
    """校验混合整例唯一原生程序的固定哈希与实际拓扑后端。

    report/runtime 为同一次原始 T1 整例的 benchmark 与生产报告；
    expected_sha256 是评估前独立冻结的 Conda 安装程序 SHA-256。返回
    程序路径、哈希和配置摘要，不运行程序、不改变数值比较。缺失或不一致
    抛 ValueError；实际源码构建记录由独立安装预检保存。原软件内部命令
    对应 mris_fix_topology -ga -seed 1234。
    """
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("mixed topology requires a previously fixed lowercase SHA-256")
    programs = report.get("native_program_sha256")
    if programs != {"mris_fix_topology_fnit": expected_sha256}:
        raise ValueError("mixed topology benchmark must declare only the fixed topology program")
    if report.get("backend") != "native" or report.get("topology_backend") != "native":
        raise ValueError("mixed topology requires the native topology route")
    own_backends = {
        "n4_backend": "simpleitk", "n4_execution": "in-process",
        "wm_backend": "torch-optimized", "wm_execution": "isolated",
        "wm_edit_backend": "python", "defects_backend": "torch",
        "inflate_backend": "torch", "sphere_finish_backend": "torch",
        "surface_placement_backend": "python",
        "normalization_controls_backend": "torch",
        "normalization_initial_bias_backend": "torch",
        "gca_inverse_backend": "torch", "gca_execution": "isolated",
        "fill_backend": "numba", "annotation_gibbs_backend": "numba",
        "remesh_scalar_storage": "python", "mni_execution": "parallel-late",
        "topology_distance_backend": "serial", "topology_state_backend": "legacy",
    }
    changed = [name for name, value in own_backends.items() if report.get(name) != value]
    if changed:
        raise ValueError("mixed topology changed the own implementation recipe: " + ", ".join(changed))
    if report.get("hemisphere_execution") != "groups":
        raise ValueError("mixed topology requires the validated groups scheduler")
    command = report.get("cli_command")
    if (not isinstance(command, list) or len(command) < 8 or
            command[1:6] != ["-X", "faulthandler", "-u", "-m", "fnit.recon_all.native_free"]):
        raise ValueError("mixed topology benchmark lacks the actual FNIT CLI command")

    def option(name: str) -> str:
        if command.count(name) != 1:
            raise ValueError("mixed topology requires one CLI " + name)
        index = command.index(name)
        if index + 1 >= len(command) or not isinstance(command[index + 1], str):
            raise ValueError("mixed topology CLI option has no value: " + name)
        return command[index + 1]

    if option("--native-optimizations") != "torch":
        raise ValueError("mixed topology must keep the own Torch GCA")
    if option("--backend") != "native" or option("--topology-backend") != "native":
        raise ValueError("mixed topology CLI and benchmark backend differ")
    native_directory = Path(option("--native-bin-dir"))
    if not native_directory.is_dir():
        raise ValueError("mixed topology program directory is missing")
    if {path.name for path in native_directory.iterdir()} != {"mris_fix_topology_fnit"}:
        raise ValueError("mixed topology directory contains an unexpected program")
    program = native_directory / "mris_fix_topology_fnit"
    if not program.is_symlink() or not program.is_file() or not os.access(program, os.X_OK):
        raise ValueError("mixed topology requires the verified Conda program link")
    conda_bin = Path(command[0]).resolve().parent
    if program.resolve().parent != conda_bin:
        raise ValueError("mixed topology program is outside the benchmark Conda environment")
    if sha(program) != expected_sha256:
        raise ValueError("mixed topology installed program SHA-256 changed")
    topology = runtime.get("topology_repair")
    if (runtime.get("backend") != "native" or not isinstance(topology, dict) or
            topology.get("backend") != "native" or topology.get("sha256") != expected_sha256 or
            Path(topology.get("binary") or "").resolve() != program.resolve()):
        raise ValueError("mixed topology runtime did not bind the fixed native topology program")
    selection = runtime.get("native_optimizations")
    if (runtime.get("n4_binary") is not None or not isinstance(selection, dict) or
            selection.get("em_backend") != "torch"):
        raise ValueError("mixed topology runtime selected another original stage program")
    scheduling = runtime.get("hemisphere_scheduling")
    if (not isinstance(scheduling, dict) or
            scheduling.get("mode") != "independent-exec-private-subjects" or
            scheduling.get("workers") != 2 or
            scheduling.get("total_thread_budget") != 4 or
            scheduling.get("execution") != "groups"):
        raise ValueError("mixed topology completed runtime used another hemisphere scheduler")
    groups = scheduling.get("groups")
    operations = ("surface", "register", "annotation", "final_white", "pial_surface")
    if (not isinstance(groups, list) or len(groups) != len(operations) or
            any(not isinstance(group, dict) or group.get("operation") != operation or
                group.get("status") != "complete"
                for group, operation in zip(groups, operations))):
        raise ValueError("mixed topology lacks the five completed hemisphere groups")
    surfaces = runtime.get("surfaces")
    for hemisphere in ("lh", "rh"):
        row = surfaces.get(hemisphere) if isinstance(surfaces, dict) else None
        seconds = row.get("topology_native_seconds") if isinstance(row, dict) else None
        if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or
                not math.isfinite(seconds) or seconds <= 0 or row.get("topology_backend") != "native"):
            raise ValueError("mixed topology lacks a completed native surface stage: " + hemisphere)
    return {"binary": str(program.resolve()), "sha256": expected_sha256,
            "allowed_original_stage_programs": ["mris_fix_topology_fnit"],
            "source_build_provenance": "not_checked_by_this_evaluator; retain the independent Conda build receipt"}


def validate_native_program_manifest(*, path: Path | None,
                                     expected_sha256: str | None) -> dict[str, str]:
    """读取评估前冻结的 sha256sum 清单，返回程序名到 SHA-256 的映射。

    path 为 Conda 独立源码构建生成的普通文本清单；expected_sha256 为其
    必填小写64位摘要。清单路径可为绝对或相对路径，只按 basename 与实际
    benchmark 程序目录绑定，不按清单路径执行或读取程序。重复名、畸形行、
    缺失摘要或读时改变抛 ValueError。没有坐标/单位或独立原软件命令。
    清单自身证明身份，源码构建来源仍须保留独立安装回执。
    """
    if (path is None or not isinstance(expected_sha256, str) or
            re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None):
        raise ValueError("native surfaces require a previously fixed program manifest SHA-256")
    if path.is_symlink() or not path.is_file():
        raise ValueError("native program manifest must be a plain file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("native program manifest SHA-256 differs")
    programs = {}
    for line in raw.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if match is None or "\x00" in match[2] or "\\" in match[2]:
            raise ValueError("invalid sha256sum native program manifest line")
        name = Path(match[2]).name
        if not name or name in {".", ".."} or name in programs:
            raise ValueError("duplicate or invalid native program manifest name")
        programs[name] = match[1]
    if not programs or sha(path) != expected_sha256:
        raise ValueError("native program manifest empty or changed during validation")
    return programs


def validate_native_surface_programs(*, report: dict, runtime: dict, subject: Path,
                                     expected_programs: dict[str, str] | None) -> dict:
    """绑定显式原生表面配方及已完成阶段证据；不改变比较阈值。

    report/runtime 来自同一原始 T1 整例，subject 为该次真实输出根；
    expected_programs 是事先固定的独立构建清单（程序名→SHA-256）。逐一
    重算声明程序摘要，并校验实际选择、双侧四组、white/pial完成报告和
    preclean→remesh3→postclean→orig发布的内容链。返回安装清单与源码
    绑定阶段证据，后者不等于操作系统 exec 审计或干净安装验收。
    网格为 surface RAS/mm；参数/时间/摘要/结构门不一致即抛 ValueError，
    不修改输出；对应固定的原生表面命令，实际源码来源由安装回执证明。
    """
    required = {"mris_fix_topology_fnit", "mris_remesh", "mris_remove_intersection",
                "mris_inflate", "mris_place_surface", "mrisp_paint", "mris_curvature_stats"}
    if (not isinstance(expected_programs, dict) or not required.issubset(expected_programs)
            or any(not isinstance(name, str) or Path(name).name != name or
                   not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for name, value in expected_programs.items())):
        raise ValueError("native surfaces require the complete fixed program manifest")
    if report.get("native_program_sha256") != expected_programs:
        raise ValueError("native surfaces declared installed programs differ from fixed manifest")
    recipe = {
        "backend": "native", "threads": 4, "topology_backend": "native",
        "topology_finish_backend": "conda-cpp", "surface_placement_backend": "native",
        "n4_backend": "simpleitk", "n4_execution": "in-process",
        "wm_backend": "torch-optimized", "wm_execution": "isolated", "wm_edit_backend": "python",
        "defects_backend": "torch", "inflate_backend": "torch", "sphere_finish_backend": "torch",
        "annotation_gibbs_backend": "numba", "gca_inverse_backend": "torch",
        "gca_candidate_chunk": 4096, "gca_reuse_atlas_context": True, "gca_execution": "isolated",
        "fill_backend": "numba", "normalization_controls_backend": "torch",
        "normalization_initial_bias_backend": "torch", "mni_execution": "parallel-late",
        "hemisphere_execution": "groups", "topology_distance_backend": "serial",
        "topology_state_backend": "legacy", "remesh_scalar_storage": "numpy"}
    changed = [name for name, value in recipe.items() if report.get(name) != value]
    if changed:
        raise ValueError("native surfaces changed the fixed recipe: " + ", ".join(changed))
    command = report.get("cli_command", [])
    def option(name):
        if command.count(name) != 1 or command.index(name) + 1 >= len(command):
            raise ValueError("native surfaces require one CLI " + name)
        return command[command.index(name) + 1]
    for name, value in recipe.items():
        if name == "gca_reuse_atlas_context":
            if command.count("--gca-reuse-atlas-context") != 1:
                raise ValueError("native surfaces lack atlas context reuse CLI")
        else:
            if option("--" + name.replace("_", "-")) != str(value):
                raise ValueError("native surfaces CLI differs from declared recipe: " + name)
    if option("--native-optimizations") != "torch" or option("--hemisphere-workers") != "2":
        raise ValueError("native surfaces require own Torch GCA and two hemisphere workers")
    if (command.count("--profile-stages") != 1 or
            not isinstance(report.get("device"), str) or
            re.fullmatch(r"cuda:[0-9]+", report["device"]) is None or
            option("--device") != report["device"]):
        raise ValueError("native surfaces require profiled stages and an explicit CUDA device")
    directory = Path(option("--native-bin-dir")).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("native program directory is missing")
    for name, digest in expected_programs.items():
        program = directory / name
        if not program.is_file() or not os.access(program, os.X_OK) or sha(program) != digest:
            raise ValueError("native installed program missing or SHA-256 changed: " + name)
    def selected(row, name, *, digest_key="sha256", path_key="binary"):
        if (not isinstance(row, dict) or row.get(digest_key) != expected_programs[name]
                or Path(row.get(path_key) or "").resolve() != (directory / name).resolve()):
            raise ValueError("native runtime program selection differs: " + name)
    selection = runtime.get("native_optimizations", {})
    if (runtime.get("backend") != "native" or runtime.get("n4_binary") is not None
            or selection.get("mode") != "torch" or selection.get("em_backend") != "torch"):
        raise ValueError("native surfaces runtime selected another volume program")
    white = Path(selection.get("white_binary") or "").name
    if white not in {"mris_place_surface", "mris_place_surface_white_fast"} or white not in expected_programs:
        raise ValueError("native surfaces selected an undeclared white program")
    selected(runtime.get("topology_repair"), "mris_fix_topology_fnit")
    selected(runtime.get("topology_repair"), "mris_remove_intersection",
             digest_key="intersection_sha256", path_key="intersection_binary")
    selected(runtime.get("remesh"), "mris_remesh")
    selected(runtime.get("white_preaparc"), white)
    selected(runtime.get("surface_metrics"), "mris_place_surface")
    selected(runtime.get("sphere_generation", {}).get("mris_inflate"), "mris_inflate")
    extra = runtime.get("extra_curvature", {})
    if (extra.get("mrisp_paint_sha256") != expected_programs["mrisp_paint"] or
            extra.get("mris_curvature_stats_sha256") != expected_programs["mris_curvature_stats"] or
            extra.get("paint_backend") != "native" or extra.get("statistics_backend") != "native"):
        raise ValueError("native curvature program selection differs")
    scheduling = runtime.get("hemisphere_scheduling", {})
    groups = scheduling.get("groups")
    operations = ("surface", "register", "annotation", "finish_surface")
    if (scheduling.get("mode") != "independent-exec-private-subjects" or
            scheduling.get("workers") != 2 or scheduling.get("total_thread_budget") != 4 or
            scheduling.get("execution") != "groups" or not isinstance(groups, list) or
            len(groups) != len(operations) or any(not isinstance(group, dict) or group.get("operation") != operation or
                group.get("status") != "complete" for group, operation in zip(groups, operations))):
        raise ValueError("native surfaces lack the four completed hemisphere groups")
    if runtime.get("mesh_validation", {}).get("status") != "passed":
        raise ValueError("native surfaces lack the completed production mesh gate")
    evidence = {}
    def positive(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    def completed_step(rows, name):
        found = [value for value in rows if isinstance(value, dict) and value.get("name") == name]
        return len(found) == 1 and positive(found[0].get("seconds")) and not found[0].get("error")
    for hemi in ("lh", "rh"):
        row = runtime.get("surfaces", {}).get(hemi, {})
        if (row.get("topology_backend") != "native" or row.get("topology_finish_backend") != "conda-cpp"
                or not all(positive(row.get(key)) for key in
                           ("topology_native_seconds", "topology_remesh_seconds", "topology_intersection_seconds"))
                or not positive(row.get("white_preaparc_report", {}).get("place_seconds"))
                or not positive(row.get("final_white_report", {}).get("seconds"))
                or row.get("pial_report", {}).get("implementation") != "conda-source-cpp"
                or not positive(row.get("pial_report", {}).get("seconds"))
                or not positive(row.get("sphere_seconds", {}).get("inflate_nofix"))):
            raise ValueError("native surfaces lack completed native placement/topology: " + hemi)
        paint_steps = groups[1].get("values", {}).get(hemi, {}).get("stages", [])
        if (not completed_step(paint_steps, "avg_curv_" + hemi) or
                not completed_step(runtime.get("stages", []), "stats_" + hemi + "_curv")):
            raise ValueError("native surfaces lack completed paint/curvature steps: " + hemi)
        receipts = {}
        for stage in ("preclean", "remesh", "intersection", "topology-finish"):
            path = subject / "scripts" / f"surface.{hemi}.{hemi}.native-{stage}.json"
            if path.is_symlink() or not path.is_file():
                raise ValueError("native surface stage receipt missing: " + str(path))
            receipt = json.loads(path.read_text())
            if receipt.get("status") != "complete":
                raise ValueError("native surface stage receipt incomplete: " + stage)
            receipts[stage] = receipt
        pre, remesh, post, finish = (receipts[name] for name in
                                    ("preclean", "remesh", "intersection", "topology-finish"))
        selected(pre, "mris_remove_intersection", digest_key="binary_sha256")
        selected(remesh, "mris_remesh", digest_key="binary_sha256")
        selected(post, "mris_remove_intersection", digest_key="binary_sha256")
        if (pre.get("input_sha256") != sha(subject / "surf" / f"{hemi}.orig.premesh") or
                pre.get("candidate_sha256") != remesh.get("input_sha256") or
                remesh.get("candidate_sha256") != post.get("input_sha256") or
                post.get("candidate_sha256") != finish.get("output_sha256") or
                finish.get("output_sha256") != sha(subject / "surf" / f"{hemi}.orig") or
                finish.get("input_sha256") != pre.get("input_sha256") or
                finish.get("published_to_orig") is not True or
                remesh.get("iterations") != 3 or post.get("mesh_structure", {}).get("status") != "passed" or
                post.get("source_intersecting_faces_after_independent_recheck") != 0 or
                post.get("geometric_intersecting_faces_after_independent_recheck") != 0):
            raise ValueError("native surface receipt content chain or strict mesh gate differs: " + hemi)
        evidence[hemi] = {"native_placement_seconds": {
            "white_preaparc": row["white_preaparc_report"]["place_seconds"],
            "final_white": row["final_white_report"]["seconds"], "pial": row["pial_report"]["seconds"]},
            "receipt_sha256": {stage: sha(subject / "scripts" / f"surface.{hemi}.{hemi}.native-{stage}.json")
                               for stage in receipts}, "orig_sha256": finish["output_sha256"]}
    return {"status": "verified_source_bound_stage_evidence", "declared_installed_programs": expected_programs,
            "selected_white_program": white, "hemispheres": evidence,
            "scope": "selected runtime bindings and completed source-bound stages; installed inventory is not execution proof",
            "os_exec_audit": "not_assessed", "source_build_provenance": "retain independent Conda build receipt"}


def validate_candidate(*, benchmark_path: Path, source_root: Path,
                       case: str, reference_manifest: dict,
                       candidate_profile: str = "python-ga",
                       expected_topology_sha256: str | None = None,
                       expected_native_programs: dict[str, str] | None = None) -> tuple[dict, dict]:
    """绑定实际完成收据/原始输入/模块哈希，返回benchmark与运行报告。

    benchmark_path为整例脚本JSON；source_root为实际冻结仓库根目录；case
    必须在reference_manifest.cases中。candidate_profile默认python-ga，
    mixed-topology-cpp仅允许一个固定GA程序，须给expected_topology_sha256；
    mixed-native-surfaces要求expected_native_programs固定完整程序映射，
    按显式原生表面配方验证。未选的程序参数默认None；摘要均为SHA-256。
    只读校验，不创建输出或修改输入。失败/不完整/源码变化/输入错配抛
    ValueError，文件和JSON错误传播。内部验证步骤没有独立官方CLI。
    """
    report = json.loads(benchmark_path.read_text())
    runtime = json.loads((benchmark_path.parent / "subject/fnit-native-free-run.json").read_text())
    if candidate_profile != "mixed-native-surfaces" and expected_native_programs is not None:
        raise ValueError("native program manifest is only accepted for mixed-native-surfaces")
    if report.get("execution") != "complete" or report.get("exit_code") != 0:
        raise ValueError("candidate raw T1 execution did not complete successfully")
    validate_candidate_cli_paths(report=report, runtime=runtime,
                                 benchmark_path=benchmark_path)
    if candidate_profile == "python-ga":
        if expected_topology_sha256 is not None:
            raise ValueError("Python GA evaluation does not take a topology program SHA-256")
        if (report.get("backend") != "python" or report.get("topology_backend") != "python-ga"
                or report.get("native_program_sha256") != {} or report.get("threads") != 4):
            raise ValueError("candidate is not the frozen four-thread Python GA profile")
        audit = report.get("execution_audit")
        if (not isinstance(audit, dict) or audit.get("status") != "complete"
                or audit.get("passed") is not True or audit.get("original_program_attempts")):
            raise ValueError("candidate Python execution audit is incomplete or blocked a program")
    elif candidate_profile == "mixed-topology-cpp":
        if report.get("threads") != 4:
            raise ValueError("mixed topology requires the frozen four-thread budget")
        report["mixed_topology_program_validation"] = validate_mixed_topology_program(
            report=report, runtime=runtime, expected_sha256=expected_topology_sha256)
    elif candidate_profile == "mixed-native-surfaces":
        if expected_topology_sha256 is not None:
            raise ValueError("native surfaces use the complete program manifest")
        report["native_surface_program_validation"] = validate_native_surface_programs(
            report=report, runtime=runtime, subject=benchmark_path.parent / "subject",
            expected_programs=expected_native_programs)
    else:
        raise ValueError("unknown candidate_profile")
    if (report.get("runtime_status") != "complete" or
            any(not isinstance(report.get(name), (int, float)) or
                isinstance(report.get(name), bool) or
                not math.isfinite(report[name]) or report[name] <= 0
                for name in ("cli_wall_seconds", "full_harness_seconds"))):
        raise ValueError("candidate complete runtime and positive whole-case wall times required")
    if runtime.get("status") != "complete":
        raise ValueError("candidate output/production mesh validation failed")
    complete = runtime.get("output_validation", {})
    if (complete.get("status") != "passed" or complete.get("expected") != 138
            or complete.get("present") != 138 or complete.get("missing")):
        raise ValueError("candidate fixed 138 output manifest is incomplete")
    if report.get("input_sha256") != reference_manifest["cases"][case]["input_sha256"]:
        raise ValueError("candidate and official original T1 hashes differ")
    if sha(Path(runtime["input"])) != report["input_sha256"]:
        raise ValueError("runtime original T1 differs from benchmark binding")
    sources = report.get("source_sha256", {})
    if not sources or "recon_all/native_free.py" not in sources:
        raise ValueError("generator source binding is missing")
    for relative, expected in sources.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid generator relative path")
        if sha(source_root / "src/fnit" / path) != expected:
            raise ValueError("generator source changed: " + relative)
    report["candidate_output_binding"] = validate_candidate_output_files(
        subject=benchmark_path.parent / "subject", runtime=runtime)
    return report, runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reference-manifest-sha256",
                        help="可选预先冻结的参考清单SHA-256；给出时必须一致")
    parser.add_argument("--case", required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, required=True)
    parser.add_argument("--label-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--candidate-profile", choices=("python-ga", "mixed-topology-cpp", "mixed-native-surfaces"),
                        default="python-ga", help="纯Python GA、单原生GA或显式原生表面配方；各模式分别绑定")
    parser.add_argument("--expected-topology-sha256",
                        help="混合模式必填：从独立 Conda 构建预检记录的拓扑程序 SHA-256")
    parser.add_argument("--expected-native-programs-manifest", type=Path,
                        help="原生表面模式必填：独立构建的 sha256sum 程序清单")
    parser.add_argument("--expected-native-programs-manifest-sha256",
                        help="原生表面模式必填：评估前冻结的程序清单 SHA-256")
    args = parser.parse_args()
    if (args.candidate_profile in {"mixed-topology-cpp", "mixed-native-surfaces"} and
            args.reference_manifest_sha256 is None):
        raise ValueError("mixed topology requires a previously fixed reference manifest SHA-256")
    expected_native_programs = None
    if args.candidate_profile == "mixed-native-surfaces":
        expected_native_programs = validate_native_program_manifest(
            path=args.expected_native_programs_manifest,
            expected_sha256=args.expected_native_programs_manifest_sha256)
    elif (args.expected_native_programs_manifest is not None or
          args.expected_native_programs_manifest_sha256 is not None):
        raise ValueError("program manifest is only accepted for mixed-native-surfaces")
    if args.threads != 4:
        raise ValueError("existing frozen comparison uses four threads")
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    # 只读取此根目录内的清单文件；来源文字source从不作为数据入口。
    from fnit.recon_all.expected_outputs import paths as expected_paths
    required = (*expected_paths(), "label/lh.cortex.label", "label/rh.cortex.label")
    manifest, reference_binding = validate_reference_manifest(
        reference_root=args.reference_root, case=args.case,
        required_paths=required,
        expected_manifest_sha256=args.reference_manifest_sha256)
    # 可选前期表面若存在，也必须在已核验清单中；不扩展固定138清单。
    for hemi in ("lh", "rh"):
        for stage in ("orig.nofix", "orig.premesh"):
            relative = "subjects/" + args.case + "/surf/" + hemi + "." + stage
            path = args.reference_root / relative
            if path.is_symlink() or (path.exists() and relative not in manifest["files"]):
                raise ValueError("optional reference surface is not a declared plain file")
    benchmark, runtime = validate_candidate(benchmark_path=args.benchmark,
        source_root=args.source_root, case=args.case, reference_manifest=manifest,
        candidate_profile=args.candidate_profile,
        expected_topology_sha256=args.expected_topology_sha256,
        expected_native_programs=expected_native_programs)
    reference = args.reference_root / "subjects" / args.case
    subject = args.benchmark.parent / "subject"
    spec = importlib.util.spec_from_file_location("_existing_whole_comparison", args.driver)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.configure_runtime()
    args.output.mkdir(parents=True)
    save = lambda name, data: driver.write(args.output / name, data)
    state = {"scope": "one complete raw T1 candidate versus archived official; not optimized/control pairing",
        "status": "running", "case": args.case, "execution_complete": True,
        "output_validation": runtime["output_validation"], "production_mesh_validation": runtime["mesh_validation"],
        "candidate_output_binding": benchmark["candidate_output_binding"],
        "benchmark_sha256": sha(args.benchmark),
        "run_sha256": sha(subject / "fnit-native-free-run.json"),
        "tested_code_version": benchmark["code_version"],
        "candidate_profile": args.candidate_profile,
        "mixed_topology_program_validation": benchmark.get("mixed_topology_program_validation"),
        "native_surface_program_validation": benchmark.get("native_surface_program_validation"),
        "native_program_manifest_sha256": (args.expected_native_programs_manifest_sha256
                                            if expected_native_programs is not None else None),
        "generator_source_sha256": benchmark["source_sha256"],
        "reference_manifest_sha256": reference_binding["manifest_sha256"],
        "reference_byte_validation": reference_binding,
        "reference": manifest["cases"][args.case],
        "comparator_sha256": {str(p): sha(p) for p in [Path(__file__),
            Path(__file__).with_name("recon_evaluation_checks.py"), args.driver, *args.scripts_dir.glob("*.py")]},
        "cli_wall_seconds": benchmark["cli_wall_seconds"],
        "full_harness_seconds": benchmark["full_harness_seconds"],
        "process_memory": {k:v for k,v in benchmark.get("process_memory",{}).items() if k!="samples"},
        "strict_reproduction": "not_assessed", "optimization_regression": "not_assessed_waiting_same_case_control",
        "overall_metric_equivalence": "not_assessed_no_confirmed_prospective_thresholds"}
    save("evaluation.json", state)
    commands, cache = [], driver.ExactDistanceCache()
    try:
        no_th3 = {str(p): driver.no_th3_subject(p) for p in (reference, subject)}
        save("no_th3_inputs.json", no_th3)
        config = {"python": sys.executable, "scripts_dir": str(args.scripts_dir), "label_table": str(args.label_table)}
        strict = driver.compare_pair(reference, subject, "candidate_vs_official", benchmark["code_version"],
            config, args.output, commands, no_th3, cache)
        state["strict_reproduction"] = strict
        region_tool = driver.tool("compare_region_stats", args.scripts_dir)
        save("all_regions_candidate_vs_official.json", all_partition_stats(
            reference=reference, candidate=subject, region_tool=region_tool))
        # 官方无FNIT生产报告；独立重算四张表面，不能把缺失报告解释成零。
        self_intersections = independent_self_intersections(subject=reference, threads=args.threads)
        save("self_intersections_official.json", self_intersections)
        state["independent_official_self_intersections"] = {
            "report_sha256": sha(args.output / "self_intersections_official.json"),
            "quality_status": self_intersections["quality_status"],
            "seconds": self_intersections["seconds"]}
        save("distance_cache.json", cache.report())
        cache.clear()
        for role, path in (("candidate", subject), ("official", reference)):
            driver.execute([sys.executable, str(args.scripts_dir / "benchmark_surface_quality_extended.py"),
                "--subject", str(path), "--output", str(args.output / ("quality_"+role)),
                "--code-version", benchmark["code_version"] if role=="candidate" else manifest["cases"][args.case]["official_version"],
                "--source-kind", "fnit" if role=="candidate" else "official", "--threads", "4",
                "--cross-timeout-seconds", "180", "--max-bbox-pairs", "20000000"], args.output / "commands.log", commands)
        driver.execute([sys.executable, str(args.scripts_dir / "plot_recon_all_comparison.py"),
            "--reference", str(reference), "--candidate", str(subject),
            "--region-report", str(args.output / "region_candidate_vs_official.json"),
            "--dice-report", str(args.output / "dice_candidate_vs_official.json"),
            "--output-dir", str(args.output / "figures"), "--code-commit", benchmark["code_version"]],
            args.output / "commands.log", commands)
        state.update(status="complete", quality={role: driver.read(args.output/("quality_"+role)/"report.json")["status"]
                    for role in ("candidate","official")})
        return 0
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        cache.clear()
        state["comparison_wall_seconds"] = time.perf_counter() - started
        state["timing_scope"] = "post-run reference/source validation, diagnostics and figures; excluded from recon-all runtime; excludes initial imports and final evaluation report publication"
        save("evaluation.json", state)


if __name__ == "__main__":
    sys.exit(main())
