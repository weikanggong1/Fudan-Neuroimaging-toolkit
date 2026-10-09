"""诊断失败生产运行保留下来的完整文件；绝不把失败整例改成成功。

只接受在 mni_mesh_parallel 网格门失败的运行。先绑定原始 T1、实际源码、
失败收据及现存138输出，再复用现有严格比较、Dice、脑区统计、no-th3、
双向点到三角面距离和扩展质量函数。诊断完成与生产成功是两个字段。
不执行重建、修补网格、复制参考或建立新的等效阈值。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import time


def sha256(path: Path) -> str:
    """读取文件计算SHA-256；缺文件或读错误直接传播，不改变输入。"""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_module(*, path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_failed_producer(*, benchmark_path: Path, source_root: Path,
                             case: str, reference_manifest: dict) -> tuple[dict, dict, dict]:
    """绑定失败收据、原始T1和全部声明源码，返回benchmark/runtime/失败网格门。

    benchmark_path为原始整例JSON，source_root为实际冻结源码根；case必须在
    官方输入清单中。均必填。只接受execution=failed、非零整数退出码、
    runtime.status=failed且failed_stage=mni_mesh_parallel。SHA、状态或几何
    输入变化抛ValueError；文件/JSON错误直接传播。此内部校验无官方CLI。
    """
    benchmark = json.loads(benchmark_path.read_text())
    subject = benchmark_path.parent / "subject"
    runtime = json.loads((subject / "fnit-native-free-run.json").read_text())
    exit_code = benchmark.get("exit_code")
    if (benchmark.get("execution") != "failed" or type(exit_code) is not int
            or exit_code == 0 or runtime.get("status") != "failed"):
        raise ValueError("requires preserved failed producer, not complete/running execution")
    if runtime.get("failed_stage") != "mni_mesh_parallel":
        raise ValueError("diagnostic supports preserved mni_mesh_parallel failure only")
    expected_input = reference_manifest["cases"][case]["input_sha256"]
    if benchmark.get("input_sha256") != expected_input:
        raise ValueError("candidate and reference original T1 hashes differ")
    if sha256(Path(runtime["input"])) != expected_input:
        raise ValueError("runtime original T1 differs from benchmark binding")
    sources = benchmark.get("source_sha256", {})
    if not sources or "recon_all/native_free.py" not in sources:
        raise ValueError("generator source binding is missing")
    for relative, expected in sources.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid generator relative path")
        if sha256(source_root / "src/fnit" / path) != expected:
            raise ValueError("generator source changed: " + relative)
    mesh = json.loads((subject / "scripts/mni-mesh-parallel.json").read_text())
    if mesh.get("status") != "failed" or mesh.get("mesh_validation", {}).get("status") != "failed":
        raise ValueError("preserved failed mesh gate is missing")
    bindings = mesh.get("mesh_inputs_sha256", {})
    expected_meshes = {f"{hemi}.{name}" for hemi in ("lh", "rh")
                       for name in ("orig", "white", "pial", "sphere.reg")}
    if not expected_meshes.issubset(bindings):
        raise ValueError("failed mesh input binding is incomplete")
    for relative, expected in bindings.items():
        path = Path(relative)
        if path.is_absolute() or len(path.parts) != 1 or relative in (".", ".."):
            raise ValueError("invalid preserved surface relative path")
        if sha256(subject / "surf" / path) != expected:
            raise ValueError("preserved failed mesh changed: " + relative)
    return benchmark, runtime, mesh


def inventory_outputs(*, subject: Path, expected_paths: tuple[str, ...]) -> dict:
    """只读清点固定138项；输出存在性不代表生产门通过。

    subject为保留被试目录，expected_paths来自实际冻结expected_outputs.paths。
    返回expected/present/missing和逐文件bytes/SHA，单位bytes；非法、重复或
    非138项清单抛ValueError。缺输出被记录，后续数值比较将停止。
    """
    if len(expected_paths) != 138 or len(set(expected_paths)) != 138:
        raise ValueError("unchanged fixed 138 output manifest required")
    files, missing = {}, []
    for relative in expected_paths:
        component = Path(relative)
        if component.is_absolute() or ".." in component.parts:
            raise ValueError("invalid output relative path")
        path = subject / component
        if not path.is_file():
            missing.append(relative)
        else:
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return {"profile": "single-t1-138", "expected": 138, "present": len(files),
            "missing": missing, "files": files,
            "interpretation": "preserved file existence only; producer execution and mesh gate remain failed"}


def require_same_resources(*, control: dict, candidate: dict, threads: int) -> None:
    """核对两自产运行身份与预算，包含权重/资产/程序；不比较跨环境官方时间。"""
    for key in ("input_sha256", "host", "cpu", "cpu_affinity", "threads", "device",
                "resource_sha256", "native_program_sha256", "source_sha256"):
        if key not in control or control[key] != candidate.get(key):
            raise ValueError("producer/control identity differs: " + key)
    if control["threads"] != threads:
        raise ValueError("comparison and producer thread budgets differ")
    if control.get("environment", {}).get("CUDA_VISIBLE_DEVICES") != candidate.get("environment", {}).get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("producer/control GPU visibility differs")


def failed_output_quality(*, subject: Path, output: Path, code_version: str,
                          mesh_failure: dict, quality_module, write) -> dict:
    """复用既有质量算子测失败目录；标签明确为failed-producer，不改原CLI门。

    输入surface RAS毫米网格、cortex索引和aparc注释；输出拓扑/连通性、
    sphere翻折和white/pial穿越JSON。阈值、180秒/半球和2000万候选预算
    沿用当前质量脚本；超限保留部分覆盖。自相交来自带输入SHA的实际
    失败生产门，与white/pial相互穿越分别报告。成功返回报告dict，缺文件
    或格式错误抛异常。本诊断无独立官方等价命令。
    """
    import importlib.metadata
    import nibabel.freesurfer.io as fs
    import numpy as np
    import torch

    output.mkdir(parents=True, exist_ok=False)
    q = quality_module
    report = {"status": "measuring", "subject": str(subject), "code_version": code_version,
              "host": socket.gethostname(), "threads": 4, "device": "cpu",
              "source_kind": "failed_fnit_preserved_output_diagnosis",
              "producer_execution_complete": False,
              "producer_failed_mesh_validation": mesh_failure["mesh_validation"],
              "candidate_run_sha256": sha256(subject / "fnit-native-free-run.json"),
              "failed_mesh_receipt_sha256": sha256(subject / "scripts/mni-mesh-parallel.json"),
              "quality_script_sha256": sha256(Path(q.__file__)),
              "wrapper_script_sha256": sha256(Path(__file__)),
              "coordinate_space": "surface RAS", "coordinate_unit": "mm",
              "cuda_initialized": torch.cuda.is_initialized(),
              "semantic_checks": q.semantic_checks(),
              "parameters": {"bbox_guard_mm": 1e-5, "normalized_plane_tolerance_mm": 1e-6,
                             "intersection_line_overlap_tolerance_mm": 1e-6,
                             "sphere_negative_area_threshold_mm2": 0,
                             "cross_timeout_seconds_per_hemisphere": 180,
                             "maximum_bbox_pairs_per_hemisphere": 20_000_000},
              "library_versions": {name: importlib.metadata.version(name) for name in
                                   ("torch", "numpy", "scipy", "numba", "nibabel")},
              "source_sha256": {}, "hemispheres": {}}
    for function in (q._topology, q.face_area_normals, q.triangles_intersect,
                     q.vertex_links, q.transverse_crossings):
        module = __import__(function.__module__, fromlist=["__file__"])
        report["source_sha256"][function.__module__] = sha256(Path(module.__file__))
    for hemi in ("lh", "rh"):
        tick = time.perf_counter()
        paths = {name: subject / "surf" / f"{hemi}.{name}"
                 for name in ("orig", "white", "pial", "sphere", "sphere.reg")}
        meshes = {name: fs.read_geometry(str(path)) for name, path in paths.items()}
        vertices, faces = meshes["orig"]
        identical = all(v.shape == vertices.shape and np.array_equal(f, faces)
                        for v, f in meshes.values())
        row = {"input_sha256": {name: sha256(path) for name, path in paths.items()},
               "ordered_faces_and_vertex_counts_preserved": identical,
               "topology": q._topology(vertices, faces),
               "vertex_links": q.vertex_links(faces, len(vertices)),
               "failed_pipeline_mesh_validation": mesh_failure["mesh_validation"].get(hemi),
               "sphere_orientation": {}}
        for name in ("sphere", "sphere.reg"):
            xyz, triangles = meshes[name]
            area, _ = q.face_area_normals(torch.as_tensor(xyz, dtype=torch.float32),
                                         torch.as_tensor(triangles.astype(np.int64)), signed_sphere=True)
            area = area.numpy()
            row["sphere_orientation"][name] = {
                "negative_faces": int((area < 0).sum()), "zero_area_faces": int((area == 0).sum()),
                "nonfinite_areas": int((~np.isfinite(area)).sum()),
                "minimum_signed_area_mm2": float(area.min()),
                "negative_face_ids_first_100": np.flatnonzero(area < 0)[:100].tolist()}
        report["hemispheres"][hemi] = row
        write(output / "report.json", report)
        label_path = subject / "label" / f"{hemi}.cortex.label"
        cortex = np.zeros(len(vertices), dtype=bool)
        cortex[fs.read_label(str(label_path))] = True
        annotation = subject / "label" / f"{hemi}.aparc.annot"
        labels, _, names = fs.read_annot(str(annotation))
        names = np.asarray([name.decode("utf-8") for name in names])
        regions = np.asarray([names[index] if index >= 0 else "unlabeled" for index in labels])
        row["input_sha256"].update(cortex_label=sha256(label_path), aparc_annotation=sha256(annotation))
        row["white_pial_crossings"] = (q.transverse_crossings(
            meshes["white"][0], meshes["pial"][0], faces, cortex, 4, 180, 20_000_000, regions)
            if identical else {"status": "not_assessed_topology_mismatch"})
        row["seconds_including_io"] = time.perf_counter() - tick
        write(output / "report.json", report)
    report.update(status="measured" if all(row["white_pial_crossings"]["status"] == "complete"
                    for row in report["hemispheres"].values()) else "partially_measured",
                  cuda_initialized_after=torch.cuda.is_initialized(),
                  scope="read-only failed-output quality diagnosis; production failure preserved; no overall equivalence judgement")
    write(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "source-root", "control-benchmark", "control-source-root", "complete-validator",
                 "reference-root", "driver", "scripts-dir", "label-table", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--reference-kind", choices=("control", "official"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("frozen diagnostic thread budget is four")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.reference_root / "TRANSFER_MANIFEST.private.json"
    manifest = json.loads(manifest_path.read_text())
    candidate, runtime, failure = validate_failed_producer(
        benchmark_path=args.benchmark, source_root=args.source_root,
        case=args.case, reference_manifest=manifest)
    complete = load_module(path=args.complete_validator, name="_complete_run_validator")
    control, control_runtime = complete.validate_candidate(
        benchmark_path=args.control_benchmark, source_root=args.control_source_root,
        case=args.case, reference_manifest=manifest)
    require_same_resources(control=control, candidate=candidate, threads=args.threads)
    driver = load_module(path=args.driver, name="_failed_whole_driver")
    driver.configure_runtime()
    expected = load_module(path=args.source_root / "src/fnit/recon_all/expected_outputs.py",
                           name="_failed_frozen_manifest")
    subject = args.benchmark.parent / "subject"
    inventory = inventory_outputs(subject=subject, expected_paths=expected.paths())
    reference = (args.control_benchmark.parent / "subject" if args.reference_kind == "control"
                 else args.reference_root / "subjects" / args.case)
    args.output.mkdir(parents=True)
    state = {"status": "running", "case": args.case,
             "scope": "failed producer preserved-output diagnosis; not a successful recon-all evaluation",
             "producer_execution": "failed", "producer_execution_complete": False,
             "producer_exit_code": candidate["exit_code"], "failed_stage": runtime["failed_stage"],
             "producer_error": runtime.get("error"), "output_inventory": inventory,
             "producer_output_validation": runtime.get("output_validation"),
             "producer_failed_mesh_validation": failure["mesh_validation"],
             "benchmark_sha256": sha256(args.benchmark),
             "run_sha256": sha256(subject / "fnit-native-free-run.json"),
             "failed_mesh_receipt_sha256": sha256(subject / "scripts/mni-mesh-parallel.json"),
             "control_benchmark_sha256": sha256(args.control_benchmark),
             "control_run_sha256": sha256(args.control_benchmark.parent / "subject/fnit-native-free-run.json"),
             "input_sha256": candidate["input_sha256"], "tested_code_version": candidate["code_version"],
             "generator_source_sha256": candidate["source_sha256"],
             "reference_manifest_sha256": sha256(manifest_path),
             "reference": manifest["cases"][args.case], "reference_kind": args.reference_kind,
             "label_table_sha256": sha256(args.label_table),
             "comparator_sha256": {str(path): sha256(path) for path in
                (Path(__file__), Path(complete.__file__), args.driver, *args.scripts_dir.glob("*.py"))},
             "producer_hardware": {key: candidate[key] for key in ("host", "cpu", "cpu_affinity", "threads", "device")},
             "cuda_visible_devices": candidate.get("environment", {}).get("CUDA_VISIBLE_DEVICES"),
             "producer_elapsed_until_failure_seconds": candidate["cli_wall_seconds"],
             "successful_whole_speedup": None,
             "process_memory": {key: value for key, value in candidate.get("process_memory", {}).items() if key != "samples"},
             "strict_reproduction": "not_assessed", "optimization_regression": "production_mesh_gate_failed",
             "overall_metric_equivalence": "not_assessed_no_confirmed_prospective_thresholds"}
    driver.write(args.output / "diagnosis.json", state)
    driver.write(args.output / "preserved_failed_mesh_receipt.json", failure)
    cache, commands, started = driver.ExactDistanceCache(), [], time.perf_counter()
    try:
        if inventory["missing"]:
            state.update(status="diagnostic_incomplete_preserved_outputs_missing")
            return 2
        no_th3 = {str(path): driver.no_th3_subject(path) for path in (reference, subject)}
        driver.write(args.output / "no_th3_inputs.json", no_th3)
        config = {"python": sys.executable, "scripts_dir": str(args.scripts_dir), "label_table": str(args.label_table)}
        label = "candidate_vs_" + args.reference_kind
        strict = driver.compare_pair(reference, subject, label, candidate["code_version"],
                                     config, args.output, commands, no_th3, cache)
        state["strict_reproduction"] = strict
        driver.write(args.output / "distance_cache.json", cache.report())
        cache.clear()
        quality = load_module(path=args.scripts_dir / "benchmark_surface_quality_extended.py",
                              name="_failed_existing_quality")
        measured = failed_output_quality(subject=subject, output=args.output / "quality_candidate",
                                        code_version=candidate["code_version"], mesh_failure=failure,
                                        quality_module=quality, write=driver.write)
        # Reference remains a genuinely complete FNIT run or archived official; existing gate is unchanged.
        driver.execute([sys.executable, str(args.scripts_dir / "benchmark_surface_quality_extended.py"),
            "--subject", str(reference), "--output", str(args.output / "quality_reference"),
            "--code-version", control["code_version"] if args.reference_kind == "control" else manifest["cases"][args.case]["official_version"],
            "--source-kind", "fnit" if args.reference_kind == "control" else "official", "--threads", "4",
            "--cross-timeout-seconds", "180", "--max-bbox-pairs", "20000000"], args.output / "commands.log", commands)
        driver.execute([sys.executable, str(args.scripts_dir / "plot_recon_all_comparison.py"),
            "--reference", str(reference), "--candidate", str(subject),
            "--region-report", str(args.output / f"region_{label}.json"),
            "--dice-report", str(args.output / f"dice_{label}.json"),
            "--output-dir", str(args.output / "figures"), "--code-commit", candidate["code_version"]],
            args.output / "commands.log", commands)
        if inventory_outputs(subject=subject, expected_paths=expected.paths()) != inventory:
            raise RuntimeError("preserved producer outputs changed during diagnosis")
        for path, original in ((args.benchmark, state["benchmark_sha256"]),
                (subject / "fnit-native-free-run.json", state["run_sha256"]),
                (subject / "scripts/mni-mesh-parallel.json", state["failed_mesh_receipt_sha256"])):
            if sha256(path) != original:
                raise RuntimeError("preserved failed producer receipt changed during diagnosis")
        state.update(status="diagnostic_complete_producer_still_failed", inputs_unchanged=True,
                     quality_candidate=measured["status"],
                     quality_reference=driver.read(args.output / "quality_reference/report.json")["status"])
        return 0
    except BaseException as error:
        state.update(status="diagnostic_failed_producer_still_failed", diagnostic_error=repr(error))
        raise
    finally:
        cache.clear()
        state["diagnostic_wall_seconds"] = time.perf_counter() - started
        state["timing_scope"] = "post-failure diagnosis/figures only; not recon-all runtime or successful speedup"
        driver.write(args.output / "diagnosis.json", state)


if __name__ == "__main__":
    raise SystemExit(main())
