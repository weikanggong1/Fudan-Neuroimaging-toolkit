"""只读比较明确指定的 SimpleITK/Torch 原始 T1→filled 运行。

不将运行目录的名称改写为串行/并行；分别验证两套实际源码、收据及
完整屏障，再复用仓库的体积、标签、LTA、CSV比较函数。不同版本和
跨时段耗时单列，不能作为受控加速或完整 recon-all 等效结论。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import time


VOLUMES = ("rawavg.mgz", "orig.mgz", "tmp/nu0.mgz", "nu.mgz", "T1.mgz",
    "brainmask.mgz", "synthseg.rca.mgz", "norm.mgz", "brain.mgz",
    "aseg.presurf.mgz", "wm.seg.mgz", "wm.asegedit.mgz", "wm.mgz", "filled.mgz")
LABEL_VOLUMES = {"synthseg.rca.mgz", "aseg.presurf.mgz", "filled.mgz"}
N4_KEYS = {"n4_backend", "n4_execution"}

# 实际 serial→filled 的直接调用入口；模块SHA同时覆盖其内部辅助函数。
# 这份静态调用核对不冒充 OS 文件访问/动态库/完整执行追踪。
STAGE_ENTRIES = {
    "input_talairach": ("recon_all/input_talairach_chain.py", "run_input_talairach_chain"),
    "n4": ("recon_all/input_n4_chain.py", "run_n4_stage"),
    "nu": ("recon_all/n4_wrapper.py", "make_nu"),
    "T1_normalize": ("recon_all/normalization/pipeline.py", "normalize_t1"),
    "brainmask": ("recon_all/mri_mask_gpu.py", "mask_volume"),
    "SynthSeg": ("recon_all/native_free.py", "_run_recon_all_python.run_synthseg_and_write"),
    "mri_em_register": ("recon_all/native_free.py", "_run_native_em_register"),
    "mri_ca_normalize": ("recon_all/ca_normalize_python.py", "run_ca_normalize"),
    "mri_cc": ("recon_all/native_free.py", "_segment_callosum"),
    "brain_second_normalize": ("recon_all/normalization/aseg_pipeline.py", "normalize_t1_aseg"),
    "entowm": ("recon_all/sclimbic.py", "mri_entowm_seg"),
    "ants_denoise": ("recon_all/ants_denoise_python.py", "denoise_volume"),
    "mri_segment": ("recon_all/native_free.py", "_run_torch_wm_segment"),
    "mri_edit_wm_with_aseg": ("recon_all/native_free.py", "_run_python_wm_edit"),
    "wm_pretess": ("recon_all/pretess_python.py", "pretess_mgh"),
    "wm_fix_ento": ("recon_all/wm_edits_gpu.py", "fix_ento_wm_gpu"),
    "wm_fix_acj": ("recon_all/wm_edits_gpu.py", "fix_ento_wm_gpu"),
    "mri_fill": ("recon_all/fill_cutting_plane_python.py", "fill_mgz"),
}
PREFIX_SUPPORT_MODULES = (
    "recon_all/input_chain.py", "recon_all/conform_gpu.py",
    "recon_all/talairach_synthmorph.py", "recon_all/normalization/normalize_3d_controls.py",
    "recon_all/normalization/normalize_neighbor_sum_torch.py",
    "recon_all/normalization/normalize_gaussian_source.py",
    "recon_all/normalization/normalize_voronoi_source.py",
    "recon_all/normalization/normalize_aseg_ridge.py",
    "synthseg_parc/synthseg.py", "synthseg_parc/segment.py",
    "synthseg_parc/precision.py", "synthstrip/pipeline.py",
    "recon_all/mri_em_register_python.py", "recon_all/gca_torch_worker.py",
    "recon_all/mri_segment.py", "recon_all/wm_torch_worker.py",
    "recon_all/edit_wm_aseg_core_python.py", "recon_all/mri_cc_python.py",
    "recon_all/n4_simpleitk.py", "recon_all/n4_itk_torch_experimental.py",
    "recon_all/n4_torch_worker.py", "recon_all/thread_budget.py",
    "recon_all/profiling.py", "recon_all/python_profile.py",
)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    return json.loads(data), data


def write(path: Path, value: dict) -> None:
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(data)
    temporary.replace(path)


def verify_manifest(root: Path, manifest: Path) -> dict:
    """校验每个冻结文件；非法相对路径/文件缺失或SHA变化直接失败。"""
    value, data = read(manifest)
    files = value["files"]
    if not files or not isinstance(files, dict):
        raise ValueError("empty frozen source manifest")
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or sha(root / path) != expected:
            raise ValueError("frozen source mismatch: " + relative)
    return {"sha256": hashlib.sha256(data).hexdigest(), "files": files,
            "metadata": {k: v for k, v in value.items() if k != "files"}}


def function_sha(path: Path, qualified_name: str) -> str:
    source = path.read_text()
    body = ast.parse(source).body
    for part in qualified_name.split("."):
        matches = [node for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   and node.name == part]
        if len(matches) != 1:
            raise ValueError("function not uniquely found: " + str(path) + ":" + qualified_name)
        selected = matches[0]
        body = selected.body
    segment = ast.get_source_segment(source, selected)
    if segment is None:
        raise ValueError("cannot capture function source")
    return hashlib.sha256(segment.encode()).hexdigest()


def profile_pair(reference: dict, candidate: dict) -> dict:
    """21键配方必须只有两个N4键变化；19个其他键不允许偷换。"""
    if len(reference) != 21 or set(reference) != set(candidate) or not N4_KEYS <= set(reference):
        raise ValueError("expected same 21 dispatch keys")
    changed = {key: [reference[key], candidate[key]] for key in reference
               if reference[key] != candidate[key]}
    if set(changed) != N4_KEYS:
        raise ValueError("exactly the two N4 dispatch keys must change")
    if [reference["n4_backend"], candidate["n4_backend"]] != ["simpleitk", "torch"]:
        raise ValueError("expected SimpleITK reference and Torch candidate")
    if [reference["n4_execution"], candidate["n4_execution"]] != ["in-process", "isolated"]:
        raise ValueError("expected in-process reference and isolated candidate")
    return {"equal_non_n4_keys": sorted(set(reference) - N4_KEYS), "equal_non_n4_key_count": 19,
            "changed_n4_keys": changed, "dispatch_keys_are_not_stage_count": True}


def verify_receipts(*, producer: dict, runtime: dict, runtime_bytes: bytes,
                    complete: dict, record: dict, case: str, raw_sha: str,
                    manifest_sha: str, backend: str, execution: str) -> None:
    """验证真实完成门/输入/有效后端；目录名字不充当算法配置证据。"""
    if (complete.get("status") != "complete_prefixes" or complete.get("case") != case
            or complete.get("full_workflow_complete") is not False or record.get("exit_code") != 0
            or producer.get("status") != "complete_prefix" or producer.get("case") != case
            or producer.get("full_workflow_complete") is not False
            or runtime.get("status") != "complete_prefix" or runtime.get("full_workflow_complete") is not False
            or runtime.get("backend") != "python" or producer.get("input_execution") != "serial"
            or runtime.get("input_execution") != "serial" or record.get("execution") != "serial"
            or producer.get("input_sha256") != raw_sha or producer.get("final_input_sha256") != raw_sha
            or producer.get("source_manifest_sha256") != manifest_sha
            or producer.get("runtime_sha256") != hashlib.sha256(runtime_bytes).hexdigest()
            or producer.get("threads") != 4 or runtime.get("threads") != 4):
        raise ValueError("invalid completion/input/source/budget: " + case)
    config = runtime["n4_configuration"]
    profile = producer["fixed_profile"]
    if (config.get("backend") != backend or config.get("execution") != execution
            or profile.get("n4_backend") != backend or profile.get("n4_execution") != execution):
        raise ValueError("requested/effective N4 configuration mismatch")
    names = [row["name"] for row in runtime["stages"]]
    if names != list(STAGE_ENTRIES) + ["filled_auto_checkpoint"]:
        raise ValueError("expected every complete serial prefix stage in source order")
    transform = producer["diagnostic_AST_transformation"]
    if (transform.get("insertion_after") != "filled_auto_checkpoint"
            or transform.get("only_added_statements") != 3
            or transform.get("original_statements_removed") != 0
            or transform.get("production_source_files_modified") is not False):
        raise ValueError("diagnostic prefix must preserve all production statements")
    audit = record["execution_audit"]
    if audit.get("blocked_events") != 0 or audit.get("helper_sha256") != producer["execution_audit_helper_sha256"]:
        raise ValueError("invalid original-program audit")
    allowed_requests = {"simpleitk", "native"} if backend == "simpleitk" else {"torch"}
    if "requested_n4_backend" in record and record["requested_n4_backend"] not in allowed_requests:
        raise ValueError("process requested N4 contradicts effective backend")
    if "n4_execution" in record and record["n4_execution"] != execution:
        raise ValueError("process N4 execution contradicts effective backend")


def prepare_role(role: dict, *, case: str, raw: Path, manifests: dict) -> dict:
    exit_path = Path(role["driver_exit_file"])
    exit_bytes = exit_path.read_bytes()
    if exit_bytes.strip() != b"0":
        raise ValueError("driver must have actual exit-code 0: " + case)
    trial = Path(role["trial_dir"])
    root, manifest_path = Path(role["source_root"]), Path(role["source_manifest"])
    key = (str(root.resolve()), str(manifest_path.resolve()))
    if key not in manifests:
        manifests[key] = verify_manifest(root, manifest_path)
    manifest = manifests[key]
    complete_path = Path(role["complete_file"])
    complete, complete_bytes = read(complete_path)
    producer, producer_bytes = read(trial / "benchmark.json")
    runtime_path = trial / "subject/fnit-native-free-run.json"
    runtime, runtime_bytes = read(runtime_path)
    index = int(trial.name.split("-", 1)[0])
    records = [record for record in complete["runs"] if record["order_index"] == index]
    if len(records) != 1:
        raise ValueError("trial does not select one actual COMPLETE run")
    record = records[0]
    command = record["command"]
    if command[command.index("--output") + 1] != str(trial):
        raise ValueError("COMPLETE record belongs to another trial")
    verify_receipts(producer=producer, runtime=runtime, runtime_bytes=runtime_bytes,
        complete=complete, record=record, case=case, raw_sha=sha(raw), manifest_sha=manifest["sha256"],
        backend=role["backend"], execution=role["n4_execution"])
    files = manifest["files"]
    if producer["diagnostic_AST_transformation"]["module_sha256"] != files["src/fnit/recon_all/native_free.py"]:
        raise ValueError("executed pipeline differs from its source manifest")
    if producer["script_sha256"] != sha(root / "tools/benchmark_recon_input_prefix.py"):
        raise ValueError("executed harness differs from frozen source")
    if complete["script_sha256"] != producer["script_sha256"]:
        raise ValueError("parent and child harness differ")
    n4 = runtime["n4_runtime"]
    if role["backend"] == "torch":
        if n4["api"]["iterations"] != 200 or n4["api"]["device"] != producer["device"]:
            raise ValueError("Torch N4 must execute all 200 iterations on the declared GPU")
        for key, relative in (("worker", "src/fnit/recon_all/n4_torch_worker.py"),
            ("full_n4", "src/fnit/recon_all/n4_itk_torch_experimental.py"),
            ("bspline_fit", "src/fnit/recon_all/n4_bspline_torch.py")):
            if n4["source_sha256"][key] != files[relative]:
                raise ValueError("N4 worker source differs from manifest: " + key)
        if n4["input_sha256"] != sha(trial / "subject/mri/orig.mgz") or n4["output_sha256"] != sha(trial / "subject/mri/tmp/nu0.mgz"):
            raise ValueError("N4 worker input/output differs from completed subject")
    elif (n4["backend"] != "simpleitk_python" or n4["used_native_programs"] is not False
          or n4["recipe"]["iterations"] != [50, 50, 50, 50]
          or n4["recipe"]["convergence_threshold"] != 0.0):
        raise ValueError("SimpleITK must retain its complete fixed recipe")
    entries = {}
    for stage, (module, name) in STAGE_ENTRIES.items():
        relative = "src/fnit/" + module
        entries[stage] = {"source_module": relative, "module_sha256": files[relative],
                          "qualified_name": name, "function_source_sha256": function_sha(root / relative, name)}
    entries["filled_auto_checkpoint"] = {"entry": "Python stdlib shutil.copyfile", "meaning": "source-semantic filled→filled.auto copy"}
    supports = {"src/fnit/" + name: files["src/fnit/" + name] for name in PREFIX_SUPPORT_MODULES}
    captures = [(exit_path, exit_bytes), (complete_path, complete_bytes), (trial/"benchmark.json", producer_bytes),
                (runtime_path, runtime_bytes), (manifest_path, manifest_path.read_bytes())]
    return dict(trial=trial, subject=trial/"subject", source_root=root, manifest=manifest,
                producer=producer, runtime=runtime, process=record, entries=entries,
                support_modules=supports, captures=captures)


def metadata_pair(first: dict, second: dict) -> dict:
    """输入/资源/预算相同；源码与跨时段环境差异记录、不伪装配对门。"""
    a, b = first["producer"], second["producer"]
    for key in ("input_sha256", "threads", "device", "CPU_affinity", "CUDA_VISIBLE_DEVICES", "hostname", "resource_sha256"):
        if a[key] != b[key]:
            raise ValueError("two roles differ in fixed input/resource/hardware/budget: " + key)
    recipe = profile_pair(a["fixed_profile"], b["fixed_profile"])
    files = [value["manifest"]["files"] for value in (first, second)]
    all_names = set(files[0]) | set(files[1])
    changes = {name: [files[0].get(name), files[1].get(name)] for name in sorted(all_names)
               if files[0].get(name) != files[1].get(name)}
    return {"recipe": recipe, "all_frozen_source_changes": changes,
        "producer_code_version": [a["code_version"], b["code_version"]],
        "same_code_version": a["code_version"] == b["code_version"],
        "same_source_manifest": first["manifest"]["sha256"] == second["manifest"]["sha256"],
        "source_manifest_sha256": [first["manifest"]["sha256"], second["manifest"]["sha256"]],
        "source_manifest_metadata": [first["manifest"]["metadata"], second["manifest"]["metadata"]],
        "prefix_stage_entrypoints": [first["entries"], second["entries"]],
        "prefix_support_module_sha256": [first["support_modules"], second["support_modules"]],
        "entrypoint_scope": "static direct serial stage calls and selected supporting modules; manifest contains every frozen file; not complete runtime reachability or OS library/file trace",
        "audit_helper_sha256": [a["execution_audit_helper_sha256"], b["execution_audit_helper_sha256"]],
        "harness_sha256": [a["script_sha256"], b["script_sha256"]],
        "fixed_hardware_budget": {key: a[key] for key in ("threads", "device", "CPU_affinity", "CUDA_VISIBLE_DEVICES", "hostname")},
        "comparison_design": "existing SimpleITK continuous prefix versus later Torch continuous prefix; shared host, resources and budget; cross-version and cross-time diagnostic",
        "controlled_speedup_established": False, "overall_metric_equivalence": "not_assessed"}


def compare(*, specification: Path, output: Path) -> dict:
    """按显式JSON spec只读比两例，输出必须是新目录；失败不写成功收据。

    每例原始T1为NIfTI，角色指定实际trial/COMPLETE/source/manifest及
    backend、n4_execution。输出含summary、逐例JSON和原字节私有收据。
    MGZ保持自身voxel/conform空间，灰度为存储值，体积CSV为mm³，LTA
    是source voxel→atlas voxel；仅同网格才做同索引和标签Dice。
    缺文件/完成门/版本/资源/配置不符抛异常；不改producer/原始影像。
    """
    import nibabel as nib
    import numpy as np
    from fnit.recon_all import compare_native_free, ca_normalize_python
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    spec, spec_bytes = read(specification)
    if spec.get("schema") != "fnit-n4-prefix-comparison-v1" or sorted(row["case"] for row in spec["cases"]) != ["sub06", "sub07"]:
        raise ValueError("explicit two-case N4 prefix specification required")
    helper_file = Path(spec["comparison_helpers_file"])
    if sha(helper_file) != spec["comparison_helpers_sha256"]:
        raise ValueError("comparison helper SHA changed")
    helper_spec = importlib.util.spec_from_file_location("fnit_prefix_readonly_helpers", helper_file)
    helpers = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helpers)
    if tuple(helpers.VOLUMES) != VOLUMES or set(helpers.LABEL_VOLUMES) != LABEL_VOLUMES:
        raise ValueError("existing comparator volume/label semantics differ")
    comparison_root = Path(spec["comparison_source_root"]).resolve()
    comparator_hashes = {}
    for module in (compare_native_free, ca_normalize_python):
        file = Path(module.__file__).resolve()
        relative = file.relative_to(comparison_root).as_posix()
        comparator_hashes[relative] = sha(file)
    color_lut = Path(spec["color_lut"])
    names = {int(parts[0]): parts[1] for line in color_lut.read_text().splitlines()
             if (parts := line.split()) and parts[0].isdigit()}
    prepared, manifests = [], {}
    for case_spec in spec["cases"]:
        case, raw = case_spec["case"], Path(case_spec["raw_t1"])
        roles = [prepare_role(case_spec[role], case=case, raw=raw, manifests=manifests) for role in ("reference", "candidate")]
        meta = metadata_pair(*roles)
        for role in roles:
            if sha(color_lut) != role["producer"]["resource_sha256"]["assets"]["FreeSurferColorLUT.txt"]:
                raise ValueError("LUT differs from producer declaration")
        for relative, digest in comparator_hashes.items():
            if digest != roles[1]["manifest"]["files"][relative]:
                raise ValueError("loaded comparison module is not candidate frozen source")
        prepared.append((case, raw, roles, meta))
    output.mkdir(parents=True)
    summary = {"status": "complete_n4_prefix_comparison", "full_workflow_complete": False,
        "overall_metric_equivalence": "not_assessed", "controlled_speedup_established": False,
        "scope": "two real raw T1 empty-directory continuous volume prefixes; SimpleITK versus explicit Torch N4",
        "script_sha256": sha(Path(__file__)), "specification_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "comparison_helpers_sha256": sha(helper_file), "comparison_modules_sha256": comparator_hashes,
        "cases": []}
    for case, raw, roles, meta in prepared:
        subjects = [role["subject"] for role in roles]
        result = {"status": summary["status"], "case": case, "full_workflow_complete": False,
            "scope": summary["scope"], "overall_metric_equivalence": "not_assessed",
            "original_T1_sha256": sha(raw), "provenance": meta, "volumes": {},
            "timing": {"controlled_speedup_established": False,
                "public_API_seconds_including_reads_writes_reference_candidate": [role["producer"]["public_API_seconds_including_reads_writes"] for role in roles],
                "fresh_CLI_seconds_reference_candidate": [role["process"]["fresh_CLI_seconds"] for role in roles],
                "diagnostic_seconds_including_imports_hashing_reference_candidate": [role["producer"]["diagnostic_seconds_including_imports_hashing"] for role in roles],
                "stage_rows_reference_candidate": [role["runtime"]["stages"] for role in roles],
                "scope": "actual synchronized stage wall includes writes; nested timers not added; different versions/time/load are diagnostic observations"},
            "memory": {"process_samples_reference_candidate": [role["process"]["samples"] for role in roles],
                "allocator_reference_candidate": [role["runtime"]["cuda_allocator"] for role in roles],
                "precision_reference_candidate": [role["runtime"]["precision"] for role in roles],
                "unknown_ownership_retained": True, "unknown_or_zero_is_not_zero_memory": True},
            "n4_runtime_reference_candidate": [role["runtime"]["n4_runtime"] for role in roles]}
        for name in VOLUMES:
            paths = [subject / "mri" / name for subject in subjects]
            before = [sha(path) for path in paths]
            images = [nib.load(str(path)) for path in paths]
            arrays = [np.asarray(image.dataobj) for image in images]
            if not all(np.isfinite(array).all() for array in arrays):
                raise ValueError("nonfinite volume: " + name)
            geometry = helpers._geometry(images)
            value = {"files_sha256": before, "files_bytes_equal": before[0] == before[1],
                     "geometry": geometry, "array_dtype": [str(array.dtype) for array in arrays]}
            if geometry["voxelwise_comparable"]:
                error = np.abs(arrays[0].astype(np.float64) - arrays[1].astype(np.float64))
                value.update(different_voxels=int(np.count_nonzero(arrays[0] != arrays[1])),
                    max_abs_error=float(error.max()), p99_abs_error=float(np.quantile(error, .99)), numeric_unit="stored voxel value")
                if name in LABEL_VOLUMES:
                    value["labels"] = helpers._dice(*arrays, label_names=names, comparator=compare_native_free, name=name)
                    value["label_pearson_not_used"] = True
                else:
                    value.update(compare_native_free._numbers(*arrays))
                if name.startswith("wm"):
                    value["WM_foreground"] = {}
                    for rule, threshold in (("WM_MIN_VAL_ge_5", 5), ("nonzero_diagnostic_only", 1)):
                        masks = [array >= threshold for array in arrays]
                        counts = [int(mask.sum()) for mask in masks]
                        common = int(np.count_nonzero(masks[0] & masks[1]))
                        value["WM_foreground"][rule] = {"threshold": threshold, "reference_voxels": counts[0],
                            "candidate_voxels": counts[1], "intersection_voxels": common,
                            "dice": 2 * common / sum(counts) if sum(counts) else 1.0}
            else:
                value.update(ordered_comparable=False, reason="shape/affine/tkr differ; no same-index numerical/Dice claim")
            if before != [sha(path) for path in paths]:
                raise RuntimeError("producer volume changed during comparison: " + name)
            result["volumes"][name] = value
        lta_paths = [subject / "mri/transforms/talairach.lta" for subject in subjects]
        lta_hashes = [sha(path) for path in lta_paths]
        matrices = [ca_normalize_python.read_voxel_lta(path) for path in lta_paths]
        lta_error = np.abs(matrices[0] - matrices[1])
        result["talairach_lta"] = {"type": 0, "meaning": "source voxel→atlas voxel; translation unit atlas voxel, not mm",
            "matrices": [matrix.tolist() for matrix in matrices], "files_sha256": lta_hashes,
            "different_elements": int(np.count_nonzero(matrices[0] != matrices[1])),
            "max_abs": float(lta_error.max()), "p99_abs_error": float(np.quantile(lta_error, .99))}
        if lta_hashes != [sha(path) for path in lta_paths]:
            raise RuntimeError("LTA changed while reading")
        csv_paths = [subject / "stats/synthseg.vol.csv" for subject in subjects]
        csv_value = helpers._csv_compare(csv_paths, subjects)
        tissue_rows = []
        for row_index, (a, b) in enumerate(zip(csv_value["reference_numeric_values"], csv_value["candidate_numeric_values"])):
            tissue_rows.append({label: {"reference_mm3": x, "candidate_mm3": y, "signed_difference_mm3": y-x,
                "absolute_difference_mm3": abs(y-x), "relative_difference": (y-x)/x if x else None}
                for label, x, y in zip(csv_value["columns"][1:], a, b)})
        csv_value["per_tissue_volume_rows"] = tissue_rows
        result["synthseg_volume_csv"] = csv_value
        if csv_value["raw_sha256"] != [sha(path) for path in csv_paths]:
            raise RuntimeError("volume CSV changed while reading")
        for role_name, role in zip(("reference", "candidate"), roles):
            for path, data in role["captures"]:
                (output / f"{case}.{role_name}.{path.name}.private").write_bytes(data)
                if path.read_bytes() != data:
                    raise RuntimeError("producer receipt changed during comparison")
        if sha(raw) != result["original_T1_sha256"]:
            raise RuntimeError("original T1 changed during comparison")
        result["all_geometry_dtype_exact"] = all(all(value["geometry"][key] for key in
            ("shape_equal", "storage_dtype_equal", "affine_exact", "vox2ras_tkr_exact", "zooms_equal"))
            for value in result["volumes"].values())
        result["pre_n4_input_values_geometry_exact"] = all(result["volumes"][name].get("different_voxels") == 0
            and all(result["volumes"][name]["geometry"][key] for key in
            ("shape_equal", "storage_dtype_equal", "affine_exact", "vox2ras_tkr_exact", "zooms_equal")) for name in ("rawavg.mgz", "orig.mgz"))
        write(output / (case + ".json"), result)
        summary["cases"].append({"case": case, "comparison_sha256": sha(output / (case + ".json")),
            "pre_n4_input_values_geometry_exact": result["pre_n4_input_values_geometry_exact"],
            "all_geometry_dtype_exact": result["all_geometry_dtype_exact"],
            "different_voxels": {name: row.get("different_voxels") for name, row in result["volumes"].items()},
            "filled_minimum_foreground_dice": result["volumes"]["filled.mgz"].get("labels", {}).get("minimum_foreground_dice"),
            "LTA_different_elements": result["talairach_lta"]["different_elements"],
            "CSV_numeric_values_equal": csv_value["numeric_values_equal"],
            "timing_observations": {key: value for key,value in result["timing"].items() if key != "stage_rows_reference_candidate"}})
    if specification.read_bytes() != spec_bytes:
        raise RuntimeError("comparison specification changed")
    summary["diagnostic_seconds_including_hashes_reads_writes"] = time.perf_counter() - started
    write(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(compare(**vars(parser.parse_args()))))
