"""复用既有迁移清单与启动审计，核对当前Python配方；不运行影像阶段。"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def function_reference(*, repository: Path, path: str, function: str | None = None) -> dict:
    """实际UTF-8源码→文件/函数SHA与行号；缺函数直接失败，不借历史SHA。"""
    raw = (repository / path).read_bytes()
    result = {"path": path, "file_sha256": digest(raw), "bytes": len(raw)}
    if function:
        nodes = ast.parse(raw).body
        for part in function.split("."):
            node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
            nodes = node.body
        first = min([node.lineno] + [n.lineno for n in getattr(node, "decorator_list", [])])
        fragment = b"".join(raw.splitlines(keepends=True)[first - 1:node.end_lineno])
        result.update(function=function, line=first, end_line=node.end_lineno,
                      function_sha256=digest(fragment))
    return result


def _runtime_without_docstrings(raw: bytes) -> str:
    tree = ast.parse(raw)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def historical_execution_audit(*, repository: Path) -> dict:
    """仅汇总既有v4实际JSON事件，保留原/公开SHA及失败状态；不改标当前版本。"""
    base = repository / "validation/recon_all/optimizations/20261010_python_profile"
    receipt = base / "reports/completed_v4"
    conversion = json.loads((receipt / "AUDIT_CONVERSION.json").read_bytes())
    published = json.loads((receipt / "PUBLIC_EXPORT_MANIFEST.json").read_bytes())["reports"]
    summary = json.loads((base / "COMPLETED_V4_SUMMARY.json").read_bytes())
    cases = {}
    for case, producer in summary["cases"].items():
        counts, files, blocked = Counter(), [], []
        paths = sorted((receipt / f"{case}-candidate/execution-audit").glob("*.events.json"))
        for path in paths:
            raw = path.read_bytes()
            rows = json.loads(raw)
            public_item = published[str(path.relative_to(receipt))]
            if digest(raw) != public_item["public_sha256"]:
                raise ValueError("historical public audit SHA changed")
            original = conversion[str(path.relative_to(receipt))]
            if len(rows) != original["event_count"]:
                raise ValueError("audit conversion event count changed")
            for row in rows:
                if row.get("blocked"):
                    blocked.append({"event": row["event"], "executable_name": Path(row["executable"]).name})
                counts[row["event"]] += 1
            files.append({"path": str(path.relative_to(repository)), "published_sha256": digest(raw),
                          "original_export_sha256": public_item["raw_sha256"],
                          "original_jsonl_sha256": original["original_jsonl_sha256"], "events": len(rows)})
        if sum(counts.values()) != producer["python_launch_audit_events"]:
            raise ValueError("published producer audit count mismatch")
        cases[case] = {"status": producer["status"], "full_workflow_complete": producer["full_workflow_complete"],
                       "failed_stage": producer["failed_stage"], "input_sha256": producer["input_sha256"],
                       "native_free_source_sha256": producer["native_free_source_sha256"],
                       "events": dict(counts), "original_stage_attempts": blocked, "files": files}
    return {"producer_code_version": summary["producer_code_version"],
            "summary_sha256": digest((base / "COMPLETED_V4_SUMMARY.json").read_bytes()),
            "cases": cases, "scope": "historical actual paths until failure; not current-code execution",
            "shared_library_or_file_access_OS_audit": "not_recorded", "clean_environment_isolation": "not_verified"}


def audit(*, repository: Path, output: Path, reviewed_commit: str) -> dict:
    """当前仓库+实际已审提交→静态调用/默认迁移/历史启动审计JSON及逐函数SHA。

    repository为仓库根；output必须是新目录；reviewed_commit为当前审核的
    实际Git提交。仅读取源码、既有v4收据和包版本；复用426c68ec迁移
    collector的无MRI/空jobs预检，不创建CUDA context、加载模型或执行FS。
    输出无坐标/体积单位；函数行号、源码字节SHA和138相对路径为结构。
    默认无外部输入；缺文件/函数、SHA/事件计数不符或预检CUDA初始化失败。
    手工条件审核不等于形式化可达性证明，历史执行不证明当前整例/OS隔离。
    """
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    legacy = repository / "validation/recon_all/optimizations/20261010_batch_python_profile_prevalidation/collect_prevalidation.py"
    spec = importlib.util.spec_from_file_location("_fnit_existing_inventory", legacy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="fnit-python-path-audit-") as directory:
        temporary = Path(directory)
        module.collect(repository=repository, output=temporary)
        inventory = json.loads((temporary / "current_inventory.json").read_bytes())
        preflight = json.loads((temporary / "prevalidation_receipt.json").read_bytes())
        source = json.loads((temporary / "source.json").read_bytes())
    assert preflight["cuda_initialized_before"] is False and preflight["cuda_initialized_after"] is False
    # 复用采集器本轮实际执行两个空任务预检；不把原目录的CLI合同标题当新测试。
    preflight["status"] = "empty_job_python_profile_prevalidation_passed"
    preflight["current_scope"] = "two current empty-job reproductions; no CLI fixture or MRI execution in this audit"
    inventory["reused_collector_sha256"] = digest(legacy.read_bytes())
    inventory["reviewed_commit"] = reviewed_commit
    for row in inventory["stages"]:
        if row["stage"] == "WM/aseg edit":
            row["acceptance_gap"] = "full source-order path search implemented; two frozen complete edits decoded exact; ten checkpoints had no active branch; active real-data branch not validated"
            row["evidence"] = "edit_wm_aseg_core_python.remove_paths_to_cortex -> wm_remove_paths.remove_mtl_paths; actual function SHA in call-chain report"
    stage_functions = {
        "N4": ("input_n4_chain", "run_n4_stage"),
        "WM segmentation": ("native_free", "_run_torch_wm_segment"),
        "WM/aseg edit": ("native_free", "_run_python_wm_edit"),
        "Topology repair": ("native_free", "_prepare_native_topology"),
        "nofix inflation": ("native_free", "_prepare_native_topology"),
        "standard inflation": ("native_free", "_run_accurate_sphere_pair"),
        "white.preaparc": ("native_free", "_place_preaparc_and_smooth"),
        "final white and pial": ("native_free", "_finish_cortical_surface"),
        "avg_curv atlas paint": ("native_free", "_run_avg_curv"),
        "curvature statistics": ("native_free", "_run_curvature_stats"),
        "defects projection": ("native_free", "_run_defects_volume"),
        "GCA registration": ("native_free", "_run_native_em_register"),
        "surface vertex metrics": ("native_free", "_run_surface_metrics"),
        "MNI postprocessing": ("mni_nonlinear_chain", "run_mni_nonlinear_chain"),
    }
    for row in inventory["stages"]:
        name, function = stage_functions[row["stage"]]
        ref = function_reference(repository=repository, path=f"src/fnit/recon_all/{name}.py", function=function)
        row["source_references"] = [ref]
        row["evidence"] = f"{ref['path']}:{ref['line']} ({function}; full function SHA captured)"
    previous = repository / "validation/recon_all/optimizations/20261010_python_profile_static/static_audit.json"
    template = json.loads(previous.read_bytes())
    stages = deepcopy(template["stages"])
    excluded = deepcopy(template["unselected_local_CPP_routes"])
    for row in [*stages, *excluded]:
        row["evidence"] = [function_reference(repository=repository, path=item["path"], function=item.get("function")) for item in row["evidence"]]
        if row.get("name") == "WM aseg/pretess/Ento-ACJ":
            row["selected_guard"] = "wm_edit_backend=python; full source-order typed/byte path search; no original program or subject whitelist"
        if row.get("name") == "两轮normalize/brainmask":
            row["selected_guard"] = "both normalization calls explicitly pass device=device; controls=torch and second initial_bias=torch; ordered ridge and feedback stay CPU"
    extra = [("wm_remove_paths", "remove_mtl_paths"), ("expected_outputs", "paths"),
             ("anatomical_stats_rows", "anatomical_stats_rows"), ("anatomical_stats_global", "anatomical_stats_global_lines"),
             ("mni_warp_inverse", "fill_inverse_fields")]
    extras = [function_reference(repository=repository, path=f"src/fnit/recon_all/{name}.py", function=function) for name, function in extra]
    doc_only = []
    for name in ("native_free", "batch"):
        path = f"src/fnit/recon_all/{name}.py"
        original = subprocess.check_output(["git", "show", reviewed_commit + ":" + path], cwd=repository)
        current = (repository / path).read_bytes()
        equal = _runtime_without_docstrings(original) == _runtime_without_docstrings(current)
        if not equal:
            raise ValueError(f"runtime AST changed from reviewed source: {path}")
        doc_only.append({"path": path, "reviewed_commit_sha256": digest(original),
                         "captured_working_tree_sha256": digest(current), "runtime_AST_without_docstrings_equal": equal})
    from fnit.recon_all.expected_outputs import paths
    outputs = paths()
    if len(outputs) != 138 or len(set(outputs)) != 138:
        raise ValueError("declared output list changed")
    runtime = historical_execution_audit(repository=repository)
    report = {"schema_version": 1, "reviewed_commit": reviewed_commit,
        "git_head_at_capture": inventory["git_head_at_capture"], "scope": "current conditional call-chain static audit; no current real MRI execution",
        "template_reuse": {"path": str(previous.relative_to(repository)), "sha256": digest(previous.read_bytes()), "reuse": "structure only; all referenced functions recaptured, no historical runtime claims copied"},
        "profile": inventory["python_profile"], "source": source,
        "stages": stages, "unselected_local_CPP_routes": excluded, "extra_current_function_references": extras,
        "docstring_only_changes": doc_only, "original_stage_selected_static": [],
        "all_outputs": {"expected_count": 138, "unique_count": 138, "relative_paths": outputs,
                        "validation": "presence after mesh/MNI join; decoded/numeric equivalence is separate", "current_execution_verified": False},
        "historical_execution_audit": runtime,
        "boundaries": {"current_runtime_audit": "not_run", "shared_library_runtime_audit": "not_run",
                       "OS_isolation": "not_verified", "precision_or_time_or_memory_benchmark": "not_run",
                       "whole_equivalence": "not_assessed", "full_source_GA_in_production_python_profile": False},
        "priority": "wire and validate the existing own source-semantics complete GA; current Python profile still selects digital topology and whole quality failed"}
    for name, value in (("current_inventory.json", inventory), ("call_chain.json", report), ("preflight.json", preflight)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"stages": len(stages), "current_function_references": sum(len(row["evidence"]) for row in stages) + len(extras),
            "original_stage_selected_static": [], "current_MRI_run": False, "historical_cases": list(runtime["cases"]),
            "expected_outputs": len(outputs), "cuda_initialized": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewed-commit", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), ensure_ascii=False))
