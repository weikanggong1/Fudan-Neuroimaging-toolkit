"""只读绑定当前21键配方/N4选择和人工调用链审核；不运行影像benchmark。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main(*, repository: Path, output: Path) -> None:
    """读取仓库与已有审计/回归，写新SHA收据；并发源码改变时拒绝发布。"""
    repository = repository.resolve()
    if output.exists():
        raise FileExistsError("use a new output directory; preserve prior receipts")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    historical = "validation/recon_all/optimizations/20261010_python_profile_static/static_audit.json"
    previous = json.loads((repository / historical).read_text())
    captured: dict[str, bytes] = {}

    def source(relative):
        if relative not in captured:
            captured[relative] = (repository / relative).read_bytes()
        return captured[relative]

    def evidence(old):
        relative = old["path"]
        raw = source(relative)
        entry = {"path": relative, "file_sha256": sha(raw)}
        name = old.get("function")
        if name:
            nodes = ast.parse(raw).body
            for part in name.split("."):
                node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
                nodes = node.body
            first = min([node.lineno] + [x.lineno for x in getattr(node, "decorator_list", [])])
            fragment = b"".join(raw.splitlines(keepends=True)[first - 1:node.end_lineno])
            entry.update(function=name, line=first, end_line=node.end_lineno,
                         function_sha256=sha(fragment))
        return entry

    # Existing manual group list is only a checklist. All source/function hashes
    # below are captured anew; historical execution results retain their version.
    groups = []
    for row in previous["stages"]:
        groups.append({"name": row["name"], "reviewed_implementation": row["implementation"],
                       "reviewed_guard": row["selected_guard"],
                       "selected_original_FS_stage": False,
                       "evidence": [evidence(x) for x in row["evidence"]]})
    excluded = []
    for row in previous["unselected_local_CPP_routes"]:
        excluded.append({"name": row["name"], "selected": False,
                         "reviewed_guard": row["guard"],
                         "evidence": [evidence(x) for x in row["evidence"]]})
    extra = ["src/fnit/recon_all/python_profile.py", "src/fnit/recon_all/input_n4_chain.py",
             "src/fnit/recon_all/n4_itk_torch_experimental.py", "src/fnit/recon_all/n4_torch_worker.py",
             "src/fnit/recon_all/native_free.py", "src/fnit/recon_all/batch.py",
             "tools/benchmark_recon_torch_end_to_end.py", "tools/recon_python_execution_audit/sitecustomize.py",
             "environment.yml", "pyproject.toml"]
    for relative in extra:
        source(relative)

    sys.path.insert(0, str(repository / "src"))
    from fnit.recon_all.python_profile import python_profile_options
    default = python_profile_options(device="cuda:0", threads=4)
    choices = [python_profile_options(device="cuda:0", threads=4,
                                     n4_backend="torch", n4_execution=execution)
               for execution in ("in-process", "isolated")]
    assert len(default) == 21
    for choice in choices:
        assert set(choice) == set(default)
        assert all(choice[key] == default[key] for key in default if key not in {"n4_backend", "n4_execution"})

    def literal_cli_default(relative, option):
        for node in ast.walk(ast.parse(source(relative))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    and any(isinstance(arg, ast.Constant) and arg.value == option for arg in node.args)):
                return ast.literal_eval(next(x.value for x in node.keywords if x.arg == "default"))
        raise AssertionError("argument not found")

    historical_paths = [historical,
        "validation/recon_all/optimizations/20261009_n4_torch_substages/reports/input_chain_a100_20261009_v2/cached_worker/summary.json",
        "validation/recon_all/optimizations/20261010_n4_continuous_prefix/reports/sub06.json",
        "validation/recon_all/optimizations/20261010_n4_continuous_prefix/reports/sub07.json",
        "validation/recon_all/optimizations/20261010_n4_whole_diagnostics/reports/summary_v2/summary.json"]
    historical_refs = []
    for relative in historical_paths:
        raw = (repository / relative).read_bytes()
        historical_refs.append({"path": relative, "sha256": sha(raw),
                                "evidence_scope": "existing report; not new execution or relabelled code version"})

    differences = subprocess.check_output(["git", "diff", "--", *captured.keys()], cwd=repository)
    tree_changed = [relative for relative, raw in captured.items() if (repository / relative).read_bytes() != raw]
    if tree_changed or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip() != head:
        raise RuntimeError("source changed during audit capture; use a fresh output: " + ", ".join(tree_changed))
    report = {"schema_version": 1, "status": "static_audit_complete",
        "scope": "manual selected-call-path review plus selector contracts; no new MRI, GPU or isolated-deployment run",
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit_at_capture": head, "tracked_diff_sha256": sha(differences),
        "has_tracked_overlay": bool(differences),
        "files": {relative: {"bytes": len(raw), "sha256": sha(raw)} for relative, raw in sorted(captured.items())},
        "profile_option_count": 21, "profile_group_count": len(groups),
        "count_semantics": "21 is the number of dispatch options, not 21 algorithm stages or output files",
        "default_profile": default, "explicit_torch_profiles": choices,
        "only_n4_keys_change": True, "current_selected_call_groups": groups,
        "unselected_local_CPP_routes": excluded,
        "generic_compiled_libraries_allowed": ["SimpleITK/ITK Python library", "PyTorch/CUDA/Triton/PTX toolchain",
            "NumPy/SciPy", "Numba/LLVM/system libm", "nibabel format IO"],
        "prohibited_runtime": ["original FS/FSL/SPM/MRtrix/AFNI stage executable",
            "original FS stage-linked extension or wrapper dispatching the prohibited stage"],
        "reviewed_original_FS_runtime_selected": False,
        "guard_scope": "reviewed parameter guards; AST and imports alone do not prove runtime noninterference",
        "execution_audit": {"scope": "benchmark sitecustomize blocks original program identity; Python children allowed",
                            "not_checked": "OS-level shared-library/file-access trace and physically clean deployment"},
        "whole_benchmark_n4_cli_default": literal_cli_default("tools/benchmark_recon_torch_end_to_end.py", "--n4-backend"),
        "known_routing_bug": {"public_profile_overrode_explicit_n4": True,
            "repair_scope": "helper accepts explicit simpleitk/torch and execution; coordinator owns API/batch/harness routing",
            "initial_public_bug_source": "native_free.run_recon_all_python: options copied over n4_backend/n4_execution",
            "default_policy": "legacy native default maps to simpleitk only in python profile; torch must be explicit",
            "parallel_input": "only simpleitk/in-process; reject torch rather than silently change scheduling"},
        "existing_real_evidence": historical_refs,
        "strict_reproduction": "not established by route contracts; existing N4 differences retained",
        "overall_metric_equivalence": "not_assessed", "new_MRI_jobs": 0,
        "new_time_or_memory_benchmark": "not_run", "new_dependencies": []}
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report_sha256": sha((output / "audit.json").read_bytes()),
                      "files": len(captured), "groups": len(groups), "option_count": len(default)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(repository=args.repository, output=args.output)
