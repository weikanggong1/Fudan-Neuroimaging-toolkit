"""诊断专用：冻结原调度只在filled检查点返回，N4仅改变拟合线程。

两条链均从同一原始T1和空subject开始；不是整例recon-all，也不读官方。
不修改冻结源码文件。保留公开wrapper的校验、21项配方和线程恢复。
"""
from __future__ import annotations

import argparse
import ast
import __future__
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def install_prefix(native_free):
    """仅在唯一直接stage(filled_auto_checkpoint)之后插入完整前段返回。"""
    original = native_free._run_recon_all_python
    source = inspect.getsource(original)
    tree = ast.parse(source)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != "_run_recon_all_python":
        raise ValueError("unexpected frozen implementation")
    matches = [index for index, node in enumerate(function.body) if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "stage" and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "filled_auto_checkpoint"]
    if len(matches) != 1:
        raise ValueError("must have exactly one direct filled_auto_checkpoint")
    index = matches[0]
    original_ast = ast.dump(tree, include_attributes=False)
    extra = ast.parse('''
report.update(status="complete_prefix", full_workflow_complete=False,
              stopping_point="filled_auto_checkpoint",
              scope="raw T1 empty-directory continuous conform-to-filled diagnostic",
              total_seconds=time.perf_counter()-started)
(subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))
return report
''').body
    for node in extra:
        ast.copy_location(node, function.body[index])
    function.body[index + 1:index + 1] = extra
    ast.fix_missing_locations(tree)
    rewritten_ast = ast.dump(tree, include_attributes=False)
    namespace = {}
    exec(compile(tree, original.__code__.co_filename, "exec",
                 flags=__future__.annotations.compiler_flag), native_free.__dict__, namespace)
    native_free._run_recon_all_python = namespace[function.name]
    report = {"original_module_sha256": sha(Path(native_free.__file__)),
        "original_function_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "original_function_AST_sha256": hashlib.sha256(original_ast.encode()).hexdigest(),
        "prefix_function_AST_sha256": hashlib.sha256(rewritten_ast.encode()).hexdigest(),
        "insertion_body_index": index + 1, "insertion_source_relative_line": function.body[index].lineno,
        "insertion_after": "stage('filled_auto_checkpoint', shutil.copyfile, filled, filled.auto)",
        "only_added_statements": 3, "original_statements_removed": 0,
        "no_production_source_file_modified": True}
    return original, report


def run_one(args):
    whole_tick = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    # Runtime依赖与实际模块均从显式冻结源选择；不下载资源，不发现原程序。
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.n4_module_directory.resolve()))
    from fnit.recon_all import n4_simpleitk, native_free, python_profile
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("prefix diagnostic expects fresh child without initialized CUDA")
    original_n4 = n4_simpleitk.correct_volume
    caller_parameters = []
    # 复用已验证Python startup audit；允许NumPy lscpu及Numba/Triton编译器。
    startup = sys.modules.get("sitecustomize")
    if startup is None or sha(Path(startup.__file__)) != sha(args.audit_helper_file):
        raise RuntimeError("declared shared execution audit was not applied at Python startup")
    audit_directory = args.output.parent / (args.output.name + "-execution-audit")
    if os.environ.get("FNIT_RECON_AUDIT_DIR") != str(audit_directory):
        raise RuntimeError("execution audit output must be this prefix's own directory")

    def n4_with_threads(*positional, **keywords):
        if "fitting_threads" in keywords:
            raise ValueError("original caller unexpectedly supplied fitting_threads")
        keywords["fitting_threads"] = args.fitting_threads
        caller_parameters.append({"args": [str(x) for x in positional],
            "kwargs": {key: str(value) if isinstance(value, Path) else value for key, value in keywords.items()}})
        return original_n4(*positional, **keywords)

    original_pipeline, transformation = install_prefix(native_free)
    n4_simpleitk.correct_volume = n4_with_threads
    report = {"scope": "raw T1 continuous empty-directory volume prefix, not full recon-all",
        "case": args.case, "fitting_threads": args.fitting_threads,
        "execution": "running", "full_workflow_complete": False,
        "input_sha256": sha(args.t1), "code_version": args.code_version,
        "source_manifest_sha256": sha(args.source_manifest),
        "script_sha256": sha(Path(__file__)), "diagnostic_AST_transformation": transformation,
        "N4_module_sha256": sha(Path(n4_simpleitk.__file__)),
        "N4_binding_sha256": sha(args.n4_module_directory / "CANDIDATE_BINDING.json"),
        "python_profile_module_sha256": sha(Path(python_profile.__file__)),
        "fixed_profile": python_profile.python_profile_options(device=args.device, threads=4),
        "device": args.device, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CPU_affinity": sorted(os.sched_getaffinity(0)), "threads": 4,
        "hostname": platform.node(), "load_average_before": list(os.getloadavg()),
        "production_default_changed": False, "overall_metric_equivalence": "not_assessed",
        "execution_audit_helper_sha256": sha(args.audit_helper_file),
        "execution_audit_scope": "shared Python-startup audit covers parent and inherited Python workers; generic Python/libs/compilers allowed; not OS/file/library isolation proof"}
    # 按原完整benchmark的声明资源清单逐项校验，不将参考图或结果当输入。
    declaration = json.loads(args.resource_declaration.read_bytes())["resource_sha256"]
    actual_resources = {}
    for kind, directory in (("weights", args.weights_dir), ("assets", args.assets_dir)):
        actual_resources[kind] = {}
        for name, expected in declaration[kind].items():
            path = directory / name
            actual = sha(path)
            if actual != expected:
                raise ValueError("declared resource SHA changed: " + kind + "/" + name)
            actual_resources[kind][name] = actual
    report["resource_sha256"] = actual_resources
    report["resource_declaration_sha256"] = sha(args.resource_declaration)
    save(args.output / "benchmark.json", report)
    try:
        api_tick = time.perf_counter()
        result = native_free.run_recon_all_python(t1=args.t1,
            subject_dir=args.output / "subject", weights_dir=args.weights_dir,
            assets_dir=args.assets_dir, native_bin_dir=None, device=args.device,
            threads=4, backend="python", profile_stages=True, cuda_allocator_cache="auto")
        if result["status"] != "complete_prefix" or result["full_workflow_complete"]:
            raise RuntimeError("unexpected prefix completion status")
        if result["stages"][-1]["name"] != "filled_auto_checkpoint":
            raise RuntimeError("prefix did not stop at the declared barrier")
        report.update(execution="complete_prefix", public_API_seconds_including_reads_writes=time.perf_counter()-api_tick,
            full_diagnostic_seconds_including_imports_hashing=time.perf_counter()-whole_tick,
            production_runtime_sha256=sha(args.output / "subject/fnit-native-free-run.json"),
            actual_N4_caller_parameters=caller_parameters, final_input_sha256=sha(args.t1),
            load_average_after=list(os.getloadavg()))
        if report["input_sha256"] != report["final_input_sha256"]:
            raise RuntimeError("raw input changed during execution")
        if len(caller_parameters) != 1 or caller_parameters[0]["kwargs"]["fitting_threads"] != args.fitting_threads:
            raise RuntimeError("N4 fitting option did not reach exactly one complete stage")
    except BaseException as error:
        report.update(execution="failed_prefix", error=repr(error),
            full_diagnostic_seconds_including_imports_hashing=time.perf_counter()-whole_tick,
            actual_N4_caller_parameters=caller_parameters)
        save(args.output / "benchmark.json", report)
        raise
    finally:
        native_free._run_recon_all_python = original_pipeline
        n4_simpleitk.correct_volume = original_n4
    save(args.output / "benchmark.json", report)
    print(json.dumps({"case": args.case, "fit": args.fitting_threads, "execution": report["execution"],
                      "prefix_API_seconds": report["public_API_seconds_including_reads_writes"]}), flush=True)


def run_group(args):
    from fnit.recon_all.profiling import ProcessTreeDeviceSampler
    args.output.mkdir(parents=True, exist_ok=False)
    for case in ("sub07", "sub06"):
        for fit in (1, 4):
            name = case + "-fit" + str(fit)
            directory = args.output / name
            command = [sys.executable, "-X", "faulthandler", "-u", str(Path(__file__)),
                "--mode", "one", "--case", case, "--fitting-threads", str(fit),
                "--t1", str(args.raw_t1_directory / (case + "_ses-test_T1w.nii.gz")),
                "--output", str(directory), "--weights-dir", str(args.weights_dir),
                "--assets-dir", str(args.assets_dir), "--device", args.device,
                "--n4-module-directory", str(args.n4_module_directory),
                "--source-manifest", str(args.source_manifest), "--code-version", args.code_version,
                "--resource-declaration", str(args.producer_directory / (case + "-candidate/benchmark.json")),
                "--audit-helper-file", str(args.audit_helper_file)]
            environment = os.environ.copy()
            environment["NUMBA_CACHE_DIR"] = str(args.output / (name + "-numba-cache"))
            audit_directory = args.output / (name + "-execution-audit")
            environment["FNIT_RECON_AUDIT_DIR"] = str(audit_directory)
            tick = time.perf_counter()
            with (args.output / (name + ".log")).open("w") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                sampler = ProcessTreeDeviceSampler(device=args.device, parent_pid=child.pid, interval=.5)
                while child.poll() is None:
                    sampler.sample_if_due()
                    time.sleep(.1)
                sampler.sample_if_due(force=True)
            receipt = {"exit_code": child.returncode, "full_process_wall_seconds": time.perf_counter()-tick,
                "command": command, "same_target_device": args.device, "samples": sampler.report(),
                "cold_cache_per_child": True, "scope": "complete raw-prefix process, no surfaces/full workflow"}
            events = []
            audit_hashes = {}
            for path in sorted(audit_directory.glob("*.jsonl")):
                events.extend(json.loads(line) for line in path.read_text().splitlines())
                audit_hashes[path.name] = sha(path)
            receipt["execution_audit"] = {"events": events, "files_sha256": audit_hashes,
                "blocked_events": sum(bool(value.get("blocked")) for value in events),
                "helper_sha256": sha(args.audit_helper_file),
                "scope": "Python audit hooks, not OS/file/library isolation"}
            save(args.output / (name + "-process.json"), receipt)
            if child.returncode:
                raise RuntimeError("prefix failure preserved: " + name)
    save(args.output / "COMPLETE.json", {"status": "complete_prefixes", "cases": 2, "runs": 4,
        "order": "sub07 fit1 then fit4; sub06 fit1 then fit4, same GPU sequential",
        "full_workflow_complete": False, "script_sha256": sha(Path(__file__))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("one", "group"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n4-module-directory", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--audit-helper-file", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--raw-t1-directory", type=Path)
    parser.add_argument("--producer-directory", type=Path)
    parser.add_argument("--resource-declaration", type=Path)
    parser.add_argument("--t1", type=Path)
    parser.add_argument("--case", choices=("sub07", "sub06"))
    parser.add_argument("--fitting-threads", type=int, choices=(1, 4))
    args = parser.parse_args()
    required = ("raw_t1_directory", "producer_directory") if args.mode == "group" else ("case", "fitting_threads", "t1", "resource_declaration")
    if any(getattr(args, name) is None for name in required):
        parser.error("missing mode-specific named arguments")
    (run_one if args.mode == "one" else run_group)(args)
