"""真实原始T1→filled连续对照；只在filled屏障返回，不是完整recon-all。

复用冻结生产调度的完整前段，不读官方中间结果、不改变任何算子。serial
保留原调度，parallel只开启已通过同输入回归的N4/SynthSeg并行。诊断在
唯一filled_auto_checkpoint之后插入3条返回语句；原语句一条不删除。
"""
from __future__ import annotations

import argparse
import __future__
import ast
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
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def install_prefix(native_free):
    """保留全部原调度语句，仅在唯一filled保存屏障后加入返回。"""
    original = native_free._run_recon_all_python
    source = inspect.getsource(original)
    tree = ast.parse(source)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != "_run_recon_all_python":
        raise ValueError("unexpected frozen pipeline")
    matches = [index for index, node in enumerate(function.body)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == "stage"
        and node.value.args and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "filled_auto_checkpoint"]
    if len(matches) != 1:
        raise ValueError("expected exactly one direct filled checkpoint")
    index = matches[0]
    baseline = ast.dump(tree, include_attributes=False)
    extra = ast.parse('''
report.update(status="complete_prefix", full_workflow_complete=False,
              stopping_point="filled_auto_checkpoint",
              scope="raw T1 empty-directory continuous volume prefix",
              total_seconds=time.perf_counter()-started)
(subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))
return report
''').body
    for node in extra:
        ast.copy_location(node, function.body[index])
    function.body[index + 1:index + 1] = extra
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, original.__code__.co_filename, "exec",
                 flags=__future__.annotations.compiler_flag), native_free.__dict__, namespace)
    native_free._run_recon_all_python = namespace[function.name]
    receipt = {"module_sha256": sha(Path(native_free.__file__)),
        "original_function_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "original_function_AST_sha256": hashlib.sha256(baseline.encode()).hexdigest(),
        "diagnostic_AST_sha256": hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        "insertion_after": "filled_auto_checkpoint", "insertion_body_index": index + 1,
        "only_added_statements": 3, "original_statements_removed": 0,
        "production_source_files_modified": False}
    return original, receipt


def one(args):
    tick = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    from fnit.recon_all import native_free, python_profile
    import torch
    startup = sys.modules.get("sitecustomize")
    if (startup is None or sha(Path(startup.__file__)) != sha(args.audit_helper_file)
            or torch.cuda.is_initialized()):
        raise RuntimeError("audit helper must load at fresh Python startup before CUDA initialization")
    original, transformation = install_prefix(native_free)
    report = {"status": "running", "case": args.case, "input_execution": args.execution,
        "full_workflow_complete": False, "overall_metric_equivalence": "not_assessed",
        "input_sha256": sha(args.t1), "code_version": args.code_version,
        "source_manifest_sha256": sha(args.source_manifest), "script_sha256": sha(Path(__file__)),
        "diagnostic_AST_transformation": transformation,
        "fixed_profile": python_profile.python_profile_options(device=args.device, threads=4,
            n4_backend="simpleitk" if args.n4_backend == "native" else args.n4_backend,
            n4_execution=args.n4_execution),
        "CPU_affinity": sorted(os.sched_getaffinity(0)), "threads": 4,
        "device": args.device, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": platform.node(), "load_before": list(os.getloadavg()),
        "execution_audit_helper_sha256": sha(args.audit_helper_file),
        "execution_audit_scope": "inherited Python audit hooks; not clean OS/file/library isolation"}
    declaration = json.loads(args.resource_declaration.read_bytes())["resource_sha256"]
    report["resource_sha256"] = {}
    for kind, directory in (("weights", args.weights_dir), ("assets", args.assets_dir)):
        actual = {name: sha(directory / name) for name in declaration[kind]}
        if actual != declaration[kind]:
            raise ValueError("declared resource hash changed: " + kind)
        report["resource_sha256"][kind] = actual
    report["resource_declaration_sha256"] = sha(args.resource_declaration)
    save(args.output / "benchmark.json", report)
    try:
        api_tick = time.perf_counter()
        result = native_free.run_recon_all_python(t1=args.t1,
            subject_dir=args.output / "subject", weights_dir=args.weights_dir,
            assets_dir=args.assets_dir, native_bin_dir=None, device=args.device,
            threads=4, backend="python", input_execution=args.execution,
            n4_backend=args.n4_backend, n4_execution=args.n4_execution,
            profile_stages=True, cuda_allocator_cache="auto")
        if (result["status"] != "complete_prefix" or result["full_workflow_complete"]
                or result["stages"][-1]["name"] != "filled_auto_checkpoint"):
            raise RuntimeError("unexpected prefix stopping condition")
        report.update(status="complete_prefix",
            public_API_seconds_including_reads_writes=time.perf_counter()-api_tick,
            diagnostic_seconds_including_imports_hashing=time.perf_counter()-tick,
            runtime_sha256=sha(args.output / "subject/fnit-native-free-run.json"),
            final_input_sha256=sha(args.t1), load_after=list(os.getloadavg()))
        if report["input_sha256"] != report["final_input_sha256"]:
            raise RuntimeError("raw T1 changed during execution")
    except BaseException as error:
        report.update(status="failed_prefix", error=repr(error),
                      diagnostic_seconds_including_imports_hashing=time.perf_counter()-tick)
        save(args.output / "benchmark.json", report)
        raise
    finally:
        native_free._run_recon_all_python = original
    save(args.output / "benchmark.json", report)
    print(json.dumps({key: report[key] for key in ("case", "status", "input_execution",
          "public_API_seconds_including_reads_writes")}), flush=True)


def pair(args):
    """同一GPU/4线程完整两个冷进程；顺序明示，不称ABBA稳定吞吐。"""
    from fnit.recon_all.profiling import ProcessTreeDeviceSampler
    args.output.mkdir(parents=True, exist_ok=False)
    order = ((args.execution,) if args.mode == "single" else
             ("serial", "parallel") if args.case == "sub07" else ("parallel", "serial"))
    records = []
    for index, execution in enumerate(order, 1):
        name = (f"{index}-{execution}" if args.mode == "pair" else
                f"{index}-{args.n4_backend}-{args.n4_execution}")
        directory = args.output / name
        command = [sys.executable, "-X", "faulthandler", "-u", str(Path(__file__).resolve()),
            "--mode", "one", "--case", args.case, "--execution", execution,
            "--output", str(directory), "--t1", str(args.t1), "--device", args.device,
            "--weights-dir", str(args.weights_dir), "--assets-dir", str(args.assets_dir),
            "--source-manifest", str(args.source_manifest), "--code-version", args.code_version,
            "--resource-declaration", str(args.resource_declaration),
            "--audit-helper-file", str(args.audit_helper_file),
            "--n4-backend", args.n4_backend, "--n4-execution", args.n4_execution]
        environment = os.environ.copy()
        environment["NUMBA_CACHE_DIR"] = str(args.output / (name + "-numba-cache"))
        audit = args.output / (name + "-execution-audit")
        environment["FNIT_RECON_AUDIT_DIR"] = str(audit)
        started = time.perf_counter()
        with (args.output / (name + ".log")).open("w") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
            sampler = ProcessTreeDeviceSampler(device=args.device, parent_pid=child.pid, interval=.5)
            while child.poll() is None:
                sampler.sample_if_due()
                time.sleep(.1)
            sampler.sample_if_due(force=True)
            sampler.close()
            sampling = sampler.report()
        events, hashes = [], {}
        for path in sorted(audit.glob("*.jsonl")):
            events.extend(json.loads(line) for line in path.read_text().splitlines())
            hashes[path.name] = sha(path)
        record = {"exit_code": child.returncode, "order_index": index, "execution": execution,
            "requested_n4_backend": args.n4_backend, "n4_execution": args.n4_execution,
            "fresh_CLI_seconds": time.perf_counter()-started, "command": command,
            "samples": sampling, "cold_numba_cache": True,
            "execution_audit": {"events": events, "files_sha256": hashes,
                "blocked_events": sum(bool(row.get("blocked")) for row in events),
                "helper_sha256": sha(args.audit_helper_file)}}
        save(args.output / (name + "-process.json"), record)
        records.append(record)
        if child.returncode:
            raise RuntimeError("failed continuous prefix preserved: " + name)
    save(args.output / "COMPLETE.json", {"status": "complete_prefixes", "case": args.case,
        "order": list(order), "runs": records, "full_workflow_complete": False,
        "stable_ABBA_speedup_established": False, "script_sha256": sha(Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("one", "pair", "single"), required=True)
    parser.add_argument("--case", choices=("sub07", "sub06"), required=True)
    parser.add_argument("--execution", choices=("serial", "parallel"), default="serial")
    for name in ("output", "t1", "weights-dir", "assets-dir", "source-manifest",
                 "resource-declaration", "audit-helper-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--n4-backend", choices=("native", "simpleitk", "torch"), default="native")
    parser.add_argument("--n4-execution", choices=("in-process", "isolated"), default="in-process")
    args = parser.parse_args()
    from fnit.recon_all.python_profile import python_profile_options
    from fnit.recon_all.native_free import _validate_input_execution
    options = python_profile_options(device=args.device, threads=4,
        n4_backend="simpleitk" if args.n4_backend == "native" else args.n4_backend,
        n4_execution=args.n4_execution)
    _validate_input_execution(args.execution, n4_backend=options["n4_backend"],
                              n4_execution=options["n4_execution"], device=args.device, threads=4)
    if args.mode == "pair" and options["n4_backend"] != "simpleitk":
        parser.error("serial/parallel pair requires complete SimpleITK; use single for Torch N4")
    (one if args.mode == "one" else pair)(args)


if __name__ == "__main__":
    main()
