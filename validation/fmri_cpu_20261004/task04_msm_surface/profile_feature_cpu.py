"""Profile one complete HCP feature API under its existing CPU task lock.

This diagnostic invokes the declared real input without changing the algorithm.
Profiler times include instrumentation and must not replace paired benchmark
wall times. Individual paths and output arrays remain in the private directory.
"""
import argparse
import cProfile
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pstats
import resource
import sys
import time


def _input_files(value):
    if isinstance(value, dict):
        for name, item in value.items():
            for role, path in _input_files(item):
                yield f"{name}.{role}", path
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            for role, path in _input_files(item):
                yield f"{index}.{role}", path
    elif isinstance(value, str) and Path(value).is_file():
        yield "file", Path(value)


def _input_hashes(case):
    result = {}
    for role, path in _input_files(case.get("inputs", {})):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[role] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--cpuset", required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    args = parser.parse_args()
    affinity = list(map(int, args.cpuset.split(",")))
    if args.threads < 1 or args.threads > len(affinity) or len(set(affinity)) != len(affinity):
        raise ValueError("Thread budget must fit distinct assigned CPU identifiers")
    manifest = json.loads(args.manifest.read_text())
    case = next(item for item in manifest["cases"] if item["id"] == args.case)
    if case["operation"] not in ("variance_normalization", "regression"):
        raise ValueError("This diagnostic covers complete VN, DR or WRN APIs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.sched_setaffinity(0, affinity[:args.threads])
    source = args.source_root.resolve()
    sys.path.insert(0, str(source / "src"))
    with args.lock_file.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        import torch
        import fnit
        Path(fnit.__file__).resolve().relative_to(source / "src")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        adapter_path = Path(__file__).with_name("adapter.py")
        specification = importlib.util.spec_from_file_location("feature_profile_adapter", adapter_path)
        adapter = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(adapter)
        input_metadata_before = _input_hashes(case)
        python_profile = cProfile.Profile(timer=time.process_time)
        usage_before = resource.getrusage(resource.RUSAGE_SELF)
        child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        load_before = list(os.getloadavg())
        started = time.perf_counter()
        python_profile.enable()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    record_shapes=False, profile_memory=False, with_stack=False) as operators:
            outputs = adapter.run_case(case, args.output_dir / "outputs", "cpu")
        python_profile.disable()
        wall = time.perf_counter() - started
        usage_after = resource.getrusage(resource.RUSAGE_SELF)
        child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        statistics = pstats.Stats(python_profile)
        input_metadata_after = _input_hashes(case)
        if input_metadata_before != input_metadata_after:
            raise RuntimeError("Input changed during complete feature profile")
        functions = []
        for (filename, line, function), (primitive, total, self_seconds, cumulative, _) in statistics.stats.items():
            if "/fnit/msm/" not in filename and function not in ("matmul", "std", "mean", "svd", "pinv"):
                continue
            functions.append({"module": Path(filename).name, "line": line, "function": function,
                              "primitive_calls": primitive, "calls": total, "self_seconds": self_seconds,
                              "inclusive_seconds": cumulative})
        events = sorted(operators.key_averages(), key=lambda item: item.self_cpu_time_total, reverse=True)
        owned = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted((source / "src/fnit/msm").glob("*.py"))}
        report = {"status": "complete_diagnostic", "case_id": args.case,
                  "scope": "Complete declared real input and saved API outputs; profiler overhead is included",
                  "formal_timing": False, "device": "cpu", "threads": torch.get_num_threads(),
                  "affinity": sorted(os.sched_getaffinity(0)), "torch_version": str(torch.__version__),
                  "diagnostic_api_seconds": wall,
                  "load_before": load_before, "load_after": list(os.getloadavg()),
                  "process_user_seconds": usage_after.ru_utime - usage_before.ru_utime,
                  "process_system_seconds": usage_after.ru_stime - usage_before.ru_stime,
                  "child_user_seconds": child_after.ru_utime - child_before.ru_utime,
                  "child_system_seconds": child_after.ru_stime - child_before.ru_stime,
                  "python_profile_clock": "time.process_time; process CPU summed across its threads, excluding child CPU and blocked wall time",
                  "operator_profile_clock": "PyTorch instrumented CPU operator elapsed; shared-host scheduling may affect it",
                  "timing_rule": "Self and inclusive rows overlap; do not sum inclusive times",
                  "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  "adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
                  "declared_input_sha256": input_metadata_before,
                  "declared_inputs_unchanged": True,
                  "source_hashes": owned,
                  "python_functions": sorted(functions, key=lambda item: item["inclusive_seconds"], reverse=True),
                  "cpu_operators": [{"operator": event.key, "calls": event.count,
                    "self_cpu_seconds": event.self_cpu_time_total / 1e6,
                    "inclusive_cpu_seconds": event.cpu_time_total / 1e6} for event in events[:30]]}
        (args.output_dir / "profile.public.json").write_text(json.dumps(report, indent=2) + "\n")
        private = {"source_root": str(source), "import_path": fnit.__file__,
                   "outputs": {key: str(path) for key, path in outputs.items()}}
        (args.output_dir / "outputs.private.json").write_text(json.dumps(private, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "formal_timing": False,
                          "diagnostic_api_seconds": wall}))


if __name__ == "__main__":
    main()
