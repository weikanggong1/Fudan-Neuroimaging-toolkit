#!/usr/bin/env python3
"""Paired, same-thread real-data benchmarks; official software is reference-only.

Adapters implement run_case(case, output_dir, device) -> {output_name: path}
and reference_command(case, output_dir, resources) -> command or command chain.
Timing reports distinguish full process wall, first adapter call, and warmed API time.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import socket
import shutil
import statistics
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".new")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def adapter(source_root, relative):
    source_root = Path(source_root).resolve()
    path = (source_root / relative).resolve()
    if not path.is_relative_to(source_root) or not path.is_file():
        raise ValueError("Adapter must be a file inside the selected source tree")
    spec = importlib.util.spec_from_file_location("_fnit_cpu_benchmark_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def worker(args):
    request = json.loads(Path(args.request).read_text())
    source_root = Path(request["source_root"]).resolve()
    sys.path[:0] = [str(source_root / "src"), str(source_root), str(source_root / "tools")]
    # Torch/adapter setup is outside the inner clock. Feature imports performed
    # by run_case are inside its first call and are removed only by API warmup.
    import torch
    if torch.get_num_threads() != request["threads"]:
        torch.set_num_threads(request["threads"])
    torch.set_num_interop_threads(1)
    budget = set(request["affinity"])
    os.sched_setaffinity(0, budget)
    plugin = adapter(request.get("adapter_root", source_root), request["adapter"])
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    case = request["case"]
    device = request["device"]
    timings, records = [], []
    repetitions = request["repetitions"]
    has_warmup = request.get("warmup", False)
    for iteration in range(repetitions + int(has_warmup)):
        destination = output_dir / ("warmup" if has_warmup and iteration == 0 else f"repeat_{iteration}")
        destination.mkdir(parents=True, exist_ok=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        before = time.perf_counter()
        outputs = plugin.run_case(case, destination, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - before
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError("Adapter must return every output file as a name:path mapping")
        outputs = {key: str(Path(value).resolve()) for key, value in outputs.items()}
        for value in outputs.values():
            if not Path(value).is_file():
                raise FileNotFoundError(value)
        timings.append(elapsed)
        records.append(outputs)
    import fnit
    report = {"backend": request["backend"], "case_id": case["id"],
              "scope": (
                  "Warmed adapter API: load + complete computation + transfer + save; "
                  "one complete warmup call excluded"
                  if has_warmup else
                  "First adapter call: includes feature imports, load, complete computation, "
                  "transfer and save; excludes process start and Torch/adapter setup"
              ),
              "warmup_seconds": timings[0] if has_warmup else None,
              "seconds": timings[1:] if has_warmup else timings,
              "median_seconds": statistics.median(timings[1:] if has_warmup else timings),
              "outputs": records[-1],
              "source_root": str(source_root), "fnit_source": fnit.__file__,
              "threads": {"torch": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
                          "numba": sys.modules["numba"].get_num_threads() if "numba" in sys.modules else None},
              "affinity": sorted(os.sched_getaffinity(0)),
              "versions": {"torch": torch.__version__, **{
                  name: importlib.metadata.version(name) for name in ("numpy", "numba", "nibabel")}},
              "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn": torch.backends.cudnn.allow_tf32},
              "cuda_peak_allocation_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None}
    if not Path(fnit.__file__).resolve().is_relative_to(source_root / "src"):
        raise RuntimeError("Wrong FNIT source imported")
    save_json(args.result, report)


def environment(threads, source_root):
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        env[key] = str(threads)
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["PYTHONPATH"] = str(Path(source_root).resolve() / "src")
    return env


def output_metadata(outputs):
    """Hash only after the timed child exits, identically for all backends."""
    return {key: {"sha256": sha256(value), "bytes": Path(value).stat().st_size}
            for key, value in outputs.items()}


def source_metadata(root):
    root = Path(root)
    paths = sorted((root / "src" / "fnit").rglob("*.py"))
    files = {str(path.relative_to(root)): sha256(path) for path in paths}
    record = {"files": files, "tree_sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest()}
    snapshot = root / "snapshot.private.json"
    if snapshot.is_file():
        record["snapshot"] = json.loads(snapshot.read_text())
    git_head = root / ".git" / "HEAD"
    if git_head.is_file():
        record["git_head"] = git_head.read_text().strip()
    return record


def input_metadata(case):
    files = set()
    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value.startswith("/"):
            path = Path(value)
            if path.is_file():
                files.add(path)
    visit(case)
    return {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(files)}


def compare_outputs(candidate, reference):
    """Report independent numeric differences; acceptance is case specific."""
    import numpy as np
    import nibabel as nib
    comparisons = {}
    for name, candidate_path in candidate.items():
        if name not in reference:
            comparisons[name] = {"status": "no_official_correspondence"}
            continue
        reference_path = reference[name]
        if str(candidate_path).endswith((".nii", ".nii.gz", ".mgz", ".gii")):
            a, b = nib.load(candidate_path), nib.load(reference_path)
            if hasattr(a, "dataobj"):
                left, right = np.asanyarray(a.dataobj), np.asanyarray(b.dataobj)
                affine_difference = float(np.max(np.abs(a.affine - b.affine)))
            else:
                left, right = a.agg_data(), b.agg_data()
                left, right = np.asarray(left), np.asarray(right)
                affine_difference = None
        elif str(candidate_path).endswith((".mat", ".txt")):
            left, right = np.loadtxt(candidate_path), np.loadtxt(reference_path)
            affine_difference = None
        else:
            comparisons[name] = {"status": "format_needs_adapter_comparison",
                                 "same_bytes": sha256(candidate_path) == sha256(reference_path)}
            continue
        if left.shape != right.shape:
            comparisons[name] = {"status": "shape_mismatch", "candidate_shape": list(left.shape),
                                 "reference_shape": list(right.shape)}
            continue
        left, right = left.astype(np.float64), right.astype(np.float64)
        candidate_finite = np.isfinite(left)
        reference_finite = np.isfinite(right)
        finite = candidate_finite & reference_finite
        nonfinite_match = (
            (candidate_finite & reference_finite)
            | (np.isnan(left) & np.isnan(right))
            | (np.isposinf(left) & np.isposinf(right))
            | (np.isneginf(left) & np.isneginf(right))
        )
        delta = left[finite] - right[finite]
        scale = right[finite]
        comparisons[name] = {"status": "compared", "shape": list(left.shape),
            "all_finite": bool(finite.all()), "affine_max_absolute_difference": affine_difference,
            "element_count": int(left.size), "finite_pair_count": int(finite.sum()),
            "candidate_finite_count": int(candidate_finite.sum()),
            "reference_finite_count": int(reference_finite.sum()),
            "nonfinite_pattern_equal": bool(nonfinite_match.all()),
            "nonfinite_pattern_mismatch_count": int((~nonfinite_match).sum()),
            "max_absolute_error": float(np.max(np.abs(delta))) if delta.size else None,
            "mean_absolute_error": float(np.mean(np.abs(delta))) if delta.size else None,
            "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
            "relative_l2_error": float(np.linalg.norm(delta) / max(np.linalg.norm(scale), 1e-30)),
            "exact_fraction": float(np.mean(delta == 0)) if delta.size else None}
    return comparisons


def run_process(argv, destination, env, affinity):
    destination.mkdir(parents=True, exist_ok=True)
    cmd = ["/usr/bin/taskset", "-c", ",".join(map(str, affinity)), *argv]
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    before = time.perf_counter()
    with (destination / "stdout.private.log").open("wb") as stdout, (destination / "stderr.private.log").open("wb") as stderr:
        proc = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr)
    elapsed = time.perf_counter() - before
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = (usage_after.ru_utime + usage_after.ru_stime
                   - usage_before.ru_utime - usage_before.ru_stime)
    run_process.last_usage = {"user_seconds": usage_after.ru_utime - usage_before.ru_utime,
                              "system_seconds": usage_after.ru_stime - usage_before.ru_stime,
                              "effective_cpu_cores": cpu_seconds / elapsed}
    if proc.returncode != 0:
        raise RuntimeError(f"Benchmark command failed ({proc.returncode}); see private logs in {destination}")
    return elapsed


def run(args):
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    source_root, baseline_root = Path(args.candidate_root).resolve(), Path(args.baseline_root).resolve()
    sys.path[:0] = [str(source_root / "src"), str(source_root), str(source_root / "tools")]
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    budgets = [int(value) for value in args.threads.split(",")]
    backends = args.backends.split(",")
    if len(set(backends)) != len(backends) or not set(backends).issubset({"official", "baseline", "candidate"}):
        raise ValueError("Backends must be a distinct subset of official,baseline,candidate")
    if not {"official", "candidate"}.issubset(backends):
        raise ValueError("Official and candidate are required for official comparisons")
    cpus = [int(value) for value in args.cpuset.split(",")]
    if not budgets or min(budgets) < 1 or max(budgets) > len(cpus):
        raise ValueError("Provide at least one distinct allowed CPU per requested thread")
    if len(set(cpus)) != len(cpus) or not set(cpus).issubset(os.sched_getaffinity(0)):
        raise ValueError("CPU affinity must contain distinct accessible CPUs")
    single_observation = getattr(args, "single_observation", False)
    if single_observation and (args.repetitions != 1 or args.api_repetitions != 0):
        raise ValueError("A single observation requires --repetitions 1 --api-repetitions 0")
    if not single_observation and args.repetitions < 2:
        raise ValueError("At least two paired repetitions are required")
    warmup_count = 0 if single_observation else 1
    # Lock covers this entire coordinated CPU suite; other research jobs remain running.
    lock_path = Path(args.lock_file or destination.parent / "nodecw10-multimodal-cpu.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        suite = {"schema_version": 1, "status": "running", "host": socket.gethostname(),
                 "manifest_sha256": sha256(manifest_path), "thread_budgets": budgets,
                 "baseline_root": str(baseline_root), "candidate_root": str(source_root),
                 "resource_metadata": input_metadata(manifest["resources"]),
                 "source_metadata": {"baseline": source_metadata(baseline_root),
                                     "candidate": source_metadata(source_root)},
                 "timing_policy": ("One complete official/candidate observation; no warmup or stable speed estimate"
                                   if single_observation else
                                   "Alternating AB/BA full-process pairs; API time separately reported; full IO on all sides"),
                 "records": []}
        for threads in budgets:
            affinity = cpus[:threads]
            os.sched_setaffinity(0, affinity)
            resources = {**manifest["resources"], "threads": threads}
            for case in manifest["cases"]:
                if args.case_id and case["id"] not in args.case_id:
                    continue
                work = destination / f"threads_{threads}" / case["id"]
                plugin = adapter(source_root, case["adapter"])
                # Post-timing numerical diagnostics use this same budget.
                if "torch" in sys.modules:
                    sys.modules["torch"].set_num_threads(threads)
                env = environment(threads, source_root)
                reference_env = manifest.get("reference_env", {})
                protected = {key for key in env if key.endswith("_NUM_THREADS")} | {
                    "OMP_DYNAMIC", "MKL_DYNAMIC", "PYTHONPATH", "VECLIB_MAXIMUM_THREADS"}
                if set(reference_env) & protected:
                    raise ValueError("Reference environment cannot override the common thread budget")
                env.update(reference_env)
                # FSL chooses its file format from this setting even when the
                # output name ends in .nii. Both sides must save the same format.
                env["FSLOUTPUTTYPE"] = "NIFTI" if case.get("output_suffix") == ".nii" else "NIFTI_GZ"
                scope = {"case_id": case["id"], "threads": threads, "affinity": affinity,
                         "input_metadata_before": input_metadata(case),
                         "official_program_metadata": [],
                         "load_before": os.getloadavg(), "full_process": {"official": [], "baseline": [], "candidate": []},
                         "worker_results": {"baseline": [], "candidate": []},
                         "process_cpu_usage": {"official": [], "baseline": [], "candidate": []}}
                output_sets = {}
                for repetition in range(args.repetitions + warmup_count):
                    order = ["official", "baseline", "candidate"] if repetition % 2 == 0 else ["candidate", "baseline", "official"]
                    order = [backend for backend in order if backend in backends]
                    for backend in order:
                        output = work / f"pair_{repetition}" / backend
                        output.mkdir(parents=True, exist_ok=True)
                        if backend == "official":
                            commands = plugin.reference_command(case, output, resources)
                            if commands and isinstance(commands[0], str):
                                commands = [commands]
                            if not commands:
                                raise ValueError("No official reference command for " + case["id"])
                            if not scope["official_program_metadata"]:
                                for command in commands:
                                    executable = Path(shutil.which(command[0]) or command[0]).resolve()
                                    scope["official_program_metadata"].append({
                                        "name": executable.name, "path_private": str(executable),
                                        "bytes": executable.stat().st_size, "sha256": sha256(executable)})
                            elapsed, cpu_usage = 0.0, []
                            for i, command in enumerate(commands):
                                elapsed += run_process(command, output / f"command_{i}", env, affinity)
                                cpu_usage.append(dict(run_process.last_usage))
                            scope["process_cpu_usage"][backend].append(cpu_usage)
                            outputs = plugin.reference_outputs(case, output, resources)
                        else:
                            root = baseline_root if backend == "baseline" else source_root
                            request = {"case": case, "adapter": case["adapter"], "adapter_root": str(source_root),
                                       "source_root": str(root), "output_dir": str(output), "threads": threads,
                                       "affinity": affinity, "device": args.device, "backend": backend,
                                       "repetitions": 1, "warmup": False}
                            request_path = output / "request.private.json"
                            result_path = output / "api-result.private.json"
                            save_json(request_path, request)
                            elapsed = run_process([args.python, str(Path(__file__).resolve()), "worker",
                                                   "--request", str(request_path), "--result", str(result_path)],
                                                  output / "process", environment(threads, root), affinity)
                            scope["process_cpu_usage"][backend].append(dict(run_process.last_usage))
                            worker_result = json.loads(result_path.read_text())
                            worker_result["output_metadata"] = output_metadata(worker_result["outputs"])
                            scope["worker_results"][backend].append(worker_result)
                            outputs = worker_result["outputs"]
                        output_sets[backend] = {key: str(value) for key, value in outputs.items()}
                        scope["full_process"][backend].append(elapsed)
                    save_json(work / "timing.private.json", scope)
                scope["load_after"] = os.getloadavg()
                scope["official_output_metadata"] = output_metadata(output_sets["official"])
                scope["accuracy"] = {
                    "candidate_vs_official": compare_outputs(output_sets["candidate"], output_sets["official"]),
                }
                if "baseline" in backends:
                    scope["accuracy"].update({
                        "baseline_vs_official": compare_outputs(output_sets["baseline"], output_sets["official"]),
                        "candidate_vs_baseline": compare_outputs(output_sets["candidate"], output_sets["baseline"])})
                if hasattr(plugin, "compare_case"):
                    scope["adapter_accuracy"] = plugin.compare_case(case, output_sets, resources)
                scope["input_metadata_after"] = input_metadata(case)
                if scope["input_metadata_after"] != scope["input_metadata_before"]:
                    raise RuntimeError("Input changed during the paired benchmark")
                scope["timing_protocol"] = "single_observation" if single_observation else "warmup_and_paired_repeats"
                if single_observation:
                    scope["single_full_process_seconds"] = {key: values[0] for key, values in scope["full_process"].items() if values}
                else:
                    scope["warmup_full_process"] = {key: values[0] for key, values in scope["full_process"].items() if values}
                    scope["median_full_process_seconds"] = {key: statistics.median(values[1:]) for key, values in scope["full_process"].items() if values}
                # API repetitions are an explicitly separate scope; they are not a speedup ratio against CLI wall.
                scope["full_process_contains_api_repetitions"] = 1
                scope["loaded_api_measurements"] = {}
                for backend, root in [("baseline", baseline_root), ("candidate", source_root)]:
                    if backend not in backends or args.api_repetitions == 0:
                        continue
                    output = work / "loaded_api" / backend
                    request = {"case": case, "adapter": case["adapter"], "adapter_root": str(source_root),
                               "source_root": str(root), "output_dir": str(output), "threads": threads,
                               "affinity": affinity, "device": args.device, "backend": backend,
                               "repetitions": args.api_repetitions, "warmup": True}
                    request_path = output / "request.private.json"
                    result_path = output / "api-result.private.json"
                    save_json(request_path, request)
                    run_process([args.python, str(Path(__file__).resolve()), "worker", "--request", str(request_path),
                                 "--result", str(result_path)], output / "process", environment(threads, root), affinity)
                    scope["loaded_api_measurements"][backend] = json.loads(result_path.read_text())
                    scope["loaded_api_measurements"][backend]["output_metadata"] = output_metadata(
                        scope["loaded_api_measurements"][backend]["outputs"])
                suite["records"].append(scope)
                save_json(destination / "suite.private.json", suite)
                print(json.dumps({"case": case["id"], "threads": threads, "status": "executed"}), flush=True)
        suite["status"] = "executed_with_numeric_comparisons"
        save_json(destination / "suite.private.json", suite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--request", required=True)
    w.add_argument("--result", required=True)
    r = sub.add_parser("run")
    for key in ["manifest", "candidate-root", "baseline-root", "output-dir", "cpuset"]:
        r.add_argument("--" + key, required=True)
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--threads", default="1,8")
    r.add_argument("--repetitions", type=int, default=3)
    r.add_argument("--api-repetitions", type=int, default=3)
    r.add_argument("--single-observation", action="store_true",
                   help="Run one full pair without warmup; requires --repetitions 1 --api-repetitions 0; reports no stable median")
    r.add_argument("--backends", default="official,baseline,candidate")
    r.add_argument("--device", default="cpu")
    r.add_argument("--lock-file")
    r.add_argument("--case-id", action="append")
    args = parser.parse_args()
    worker(args) if args.command == "worker" else run(args)


if __name__ == "__main__":
    main()
