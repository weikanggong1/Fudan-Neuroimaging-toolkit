#!/usr/bin/env python3
"""Complete CPU InvWarp cases with strictly identical native oracle reuse.

Official programs are isolated benchmark references. A reused output supplies
accuracy only: its earlier clock is never a new timing observation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys


def normalized_commands(commands):
    """Replace only declared output names, including their later input uses."""
    outputs = {}
    for command in commands:
        for token in command:
            if token.startswith("--out="):
                outputs.setdefault(token.split("=", 1)[1], f"<OUTPUT_{len(outputs)}>")
    normalized = []
    for command in commands:
        row = []
        for token in command:
            if "=" in token:
                flag, value = token.split("=", 1)
                token = flag + "=" + outputs.get(value, value)
            else:
                token = outputs.get(token, token)
            row.append(token)
        normalized.append(row)
    return normalized


def contract_digest(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def program_metadata(commands, env, harness):
    programs, dependencies = [], {}
    for command in commands:
        executable = Path(shutil.which(command[0]) or command[0]).resolve()
        programs.append({"name": executable.name, "path_private": str(executable),
                         "bytes": executable.stat().st_size, "sha256": harness.sha256(executable)})
        result = subprocess.run(["ldd", str(executable)], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if "not found" in result.stdout:
            raise RuntimeError("Native dependency missing")
        if result.returncode and not any(text in result.stdout for text in ("not a dynamic executable", "statically linked")):
            raise RuntimeError("Unable to audit native dynamic dependencies")
        for line in result.stdout.splitlines():
            match = re.search(r"=>\s+(/\S+)", line) or re.match(r"\s*(/\S+)", line)
            if match:
                path = Path(match.group(1)).resolve()
                dependencies[str(path)] = {"bytes": path.stat().st_size, "sha256": harness.sha256(path)}
    return programs, dependencies


def validate_oracle(outputs, case, harness):
    import nibabel as nib
    import numpy as np
    reference = nib.load(case["reference"])
    for path in outputs.values():
        image = nib.load(path)
        if image.shape != (*reference.shape, 3) or not np.allclose(image.affine, reference.affine):
            raise RuntimeError("Native inverse output grid violates the declared contract")
        if not np.isfinite(np.asanyarray(image.dataobj)).all():
            raise RuntimeError("Non-finite native oracle; preserve it for inspection")
    return harness.output_metadata(outputs)


def public_report(suite):
    """Explicit aggregate whitelist; cached native clocks never enter it."""
    records = []
    for record in suite["records"]:
        worker = record["worker_results"]["candidate"][0]
        records.append({key: record[key] for key in (
            "case_id", "threads", "affinity", "reference_reused", "reference_contract_sha256",
            "timing_protocol", "native_timing_scope", "single_full_process_seconds", "process_cpu_usage",
            "accuracy", "adapter_accuracy", "official_output_metadata")})
        records[-1].update(candidate_api_single_seconds=worker["median_seconds"],
                           candidate_output_metadata=worker["output_metadata"],
                           candidate_maximum_rss_kib=worker["maximum_rss_kib"],
                           input_sha256=sorted({value["sha256"] for value in record["input_metadata_before"].values()}),
                           official_programs=[{key: value[key] for key in ("name", "sha256", "bytes")}
                                              for value in record["official_program_metadata"]])
    return {"schema_version": 1, "status": suite["status"], "timing_policy": suite["timing_policy"],
            "candidate_source_tree_sha256": suite["source_metadata"]["candidate"]["tree_sha256"],
            "benchmark_script_sha256": suite["benchmark_script_sha256"],
            "adapter_source_files_sha256": suite.get("adapter_source_files_sha256", {}),
            "native_output_contract": suite.get("native_output_contract"),
            "expected_complete_observations": suite["expected_complete_observations"],
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--adapter-root", help="Independent immutable benchmark tools; runtime remains candidate-root")
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-cache-dir", required=True)
    parser.add_argument("--cpuset", required=True)
    parser.add_argument("--threads", default="1,8")
    parser.add_argument("--lock-file", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    source, baseline = Path(args.candidate_root).resolve(), Path(args.baseline_root).resolve()
    adapter_root = Path(args.adapter_root).resolve() if args.adapter_root else source
    destination, cache = Path(args.output_dir).resolve(), Path(args.reference_cache_dir).resolve()
    if destination.exists():
        raise FileExistsError("Preserve existing runs; select a new output directory")
    cpus = [int(value) for value in args.cpuset.split(",")]
    budgets = [int(value) for value in args.threads.split(",")]
    if len(set(cpus)) != len(cpus) or not set(cpus).issubset(os.sched_getaffinity(0)) or min(budgets) < 1 or max(budgets) > len(cpus):
        raise ValueError("Provide distinct accessible CPUs for all thread budgets")
    manifests = [(Path(path).resolve(), json.loads(Path(path).read_text())) for path in args.manifests]
    identifiers = [case["id"] for _, manifest in manifests for case in manifest["cases"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Case IDs must be unique across manifests")
    destination.mkdir(parents=True)
    cache.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(source / "src"), str(source), str(source / "tools")]
    import benchmark_multimodal_cpu as harness
    suite = {"schema_version": 1, "status": "waiting_for_group_lock", "host": socket.gethostname(),
             "source_metadata": {"candidate": harness.source_metadata(source), "baseline": harness.source_metadata(baseline)},
             "benchmark_script_sha256": harness.sha256(__file__), "thread_budgets": budgets,
             "adapter_root_private": str(adapter_root),
             "adapter_source_files_sha256": {case["adapter"]: harness.sha256(adapter_root / case["adapter"])
                                               for _, manifest in manifests for case in manifest["cases"]},
             "native_output_contract": "FSL 6.0.7.4 inverse is relative; absolute outputs include a complete invwarp + convertwarp --rel --absout chain",
             "manifest_sha256": [harness.sha256(path) for path, _ in manifests],
             "expected_complete_observations": len(identifiers) * len(budgets),
             "timing_policy": "Complete single observations; native oracle reuse has no new native clock and no paired speed ratio",
             "records": []}
    harness.save_json(destination / "suite.private.json", suite)
    native_environments = {threads: harness.environment(threads, source) for threads in budgets}
    try:
        with Path(args.lock_file).open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            suite["status"] = "running"
            for threads in budgets:
                affinity = cpus[:threads]
                os.sched_setaffinity(0, affinity)
                for _, manifest in manifests:
                    resources = {**manifest["resources"], "threads": threads}
                    for case in manifest["cases"]:
                        work = destination / f"threads_{threads}" / case["id"]
                        official_dir, candidate_dir = work / "official", work / "candidate"
                        official_dir.mkdir(parents=True)
                        candidate_dir.mkdir()
                        plugin = harness.adapter(adapter_root, case["adapter"])
                        if "torch" in sys.modules:
                            sys.modules["torch"].set_num_threads(threads)
                        env = dict(native_environments[threads])
                        reference_env = manifest.get("reference_env", {})
                        protected = {key for key in env if key.endswith("_NUM_THREADS")} | {"OMP_DYNAMIC", "MKL_DYNAMIC", "PYTHONPATH"}
                        if set(reference_env) & protected:
                            raise ValueError("Reference cannot override common thread budget")
                        env.update(reference_env)
                        env["FSLOUTPUTTYPE"] = "NIFTI" if case.get("output_suffix") == ".nii" else "NIFTI_GZ"
                        commands = plugin.reference_command(case, official_dir, resources)
                        if commands and isinstance(commands[0], str):
                            commands = [commands]
                        before = harness.input_metadata(case)
                        programs, dependencies = program_metadata(commands, env, harness)
                        binding = {"argv_without_output_names": normalized_commands(commands), "inputs": before,
                                   "programs": programs, "dependencies": dependencies, "threads": threads,
                                   "affinity": affinity, "environment_sha256": contract_digest(dict(env)),
                                   "output_suffix": case.get("output_suffix", ".nii.gz")}
                        key = contract_digest(binding)
                        entry_path = cache / (key + ".private.json")
                        record = {"case_id": case["id"], "threads": threads, "affinity": affinity,
                                  "input_metadata_before": before, "official_program_metadata": programs,
                                  "reference_contract_sha256": key, "reference_reused": entry_path.exists(),
                                  "full_process": {"official": [], "baseline": [], "candidate": []},
                                  "process_cpu_usage": {"official": [], "baseline": [], "candidate": []},
                                  "worker_results": {"baseline": [], "candidate": []},
                                  "timing_protocol": "single_observation", "loaded_api_measurements": {},
                                  "full_process_contains_api_repetitions": 1}
                        if entry_path.exists():
                            entry = json.loads(entry_path.read_text())
                            if entry["status"] != "complete_native_oracle" or entry["binding"] != binding:
                                raise RuntimeError("Oracle contract changed")
                            official = entry["outputs_private"]
                            if harness.output_metadata(official) != entry["output_metadata"]:
                                raise RuntimeError("Cached oracle file changed; do not reuse")
                        else:
                            elapsed, usages = 0.0, []
                            for index, command in enumerate(commands):
                                elapsed += harness.run_process(command, official_dir / f"command_{index}", env, affinity)
                                usages.append(dict(harness.run_process.last_usage))
                            official = plugin.reference_outputs(case, official_dir, resources)
                            metadata = validate_oracle(official, case, harness)
                            if harness.input_metadata(case) != before:
                                raise RuntimeError("Native inputs changed during execution")
                            record["full_process"]["official"] = [elapsed]
                            record["process_cpu_usage"]["official"] = [usages]
                            harness.save_json(entry_path, {"status": "complete_native_oracle", "binding": binding,
                                              "outputs_private": official, "output_metadata": metadata,
                                              "original_case_id_private": case["id"], "original_native_seconds": elapsed})
                        request = {"case": case, "adapter": case["adapter"], "adapter_root": str(adapter_root),
                                   "source_root": str(source), "output_dir": str(candidate_dir), "threads": threads,
                                   "affinity": affinity, "device": "cpu", "backend": "candidate", "repetitions": 1, "warmup": False}
                        request_path, result_path = candidate_dir / "request.private.json", candidate_dir / "api-result.private.json"
                        harness.save_json(request_path, request)
                        elapsed = harness.run_process([args.python, str(source / "tools/benchmark_multimodal_cpu.py"), "worker",
                                                       "--request", str(request_path), "--result", str(result_path)],
                                                      candidate_dir / "process", harness.environment(threads, source), affinity)
                        worker = json.loads(result_path.read_text())
                        worker["output_metadata"] = harness.output_metadata(worker["outputs"])
                        record["full_process"]["candidate"] = [elapsed]
                        record["process_cpu_usage"]["candidate"] = [dict(harness.run_process.last_usage)]
                        record["worker_results"]["candidate"] = [worker]
                        record["official_output_metadata"] = harness.output_metadata(official)
                        record["accuracy"] = {"candidate_vs_official": harness.compare_outputs(worker["outputs"], official)}
                        record["adapter_accuracy"] = plugin.compare_case(case, {"official": official, "candidate": worker["outputs"]}, resources)
                        record["input_metadata_after"] = harness.input_metadata(case)
                        if record["input_metadata_after"] != before:
                            raise RuntimeError("Inputs changed during the complete case")
                        record["single_full_process_seconds"] = {backend: values[0] for backend, values in record["full_process"].items() if values}
                        record["native_timing_scope"] = "reused precision oracle; native clock omitted" if record["reference_reused"] else "one fresh complete native process chain"
                        harness.save_json(work / "timing.private.json", record)
                        suite["records"].append(record)
                        harness.save_json(destination / "suite.private.json", suite)
                        harness.save_json(destination / "report.partial.public.json", public_report(suite))
                        print(json.dumps({"case_id": case["id"], "threads": threads, "reference_reused": record["reference_reused"], "status": "completed"}), flush=True)
            suite["status"] = "executed_with_numeric_comparisons"
            harness.save_json(destination / "suite.private.json", suite)
            harness.save_json(destination / "report.public.json", public_report(suite))
    except Exception as error:
        suite.update(status="failed", error_type=type(error).__name__)
        harness.save_json(destination / "suite.private.json", suite)
        raise


if __name__ == "__main__":
    main()
