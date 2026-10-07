"""Tampered provenance must fail; these are gate contracts, not MRI benchmarks."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("cpu_abba_gate", Path(__file__).with_name("compare_cpu_abba.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def evidence():
    common = {"mris_register_run.py": "run-sha", "_average_cpu_cpp.py": "helper-sha"}
    source = {"src/fnit/recon_all/" + name: sha for name, sha in common.items()}
    source.update({gate.AVERAGE_KEY: "candidate-wrapper-sha", gate.CPP_KEY: "cpp-sha"})
    records, queued, jobs, libraries = {}, {}, [], {}
    for name in gate.ARMS:
        job = {"id": name, "argv": ["python", "worker.py", "--threads", "8"],
               "env": {"PYTHONPATH": "/private/frozen/source/src"}}
        jobs.append(job)
        queued[name] = {
            "job": deepcopy(job), "job_sha256": hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
            "argv": ["taskset", "-c", "3,7,11,15,19,23,27,31", "/usr/bin/time", "-v", "-o", "time.txt", *job["argv"]],
            "max_cpu_threads": 8, "cpu_affinity": "3,7,11,15,19,23,27,31", "hostname": "nodecw7",
            "environment": {**{key: "8" for key in gate.THREAD_VARIABLES},
                            "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": job["env"]["PYTHONPATH"]},
        }
        record = {"averaging_source_sha256": gate.BASELINE_SHA256 if name.startswith("baseline") else "candidate-wrapper-sha",
                  "common_source_sha256": deepcopy(common), "versions": {"torch": "fixed-version"},
                  "threads": 8, "affinity": list(gate.AFFINITY), "hostname": "nodecw7",
                  "averaging_calls": [{"iterations": 16}]}
        if name.startswith("candidate"):
            library = name + ".so"
            backend = {"library": library, "backend": "cpp", "source_sha256": "cpp-sha",
                       "requested_threads": 8, "actual_threads": 8, "compiler": {"binary_sha256": "compiler-sha"},
                       "flags": ["no-fast-math"], "abi": 1, "cache_key": name + "-key"}
            record["averaging_calls"][0]["CPU_backend"] = backend
            libraries[library] = {key: deepcopy(backend[key]) for key in ("source_sha256", "compiler", "flags", "abi")}
            libraries[library]["key"] = name + "-key"
        records[name] = record
    return {"source_files": source, "records": records, "queued": queued, "jobs": jobs,
            "baseline_sha": gate.BASELINE_SHA256, "worker_sha": gate.WORKER_SHA256,
            "compiled_libraries": libraries}


def test_bound_evidence_passes_without_claiming_observed_baseline_team(evidence):
    result = gate.identity_resource_checks(**evidence)
    assert result["source_identity_gate_passed"]
    assert result["equal_CPU_resources_gate_passed"]
    assert result["baseline_NumBa_thread_evidence"]["observed_per_call_team_size"] == "not_recorded"


@pytest.mark.parametrize("tamper", ["common_module", "wrapper", "cpp_source", "compiler", "library_key", "thread_team", "empty_calls", "baseline_code"])
def test_changed_source_or_backend_cannot_pass(evidence, tamper):
    record = evidence["records"]["candidate1"]
    backend = record["averaging_calls"][0]["CPU_backend"]
    if tamper == "common_module":
        record["common_source_sha256"]["mris_register_run.py"] = "different-code"
    elif tamper == "wrapper":
        record["averaging_source_sha256"] = "different-wrapper"
    elif tamper == "cpp_source":
        backend["source_sha256"] = "different-cpp"
    elif tamper == "compiler":
        evidence["compiled_libraries"]["candidate1.so"]["compiler"]["binary_sha256"] = "different-compiler"
    elif tamper == "library_key":
        evidence["compiled_libraries"].clear()
    elif tamper == "thread_team":
        backend["actual_threads"] = 4
    elif tamper == "empty_calls":
        record["averaging_calls"] = []
    elif tamper == "baseline_code":
        evidence["baseline_sha"] = "unreviewed-baseline"
    assert not gate.identity_resource_checks(**evidence)["source_identity_gate_passed"]


@pytest.mark.parametrize("tamper", ["job_environment", "payload", "executed_argv", "record_cores", "queue_budget", "numba_budget", "hidden_GPU", "order"])
def test_changed_jobs_or_CPU_budget_cannot_pass(evidence, tamper):
    queue = evidence["queued"]["baseline1"]
    if tamper == "job_environment":
        evidence["jobs"][0]["env"]["NUMBA_NUM_THREADS"] = "4"
    elif tamper == "payload":
        queue["job"]["argv"][-1] = "4"
    elif tamper == "executed_argv":
        queue["argv"][-1] = "4"
    elif tamper == "record_cores":
        evidence["records"]["baseline1"]["affinity"][-1] = 3  # duplicate core, not a fresh eight-core budget
    elif tamper == "queue_budget":
        queue["max_cpu_threads"] = 4
    elif tamper == "numba_budget":
        queue["environment"]["NUMBA_NUM_THREADS"] = "4"
    elif tamper == "hidden_GPU":
        queue["environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    elif tamper == "order":
        evidence["jobs"] = list(reversed(evidence["jobs"]))
    assert not gate.identity_resource_checks(**evidence)["equal_CPU_resources_gate_passed"]


@pytest.mark.parametrize("tamper", [None, "official_budget", "official_environment"])
def test_historical_native_reference_requires_same_CPU_resources(evidence, tamper):
    queue = deepcopy(evidence["queued"]["baseline1"])
    if tamper == "official_budget":
        queue["max_cpu_threads"] = 4
    elif tamper == "official_environment":
        queue["environment"]["NUMBA_NUM_THREADS"] = "16"
    assert all(gate.reference_resource_checks(queue).values()) is (tamper is None)
