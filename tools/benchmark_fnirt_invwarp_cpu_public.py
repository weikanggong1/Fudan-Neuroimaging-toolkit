#!/usr/bin/env python3
"""Complete same-budget CPU comparisons using one verified public CC0 T1.

FSL programs are isolated reference programs in this benchmark tool. They are
never used by the FNIT production APIs. Preparation is timed separately from
the paired FNIRT/InvWarp processes, and all stages honor one common CPU lock.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PUBLIC_T1_SHA256 = "f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a"
PUBLIC_T1_URL = "https://openneuro.org/datasets/ds000114/versions/1.0.2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".new")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_logged(argv, directory, env):
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (directory / "stdout.private.log").open("wb") as stdout, (directory / "stderr.private.log").open("wb") as stderr:
        result = subprocess.run(argv, env=env, stdout=stdout, stderr=stderr)
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f"Reference/benchmark phase failed ({result.returncode}); inspect its private logs")
    return elapsed


def summarize_suite(path):
    """Copy explicit aggregate fields only; no private input/output paths."""
    suite = json.loads(Path(path).read_text())
    if suite["status"] != "executed_with_numeric_comparisons":
        raise ValueError("Cannot publish an unfinished suite")
    records = []
    for record in suite["records"]:
        summary = {
            "case_id": record["case_id"], "threads": record["threads"],
            "affinity": record["affinity"],
            "timing_protocol": record.get("timing_protocol", "warmup_and_paired_repeats"),
            "process_cpu_usage": record["process_cpu_usage"],
            "candidate_api_seconds": [value["median_seconds"] for value in record["worker_results"]["candidate"]],
            "accuracy": record["accuracy"], "adapter_accuracy": record["adapter_accuracy"],
            "official_programs": [{key: program[key] for key in ("name", "sha256", "bytes")}
                                  for program in record["official_program_metadata"]],
            "maximum_rss_kib": [value["maximum_rss_kib"] for value in record["worker_results"]["candidate"]],
            "official_output_metadata": record["official_output_metadata"],
            "worker_output_metadata": {
                backend: [value["output_metadata"] for value in values]
                for backend, values in record["worker_results"].items() if values
            },
        }
        if summary["timing_protocol"] == "single_observation":
            summary["single_full_process_seconds"] = record["single_full_process_seconds"]
        else:
            summary.update({
                "warmup_seconds": record["warmup_full_process"],
                "paired_full_process_seconds": {
                    key: values[1:] for key, values in record["full_process"].items() if values},
                "median_full_process_seconds": record["median_full_process_seconds"],
            })
        records.append(summary)
    return {
        "status": suite["status"], "timing_policy": suite["timing_policy"],
        "source_tree_sha256": suite["source_metadata"]["candidate"]["tree_sha256"],
        "backend_source_tree_sha256": {
            backend: metadata["tree_sha256"]
            for backend, metadata in suite["source_metadata"].items()
        },
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-root", "baseline-root", "fsl-dir", "output-dir", "cpuset", "lock-file"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--threads", default="1,8")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--inverse-single-observation", action="store_true",
                        help="Run each inverse budget once in full, without warmup or a median")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    candidate = Path(args.candidate_root).resolve()
    baseline = Path(args.baseline_root).resolve()
    fsl = Path(args.fsl_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; preserve previous attempts")
    output.mkdir(parents=True)
    state = {"status": "queued", "completed_phases": []}
    def status(phase):
        state["status"] = phase
        write_json(output / "status.private.json", state)
    status("preflight")
    try:
        cpus = [int(value) for value in args.cpuset.split(",")]
        budgets = [int(value) for value in args.threads.split(",")]
        if args.repetitions < 2 or min(budgets) < 1 or max(budgets) > len(cpus):
            raise ValueError("Use at least two repetitions and sufficient CPUs for every thread budget")
        if len(set(cpus)) != len(cpus) or not set(cpus).issubset(os.sched_getaffinity(0)):
            raise ValueError("CPU set must contain distinct accessible CPUs")
        public_t1 = baseline / "examples/data/sub-01_T1w.nii.gz"
        if sha256(public_t1) != PUBLIC_T1_SHA256:
            raise ValueError("Input is not the verified, defaced public CC0 example")
        sources = json.loads((baseline / "examples/data/SOURCES.json").read_text())
        if sources["source_license"] != "CC0":
            raise ValueError("Public example source license mismatch")
        reference = fsl / "data/standard/MNI152_T1_2mm.nii.gz"
        mask = fsl / "data/standard/MNI152_T1_2mm_brain_mask.nii.gz"
        for path in (reference, mask, candidate / "tools/benchmark_multimodal_cpu.py"):
            if not path.is_file():
                raise FileNotFoundError(path)
        import nibabel as nib
        import numpy as np
        from dataclasses import asdict
        sys.path[:0] = [str(candidate / "src")]
        from fnit.fnirt import resolve_fnirt_config
        input_image = nib.load(public_t1)
        reference_image = nib.load(reference)
        mask_image = nib.load(mask)
        if len(input_image.shape) != 3 or len(reference_image.shape) != 3 or mask_image.shape != reference_image.shape or not np.allclose(mask_image.affine, reference_image.affine):
            raise ValueError("Expected full 3D T1, MNI template and same-grid accuracy mask")
        env = dict(os.environ)
        env.update({"FSLDIR": str(fsl), "LD_LIBRARY_PATH": str(fsl / "lib"), "FSLOUTPUTTYPE": "NIFTI_GZ"})
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[key] = str(max(budgets))
        env.update({"OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE"})
        env["PYTHONPATH"] = str(candidate / "src")
        lock_path = Path(args.lock_file)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        affine = output / "shared_initial_affine.mat"
        linear_image = output / "shared_linear_t1.nii.gz"
        status("waiting_for_preparation_lock")
        with lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            status("preparing_shared_official_affine")
            argv = ["/usr/bin/taskset", "-c", ",".join(map(str, cpus[:max(budgets)])),
                    str(fsl / "bin/flirt"), "-in", str(public_t1), "-ref", str(reference),
                    "-omat", str(affine), "-out", str(linear_image), "-dof", "12", "-cost", "corratio",
                    "-interp", "trilinear", "-searchrx", "-90", "90", "-searchry", "-90", "90", "-searchrz", "-90", "90"]
            preparation_seconds = run_logged(argv, output / "shared_affine_preparation", env)
        if not affine.is_file() or not linear_image.is_file():
            raise RuntimeError("Official affine preparation did not save all outputs")
        matrix = np.loadtxt(affine)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
            raise ValueError("Shared official affine is malformed or singular")
        resources = {"fsl_dir": str(fsl), "invwarp_supports_niter": False}
        reference_env = {key: env[key] for key in ("FSLDIR", "LD_LIBRARY_PATH", "FSLOUTPUTTYPE")}
        fnirt_case = {
            "id": "public_cc0_t1_default", "adapter": "tools/benchmark_multimodal_cpu_fnirt.py",
            "input": str(public_t1), "reference": str(reference), "affine": str(affine),
            "accuracy_mask": str(mask), "preset": "default", "overrides": {},
            "full_pull_jacobian": False, "output_suffix": ".nii.gz",
        }
        def run_suite(name, case, single_observation=False):
            manifest = output / (name + ".manifest.private.json")
            write_json(manifest, {"resources": resources, "reference_env": reference_env, "cases": [case]})
            status("waiting_or_running_" + name)
            argv = [args.python, str(candidate / "tools/benchmark_multimodal_cpu.py"), "run",
                    "--manifest", str(manifest), "--candidate-root", str(candidate), "--baseline-root", str(baseline),
                    "--output-dir", str(output / name), "--cpuset", args.cpuset, "--threads", args.threads,
                    "--repetitions", "1" if single_observation else str(args.repetitions), "--api-repetitions", "0",
                    "--backends", "official,candidate", "--lock-file", str(lock_path)]
            if single_observation:
                argv.append("--single-observation")
            run_logged(argv, output / (name + "_controller"), env)
            state["completed_phases"].append(name)
            return output / name / "suite.private.json"
        fnirt_suite = run_suite("fnirt", fnirt_case)
        # Hold the exact same official forward field fixed for both inverse solvers.
        coefficients = output / "fnirt" / f"threads_{budgets[0]}" / fnirt_case["id"] / f"pair_{args.repetitions}" / "official/cout.nii.gz"
        inverse_case = {
            "id": "public_cc0_t1_fixed_official_coefficients", "adapter": "tools/benchmark_multimodal_cpu_invwarp.py",
            "reference": str(public_t1), "warp": str(coefficients), "warp_convention": "auto",
            "output_convention": "relative", "iterations": 30, "tolerance_mm": 0.01,
            "accuracy_mask": str(public_t1), "output_suffix": ".nii.gz",
            "accuracy_region_description": "positive defaced native T1 foreground, including nearby skull",
        }
        inverse_suite = run_suite("invwarp", inverse_case, args.inverse_single_observation)
        status("summarizing_public_numeric_fields")
        report = {
            "schema_version": 1, "status": "executed_with_numeric_comparisons",
            "input": {"dataset": "OpenNeuro ds000114 v1.0.2", "source_url": PUBLIC_T1_URL, "license": "CC0",
                      "example": "examples/data/sub-01_T1w.nii.gz", "sha256": PUBLIC_T1_SHA256,
                      "shape": list(input_image.shape), "voxel_size_mm": [float(value) for value in input_image.header.get_zooms()],
                      "description": "Published defaced T1 derivative; full original grid, no benchmark crop"},
            "benchmark_script_sha256": sha256(__file__),
            "reference_sha256": sha256(reference), "reference_accuracy_mask_sha256": sha256(mask),
            "fnirt_configuration": asdict(resolve_fnirt_config("default")),
            "initial_affine": {"sha256": sha256(affine), "preparation_seconds": preparation_seconds,
                               "preparation_thread_budget": max(budgets), "dof": 12, "cost": "corratio",
                               "program_sha256": sha256(fsl / "bin/flirt"),
                               "scope": "Shared official FLIRT preprocessing, excluded from each FNIRT process time"},
            "repetitions_after_warmup": args.repetitions,
            "inverse_timing_protocol": "single_observation" if args.inverse_single_observation else "warmup_and_paired_repeats",
            "fnirt": summarize_suite(fnirt_suite), "invwarp": summarize_suite(inverse_suite),
            "inverse_contract": "Same official forward coefficient field; FNIT fixed-point stopping and FSL native default stopping differ",
        }
        write_json(output / "report.public.json", report)
        status("executed_with_numeric_comparisons")
    except Exception as error:
        state["error_type"] = type(error).__name__
        status("failed")
        raise


if __name__ == "__main__":
    main()
