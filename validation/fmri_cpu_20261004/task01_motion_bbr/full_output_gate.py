"""Require complete candidate/baseline output equality outside speed timing."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

from benchmark_baseline import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    candidate = plan["candidate_manifest"]
    baseline = plan["baseline_manifest"]
    controller = json.loads(Path(plan["candidate_controller_status"]).read_text())
    if controller["status"] != "completed" or len(controller["jobs"]) != len(candidate["jobs"]):
        raise RuntimeError("Candidate's complete default queue is not finished")
    if any(job.get("exit_code") != 0 for job in controller["jobs"]):
        raise RuntimeError("Candidate queue contains a failed job")
    actual_manifest_path = Path(candidate["task_workspace"]) / "run_manifest.private.json"
    if json.loads(actual_manifest_path.read_text()) != candidate or digest(actual_manifest_path) != controller["manifest_sha256"]:
        raise RuntimeError("Candidate controller and analysis do not bind the same manifest")
    if {job["name"] for job in candidate["jobs"]} != {job["name"] for job in controller["jobs"]}:
        raise RuntimeError("Candidate controller job names differ from the frozen manifest")
    for job in candidate["jobs"]:
        receipt = next(row for row in controller["jobs"] if row["name"] == job["name"])
        expected_command = hashlib.sha256(json.dumps(job["command"], separators=(",", ":")).encode()).hexdigest()
        if receipt["command_sha256"] != expected_command:
            raise RuntimeError("Candidate's executed command differs from its frozen manifest")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.sched_setaffinity(0, {candidate["cpu_group"][0]})
    sys.path.insert(0, plan["analysis_helpers"])
    from analyze_complete import compare_prefixes, publish
    publish(output / "status.safe.json", {"status": "waiting_for_cpu_lease", "pid": os.getpid()})
    records = []
    with Path(candidate["cpu_lock"]).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        import nibabel as nib
        import numpy as np
        inputs = json.loads(Path(plan["inputs"]).read_text())
        publish(output / "status.safe.json", {"status": "checking_complete_outputs", "pid": os.getpid()})
        for job in candidate["jobs"]:
            report = json.loads((Path(job["output_dir"]) / "report.safe.json").read_text())
            old_job = next(row for row in baseline["jobs"] if row["name"] == job["name"])
            old_report = json.loads((Path(old_job["output_dir"]) / "report.safe.json").read_text())
            if report["status"] != "complete" or old_report["status"] != "complete":
                raise RuntimeError("Candidate/baseline API did not finish")
            if report["input_sha256"] != old_report["input_sha256"]:
                raise RuntimeError("Candidate/baseline input hashes differ")
            if report["adapter_sha256"] != candidate["adapter_sha256"] or old_report["adapter_sha256"] != baseline["adapter_sha256"]:
                raise RuntimeError("Candidate/baseline adapters differ from their frozen manifests")
            if report["cpu_threads"] != job["threads"] or report["cpu_affinity"] != [int(value) for value in job["cpu_list"].split(",")]:
                raise RuntimeError("Candidate did not use its assigned CPU thread and affinity budget")
            source = Path(job["command"][job["command"].index("--source") + 1])
            for path, expected in candidate["frozen_source_sha256"].items():
                if digest(source / path) != expected or report["source_sha256"].get(path) != expected:
                    raise RuntimeError("Candidate's actual source differs from its frozen manifest")
            old_prefix = Path(old_job["output_dir"]) / "repeat_0/result"
            mask = inputs[report["case"]].get("brain_mask") if report["function"] != "bbr" else None
            for repeat, current in enumerate(report["repeat_records"]):
                prefix = Path(job["output_dir"]) / f"repeat_{repeat}/result"
                precision = compare_prefixes(prefix, old_prefix, report["function"], mask)
                actual_image = nib.load(str(prefix) + ".nii.gz")
                old_image = nib.load(str(old_prefix) + ".nii.gz")
                differences = [key for key in actual_image.header.keys()
                               if not np.array_equal(actual_image.header[key], old_image.header[key])]
                equality = {
                    "image_values": precision["images"]["whole_grid"]["all_values_bit_equal"],
                    "binary_header": precision["images"]["header_binary_equal"],
                    "matrices": precision["matrices"]["all_values_bit_equal"],
                    "source_unchanged": current["source_unchanged"],
                    "cost_evaluations": current["cost_evaluations"] == old_report["cost_evaluations"],
                }
                if report["function"] == "bbr":
                    equality["boundary_points"] = current["boundary_points"] == old_report["boundary_points"]
                    equality["phase_cost_evaluations"] = current["phase_cost_evaluations"] == old_report["phase_cost_evaluations"]
                    equality["initial_cost"] = current["initial_cost"] == old_report["initial_cost"]
                    equality["final_cost"] = current["final_cost"] == old_report["final_cost"]
                else:
                    equality["parameters"] = precision["parameters"]["all_values_bit_equal"]
                    frames = report["input_shape"][3]
                    matrix_files = sorted(Path(str(prefix) + ".mat").glob("MAT_*"))
                    equality["complete_standard_outputs"] = (
                        actual_image.shape == tuple(report["input_shape"])
                        and [path.name for path in matrix_files] == [f"MAT_{frame:04d}" for frame in range(frames)]
                        and np.loadtxt(str(prefix) + ".par").shape == (frames, 6))
                    equality["normal_matrix_files"] = all(
                        path.read_bytes() == (Path(str(old_prefix) + ".mat") / path.name).read_bytes()
                        for path in matrix_files)
                    equality["normal_parameter_file"] = (
                        Path(str(prefix) + ".par").read_bytes() == Path(str(old_prefix) + ".par").read_bytes())
                    expected_rms = ("_abs.rms", "_rel.rms", "_abs_mean.rms", "_rel_mean.rms") if "--rms" in job["command"] else ()
                    equality["rms"] = all(
                        suffix in precision["rms"] and precision["rms"][suffix]["all_values_bit_equal"]
                        and np.atleast_1d(np.loadtxt(str(prefix) + suffix)).size == (
                            frames if suffix == "_abs.rms" else frames - 1 if suffix == "_rel.rms" else 1)
                        for suffix in expected_rms)
                records.append({"job": job["name"], "function": report["function"], "case": report["case"],
                                "threads": report["cpu_threads"], "repeat": repeat,
                                "comparison": "final_candidate_v4_vs_frozen_baseline",
                                "precision": precision, "equality": equality,
                                "different_header_field_names": differences})
                publish(output / "precision.safe.json", records)
        accepted = len(records) == 10 and all(all(row["equality"].values()) for row in records)
        result = {"status": "completed" if accepted else "failed", "accepted": accepted,
                  "pid": os.getpid(), "full_comparisons": len(records),
                  "all_full_images_bit_equal": all(row["equality"]["image_values"] for row in records),
                  "all_headers_binary_equal": all(row["equality"]["binary_header"] for row in records),
                  "all_matrices_bit_equal": all(row["equality"]["matrices"] for row in records),
                  "all_algorithm_counts_equal": all(row["equality"]["cost_evaluations"] for row in records),
                  "candidate_source_sha256": candidate["frozen_source_sha256"],
                  "scope": "Six complete default CPU1/8 jobs, including first/warm calls; every image value, matrix, parameter and requested RMS, binary header and algorithm count."}
        publish(output / "status.safe.json", result)
        if not accepted:
            raise RuntimeError("Full output gate failed; inspect preserved complete precision records")
    print(json.dumps({"status": result["status"], "accepted": accepted, "comparisons": len(records)}))


if __name__ == "__main__":
    main()
