"""从带原始/公开SHA的完整连续前段收据汇总，不重跑影像算法。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(*, reports: Path, output: Path) -> dict:
    export = json.loads((reports / "PUBLIC_EXPORT_MANIFEST.json").read_bytes())
    for name, entry in export["files"].items():
        if digest(reports / name) != entry["published_sha256"]:
            raise ValueError("published receipt SHA mismatch: " + name)
    stage_rows, error_rows, label_rows, cases = [], [], [], []
    for case in ("sub07", "sub06"):
        compare = json.loads((reports / (case + "-comparison.json")).read_bytes())
        if compare["status"] != "complete_prefix_diagnostic" or compare["full_workflow_complete"]:
            raise ValueError("expected continuous-prefix comparison")
        runs = []
        for fit, side in ((1, "reference"), (4, "candidate")):
            name = case + "-fit" + str(fit)
            producer = json.loads((reports / (name + "-benchmark.json")).read_bytes())
            runtime = json.loads((reports / (name + "-runtime.json")).read_bytes())
            process = json.loads((reports / (name + "-process.json")).read_bytes())
            if (export["files"][name + "-benchmark.json"]["raw_sha256"] != compare[side + "_producer_sha256"]
                    or export["files"][name + "-runtime.json"]["raw_sha256"] != compare[side + "_runtime_sha256"]
                    or producer["execution"] != "complete_prefix" or process["exit_code"] != 0
                    or process["execution_audit"]["blocked_events"] != 0):
                raise ValueError("producer/status/audit/raw SHA mismatch: " + name)
            for stage in runtime["stages"]:
                stage_rows.append({"case": case, "fitting_threads": fit, "stage": stage["name"],
                    "synchronized_stage_seconds": stage["seconds"], "function_seconds": stage["function_seconds"],
                    "cuda_synchronized": stage["cuda_synchronized"]})
            runs.append({"fitting_threads": fit,
                "complete_prefix_API_seconds": producer["public_API_seconds_including_reads_writes"],
                "process_wall_seconds_including_imports_hashing": process["full_process_wall_seconds"],
                "actual_N4_call": producer["actual_N4_caller_parameters"],
                "actual_model_forwards": [stage["actual_forwards"] for stage in runtime["stages"] if stage.get("actual_forwards")],
                "sampler": process["samples"], "execution_audit": process["execution_audit"],
                "load_before": producer["load_average_before"], "load_after": producer["load_average_after"]})
        for name, value in compare["volumes"].items():
            error_rows.append({"case": case, "volume": name, "different_voxels": value.get("different_voxels"),
                "max_abs": value.get("max_abs"), "p99_abs": value.get("p99_abs_error"), "mae": value.get("mae"),
                "affine_max_abs": value["affine_max_abs"], "vox2ras_tkr_max_abs": value["vox2ras_tkr_max_abs"],
                "reference_dtype": value["reference_dtype"], "candidate_dtype": value["candidate_dtype"],
                "foreground_dice": value.get("foreground_dice")})
            for label, row in value.get("per_label", {}).items():
                label_rows.append({"case": case, "volume": name, "label": int(label), **row})
        cases.append({"case": case, "runs": runs,
            "observed_prefix_seconds_saved": runs[0]["complete_prefix_API_seconds"]-runs[1]["complete_prefix_API_seconds"],
            "observed_prefix_reduction_percent": 100*(1-runs[1]["complete_prefix_API_seconds"]/runs[0]["complete_prefix_API_seconds"]),
            "talairach_lta": compare["talairach_lta"], "volumes": compare["volumes"],
            "existing_stage_input_orig_comparison": compare.get("existing_stage_input_orig_comparison"),
            "comparison_diagnostic_seconds": compare["diagnostic_seconds_including_reads"]})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("stage_times.csv", stage_rows), ("volume_errors.csv", error_rows), ("label_dice.csv", label_rows)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    result = {"status": "complete_prefix_diagnostic", "cases": cases,
        "scope": "same raw T1, same GPU5/CPU100-103/four threads, fresh empty outputs, sequential fit1/fit4 continuous prefix",
        "full_workflow_complete": False, "whole_pipeline_speedup": "not_measured",
        "overall_metric_equivalence": "not_assessed", "strict_reproduction": "per-volume/LTA/label differences retained; not replaced by correlation",
        "production_default": "fitting_threads=1 unchanged", "ten_minute_full_workflow_goal": "not_evaluated_by_prefix",
        "memory_limit": "ownership unresolved tree value remains null; target device is a sampled conservative upper bound, not continuous peak",
        "auditing_limit": "Python-startup execution audit, not OS/file/library isolation",
        "timing_limit": "one sequential raw-prefix pair per case on shared host; full stage ABBA separately published",
        "public_export_manifest_sha256": digest(reports / "PUBLIC_EXPORT_MANIFEST.json"),
        "summarizer_sha256": digest(Path(__file__))}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    value = summarize(**vars(parser.parse_args()))
    print(json.dumps({"status": value["status"], "cases": len(value["cases"])}))
