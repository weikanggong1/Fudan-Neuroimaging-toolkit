"""只读收集四组真实回执、同期已验证PID行与公开输入脑图，不执行N4。"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, run = args.fnit_root, args.fnit_root / "runs" / args.task
    execution = run / "execution_shared_v2"
    report_path = execution / "report.private.json"
    report = json.loads(report_path.read_text())
    assert (run / "exit_execution_shared_v2").read_text().strip() == "0"
    assert report["status"] == "completed_four_full200_guarded_n4_regressions" and len(report["rows"]) == 4
    guard = json.loads((run / "guard_contracts_v2/guard_contracts.public.json").read_text())
    assert guard["tests"]["tests"] == 6 and guard["tests"]["failures"] == guard["tests"]["errors"] == 0
    assert guard["module_sha256"]["fnit.recon_all.n4_torch_worker"] == report["module_sha256"]["fnit.recon_all.n4_torch_worker"]
    inputs = json.loads((run / "inputs.private.json").read_text())
    summary = {"status": "verified_four_real_full200_n4_calls", "scope": report["scope"],
        "source_commit": report["source_commit"], "source_manifest_sha256": report["resource_protocol"]["source_manifest_sha256"],
        "wheel_sha256": report["wheel_sha256"], "wheel_bytes": report["wheel_bytes"],
        "private_report_sha256": sha(report_path), "collection_script_sha256": sha(__file__),
        "resource_protocol_sha256": report["resource_protocol_sha256"],
        "module_sha256": report["module_sha256"], "nonexclusive": True,
        "threads": report["threads"], "cpu_affinity": report["cpu_affinity"],
        "GPU_uuid": report["gpu_uuid"], "physical_gpu": report["physical_gpu"],
        "precision_before": report["precision_before"], "precision_after": report["precision_after"],
        "half_requested": False, "whole_reconstruction_memory_20gb": "not_assessed",
        "whole_reconstruction_10min": "not_assessed", "production_default_changed": False,
        "guard_contracts": guard, "rows": []}
    audit = {"scope": "actual four child Python startup records and direct-exec audit; guard activation bound to inherited flag/source plus six installed negative CPU contracts, not syscall sandbox", "workers": []}
    for row in report["rows"]:
        worker = row["worker"]
        assert worker["api"]["iterations"] == 200
        assert worker["input_sha256"] == row["input_sha256"]
        output = execution / row["case"] / row["mode"] / "nu0.mgz"
        assert sha(output) == worker["output_sha256"]
        assert all(x["different_voxels"] == 0 and x["same_affine"] and x["same_dtype"] for x in row["comparisons"].values())
        audit_file = execution / row["case"] / row["mode"] / "execution-audit" / (str(worker["worker_pid"])+".jsonl")
        events = [json.loads(line) for line in audit_file.read_text().splitlines()]
        starts = [x for x in events if x["event"] == "python_start"]
        assert len(starts) == 1 and starts[0]["pid"] == worker["worker_pid"]
        parent_pid = starts[0]["parent_pid"]
        assert not any(x.get("blocked") for x in events)
        audit["workers"].append({"case": row["case"], "mode": row["mode"],
            "worker_pid": worker["worker_pid"], "parent_pid": parent_pid, "startup_count": 1,
            "direct_exec_event_count": sum(x["event"] != "python_start" for x in events),
            "blocked_attempts": 0, "audit_file_sha256": sha(audit_file)})
        samples = row["monitoring"]["sampler_report"]["samples"]
        child_values, pair_values = [], []
        for sample in samples:
            entries = {x["local_pid"]:x for x in sample["processes"]}
            if worker["worker_pid"] in entries:
                child_values.append(entries[worker["worker_pid"]]["bytes"])
            if parent_pid in entries and worker["worker_pid"] in entries:
                parent, child = entries[parent_pid], entries[worker["worker_pid"]]
                pair_values.append({"monotonic": sample["monotonic"], "parent": parent, "child": child,
                                    "same_snapshot_pair_bytes": parent["bytes"]+child["bytes"]})
        m = row["monitoring"]
        if row["mode"] == "initialized_api":
            assert pair_values and row["parent_stats"]["retained_real_input_bytes"] == 67108864
            assert worker["parent_cuda_initialized_before_exec"] is True
        assert child_values
        summary["rows"].append({"case": row["case"], "mode": row["mode"],
            "input_sha256": row["input_sha256"], "output_sha256": worker["output_sha256"],
            "iterations": 200, "comparisons": row["comparisons"],
            "wall_seconds_including_import_exec_transfer_io_exit": row["wall_seconds_including_import_exec_transfer_io_exit"],
            "worker_file_api_seconds": worker["api"]["total_seconds"],
            "parent_initialization_seconds_outside_api_wall": row["parent_initialization_seconds_outside_api_wall"],
            "parent_stats": row["parent_stats"], "child_stats_valid": worker["cuda_allocator"]["torch_stats_known_valid"],
            "child_allocated_peak_bytes": worker["allocated_peak_bytes"], "child_reserved_peak_bytes": worker["reserved_peak_bytes"],
            "sample_count": len(samples), "unknown_total_sample_count": sum(x["tree_total_bytes"] is None for x in samples),
            "failed_current_frame_count": sum(x["current_ownership"] == "unavailable" for x in m["frames"]),
            "sampling_interval_seconds": m["sampling_interval_seconds"], "max_observed_interval_seconds": m["max_observed_interval_seconds"],
            "verified_child_window_max_bytes": max(child_values), "verified_parent_child_pair_sample_count": len(pair_values),
            "verified_parent_child_window_max_bytes": max((x["same_snapshot_pair_bytes"] for x in pair_values),default=None),
            "verified_parent_child_pairs": pair_values, "full_sequence_tree_peak_bytes": m["full_sequence_peak_bytes"],
            "memory_scope": "only identity-verified owned rows summed within one sampler snapshot; resident foreign process remains unresolved; not whole-tree/continuous peak", 
            "historical_to_native": row["historical_to_native"],
            "historical_reference_scope": row["historical_reference_scope"]})
    assert summary["precision_before"] == summary["precision_after"]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    # 原始typed快照保留，仅把私有根路径替换为可识别占位符。
    report_text = json.dumps(report,indent=2).replace(str(root),"FNIT_ROOT")
    (args.output/"report.public.json").write_text(report_text+"\n")
    (args.output/"summary.public.json").write_text(json.dumps(summary,indent=2)+"\n")
    (args.output/"execution_audit.public.json").write_text(json.dumps(audit,indent=2)+"\n")
    (args.output/"guard_contracts.public.json").write_text(json.dumps(guard,indent=2)+"\n")
    for name in ["resource_protocol_v2.private.json","resource_gate_v2.private.json","WAIT_empty_gpu_v1.private.json"]:
        text = (run/name).read_text().replace(str(root),"FNIT_ROOT")
        (args.output/name.replace(".private.",".public.")).write_text(text)
    figure, axes = plt.subplots(2,3,figsize=(10,7),layout="constrained")
    for i, case in enumerate(inputs["cases"]):
        original = np.asarray(nib.as_closest_canonical(nib.load(case["input"])).dataobj)
        result = np.asarray(nib.as_closest_canonical(nib.load(str(execution/case["case"]/"initialized_api/nu0.mgz"))).dataobj)
        old = np.asarray(nib.as_closest_canonical(nib.load(case["old_workers"][1]["output"])).dataobj)
        assert np.array_equal(result,old)
        z = original.shape[2]//2
        axes[i,0].imshow(original[:,:,z].T,origin="lower",cmap="gray",vmin=0,vmax=180)
        axes[i,1].imshow(result[:,:,z].T,origin="lower",cmap="gray",vmin=0,vmax=180)
        axes[i,2].imshow((result.astype(float)-old)[:,:,z].T,origin="lower",cmap="coolwarm",vmin=-1,vmax=1)
        for ax in axes[i]:ax.axis("off")
        axes[i,0].set_title(case["case"]+" | FNIT orig")
        axes[i,1].set_title("Guarded N4 | 200 iterations")
        axes[i,2].set_title("New minus old worker | all voxels = 0")
    figure.savefig(args.output/"n4_real_comparison.public.png",dpi=150)
    plt.close(figure)
    manifest={"scope":"typed real-run proofs and derived brain figure only; no source images/weights", "files":{}}
    for path in sorted(args.output.iterdir()):
        manifest["files"][path.name]={"bytes":path.stat().st_size,"sha256":sha(path)}
    (args.output/"PUBLIC_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"status":summary["status"],"rows":len(summary["rows"]),
        "same_snapshot_pair_counts":[x["verified_parent_child_pair_sample_count"] for x in summary["rows"]],
        "guard_tests_passed":6,"figure_sha256":sha(args.output/"n4_real_comparison.public.png")}))


if __name__ == "__main__":
    main()
