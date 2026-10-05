"""Re-extract saved scalar timings and hashes; does not import Torch or run MRI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_report(relative: str):
    path = REPOSITORY / relative
    return json.loads(path.read_text()), {
        "repository_relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main():
    memory, memory_binding = read_report("validation/smri_cpu/seg_memory_20261005/CPU_FULL.public.json")
    precision, precision_binding = read_report("validation/smri_cpu/seg_tf32_20261005/CPU_FULL.public.json")
    native, native_binding = read_report(
        "validation/smri_cpu/seg_memory_20261005/OFFICIAL_NODE7_FAST_BA_REPEAT.public.json")
    join, join_binding = read_report("validation/smri_cpu/seg_memory_20261005/CPU_JOIN_STAGE.public.json")
    reduction, reduction_binding = read_report(
        "validation/smri_cpu_20261004/t2_seg/reduction_diagnostic.public.json")
    sources = {
        path.name: sha256(path)
        for path in sorted((REPOSITORY / "src/fnit/synthseg_parc").glob("*.py"))
    }
    latest_workers = [worker for worker in precision["jobs"].values() if worker["arm"] == "candidate"]
    for worker in latest_workers:
        assert worker["source_files"] == sources
    input_sha = native["source_binding"]["input_sha256"]
    records = []
    for job in memory["jobs"]:
        worker = job["worker"]
        assert worker["input_sha256"] == input_sha
        api = worker["api_seconds"]
        row = {
            "report": memory_binding["repository_relative_path"],
            "name": job["name"], "arm": job["arm"], "mode": job["mode"],
            "outer_cold_seconds": job["wall_seconds"],
            **{key: worker[key] for key in (
                "preflight_seconds", "construct_seconds", "api_seconds", "save_seconds",
                "worker_seconds", "maximum_rss_kib", "cpu_join_calls")},
            "source_files": worker["source_files"],
            "outer_minus_api_seconds": job["wall_seconds"] - api,
            "exclusive_outer_other_seconds": job["wall_seconds"] - api
                - worker["preflight_seconds"] - worker["construct_seconds"] - worker["save_seconds"],
            "api_fraction_of_outer": api / job["wall_seconds"],
            "api_internal_stage_shares": "not_measured_in_this_current_full_record",
        }
        records.append(row)
    queue_rows = {row["name"]: row for row in precision["queue"]["jobs"]}
    precision_records = []
    for name, worker in precision["jobs"].items():
        assert worker["input_sha256"] == input_sha
        precision_records.append({
            "report": precision_binding["repository_relative_path"], "name": name,
            "arm": worker["arm"], "mode": worker["mode"],
            "outer_cold_seconds": queue_rows[name]["wall_seconds"],
            **{key: worker[key] for key in (
                "preflight_seconds", "constructor_seconds", "api_seconds", "save_seconds",
                "worker_seconds", "maximum_rss_kib")},
            "load_after": queue_rows[name]["load_after"],
            "api_internal_stage_shares": "not_measured_in_this_current_full_record",
        })
    native_records = []
    for mode, row in native["official_modes"].items():
        native_records.append({
            "mode": mode, "wall_seconds": row["wall_seconds"],
            "record_sha256": row["record_sha256"],
            "used_for_stable_speed_claim": False,
            "interpretation": "preserved_outlier_excluded" if mode == "seg33" else "single_cold_observation",
        })
    repeat = native["seg33_declared_native_repeat"]
    native_records.append({"mode": "seg33_repeat", "wall_seconds": repeat["wall_seconds"],
                           "record_sha256": repeat["record_sha256"],
                           "interpretation": "normal_single_cold_observation_not_a_replicate_median"})
    legacy_profiles = [
        {
            "fnit_relative_path": "runs/smri_cpu_20261004/t2_seg/profiles/t2_synthseg_case01_profile_initial.json",
            "sha256": "86dfb40200277e97941b71c043166d81a5f5cc5ff605a5da345327c6521dddb6",
            "read_only_remote_record": True, "hostname": "nodecw10", "input_alias": "case01",
            "input_sha256": "afd1a20fe75fdea44313f0eda05020b916c87234e7a2045f7ccc6bb7c6e90b19",
            "network_shape": [192, 224, 224], "api_seconds": 147.199004,
            "module_conv_inclusive_seconds": 96.123398,
            "preprocess_seconds": 9.651901, "postprocess_seconds": 15.087281,
            "lcc_seconds_nested_in_postprocess": 11.862757, "soft_volumes_seconds": 0.516414,
            "top_module_conv_seconds": {"up.3.conv0": 47.214438, "up.3.conv1": 12.324698,
                                        "down.0.conv1": 11.538015, "up.2.conv0": 8.589036},
        },
        {
            "fnit_relative_path": "runs/smri_cpu_20261004/t2_seg/profiles/t2_parc_case01_profile_initial.json",
            "sha256": "378ea0760ac78bf4e423999908842893d6e729f3d320969ff08a1513ea89f574",
            "read_only_remote_record": True, "hostname": "nodecw10", "input_alias": "case01",
            "input_sha256": "afd1a20fe75fdea44313f0eda05020b916c87234e7a2045f7ccc6bb7c6e90b19",
            "network_shape": [192, 224, 224], "fast": True, "api_seconds": 59.238149,
            "module_conv_inclusive_seconds": 23.993880,
            "preprocess_seconds": 10.103496, "postprocess_seconds": 3.619563,
            "lcc_seconds_nested_in_postprocess": 1.748383,
        },
    ]
    audit = {
        "schema": "fnit_synthseg_cpu_hotspot_readonly_audit/v1", "date": "2026-10-06",
        "scope": "read-only source and saved real timing audit; no new inference or benchmark",
        "audit_worktree_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "root_integrated_commit_observed": "a5a21bdc76ea4c4581b97e3ef85dd2c60fea07f3",
        "accepted_tf32_source_commit": "46eead65807265395982e6968b0ce48c750c1672",
        "current_source_files": sources,
        "current_sources_match_complete_precision_candidate_workers": True,
        "input": {"dataset": "OpenNeuro ds003138 v1.0.1", "license": "CC0",
                  "alias": "case02", "sha256": input_sha, "network_shape": [192, 224, 256]},
        "evidence": [memory_binding, precision_binding, native_binding, join_binding, reduction_binding],
        "cpu_full_memory_records": records, "cpu_full_latest_precision_records": precision_records,
        "native_records": native_records, "native_source_binding": native["source_binding"],
        "native_backend_operator_trace": "not_available_in_saved_formal_reports",
        "timing_boundary_limitations": native["timing_limitations"],
        "partial_join_stage": {"scope": join["scope"], "capture": join["capture"],
                               "summary": join["summary"], "not_complete_inference": True},
        "rejected_real_first_two_layer_diagnostic": reduction["first_two_real_layers"],
        "legacy_profiles": legacy_profiles,
        "legacy_profile_limitations": [
            "Different case, network grid, source version and host from current complete timing",
            "Module hooks would disqualify the current no-hooks CPU join optimization",
            "External F.conv3d Gaussian blur is not in Module convolution totals",
            "LCC is nested in postprocess; do not add twice",
            "Plus child loading was done before the old API clock",
            "Old LCC graph timing predates the current CPU scipy implementation",
        ],
        "unmeasured_current_api_stages": [
            "T1 input read/resample/orient/normalize", "Plus lazy model load",
            "original CNN", "flipped CNN", "33-channel blur twice", "69-channel blur",
            "postprocess and nested LCC", "tie argmax", "soft volumes",
            "slab pad/conv/output copy exclusive fractions",
        ],
        "next_minimum_profile": {
            "status": "proposed_not_executed_in_this_report", "full_fnit_arms": 1,
            "mode": "ordinary33", "fresh_process": True, "threads": 8,
            "affinity": "32,36,40,44,48,52,56,60", "common_cpu_lock": True,
            "cuda_visible_devices": "", "official_arms": 0,
            "observation": "private Python function/forward timers; no Module hooks; preserve existing arithmetic",
            "required_gates": ["source/weight/input/config hashes", "8 actual CPU join calls",
                               "complete maps/headers/CSV numerical and saved-file SHA equality",
                               "source unchanged", "exclusive versus nested clocks", "RSS and instrumentation overhead"],
            "performance_claim": "diagnostic only; do not replace formal uninstrumented wall times",
        },
        "cpu_official_speed_goal_met": False, "new_production_change": False,
        "new_mri_execution": False, "new_gpu_execution": False,
    }
    (HERE / "AUDIT.public.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
