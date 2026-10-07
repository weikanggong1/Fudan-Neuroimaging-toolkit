"""Create a local private plan from already verified persisted metadata.

No network, MRI loading, registration, sampler or optimizer is executed.
The live observation is a prior read-only SHA check, not a numerical run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from common import digest, save_exclusive, validate_plan


def value_bytes(path):
    path = Path(path)
    return {"bytes": int(path.stat().st_size), "sha256": digest(path)}


def prepare(old_path, score_path, observation_path, output_directory):
    original_sha = "05a45563f703420f2d70a22e5cf551a3c0e263c28bb94543e875a003ea884bf8"
    score_sha = "c06d652d5f851cb9c6bcf91af4ccce6b40a325745e96ee661b03069d3e03ca8a"
    if digest(old_path) != original_sha or digest(score_path) != score_sha:
        raise ValueError("the exact previously reviewed PLAN and score are required")
    old = json.loads(Path(old_path).read_text())
    score = json.loads(Path(score_path).read_text())
    observation = json.loads(Path(observation_path).read_text())
    actual = {key: {name: item[name] for name in ("bytes", "sha256")}
              for key, item in old["bindings"].items()}
    actual_sha = hashlib.sha256(json.dumps(actual, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
    if (observation["original_518_bindings_checked"] != 518
            or observation["original_518_wrong_keys"]
            or observation["actual_binding_values_SHA256"] != actual_sha
            or observation["original_plan"] != value_bytes(old_path)):
        raise ValueError("the live read-only source/input/reference observation differs")
    canonical = Path(old["workspace_directory"]).parents[4]
    if canonical.name != "FNIT":
        raise ValueError("the existing canonical FNIT root was not found")
    relative = Path("smri_cpu_20261004/remaining_20261006/robust-support-points-20261006-v1")
    workspace, run = canonical / "workspaces" / relative, canonical / "runs" / relative
    expected = {mode: {key: score["result"]["stages"][mode][key]
                       for key in ("warp_relative_L2", "warp_nonzero_support_difference_voxels")}
                for mode in ("rigid", "affine")}
    binding = {key: dict(item) for key, item in old["bindings"].items()}
    binding["original_inverse_real_PLAN"] = {
        "path": str(Path(old["workspace_directory"]) / "PLAN.private.json"),
        **value_bytes(old_path),
    }
    binding.update({key: dict(item) for key, item in observation["extras"].items()})
    source = Path(__file__).parent
    harness = {name: {"path": str(workspace / name), **value_bytes(source / name)}
               for name in ("common.py", "inspect_saved_points.py", "run_prepared.py")}
    formal = dict(old["gates"])
    formal.update(warped_nonzero_support_difference_voxels=0,
                  total=20, persisted_passed=17, persisted_failed=3)
    plan = {
        "schema": 1, "status": "prepared_not_run",
        "scope": "four saved-geometry baseline resamples, then at most four original single-point linear inspections",
        "canonical_root": str(canonical), "workspace_directory": str(workspace),
        "run_directory": str(run), "source_directory": old["source_directory"],
        "legacy_candidate_directory": old["legacy_candidate_directory"],
        "moving": old["moving"], "fixed": old["fixed"],
        "official_directory": old["official_directory"],
        "saved_output_directory": str(Path(old["run_directory"]) / "stages"),
        "baseline_score": observation["extras"]["baseline_score"]["path"],
        "python": old["python"], "worker": str(workspace / "inspect_saved_points.py"),
        "physical_cores": [int(value) for value in old["physical_cores"]],
        "common_CPU_lock": old["common_CPU_lock"],
        "parameters": {"spatial_chunk_size": 131072}, "expected_target_shape": [39, 45, 56],
        "expected_baseline": expected, "unchanged_formal_gates": formal,
        "limits": {"lock_wait_seconds": 120, "child_seconds": 120, "outer_seconds": 300,
                   "terminate_wait_seconds": 2, "kill_wait_seconds": 2,
                   "address_space_bytes": 20000000000, "scientific_attempts": 1},
        "observed_canonical_metadata": observation["canonical_meta"],
        "observed_canonical_repo_HEAD": observation["canonical_repo_HEAD"],
        "metadata_observation_only_not_runtime_lock": True,
        "original_binding_values_SHA256": actual_sha,
        "read_only_observation": value_bytes(observation_path),
        "data_origin": old["data_origin"],
        "affine_inherits_own_different_rigid_MGH": True,
        "coordinate_value_privacy": "point indices, pull coordinates/matrices and all atlas samples only in mode600 private worker report; public summary only classification/counts",
        "first_failure_policy": "stop without retry or next numerical branch; retain failure receipt",
        "existing_outputs_old_sources_and_original_23_public_files_modified": False,
        "new_registration_official_GEMS_GPU_calls": 0,
        "bindings": binding, "harness_bindings": harness,
    }
    validate_plan(plan)
    output = Path(output_directory)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.umask(0o077)
    save_exclusive(output / "PLAN.private.json", plan)
    private = value_bytes(output / "PLAN.private.json")
    public = {
        "schema": 1, "status": "prepared_not_run", "private_PLAN": private,
        "source_commit_basis": "2715a5e3d02730459cb443cf9808f8522f5049b0",
        "observed_canonical_repo_HEAD": observation["canonical_repo_HEAD"],
        "canonical_metadata_read_SHA256": observation["canonical_meta"],
        "canonical_index_semantics": "preparation-time observation; shared index can advance normally before execution",
        "previous_518_actual_source_input_reference_bindings_exact": True,
        "previous_518_binding_values_SHA256": actual_sha,
        "fixed_binding_count": len(binding),
        "fixed_harness_count": len(harness),
        "original_PLAN_SHA256": original_sha, "baseline_score_SHA256": score_sha,
        "expected_baseline": expected, "unchanged_formal_gates": formal,
        "scope": plan["scope"], "limits": plan["limits"],
        "CPU_threads": 8, "GPU_hidden": True, "no_registration_or_official_command": True,
        "actual_numerical_jobs": 0, "actual_uploads": 0,
        "single_point_evaluation_bound": 4,
        "affine_inherits_own_different_rigid_MGH": True,
        "public_point_coordinates_atlas_samples_or_MRI_arrays": False,
        "frozen_harness": {name: {key: value[key] for key in ("bytes", "sha256")}
                           for name, value in harness.items()},
    }
    save_exclusive(output / "PLAN.public.json", public)
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-plan", type=Path, required=True)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--live-observation", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.old_plan, args.baseline_score, args.live_observation,
                     args.output_directory)
    print(json.dumps({"status": result["status"], "private_PLAN": result["private_PLAN"],
                      "fixed_binding_count": result["fixed_binding_count"]}, indent=2))


if __name__ == "__main__":
    main()
