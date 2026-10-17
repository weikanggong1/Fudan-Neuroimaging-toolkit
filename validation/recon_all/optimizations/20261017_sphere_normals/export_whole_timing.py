"""从四次已有整例的 SHA 绑定元数据提取公开计时账，不执行影像计算。

输入为私有 evidence JSON，结构和具名调用见 TIMING_METHOD.md。只接受本轮
两例 A/B、B/A 的完整契约；读取/哈希/结构错误抛异常，不输出部分成功。
输出为普通字典及新建 JSON/CSV；时间单位为秒，无坐标或影像数据输出。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


STAGE_NAMES = frozenset("""input_talairach n4 nu T1_normalize brainmask SynthSeg
mri_em_register mri_ca_normalize mri_cc brain_second_normalize entowm ants_denoise
mri_segment mri_edit_wm_with_aseg wm_pretess wm_fix_ento wm_fix_acj mri_fill
filled_auto_checkpoint mni_aux brain_finalsurfs surface_hemisphere_group defects_lh
defects_rh register_hemisphere_group annotation_hemisphere_group
finish_surface_hemisphere_group finish_metrics_lh finish_metrics_rh exvivo_annotations
jacobian_lh contrast_lh jacobian_rh contrast_rh ribbon relabel_hypointensities
aseg_ribbon_fix project_aparc_volumes project_wmparc brain_volume_stats aseg_stats
wmparc_stats stats_lh_aparc stats_lh_aparc.a2009s stats_lh_aparc.DKTatlas
stats_lh_aparc.pial stats_lh_BA_exvivo stats_lh_BA_exvivo.thresh stats_lh_w-g.pct
stats_lh_curv stats_rh_aparc stats_rh_aparc.a2009s stats_rh_aparc.DKTatlas
stats_rh_aparc.pial stats_rh_BA_exvivo stats_rh_BA_exvivo.thresh stats_rh_w-g.pct
stats_rh_curv mni_mesh_parallel""".split())
GROUP_NAMES = ("volume_prefix", "surface_group", "register_group", "annotation_group",
               "finish_surface_group", "stats", "mni_mesh_parallel", "other_serial_stages",
               "CLI_residual")
NAMED_GROUPS = {"surface_hemisphere_group": "surface_group",
    "register_hemisphere_group": "register_group", "annotation_hemisphere_group": "annotation_group",
    "finish_surface_hemisphere_group": "finish_surface_group", "mni_mesh_parallel": "mni_mesh_parallel"}
FLAGS = ("backend", "topology_backend", "topology_finish_backend", "topology_distance_backend",
    "topology_state_backend", "defects_backend", "sphere_normals_backend", "wm_edit_backend",
    "wm_backend", "n4_backend", "n4_execution", "input_execution", "normalization_controls_backend",
    "normalization_initial_bias_backend", "inflate_backend", "sphere_finish_backend",
    "surface_placement_backend", "annotation_gibbs_backend", "remesh_scalar_storage", "mni_execution",
    "wm_execution", "gca_inverse_backend", "gca_candidate_chunk", "gca_reuse_atlas_context",
    "gca_execution", "hemisphere_execution", "fill_backend")
FLAG_VALUES = frozenset(("native", "conda-cpp", "serial", "legacy", "torch", "numba", "python",
    "torch-optimized", "simpleitk", "in-process", "numpy", "parallel-late", "isolated", "groups"))
CHANGED_FILES = frozenset(("recon_all/native_free.py", "recon_all/mris_register_smoothwm.py",
    "recon_all/mris_register_smoothwm_run.py", "recon_all/mris_register_line_search.py",
    "recon_all/mris_register_average_numba.py"))
BUDGET_FIELDS = ("host", "cpu", "cpu_affinity", "threads", "device", "environment",
                 "resource_sha256", "native_program_sha256", "n4_runtime_dependencies")
CLOCKS = ("CLI", "full_harness", "API", "supervisor_case_outer", "supervisor_API", "pair_supervisor_outer")
CLOCK_SCOPES = {
    "CLI": "benchmark child interpreter startup through exit, including validation, model loads, transfers, compute and IO",
    "full_harness": "benchmark main entry including parsing, preflight/hash checks and prior report writes; excludes pre-main imports and final metadata publication",
    "API": "reconstruction API entry through validation, thread restoration and output IO; excludes CLI startup and final public metadata publication",
    "supervisor_case_outer": "supervisor Popen of benchmark interpreter through its exit and final publication",
    "supervisor_API": "single-case supervisor run_cases including input validation and prior report writes; excludes interpreter imports and final publication",
    "pair_supervisor_outer": "pair driver Popen of supervisor interpreter through its exit, including imports and final publication"}
FP32_EXCEPTIONS = ["SynthStrip", "SynthSeg", "Talairach affine", "MNI nonlinear CUDA",
                   "MNI152 affine matmul", "EntoWM/MCA-dura/vsinus cuDNN"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA-256")
    return value


def seconds(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and value >= 0, "invalid nonnegative timing or count")
    return value


def load_ref(record):
    require(isinstance(record, dict), "missing evidence binding")
    expected = digest(record.get("sha256"))
    require(sha(record["path"]) == expected, "evidence SHA mismatch")
    value = json.loads(Path(record["path"]).read_text())
    require(isinstance(value, dict), "evidence must be a JSON object")
    return value


def mapping_summary(value):
    require(isinstance(value, dict) and value, "missing hash mapping")
    for item in value.values():
        digest(item)
    return {"entries": len(value), "canonical_mapping_sha256": canonical_sha(value)}


def policy(benchmark, runtime):
    flags = {name: benchmark[name] for name in FLAGS}
    for name, value in flags.items():
        require((name == "gca_candidate_chunk" and type(value) is int and value == 4096) or
                (name == "gca_reuse_atlas_context" and value is True) or
                (isinstance(value, str) and value in FLAG_VALUES), "unknown strategy flag")
    precision = runtime["precision"]
    require(precision["matmul_tf32_default"] is True and precision["cudnn_tf32_default"] is True
            and precision["fp16_or_bf16_enabled"] is False and
            precision["fp32_exceptions"] == FP32_EXCEPTIONS, "precision policy changed")
    require(runtime["native_optimizations"]["mode"] == "torch", "native optimization mode changed")
    worker_groups = []
    for group in runtime["hemisphere_scheduling"]["groups"]:
        require(group["operation"] in ("surface", "register", "annotation", "finish_surface"), "unknown worker group")
        require(group["status"] == "complete" and group["workers_requested"] == 2 and
                group["worker_threads"] == 2 and group["total_thread_budget"] == 4, "worker budget changed")
        expected_allocator = "enabled" if group["operation"] == "surface" else "disabled"
        require(group["worker_allocator_selection"]["selected_policy"] == expected_allocator, "allocator policy changed")
        require(set(group["workers"]) == {"lh", "rh"}, "missing hemisphere worker")
        for worker in group["workers"].values():
            budget = worker["thread_budget"]
            require(worker["status"] == "complete" and worker["logical_device"] == "cuda:0" and
                    budget["torch"]["effective"] == budget["numba"]["effective"] == 2 and
                    budget["torch"]["interop_unchanged"] == 1, "actual worker resources changed")
            actual = worker["precision_actual_at_entry"]
            require(actual["matmul_tf32"] is True and actual["cudnn_tf32"] is True and
                    all(actual["autocast"][device]["enabled"] is False for device in ("cpu", "cuda")),
                    "actual worker precision changed")
        worker_groups.append({"operation": group["operation"], "workers": 2, "threads_per_worker": 2,
                              "interop_threads_per_worker": 1, "allocator_policy": expected_allocator})
    require(len(worker_groups) == 4, "missing worker groups")
    return {"flags": flags, "native_optimizations": "torch", "TF32_default": True, "FP16_BF16_enabled": False,
            "validated_stage_FP32_exceptions_preserved": FP32_EXCEPTIONS,
            "worker_groups": worker_groups}


def stage_account(benchmark):
    stages, groups = [], {name: [] for name in GROUP_NAMES[:-1]}
    seen_surface = False
    for stage in benchmark["stages"]:
        name = stage["name"]
        require(name in STAGE_NAMES and stage["profiling_nesting_depth"] == 0 and
                stage["cuda_synchronized"] is True, "unsupported or nested parent stage")
        if name == "surface_hemisphere_group":
            seen_surface = True
        group = NAMED_GROUPS.get(name, "stats" if name.startswith("stats_") else
                                 "other_serial_stages" if seen_surface else "volume_prefix")
        value = seconds(stage["seconds"])
        stages.append({"stage": name, "group": group, "seconds": value})
        groups[group].append(value)
    require(len(stages) == len(STAGE_NAMES) and {x["stage"] for x in stages} == STAGE_NAMES,
            "parent stage coverage changed")
    totals = {name: math.fsum(values) for name, values in groups.items()}
    totals["CLI_residual"] = benchmark["cli_wall_seconds"] - math.fsum(x["seconds"] for x in stages)
    require(totals["CLI_residual"] >= 0, "stages exceed CLI clock")
    return stages, totals


def extract(evidence):
    """校验私有输入和四次运行，返回仅含白名单元数据的公开 JSON 字典。"""
    require(evidence.get("schema") == "fnit-whole-timing-inputs-v1", "unknown evidence schema")
    pair = load_ref(evidence["pair_report"])
    require(pair.get("status") == "complete" and pair.get("scope") ==
            "two-raw-T1-four-empty-directory-paired-whole-runs", "pair is incomplete")
    require(len(pair["runs"]) == len(evidence["runs"]) == 4, "four runs required")
    require([(x["case_index"], x["arm"]) for x in pair["runs"]] ==
            [(0, "baseline"), (0, "candidate"), (1, "candidate"), (1, "baseline")], "run order changed")
    source_integrity = pair["integrity_after_runs"]["source_sha256_unchanged"]
    input_integrity = pair["integrity_after_runs"]["raw_inputs_unchanged"]
    require(set(source_integrity) == {"baseline", "candidate"} and len(input_integrity) == 2 and
            all(value is True for value in [*source_integrity.values(), *input_integrity.values()]),
            "post-run integrity failed")
    require(pair["cache_scope"] == "empty per arm at pair start; reused within that arm across both subjects" and
            pair["candidate_only_benchmark_arguments"] == ["--sphere-normals-backend", "torch"],
            "cache or candidate argument contract changed")
    records, benchmarks, runtimes, seen_arm = [], [], [], set()
    source_summaries = {}
    for index, (row, bound) in enumerate(zip(pair["runs"], evidence["runs"])):
        require(bound["execution_index"] == index, "evidence order mismatch")
        benchmark, runtime, supervisor = (load_ref(bound[key]) for key in ("benchmark", "runtime", "supervisor"))
        require(row["benchmark_sha256"] == bound["benchmark"]["sha256"] and
                row["supervisor_report_sha256"] == bound["supervisor"]["sha256"] and
                row["supervisor_case"]["runtime_report_sha256"] == bound["runtime"]["sha256"],
                "controller/report binding mismatch")
        require(len(supervisor["cases"]) == 1 and supervisor["status"] == "complete", "supervisor incomplete")
        case = supervisor["cases"][0]
        require(case["benchmark_sha256"] == bound["benchmark"]["sha256"] and
                case["runtime_report_sha256"] == bound["runtime"]["sha256"], "supervisor/report binding mismatch")
        arm = row["arm"]
        require(benchmark["sphere_normals_backend"] == ("numba" if arm == "baseline" else "torch"),
                "sphere normals arm mismatch")
        source = pair["sources"][arm]
        require(re.fullmatch(r"[0-9a-f]{40}", source["version"]) and
                source["version"] == row["code_version"] == benchmark["code_version"] == supervisor["code_version"],
                "source version mismatch")
        require(row["input_sha256"] == benchmark["input_sha256"] == case["input_sha256"] ==
                pair["cases"][row["case_index"]]["sha256"], "input mismatch")
        require(row["exit_code"] == benchmark["exit_code"] == case["exit_code"] == 0 and
                benchmark["execution"] == benchmark["runtime_status"] == runtime["status"] == "complete",
                "scientific execution incomplete")
        completeness = benchmark["output_completeness"]
        require(completeness["expected"] == completeness["present"] == 138 and not completeness["missing"] and
                completeness["status"] == benchmark["grid_quality"]["status"] == "passed", "output or grid gate failed")
        require(benchmark["threads"] == 4 and benchmark["device"] == "cuda:0" and
                len(benchmark["cpu_affinity"]) == 4, "CPU/GPU budget changed")
        require(runtime["gca_registration"]["stage_result"]["gpu_name"] == "NVIDIA H100 PCIe" and
                runtime["white_matter_chain"]["segmentation_runtime"]["gpu_name"] == "NVIDIA H100 PCIe",
                "unsupported hardware family")
        files = benchmark["source_sha256"]
        require(len(files) == 639 and all(source["sha256"].get("src/fnit/" + name) == value
                for name, value in files.items()), "639-source mapping binding failed")
        source_summary = {"commit": source["version"], "benchmark_FNIT_source_mapping": mapping_summary(files),
                          "pair_preflight_source_mapping": mapping_summary(source["sha256"]),
                          "benchmark_mapping_matches_pair_preflight": True}
        manifest_ref = evidence.get("source_manifests", {}).get(arm)
        if manifest_ref:
            manifest = load_ref(manifest_ref)
            require(manifest["git_commit"] == source["version"] and all(manifest["files"].get("src/fnit/" + key) == value
                    for key, value in files.items()), "archive/source mapping mismatch")
            source_summary["provided_archive_manifest"] = {"file_sha256": manifest_ref["sha256"],
                "mapping": mapping_summary(manifest["files"]), "639_mapping_matches": True}
        else:
            source_summary["provided_archive_manifest"] = "not_provided"
        if arm in source_summaries:
            require(source_summaries[arm] == source_summary, "same-arm source changed")
        source_summaries[arm] = source_summary
        stages, groups = stage_account(benchmark)
        times = {"CLI": seconds(benchmark["cli_wall_seconds"]),
            "full_harness": seconds(benchmark["full_harness_seconds"]), "API": seconds(benchmark["api_wall_seconds"]),
            "supervisor_case_outer": seconds(case["outer_case_wall_seconds"]),
            "supervisor_API": seconds(supervisor["supervisor_wall_seconds"]),
            "pair_supervisor_outer": seconds(row["outer_supervisor_wall_seconds"])}
        require(times["CLI"] == row["benchmark_cli_wall_seconds"] == case["cli_wall_seconds"] and
                times["full_harness"] == row["benchmark_full_harness_seconds"] == case["full_harness_seconds"] and
                times["supervisor_case_outer"] == row["supervisor_case_outer_wall_seconds"], "timing copy mismatch")
        memory = benchmark["process_memory"]
        failures = memory.get("failed_samples")
        require(isinstance(failures, list), "unknown sampler failure schema")
        require(memory["status"] == "ownership_unresolved" and memory["peak_tree_total_bytes"] is None,
                "unexpected owned-memory evidence; review adapter")
        record = {"execution_index": index, "case_index": row["case_index"], "arm": arm,
            "source_commit": source["version"], "input_sha256": digest(row["input_sha256"]),
            "report_sha256": {key: bound[key]["sha256"] for key in ("benchmark", "runtime", "supervisor")},
            "scientific_exit_code": 0, "execution": "complete", "outputs": {"expected": 138, "present": 138},
            "production_grid_gate": "passed", "seconds": times, "nonoverlapping_parent_groups_seconds": groups,
            "parent_stages": stages, "strategy": policy(benchmark, runtime),
            "cache_order": {"arm_case_ordinal": 2 if arm in seen_arm else 1,
                "arm_cache_reused_from_previous_case": arm in seen_arm,
                "cache_scope": "empty per arm at pair start; reused within arm across two cases"},
            "memory": {"ownership_status": "ownership_unresolved", "owned_simultaneous_peak_bytes": None,
                "target_device_sampled_peak_bytes_including_other_jobs": seconds(memory["peak_target_device_used_bytes"]),
                "target_compute_process_sum_sampled_peak_bytes_including_other_jobs": seconds(memory["peak_target_compute_process_sum_bytes"]),
                "sampling_interval_seconds": seconds(memory["sampling_interval_seconds"]),
                "maximum_observed_interval_seconds": seconds(memory["max_observed_interval_seconds"]),
                "failed_samples": len(failures), "20GB_owned_memory_acceptance": "not_assessed"}}
        records.append(record); benchmarks.append(benchmark); runtimes.append(runtime); seen_arm.add(arm)
    checks = {name: all(name in item and item[name] == benchmarks[0][name] for item in benchmarks)
              for name in BUDGET_FIELDS}
    require(all(checks.values()), "resource or budget mismatch")
    require(set(benchmarks[0]["resource_sha256"]) == {"weights", "assets"}, "unknown resource mapping scope")
    strategy_flags = [{name: row["strategy"]["flags"][name] for name in FLAGS if name != "sphere_normals_backend"}
                      for row in records]
    require(all(flags == strategy_flags[0] for flags in strategy_flags), "non-target strategy changed")
    uuids = [worker["actual_cuda_device"]["uuid"] for run in runtimes
             for group in run["hemisphere_scheduling"]["groups"] for worker in group["workers"].values()]
    require(len(set(uuids)) == 1 and isinstance(uuids[0], str) and uuids[0], "physical GPU mismatch")
    for arm in ("baseline", "candidate"):
        selected = [row for row in pair["runs"] if row["arm"] == arm]
        for variable, kind in (("NUMBA_CACHE_DIR", "numba"), ("TRITON_CACHE_DIR", "triton")):
            require(all(row["environment_override"][variable] == pair["fresh_caches"][arm][kind] for row in selected),
                    "cache binding mismatch")
            require(pair["fresh_caches"]["baseline"][kind] != pair["fresh_caches"]["candidate"][kind], "arm caches overlap")
    changed = sorted(key for key in set(benchmarks[0]["source_sha256"]) | set(benchmarks[1]["source_sha256"])
                     if benchmarks[0]["source_sha256"].get(key) != benchmarks[1]["source_sha256"].get(key))
    require(set(changed) == CHANGED_FILES, "scientific source changes require review")
    comparisons = []
    for case_index in (0, 1):
        by_arm = {row["arm"]: row for row in records if row["case_index"] == case_index}
        a, b = by_arm["baseline"], by_arm["candidate"]
        rows = []
        def add(scope, name, av, bv):
            rows.append({"scope": scope, "measure": name, "baseline_seconds": av, "candidate_seconds": bv,
                         "observed_saved_seconds": av - bv, "observed_reduction_percent": 100 * (av - bv) / av if av else None})
        for name in CLOCKS:
            add("separate_clock_not_additive", name, a["seconds"][name], b["seconds"][name])
        for name in GROUP_NAMES:
            add("nonoverlapping_parent_partition", name, a["nonoverlapping_parent_groups_seconds"][name],
                b["nonoverlapping_parent_groups_seconds"][name])
        amap, bmap = ({x["stage"]: x["seconds"] for x in row["parent_stages"]} for row in (a, b))
        for name in (x["stage"] for x in a["parent_stages"]):
            add("individual_parent_stage_included_in_partition", name, amap[name], bmap[name])
        comparisons.append({"case_index": case_index, "input_sha256": a["input_sha256"], "measurements": rows})
    at = math.fsum(row["seconds"]["CLI"] for row in records if row["arm"] == "baseline")
    bt = math.fsum(row["seconds"]["CLI"] for row in records if row["arm"] == "candidate")
    return {"schema": "fnit-whole-timing-public-v1", "extraction_scope": "existing SHA-bound metadata only; no image or GPU computation",
        "exporter_sha256": sha(__file__), "evidence_bindings_sha256": canonical_sha(evidence),
        "pair_report_sha256": evidence["pair_report"]["sha256"], "sources": source_summaries,
        "configuration_sha256": {name: digest(pair[name]) for name in ("cases_json_sha256",
            "common_arguments_json_sha256", "dependency_pythonpath_json_sha256", "python_interpreter_sha256",
            "script_sha256")},
        "changed_scientific_files": {"src/fnit/" + name: {"baseline": benchmarks[0]["source_sha256"][name],
            "candidate": benchmarks[1]["source_sha256"][name]} for name in changed},
        "input_and_source_integrity_unchanged_after_runs": True,
        "resources": {"GPU_family": "H100 PCIe", "logical_device": "cuda:0", "total_CPU_threads": 4,
            "CPU_affinity_count": 4, "actual_same_physical_GPU_across_all_workers": True,
            "same_budget_and_resource_checks": checks, "same_strategy_except_sphere_normals": True,
            "resource_mapping": {role: mapping_summary(benchmarks[0]["resource_sha256"][role])
                                 for role in ("weights", "assets")},
            "native_program_mapping": mapping_summary(benchmarks[0]["native_program_sha256"]),
            "N4_runtime_dependencies_canonical_sha256": canonical_sha(benchmarks[0]["n4_runtime_dependencies"]),
            "environment_canonical_sha256": canonical_sha(benchmarks[0]["environment"])},
        "clock_scopes_seconds": CLOCK_SCOPES,
        "partition_scope": "only 59 depth-zero synchronized parent stages; CLI residual closes partition; hemisphere workers and nested timers are not added",
        "runs": records, "within_case_comparisons": comparisons,
        "observed_two_case_CLI_sum": {"baseline_seconds": at, "candidate_seconds": bt,
            "saved_seconds": at - bt, "reduction_percent": 100 * (at - bt) / at},
        "interpretation": {"shared_environment": True, "stable_throughput": "not_assessed",
            "individual_runs_are_not_repeated_same_input_trials": True,
            "non_target_stage_drift_is_not_attributed_to_optimization": True,
            "ten_minute_target_met": False, "overall_metric_equivalence": "not_assessed_by_timing_export",
            "20GB_owned_memory_acceptance": "not_assessed"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_json.exists() and not args.output_csv.exists(), "outputs must not exist")
    evidence = json.loads(args.evidence_json.read_text())
    result = extract(evidence)
    with args.output_json.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    columns = ["case_index", "input_sha256", "scope", "measure", "baseline_seconds", "candidate_seconds",
               "observed_saved_seconds", "observed_reduction_percent", "baseline_benchmark_sha256", "candidate_benchmark_sha256"]
    with args.output_csv.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for comparison in result["within_case_comparisons"]:
            by_arm = {row["arm"]: row for row in result["runs"] if row["case_index"] == comparison["case_index"]}
            for row in comparison["measurements"]:
                writer.writerow({"case_index": comparison["case_index"], "input_sha256": comparison["input_sha256"],
                    **row, "baseline_benchmark_sha256": by_arm["baseline"]["report_sha256"]["benchmark"],
                    "candidate_benchmark_sha256": by_arm["candidate"]["report_sha256"]["benchmark"]})
    print(json.dumps({"output_json_sha256": sha(args.output_json), "output_csv_sha256": sha(args.output_csv)}))


if __name__ == "__main__":
    main()
