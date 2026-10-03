"""Extract original completed-cohort ranges; no solver or metric recomputation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys


CONFIG_SHA = "f8eba1cbf3eab2baa549172b32be2c9d3b1014ad478f6e3702d7987378f52f8c"
ANALYSIS_SHA = "a443c401b68bb8b88f1c31aedd37b534ddf6ea3a5e9d7d4d08bccf4b1ffc8521"
CASE_IDS = [f"sub-CON{number:02d}" for number in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11)]
PAIRED_CASES = CASE_IDS[:2]
MATRIX_FIELDS = ("count_relative_l1", "sift2_fbc_relative_l1", "count_support_dice",
                 "count_pearson", "mean_length_common_normalized_mae", "mean_fa_common_normalized_mae")
POP_FIELDS = ("accepted_fraction_absolute_difference", "length_ks", "endpoint_8mm_histogram_pearson",
              "tdi_native_voxel_pearson", "tdi_four_voxel_block_pearson")
RANGE_KEYS = {"official_min_max", "official_values", "fnit_vs_official", "official_defined_count",
              "official_comparison_count", "inside_count", "criterion", "threshold",
              "comparison_accepted", "accepted_count", "status"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def same(left, right):
    # Preserve nonfinite values if present; NaN itself is not equal to itself.
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


class OriginalFiles:
    """Keep exact bytes and identity; reread every original before publishing."""

    def __init__(self):
        self.raw = {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_file(), f"missing original file: {path}")
        raw = path.read_bytes()
        identity = {"path": str(path), "sha256": sha(raw), "size_bytes": len(raw)}
        if expected is not None:
            require(identity["sha256"] == expected["sha256"], f"original SHA differs: {path}")
            if "size_bytes" in expected:
                require(len(raw) == expected["size_bytes"], f"original size differs: {path}")
            if "path" in expected:
                require(path == Path(expected["path"]).resolve(), f"original path differs: {path}")
        if path in self.raw:
            require(raw == self.raw[path], f"original changed during extraction: {path}")
        self.raw[path] = raw
        return raw, identity

    def json(self, path, expected=None):
        raw, identity = self.read(path, expected)
        value = json.loads(raw)
        require(isinstance(value, dict), f"original JSON object required: {path}")
        return value, identity

    def bound(self, record):
        require(Path(record["path"]).is_absolute(), "bound original path must be absolute")
        return self.json(record["path"], record)

    def verify_unchanged(self):
        for path, raw in self.raw.items():
            require(path.read_bytes() == raw, f"original changed before publication: {path}")


def counts(records):
    decisions = [item for record in records for item in record["comparison_accepted"]]
    require(all(item is None or type(item) is bool for item in decisions), "invalid original decision")
    return {"passed": sum(item is True for item in decisions),
            "failed": sum(item is False for item in decisions),
            "not_assessed": sum(item is None for item in decisions), "total": len(decisions)}


def descriptive(values):
    require(all(value is None or type(value) in (int, float) for value in values), "non-numeric metric value")
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return {"finite_only_median": statistics.median(finite) if finite else None,
            "n_total": len(values), "n_finite": len(finite),
            "n_null": sum(value is None for value in values),
            "n_nonfinite": sum(value is not None and not math.isfinite(value) for value in values)}


def check_ranges(ranges, fields, self_repeat=False):
    require(set(ranges) == set(fields), "missing or unexpected original metric field")
    for field in fields:
        record = ranges[field]
        required = RANGE_KEYS - {"fnit_vs_official"} if self_repeat else RANGE_KEYS
        require(required <= set(record), f"missing original range member: {field}")
        require(len(record["official_values"]) == record["official_comparison_count"] == 10 and
                len(record["official_min_max"]) == 2, f"official repeat coverage differs: {field}")
        require(record["criterion"] in {"<= official_max", ">= official_min"}, "unknown original criterion")
        require(record["status"] in {"passed", "failed", "not_assessed"}, "invalid original metric status")
        decision_count = counts([record])
        require(record["accepted_count"] == decision_count["passed"], "original accepted count differs")
        if self_repeat:
            require(record["status"] == "not_assessed" and record["fnit_within"] == [] and
                    record["comparison_accepted"] == [], "FNIT single seed must remain not_assessed")
        else:
            require(len(record["fnit_vs_official"]) == len(record["comparison_accepted"]) == 5,
                    f"cross-arm coverage differs: {field}")
            descriptive(record["fnit_vs_official"])
        descriptive(record["official_values"])


def expected_cli(config, case, job, anatomy):
    arguments = ["UKBConnectome_pipeline", "--bids-root", case["bids_root"], "--subject", case["subject"],
                 "--freesurfer-subject-dir", anatomy, "--output-dir", str(job / "connectome"),
                 "--device", config["device"], "--n-seeds", str(config["n_seeds"]), "--seed", str(config["seed"])]
    for name in ("session", "run", "acquisition", "direction"):
        if case.get(name) is not None:
            arguments += ["--" + name, case[name]]
    return arguments + ["--atlas", *config["atlases"], *config["atlas_options"]]


def completed_producer(originals, config, case, version, binding):
    job = Path(config["run_root"]) / version / case["case_id"]
    gpu, gpu_identity = originals.json(job / "gpu_report.json")
    wall, wall_identity = originals.json(job / "raw_bids_wall.json")
    label = f"{version}/{case['case_id']}"
    require(gpu.get("status") == "completed" and gpu.get("case_id") == case["case_id"] and
            gpu.get("version") == version and gpu.get("exit_code") == 0, f"producer incomplete: {label}")
    expected = config["declared_source_manifests"][version]
    for when in ("source_before", "source_after"):
        require(gpu[when]["source_fingerprint"] == expected["source_fingerprint"] and
                gpu[when]["source_sha256"] == expected["source_sha256"], f"full source binding differs: {label}/{when}")
    require(Path(gpu["wall_report"]).resolve() == Path(wall_identity["path"]), "producer wall path differs")
    require(gpu["raw_input_provenance"] == case["input_files"], "producer raw input binding differs")
    require(gpu["anatomy"] == gpu["anatomy_after"] == binding["anatomy"]["files"], "completed anatomy binding differs")
    require(wall.get("mode") == "wall" and wall.get("status") == "completed" and wall.get("exit_code") == 0 and
            wall["outputs"]["status"] == "complete", "original wall/output completion differs")
    initial = wall["initial_output_state"]
    require(initial["output_directory_existed"] is False and not initial.get("preexisting_state_files") and
            not initial.get("preexisting_run_state"), "original CLI output was not fresh")
    stages = wall["preprocessing"]
    reverse = any(item["kind"] in ("reverse_pe", "reverse_dwi") for item in case["input_files"])
    require(len(stages) == 1 and stages[0]["eddy"] == "completed" and stages[0]["recon_all"] == "supplied" and
            stages[0]["topup"] == ("completed" if reverse else "no_reverse_pe"), "original preprocessing contract differs")
    require(wall["actual_eddy_gp_seeds"] == [config["eddy_gp_seed"]] and not wall["official_recon_all_calls"],
            "original seed/anatomy execution differs")
    require(wall["cli_arguments"] == expected_cli(config, case, job, binding["anatomy"]["directory"]),
            "original CLI differs from frozen parameters")
    seconds = wall["total_runtime_seconds"]
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0 and
            gpu["raw_dwi_cli_total_runtime_seconds"] == seconds, "original wall/GPU timer differs")
    keys = {"raw_dwi": "raw/image", "bval": "raw/bval", "bvec": "raw/bvec",
            "dataset_description": "raw/dataset_description", "reverse_pe": "raw/reverse",
            "reverse_dwi": "raw/reverse", "reverse_bval": "raw/reverse_bval"}
    for declared in case["input_files"]:
        kind = declared["kind"]
        if kind in keys:
            observed = [wall["inputs"].get(keys[kind], {})]
        elif kind in ("dwi_json", "reverse_json"):
            observed = [value for name, value in wall["inputs"].items() if name.startswith("raw_metadata/")]
        else:
            continue
        require(any(Path(value.get("path", "")).resolve() == Path(declared["path"]).resolve() and
                    value.get("sha256", "").lower() == declared["sha256"].lower() for value in observed),
                f"original wall selected input differs: {declared['path']}")
    memory = gpu["memory_budget"]
    require(memory["limit_bytes"] == 20_000_000_000 and
            {"process_tree", "allocated_bytes", "reserved_bytes"} <= set(memory["measurements"]),
            "original memory ledger fields missing")
    require(memory["measurements"]["process_tree"] == wall["gpu_process_memory"]["peak_process_tree_bytes"],
            "original process-tree ledger differs")
    require(all(same(value, wall["cuda_allocator"][key]) for key, value in memory["measurements"].items()
                if key != "process_tree"), "original allocator ledger differs")
    summary = {"actual_gpu_report": gpu_identity, "actual_wall_report": wall_identity,
               "source_fingerprint": expected["source_fingerprint"],
               "source_file_count": len(expected["source_sha256"]), "all_source_sha_before_after_match": True,
               "raw_dwi_cli_total_runtime_seconds": seconds, "memory_budget": memory,
               "wall_timing_scope": wall["timing_scope"],
               "post_timing_result_export_seconds": wall["post_timing_result_export"]["seconds"],
               "gpu_lock_queue_seconds": gpu["gpu_lock_queue_seconds"],
               "worker_wall_seconds": gpu["worker_wall_seconds"], "start_utc": gpu["start_utc"], "end_utc": gpu["end_utc"]}
    return summary


def build(configuration, analysis_report, audit):
    originals = OriginalFiles()
    config, config_identity = originals.json(configuration, {"sha256": CONFIG_SHA})
    audit["configuration"] = config_identity
    manifest, manifest_identity = originals.bound(config["raw_manifest"])
    bindings, binding_identity = originals.bound(config["input_bindings"])
    cases = {case["case_id"]: case for case in manifest["cases"]}
    require(set(cases) == set(bindings["cases"]) == set(CASE_IDS), "actual ten-case binding differs")
    planned = [(item["version"], item["case_id"]) for item in config["execution_order"]]
    require(len(planned) == len(set(planned)) == 12 and
            set(planned) == {("candidate", case) for case in CASE_IDS} | {("baseline", case) for case in PAIRED_CASES},
            "frozen ten-candidate/two-baseline plan differs")
    phase, phase_identity = originals.json(Path(config["run_root"]) / "status.json")
    audit["phase_observation"] = {"identity": phase_identity, "status": phase["status"],
                                  "producer_status": {f"{version}/{case}": phase.get("cases", {}).get(
                                      f"{version}/{case}", {}).get("status", "not_started") for version, case in planned},
                                  "full_cohort_report_exists": Path(analysis_report).is_file()}
    require(phase["status"] == "execution_completed" and len(phase["cases"]) == 12 and
            all(value == "completed" for value in audit["phase_observation"]["producer_status"].values()),
            "all ten candidates and two baselines must actually complete before extraction")
    report, report_identity = originals.json(analysis_report)
    require(Path(analysis_report).resolve().parent.name == "full_cohort" and report["status"] == "analysis_completed",
            "actual full_cohort analysis_completed report required")
    require(report["configuration"] == {"path": config_identity["path"], "sha256": CONFIG_SHA} and
            report["raw_manifest"] == config["raw_manifest"] and report["input_bindings"] == config["input_bindings"] and
            report["analysis_source_sha256"] == ANALYSIS_SHA, "original analysis source/input/config binding differs")
    require(report["coverage"]["full_ten_case_cohort"] is True and len(report["coverage"]["selected_cases"]) == 10 and
            set(report["coverage"]["selected_cases"]) == set(report["cases"]) == set(CASE_IDS), "full analysis coverage differs")
    producers = {f"{version}/{case}": completed_producer(originals, config, cases[case], version, bindings["cases"][case])
                 for version, case in planned}
    excerpt, atlas_counts, medians = {}, [], []
    pooled = {(atlas, field): {"official_values": [], "fnit_vs_official": []}
              for atlas in config["atlases"] for field in MATRIX_FIELDS}
    for case_id in CASE_IDS:
        entry = report["cases"][case_id]
        producer = producers[f"candidate/{case_id}"]
        require(entry["source_fingerprint"] == producer["source_fingerprint"] and
                entry["actual_gpu_report"] == {key: producer["actual_gpu_report"][key] for key in ("path", "sha256")} and
                entry["actual_wall_report"] == {key: producer["actual_wall_report"][key] for key in ("path", "sha256")},
                "original analysis producer report SHA binding differs")
        require(entry["official_reference_manifest"] == bindings["cases"][case_id]["official_reference_manifest"],
                "official analysis binding differs")
        originals.bound(entry["official_reference_manifest"])
        directory = Path(analysis_report).resolve().parent / case_id
        expected_names = {"matrix_envelope.json", "population_envelope.json", "population.png"}
        require(set(entry["outputs"]) == expected_names, "original analysis output set differs")
        matrix, matrix_identity = originals.json(directory / "matrix_envelope.json", entry["outputs"]["matrix_envelope.json"])
        pop, pop_identity = originals.json(directory / "population_envelope.json", entry["outputs"]["population_envelope.json"])
        _, png_identity = originals.read(directory / "population.png", entry["outputs"]["population.png"])
        for value in (matrix, pop):
            require(value["n_seed_attempts"] == config["n_seeds"] and value["official_seeds"] == list(range(5)) and
                    value["fnit_seeds"] == [config["seed"]] and value["fnit_reproducibility_status"] == "not_assessed",
                    "original single-seed/repeat contract differs")
        require(matrix["case_id"] == case_id and set(matrix["profiles"]) == set(config["atlases"]) and
                matrix["raw_case_binding"]["fnit_gpu_report_sha256"] == [producer["actual_gpu_report"]["sha256"]] and
                matrix["raw_case_binding"]["raw_manifest_sha256"] == manifest_identity["sha256"] and
                matrix["raw_case_binding"]["official_manifest_sha256"] == entry["official_reference_manifest"]["sha256"],
                "original matrix case/source/reference binding differs")
        require("stored point-visit" in pop["metric_policy"]["tdi"] and "not MRtrix tckmap" in pop["metric_policy"]["tdi"],
                "original point-visit definition missing")
        profiles, records = {}, []
        for atlas in config["atlases"]:
            profile = matrix["profiles"][atlas]
            check_ranges(profile["ranges"], MATRIX_FIELDS)
            check_ranges(profile["fnit_reproducibility_ranges"], MATRIX_FIELDS, self_repeat=True)
            require(profile["comparison_counts"] == {"official": 10, "fnit": 0, "cross": 5} and
                    profile["fnit_reproducibility_status"] == "not_assessed", "original profile repeat coverage differs")
            profiles[atlas] = {key: value for key, value in profile.items() if key not in {"pairwise", "provenance"}}
            tally = counts(profile["ranges"].values())
            atlas_counts.append({"case_id": case_id, "atlas": atlas, **tally,
                                 "original_status": profile["matrix_envelope_status"]})
            records.extend(profile["ranges"].values())
            for field in MATRIX_FIELDS:
                record = profile["ranges"][field]
                for arm in ("official_values", "fnit_vs_official"):
                    values = record[arm]
                    medians.append({"scope": "one_case_one_atlas", "case_id": case_id, "atlas": atlas,
                                    "field": field, "arm": arm, **descriptive(values)})
                    pooled[atlas, field][arm].extend(values)
        check_ranges(pop["ranges"], POP_FIELDS)
        check_ranges(pop["fnit_reproducibility_ranges"], POP_FIELDS, self_repeat=True)
        require(counts(records) == entry["matrix_decisions"] and counts(pop["ranges"].values()) == entry["population_decisions"],
                "original case decisions disagree with original ranges")
        require(entry["matrix_status"] == matrix["matrix_envelope_status"] and
                entry["population_status"] == pop["population_envelope_status"] and
                entry["fnit_self_repeat_status"] == "not_assessed: one actual candidate seed", "original case statuses differ")
        timing = entry["timing"]
        require(timing["candidate_raw_dwi_cli_seconds"] == producer["raw_dwi_cli_total_runtime_seconds"] and
                timing["candidate_memory"] == producer["memory_budget"], "original candidate timing binding differs")
        if case_id in PAIRED_CASES:
            baseline = producers[f"baseline/{case_id}"]
            require(timing["baseline_timing_status"] == "assessed" and timing["baseline_status"] == "completed" and
                    timing["baseline_raw_dwi_cli_seconds"] == baseline["raw_dwi_cli_total_runtime_seconds"] and
                    timing["baseline_memory"] == baseline["memory_budget"] and timing["baseline_observation"] ==
                    {key: baseline["actual_gpu_report"][key] for key in ("path", "sha256")}, "original paired timing binding differs")
        else:
            require(timing["baseline_timing_status"] == "not_assessed" and timing["baseline_status"] == "not_scheduled",
                    "unpaired case must remain not_assessed")
        excerpt[case_id] = {"original_case_report": entry, "original_matrix_file": matrix_identity,
                            "original_population_file": pop_identity, "original_png_file": png_identity,
                            "matrix": {**{key: value for key, value in matrix.items() if key != "profiles"}, "profiles": profiles},
                            "population": {key: value for key, value in pop.items() if key not in {"pairs", "pairwise"}}}
    for (atlas, field), arms in pooled.items():
        for arm, values in arms.items():
            medians.append({"scope": "pooled_original_pairs_ten_cases_one_atlas", "case_id": "all_ten",
                            "atlas": atlas, "field": field, "arm": arm, **descriptive(values)})
    originals.verify_unchanged()
    case_decisions = {case_id: {family: entry["original_case_report"][family + "_decisions"]
                               for family in ("matrix", "population")} for case_id, entry in excerpt.items()}
    cohort_counts = {family: {key: sum(case_decisions[case][family][key] for case in CASE_IDS)
                              for key in ("passed", "failed", "not_assessed", "total")}
                     for family in ("matrix", "population")}
    return {"status": "extraction_completed", "created_utc": utc(), "configuration": config_identity,
            "original_full_cohort_report": report_identity, "original_phase_report": phase_identity,
            "raw_manifest": manifest_identity, "input_bindings": binding_identity,
            "scope": "original ranges and decisions only; descriptive finite-value medians do not change gates",
            "null_nonfinite_policy": "original records retained; descriptive finite-only medians report all coverage counts",
            "memory_scope": "original sampled whole raw-DWI process-tree/allocator ledger; no continuous or component-peak extrapolation",
            "cases": excerpt, "producers": producers, "cohort_decision_counts": cohort_counts,
            "paired_timing": {case: report["cases"][case]["timing"] for case in PAIRED_CASES}}, atlas_counts, medians, originals


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=True)
        stream.write("\n")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--configuration", type=Path, required=True, help="unchanged frozen configuration")
    parser.add_argument("--analysis-report", type=Path, required=True, help="actual v3 full_cohort/report.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh excerpt directory; created only after all guards")
    parser.add_argument("--receipt", type=Path, required=True, help="new success/failure receipt outside output directory")
    args = parser.parse_args()
    require(not args.output_dir.exists() and not args.receipt.exists(), "output and receipt must both be fresh")
    require(args.output_dir.resolve() not in args.receipt.resolve().parents, "receipt must be outside result directory")
    audit = {"start_utc": utc(), "command": [sys.executable, *sys.argv], "python": platform.python_version(),
             "hostname": platform.node(), "uid": os.getuid(), "script_sha256": sha(Path(__file__).read_bytes()),
             "scope": "stdlib read-only extraction; no MRI, solver, plotting, GPU or acceptance-threshold calculation"}
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicitly hide GPUs for CPU-only extraction")
        result, atlas_counts, medians, originals = build(args.configuration, args.analysis_report, audit)
        originals.verify_unchanged()
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "ranges_excerpt.json", result)
        write_csv(args.output_dir / "case_atlas_counts.csv", atlas_counts)
        write_csv(args.output_dir / "matrix_descriptive_medians.csv", medians)
        case_rows = [{"case_id": case_id, "family": family, **entry["original_case_report"][family + "_decisions"],
                      "original_status": entry["original_case_report"][family + "_status"]}
                     for case_id, entry in result["cases"].items() for family in ("matrix", "population")]
        write_csv(args.output_dir / "case_counts.csv", case_rows)
        atlas_rows = [{"atlas": atlas, **{key: sum(row[key] for row in atlas_counts if row["atlas"] == atlas)
                                           for key in ("passed", "failed", "not_assessed", "total")}}
                      for atlas in dict.fromkeys(row["atlas"] for row in atlas_counts)]
        write_csv(args.output_dir / "atlas_cohort_counts.csv", atlas_rows)
        for case_id in PAIRED_CASES:
            original = Path(result["cases"][case_id]["original_png_file"]["path"])
            (args.output_dir / (case_id + "_population.png")).write_bytes(originals.raw[original])
        originals.verify_unchanged()
        audit.update(status="extraction_completed", original_files_rechecked=len(originals.raw),
                     output_files={path.name: {"size_bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}
                                   for path in sorted(args.output_dir.iterdir())})
        exit_code = 0
    except Exception as error:
        audit.update(status="failed", error=f"{type(error).__name__}: {error}",
                     output_directory_exists=args.output_dir.exists())
        exit_code = 1
    audit["end_utc"] = utc()
    write_json(args.receipt, audit)
    print(json.dumps({"status": audit["status"], "receipt": str(args.receipt), "error": audit.get("error")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
