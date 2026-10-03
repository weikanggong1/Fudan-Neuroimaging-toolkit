"""Check available actual metadata schemas; never mark them full-cohort evidence."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("extract_completed_cohort", Path(__file__).with_name("extract_completed_cohort.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    tool.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not args.output.exists(), "CPU-only fresh receipt required")
    originals = tool.OriginalFiles()
    config, identity = originals.json(args.accuracy_root / "formal_frozen_v1/accuracy_configuration.json",
                                      {"sha256": tool.CONFIG_SHA})
    manifest, _ = originals.bound(config["raw_manifest"])
    bindings, _ = originals.bound(config["input_bindings"])
    cases = {case["case_id"]: case for case in manifest["cases"]}
    result = {"status": "available_actual_schema_checked", "scope": "two existing actual case reports and four completed metadata producers; not final cohort extraction",
              "created_utc": tool.utc(), "configuration": identity, "command": [sys.executable, *sys.argv],
              "script_sha256": tool.sha(Path(__file__).read_bytes()), "extractor_sha256": tool.sha(Path(tool.__file__).read_bytes()),
              "cases": {}, "producers": {}}
    for case_id in tool.PAIRED_CASES:
        for version in ("candidate", "baseline"):
            result["producers"][f"{version}/{case_id}"] = tool.completed_producer(
                originals, config, cases[case_id], version, bindings["cases"][case_id])
        directory = args.accuracy_root / "root_matrix_analysis_v3" / case_id
        report, report_identity = originals.json(directory / "report.json")
        tool.require(report["status"] == "analysis_completed" and report["analysis_source_sha256"] == tool.ANALYSIS_SHA,
                     "existing actual case analysis is incomplete")
        entry = report["cases"][case_id]
        matrix, matrix_identity = originals.json(directory / case_id / "matrix_envelope.json", entry["outputs"]["matrix_envelope.json"])
        pop, pop_identity = originals.json(directory / case_id / "population_envelope.json", entry["outputs"]["population_envelope.json"])
        _, png_identity = originals.read(directory / case_id / "population.png", entry["outputs"]["population.png"])
        records = []
        by_atlas = {}
        for atlas, profile in matrix["profiles"].items():
            tool.check_ranges(profile["ranges"], tool.MATRIX_FIELDS)
            tool.check_ranges(profile["fnit_reproducibility_ranges"], tool.MATRIX_FIELDS, self_repeat=True)
            records.extend(profile["ranges"].values())
            by_atlas[atlas] = tool.counts(profile["ranges"].values())
        tool.check_ranges(pop["ranges"], tool.POP_FIELDS)
        tool.check_ranges(pop["fnit_reproducibility_ranges"], tool.POP_FIELDS, self_repeat=True)
        tool.require(tool.counts(records) == entry["matrix_decisions"] and tool.counts(pop["ranges"].values()) == entry["population_decisions"],
                     "actual original decisions differ")
        result["cases"][case_id] = {"report": report_identity, "matrix": matrix_identity, "population": pop_identity,
                                    "png": png_identity, "matrix_counts": entry["matrix_decisions"],
                                    "population_counts": entry["population_decisions"], "by_atlas": by_atlas,
                                    "original_fnit_self_repeat_status": entry["fnit_self_repeat_status"],
                                    "original_baseline_timing_status_at_case_analysis": entry["timing"]["baseline_timing_status"]}
    originals.verify_unchanged()
    result["original_files_rechecked"] = len(originals.raw)
    tool.write_json(args.output, result)
    print(json.dumps({"status": result["status"], "cases": list(result["cases"])}))


if __name__ == "__main__":
    main()
