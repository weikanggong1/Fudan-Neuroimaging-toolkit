"""Fresh posthoc array analysis; preserve failed producer and its original clocks."""
import argparse
import csv
import json
from pathlib import Path

from run_same_input import (STAGES, compare_files, input_specs, reference_metadata,
                            reference_prefix, require_fresh_output, sha256,
                            verify_specs, write_json)


def analyze(manifest_path, producer_code_root, existing_root, output_root, completed_producer=False):
    manifest_path = Path(manifest_path).resolve()
    code = Path(producer_code_root).resolve()
    existing = Path(existing_root).resolve()
    producer_path = existing / "report.public.json"
    producer = json.loads(producer_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (producer["case_id"] != manifest["case_id"] or producer.get("input_guards_equal") is not True
            or producer["input_sha256_before"] != producer["input_sha256_after"]):
        raise ValueError("Original producer inputs/source were not unchanged")
    if completed_producer:
        if producer["status"] != "operator_comparison_complete" or len(producer.get("differences", {})) != 11:
            raise ValueError("This receipt requires a completed eleven-output producer")
    elif producer["status"] != "validation_failed" or producer.get("error_type") != "TypeError":
        raise ValueError("This repair analyzes the preserved GIFTI-loader failure only")
    if producer.get("reference_operators_seconds") is None or producer.get("FNIT_sampler_api_seconds") is None:
        raise ValueError("Both original sampling roles must have completed")
    specs = input_specs(manifest)
    specs.update({"FNIT_workbench": manifest["fnit_workbench"],
                  "reference_container": manifest["reference"]["container"],
                  "manifest": {"path": str(manifest_path), "sha256": producer["input_sha256_before"]["manifest"]},
                  "driver": {"path": str(code / "run_same_input.py"), "sha256": producer["input_sha256_before"]["driver"]}})
    from fnit.fmri import surface_fmriprep, sampling_reference, surface, assets_setup, surface_prepare
    from fnit import _hemisphere_parallel
    for key, module in (("surface_fmriprep_source", surface_fmriprep), ("sampling_reference_source", sampling_reference),
                        ("surface_contract_source", surface), ("assets_source", assets_setup),
                        ("hemisphere_parallel_source", _hemisphere_parallel), ("surface_prepare_source", surface_prepare)):
        specs[key] = {"path": module.__file__, "sha256": producer["input_sha256_before"][key]}
    verify_specs(specs)
    prepared = json.loads((existing / "prepared_inputs.private.json").read_text())
    if {key: entry["sha256"] for key, entry in prepared.items()} != producer["preparation_derived_input_sha256_before"]:
        raise ValueError("Prepared file binding differs from original producer")
    for key, entry in prepared.items():
        specs["prepared." + key] = entry
    names = [f"{h}.{stage}.func.gii" for h in ("L", "R") for stage in STAGES]
    names.append("space-fsLR_den-91k_bold.dtseries.nii")
    for role in ("FNIT", "reference"):
        for name in names:
            path = existing / role / name
            expected_sha = producer["output_sha256"][role][name] if completed_producer else sha256(path)
            specs[role + "." + name] = {"path": str(path), "sha256": expected_sha}
    for key, path in (("producer_report", producer_path), ("posthoc_code", Path(__file__).resolve()),
                      ("comparison_code", Path(__file__).with_name("run_same_input.py"))):
        specs[key] = {"path": str(path), "sha256": sha256(path)}
    before = verify_specs(specs)
    prefix = reference_prefix(manifest, 4)
    metadata = reference_metadata(prefix, manifest["reference"]["workbench"], 4)
    if metadata != producer["programs"]["reference"]:
        raise ValueError("Original reference programs changed")
    output = require_fresh_output(output_root, [entry["path"] for entry in specs.values()], [existing])
    output.mkdir(parents=True)
    differences = {name: compare_files(existing / "FNIT" / name, existing / "reference" / name) for name in names}
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["artifact", "different_values", "maximum_absolute_error",
                                                  "root_mean_squared_error", "within_declared_absolute_tolerance"])
        writer.writeheader()
        for name, result in differences.items():
            writer.writerow({"artifact": name, **{key: result[key] for key in writer.fieldnames[1:]}})
    after = {key: sha256(entry["path"]) for key, entry in specs.items()}
    reference_after = reference_metadata(prefix, manifest["reference"]["workbench"], 4)
    equal = after == before and reference_after == metadata
    report = {"schema": "fnit.surface_sampler_same_input.posthoc.v2", "case_id": producer["case_id"],
              "scope": "new guarded array analysis of completed real sampling outputs; no sampler or MRI rerun",
              "status": "posthoc_operator_comparison_complete" if equal else "input_changed_during_diagnostic",
              "scientific_equivalence": "not_assessed", "original_producer_status": producer["status"],
              "original_producer_error_type": producer.get("error_type"), "original_producer_report_sha256": before["producer_report"],
              "original_producer_driver_sha256": before["driver"], "input_guards_equal": equal,
              "input_sha256_before": before, "input_sha256_after": after,
              "same_workbench_binary": producer["same_workbench_binary"], "programs": producer["programs"],
              "expected_frames": producer["expected_frames"], "tr_seconds": producer["tr_seconds"],
              "differences": differences,
              "all_stage_arrays_equal": all(result["different_values"] == 0 for result in differences.values()),
              "all_within_declared_absolute_tolerance": all(result["within_declared_absolute_tolerance"] for result in differences.values()),
              "original_timing_seconds": {key: producer[key] for key in ("preparation_seconds_excluded_from_sampler",
                  "FNIT_sampler_api_seconds", "reference_operators_seconds", "FNIT_operator_seconds", "reference_operator_seconds")},
              "timing_boundaries": producer["timing_boundaries"],
              "original_load_observations": {key: producer[key] for key in ("load_before", "load_between_roles", "load_after")},
              "cpu_threads": 4, "parallel_hemispheres": False}
    if completed_producer:
        report["scope"] = "new guarded before/after verification of completed producer's bound real output bytes; no sampler or MRI rerun"
        report["producer_declared_output_sha256_verified"] = True
    write_json(output / "report.public.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--producer-code-root", required=True)
    parser.add_argument("--existing-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    report = analyze(args.manifest, args.producer_code_root, args.existing_root, args.output_root)
    print(json.dumps({"status": report["status"], "all_stage_arrays_equal": report["all_stage_arrays_equal"]}))
    raise SystemExit(0 if report["status"] == "posthoc_operator_comparison_complete" else 2)
