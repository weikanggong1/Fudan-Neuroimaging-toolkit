"""Export an explicit public whitelist; never copy MRI, arrays or private paths."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def runtime_record(value):
    keys = ("source_commit", "source_archive_sha256", "source_archives", "source_file_sha256",
            "configuration_isolation", "builder_sha256", "binary_sha256", "binary_size",
            "runtime_files", "license", "arch", "configure_args", "build_target", "jobs",
            "dependency_linking", "installed_mrtrix_used")
    record = {key: value[key] for key in keys}
    record["build_records"] = [{"stage": row["log"].removesuffix(".log"),
                                 "seconds": row["seconds"], "returncode": row["returncode"]}
                                for row in value["build_records"]]
    record["compiler"] = "GCC/G++ 7.5.0; isolated CFFF validation environment"
    return record


def command_record(row):
    return {key: row[key] for key in ("binary_sha256", "wall_seconds", "returncode")}


def export(e2e_path, scale_path, output_dir):
    e2e, scale = json.loads(e2e_path.read_text()), json.loads(scale_path.read_text())
    if e2e["status"] != "completed" or scale["status"] != "completed":
        raise ValueError("only complete measurements may be exported")
    tracking = e2e["tracking_reference"]
    strict = tracking["strict_one_thread"]
    parameters = {key: e2e["parameters"][key] for key in (
        "atlas", "reuse_atlas", "n_seeds", "seed", "tracking_threads", "reference_threads",
        "strict_tracking_seeds", "reference_repeats", "assignment_radius")}
    result = {key: e2e[key] for key in (
        "status", "scope", "raw_topup_eddy_recon_all_rerun", "timing_scope", "benchmark_sha256",
        "environment", "stages", "full_call_seconds", "accepted_streamlines", "cuda_allocator_peak", "tf32")}
    result.update(
        schema_version=1, dataset=dict(name="OpenNeuro ds004666", license="CC0",
            doi="10.18112/openneuro.ds004666.v1.0.8",
            license_source="https://raw.githubusercontent.com/OpenNeuroDatasets/ds004666/master/dataset_description.json"),
        hardware=dict(gpu="NVIDIA A100-SXM4-80GB", visible_gpu_count=1, cpu_threads=8),
        private_report_sha256=hashlib.sha256(e2e_path.read_bytes()).hexdigest(),
        input_sha256={key: {field: row[field] for field in ("bytes", "sha256")}
                      for key, row in e2e["input_before"].items()},
        input_and_source_unchanged=(e2e["input_before"] == e2e["input_after"]
                                    and e2e["source_before"] == e2e["source_after"]),
        source_sha256=e2e["source_before"], parameters=parameters,
        runtime=runtime_record(e2e["native_tracking"]["runtime"]),
        native_input_geometry=e2e["native_tracking"]["input_images"],
        native_command_seconds=e2e["native_tracking"]["command_seconds"],
        native_adapter_seconds=e2e["native_tracking"]["adapter_seconds"],
        strict_one_thread={key: strict[key] for key in (
            "requested_seed_budget", "candidate_seconds", "candidate_digest", "reference_digest",
            "ordered_points_and_offsets_equal")},
        official_tracking_commands={key: command_record(row) for key, row in tracking["commands"].items()},
        threaded_independent_repeats=[{key: row[key] for key in (
            "seed", "accepted_streamlines", "requested_seed_budget", "length_definition",
            "mean_polyline_length_mm", "mean_candidate_polyline_length_mm", "length_ks", "header_total_count")}
                                      for row in tracking["threaded_independent_repeats"]],
        repeat_envelope_acceptance_claim=None,
        fixed_tck_reference={key: e2e["fixed_tck_reference"][key] for key in (
            "scope", "per_track_vectors", "atlas_matrices", "native_tracking_equivalence_assessed_here")},
        cache_reuse={key: e2e["atlas_reuse"][key] for key in (
            "seconds", "atlas", "paths_equal", "weights_equal")},
        scientific_scope="Real corrected-DWI-to-two-atlas SC execution; tracking equivalence on fixed FNIT-produced inputs and fixed-TCK downstream comparison; not a new full official preprocessing/anatomy pipeline comparison.",
    )
    result["cache_reuse"]["core"] = e2e["atlas_reuse"]["cache_status"]["core"]
    commands = e2e["fixed_tck_reference"]["commands"]
    result["fixed_tck_reference"]["official_commands"] = {
        key: command_record(row) if "wall_seconds" in row else
        {metric: command_record(item) for metric, item in row.items()}
        for key, row in commands.items()}
    scale_public = dict(status=scale["status"], benchmark_sha256=scale["benchmark_sha256"],
                        private_report_sha256=hashlib.sha256(scale_path.read_bytes()).hexdigest(),
                        scope="unchanged real files; adapter includes SHA checks/TCK read/packed H2D; command timings are separate",
                        parameters={key: scale["parameters"][key] for key in (
                            "n_seeds", "strict_seeds", "threads", "repeats", "device")},
                        inputs=scale["inputs"], rows=[{key: row[key] for key in (
                            "mode", "seed", "seeds", "threads", "adapter_seconds", "native_command_seconds",
                            "official_command_seconds", "candidate_digest", "reference_digest",
                            "ordered_points_and_offsets_equal", "cuda_allocator")} for row in scale["rows"]])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, report in (("results.public.json", result), ("tracking_100k.public.json", scale_public)):
        rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        # This whitelist must not accidentally grow into a path-preserving dump.
        if re.search(r'"(?:path|command|argv|cxx_argv|tck_file|user|account|hostname|address)"\s*:', rendered):
            raise ValueError("private field found in public whitelist")
        if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', rendered):
            raise ValueError("network address found in public whitelist")
        (output_dir / name).write_text(rendered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e-report", type=Path, required=True)
    parser.add_argument("--scale-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.e2e_report, args.scale_report, args.output_dir)
