"""Summarize completed FLIRT CPU suite records without private file paths."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


SOURCE_HASHES = {
    "src/fnit/flirt/core.py": "21efb96d6f48797f7dc90542fad6dfa75cd7580fa77fed346c90615f0280f4ef",
    "src/fnit/flirt/_cpu.py": "0867dc5ce59a773aec47d4d5d0249f6435b4e2cfca2778949397b4858d75591d",
    "src/fnit/flirt/_cpu_simd.py": "03c70f8c6c9239b5d66eff2f992c60f4e1aca2bb471067be2baeec6d9a3ac04f",
}


def fit_details(record):
    details = []
    for worker in record["worker_results"]["candidate"]:
        image = Path(worker["outputs"]["moved"])
        qc_path = image.parent / "fnit_qc.json"
        if not qc_path.is_file():
            continue
        qc = json.loads(qc_path.read_text())
        details.append({key: qc[key] for key in (
            "cost_value", "cost_evaluations", "phase_cost_evaluations", "phase_timings_seconds",
            "execution", "degrees_of_freedom", "angular_search", "input_weight_used",
            "reference_weight_used", "initial_matrix_used") if key in qc})
    return details


def public_record(record):
    single = record["timing_protocol"] == "single_observation"
    full = {key: values for key, values in record["full_process"].items() if values}
    diagnostics = fit_details(record)
    result = {
        "case_id": record["case_id"], "threads_ceiling": record["threads"],
        "affinity": record["affinity"], "timing_protocol": record["timing_protocol"],
        "full_process_seconds": full,
        "input_files": [{"filename": Path(path).name, **metadata}
                        for path, metadata in record["input_metadata_before"].items()],
        "official_programs": [{key: item[key] for key in ("name", "bytes", "sha256")}
                              for item in record["official_program_metadata"]],
        "process_cpu_usage": {key: values for key, values in record["process_cpu_usage"].items() if values},
        "load_before": record["load_before"], "load_after": record["load_after"],
        "accuracy": record["adapter_accuracy"],
        "official_output_metadata": record["official_output_metadata"],
        "candidate_output_metadata": [worker["output_metadata"]
                                      for worker in record["worker_results"]["candidate"]],
        "candidate_rss_kib": [worker["maximum_rss_kib"]
                              for worker in record["worker_results"]["candidate"]],
        "runtime_versions": record["worker_results"]["candidate"][-1]["versions"],
        "actual_thread_settings": record["worker_results"]["candidate"][-1]["threads"],
        "fnit_diagnostics": diagnostics,
    }
    if single:
        result["single_full_process_seconds"] = record["single_full_process_seconds"]
    else:
        result["warmup_full_process_seconds"] = record["warmup_full_process"]
        result["median_full_process_seconds"] = record["median_full_process_seconds"]
        api = record["loaded_api_measurements"]["candidate"]
        result["loaded_api_candidate"] = {key: api[key] for key in (
            "scope", "warmup_seconds", "seconds", "median_seconds", "maximum_rss_kib", "output_metadata")}
        fingerprints = [{name: metadata["sha256"] for name, metadata in row.items()}
                        for row in result["candidate_output_metadata"]]
        api_fingerprint = {name: metadata["sha256"] for name, metadata in api["output_metadata"].items()}
        result["candidate_repetition_checks"] = {
            "process_outputs_sha256_identical": all(row == fingerprints[0] for row in fingerprints),
            "loaded_api_outputs_sha256_identical": api_fingerprint == fingerprints[0],
            "final_cost_identical": len({detail.get("cost_value") for detail in diagnostics}) == 1,
            "cost_evaluations_identical": len({detail.get("cost_evaluations") for detail in diagnostics}) == 1,
        }
        measured = diagnostics[1:]
        phase_names = set().union(*(detail.get("phase_timings_seconds", {}) for detail in measured))
        result["median_fnit_phase_seconds"] = {
            name: statistics.median(detail["phase_timings_seconds"][name] for detail in measured
                                    if name in detail.get("phase_timings_seconds", {}))
            for name in sorted(phase_names)}
    return result


def table(records, main):
    if main:
        rows = ["| 模式 | CPU 上限 | FSL 完整进程中位数 | FNIT 完整进程中位数 | FNIT 预热 API 中位数 | 位移 RMS | Pearson |",
                "|---|---:|---:|---:|---:|---:|---:|"]
    else:
        rows = ["| 功能观察 | CPU 上限 | FSL 完整进程 | FNIT 完整进程 | 位移 RMS | Pearson |",
                "|---|---:|---:|---:|---:|---:|"]
    for record in records:
        metric = record["accuracy"]["candidate_vs_official"]
        rms = metric["world_grid_displacement_mm"]["rms"]
        pearson = metric["warped"]["pearson_r"]
        times = record["median_full_process_seconds" if main else "single_full_process_seconds"]
        label = record["case_id"].removeprefix("flirt_")
        for suffix, description in (
            ("_default", "，默认搜索"), ("_explicit_batched", "，显式 batched"),
            ("_input_weight", "，input 权重"), ("_reference_weight", "，reference 权重"),
            ("_both_weights", "，两侧权重"), ("_init", "，初始矩阵"),
            ("_nosearch", "，无角度搜索"), ("_known_matrix", "，已知矩阵"),
            ("_usesqform", "，qform/sform")):
            if label.endswith(suffix):
                label = label.removesuffix(suffix) + description
                break
        label = label.replace("12_corratio", "12/corratio").replace("6_normmi", "6/normmi")
        cells = [label, str(record["threads_ceiling"]),
                 f"{times['official']:.3f} s", f"{times['candidate']:.3f} s"]
        if main:
            cells.append(f"{record['loaded_api_candidate']['median_seconds']:.3f} s")
        cells.extend((f"{rms:.5f} mm", f"{pearson:.8f}"))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tables", type=Path)
    args = parser.parse_args()
    groups, states, benchmark_hashes = {}, {}, {}
    for scope in ("main", "functions"):
        path = args.run_root / scope / "suite.private.json"
        if path.exists():
            suite = json.loads(path.read_text())
            assert Path(suite["candidate_root"]).name == "candidate_all_v16", "final source only"
            for relative, expected in SOURCE_HASHES.items():
                assert hashlib.sha256((Path(suite["candidate_root"]) / relative).read_bytes()).hexdigest() == expected
            for relative in ("tools/benchmark_multimodal_cpu.py", "tools/benchmark_multimodal_cpu_flirt.py"):
                benchmark_hashes[relative] = hashlib.sha256((Path(suite["candidate_root"]) / relative).read_bytes()).hexdigest()
            states[scope] = suite["status"]
            groups[scope] = [public_record(record) for record in suite["records"]]
        else:
            states[scope], groups[scope] = "not_started", []
    complete = (len(groups["main"]) == 4 and len(groups["functions"]) == 22
                and all(state == "executed_with_numeric_comparisons" for state in states.values()))
    report = {
        "schema_version": 1, "date": "2026-10-04", "status": "complete" if complete else "in_progress",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_status": states, "run_id": args.run_root.name,
        "hardware": json.loads((args.run_root / "hardware.public.json").read_text()),
        "dataset": "OpenNeuro ds000114 v1.0.2; FNIT defaced public T1w examples",
        "dataset_license": "CC0",
        "license_metadata": "https://raw.githubusercontent.com/OpenNeuroDatasets/ds000114/master/dataset_description.json",
        "official_version": "FSL 6.0.7.4", "candidate_snapshot": "candidate_all_v16",
        "candidate_archive_sha256": "6c98418a54081c38c84421e03de74405b04790ccf46d5b83352bfdc3b81540cd",
        "source_sha256": SOURCE_HASHES,
        "benchmark_source_sha256": benchmark_hashes,
        "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing_scope": {
            "full_process": "Fresh full official CLI or one full FNIT API worker; process start, imports, read, complete calculation and save",
            "main": "One complete warmup pair excluded, then three alternating AB/BA pairs; median of three",
            "loaded_api": "One full warmup excluded, then three complete read/calculate/save API calls; excludes interpreter and Torch/adapter setup",
            "functions": "One complete pair per case/thread budget; no warmup, no stable speed estimate",
            "threads": "Same allowed physical cores and thread ceilings; actual process CPU usage reported separately",
            "phases": "FNIT phase diagnostics only; default native CLI does not supply equivalent phase clocks",
        },
        "precision": "Float32 images/coordinates/ordered statistics; float64 matrix construction and NMI histogram; unchanged search/iteration budgets",
        "gpu_report": "../../validation/multimodal_cpu_20261004/gpu_final_20261004.public.json",
        "main_expected_records": 4, "function_expected_records": 22,
        "main": groups["main"], "functions": groups["functions"],
        "excluded": "Private zero-weight prototype was not merged or used for these timings",
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    assert all(value not in serialized for value in ("/cwStorage/", "/mnt/c/Users/", ".sock"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    if args.tables:
        args.tables.write_text("# FLIRT CPU " + report["status"] + "\n\n"
                              + table(groups["main"], True) + "\n\n"
                              + table(groups["functions"], False) + "\n")
    print(json.dumps({"status": report["status"], "main_records": len(groups["main"]),
                      "function_records": len(groups["functions"])}))


if __name__ == "__main__":
    main()
