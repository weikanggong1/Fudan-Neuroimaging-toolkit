"""Export aggregate task04 results and compare native8 with native1 gold.

Private manifests, original images and per-vertex arrays remain on the server.
The public report includes only timing, numerical summaries and source hashes.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time


SUITES = {
    "initial_registration_cpu1_cpu8_attempt03": "registration.manifest.private.json",
    "candidate_registration_cpu1_cpu8": "registration.manifest.private.json",
    "features_node10_cpu1_cpu8_attempt02": "features.manifest.private.json",
    "candidate_features_node10_cpu1_cpu8": "features.manifest.private.json",
    "fixed_projection_cpu1_cpu8": "projection.manifest.private.json",
    "full_surface_cpu1_cpu8_attempt02": "surface_full180.manifest.private.json",
}


def numeric_tree(value):
    if isinstance(value, dict):
        return {key: result for key, item in value.items()
                if "/" not in key and "sub-" not in key
                if (result := numeric_tree(item)) is not None}
    if isinstance(value, list):
        return [numeric_tree(item) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if value in ("L", "R", "cpu", "cuda:0", "cuda:1", "optimized", "reference"):
        return value
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module_path = Path(__file__).with_name("adapter.py")
    spec = importlib.util.spec_from_file_location("task04_metrics", module_path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    report = {"schema_version": 1, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "scope": "Complete declared real inputs; single observations, not stable speed estimates",
              "baseline_commit": "cc9402734faeba93b3a13c29932fa1392eaccf62",
              "suites": {}, "coverage_gaps": ["CA/CAT genuine same-subject myelin and iteration bias provenance not available"],
              "privacy": "Aggregate metrics only; no new individual images or per-vertex arrays"}
    for name, manifest_name in SUITES.items():
        source = args.run_root / name / "suite.private.json"
        if not source.is_file():
            report["suites"][name] = {"status": "not_started_or_waiting_for_lock"}
            continue
        suite = json.loads(source.read_text())
        manifest = json.loads((args.run_root / "preparation" / manifest_name).read_text())
        cases = {case["id"]: case for case in manifest["cases"]}
        out = {"status": suite["status"], "host": suite["host"],
               "manifest_sha256": suite["manifest_sha256"], "records": [], "native8_vs_native1": {}}
        out["runtime_source_at_lock_acquisition"] = {
            backend: {"tree_sha256": metadata["tree_sha256"], "owned_files": {
                path: digest for path, digest in metadata["files"].items()
                if path.startswith("src/fnit/msm/") or path.startswith("src/fnit/fmri/surface")}}
            for backend, metadata in suite["source_metadata"].items()}
        completed = {}
        for row in suite["records"]:
            clean = {key: row.get(key) for key in (
                "case_id", "threads", "affinity", "full_process", "process_cpu_usage", "load_before", "load_after", "adapter_accuracy")}
            clean["api"] = {backend: [{key: worker.get(key) for key in (
                "scope", "seconds", "maximum_rss_kib", "threads", "affinity", "versions", "tf32", "cuda_peak_allocation_bytes", "output_metadata")}
                for worker in workers] for backend, workers in row["worker_results"].items()}
            official_dir = args.run_root / name / f"threads_{row['threads']}" / row["case_id"] / "pair_0" / "official"
            for filename in ("reference_report.public.json", "results/run.public.json"):
                path = official_dir / filename
                if path.is_file():
                    clean["original_stage_numeric_report"] = numeric_tree(json.loads(path.read_text()))
            for backend, workers in row["worker_results"].items():
                if not workers:
                    continue
                output_paths = workers[0]["outputs"]
                for role in ("registration_report", "timing_report"):
                    if role in output_paths:
                        clean.setdefault("fnit_stage_numeric_report", {}).setdefault(backend, {})[role] = numeric_tree(
                            json.loads(Path(output_paths[role]).read_text()))
            out["records"].append(clean)
            completed[(row["case_id"], row["threads"])] = official_dir
        for case_id, case in cases.items():
            if (case_id, 1) in completed and (case_id, 8) in completed:
                resources = {**manifest["resources"], "threads": 8}
                first = plugin.reference_outputs(case, completed[(case_id, 1)], resources)
                second = plugin.reference_outputs(case, completed[(case_id, 8)], resources)
                out["native8_vs_native1"][case_id] = plugin.compare_case(
                    case, {"candidate": second, "official": first}, resources)["candidate_vs_official"]
        report["suites"][name] = out
    nodes = args.run_root / "original_nodes_spectra_node10_cpu1_cpu8/run.private.json"
    if nodes.is_file():
        data = json.loads(nodes.read_text())
        branch = {key: data[key] for key in ("status", "scope", "records")}
        branch["manifest_sha256"] = data["manifest_sha256"]
        branch["numeric_comparisons"] = []
        for row in data["records"]:
            native_node = nodes.parent / f"threads_{row['threads']}" / row["case_id"] / "individual_maps_ts.txt"
            if row["returncode"] != 0 or not row["original_nodes_written"] or not native_node.is_file():
                continue
            for name in ("features_node10_cpu1_cpu8_attempt02", "candidate_features_node10_cpu1_cpu8"):
                suite_path = args.run_root / name / "suite.private.json"
                if not suite_path.is_file():
                    continue
                suite = json.loads(suite_path.read_text())
                # The spectra branch changes only nTPsForSpectra. Pair it with
                # the same declared source BOLD, VN, maps, components and area.
                if suite["manifest_sha256"] != data["manifest_sha256"]:
                    branch["numeric_comparisons"].append({"suite": name,
                        "case_id": row["case_id"], "threads": row["threads"],
                        "status": "unmatched_input_manifest"})
                    continue
                manifest = json.loads((args.run_root / "preparation" / "features.manifest.private.json").read_text())
                case = next(case for case in manifest["cases"] if case["id"] == row["case_id"])
                for completed in suite["records"]:
                    if (completed["case_id"], completed["threads"]) != (row["case_id"], row["threads"]):
                        continue
                    for backend, workers in completed["worker_results"].items():
                        if backend == "official" or not workers:
                            continue
                        result = plugin.compare_case(case, {
                            "candidate": {"nodes": workers[0]["outputs"]["nodes"]},
                            "official": {"nodes": native_node}}, manifest["resources"])
                        branch["numeric_comparisons"].append({"suite": name,
                            "case_id": row["case_id"], "threads": row["threads"],
                            "backend": backend, "status": "compared_complete_saved_nodes",
                            "scope": "Full declared frame/component matrix; no spectra speed ratio",
                            "nodes": result["candidate_vs_official"]["nodes"]})
        report["original_nodes_branch"] = branch
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "aggregate_report_written", "suite_states": {
        name: data["status"] for name, data in report["suites"].items()}}))


if __name__ == "__main__":
    main()
