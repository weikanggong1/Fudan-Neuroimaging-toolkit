"""Prepare full-real-data functional jobs without starting server processes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_baseline import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--cpu-lock", type=Path, required=True)
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    group = [int(value) for value in args.cpu_list.split(",")]
    if len(group) != 8 or len(set(group)) != 8:
        raise ValueError("Provide the assigned eight distinct physical CPU IDs")
    baseline_revision = "cc9402734faeba93b3a13c29932fa1392eaccf62"
    jobs = []

    def add(name, backend, source_kind, threads, function, case, options):
        source = args.baseline_source if source_kind == "baseline" else args.candidate_source
        revision = baseline_revision if source_kind == "baseline" else args.candidate_revision
        output = args.run_root / name
        common = ["--case-json", str(args.case_json), "--case", case, "--source", str(source),
                  "--source-revision", revision, "--output-dir", str(output), "--threads", str(threads),
                  "--cpu-list", str(group[0]) if threads == 1 else args.cpu_list,
                  "--cpu-lock", str(args.cpu_lock)]
        if function in ("cli_contract", "wrapper_parity"):
            command = [str(args.python), str(args.task_dir / "feature_checks.py"), "--kind", function] + common
        else:
            command = [str(args.python), str(args.task_dir / "benchmark_baseline.py"),
                       "--function", function, "--backend", backend] + common + options
        jobs.append({"name": name, "backend": backend, "source_kind": source_kind,
                     "function": function, "case": case, "threads": threads,
                     "cpu_list": common[common.index("--cpu-list") + 1], "output_dir": str(output),
                     "command": command, "full_real_frames_required": True,
                     "adapter_sha256": digest(Path(__file__).with_name("feature_checks.py" if function in ("cli_contract", "wrapper_parity") else "benchmark_baseline.py"))})

    # Native-supported modes retain all 180 frames and the full spatial grid.
    native_variants = (
        ("middle_linear_stage1", ["--reference-mode", "middle", "--interpolation", "linear", "--stages", "1"]),
        ("external_linear_stage2", ["--reference-mode", "external", "--interpolation", "linear", "--stages", "2"]),
        ("middle_spline_stage3", ["--reference-mode", "middle", "--interpolation", "spline", "--stages", "3"]),
    )
    for threads in (1, 8):
        for variant, options in native_variants:
            add(f"{variant}_official_cpu{threads}", "official", "baseline", threads, "mcflirt", "mc180", options + ["--native-trace"])
            add(f"{variant}_candidate_cpu{threads}", "fnit", "candidate_v4", threads, "mcflirt", "mc180", options)
            if threads == 8:
                add(f"{variant}_baseline_cpu8", "fnit", "baseline", 8, "mcflirt", "mc180", options)

    # API iteration extensions have no equivalent MCFLIRT CLI switch.
    for variant, options in (
            ("two_first_stage_iterations", ["--stage-iterations", "2", "1", "1", "--estimate-only"]),
            ("skip_first_stage_image_estimate", ["--stage-iterations", "0", "1", "1", "--estimate-only", "--input-mode", "image"])):
        for kind in ("baseline", "candidate_v4"):
            add(f"{variant}_{kind}_cpu8", "fnit", kind, 8, "mcflirt", "mc180", options)

    # Complete default schedule is used for reference/batch/automatic cases.
    for variant, options in (
            ("reference_image_array_init", ["--execution", "reference", "--input-mode", "image", "--initialization", "array"]),
            ("batched_one_candidate", ["--candidate-batch-size", "1"]),
            ("no_grid_image_array_init", ["--no-grid-search", "--input-mode", "image", "--initialization", "array"]),
            ("automatic_initialization", ["--initialization", "automatic"])):
        for kind in ("baseline", "candidate_v4"):
            add(f"bbr_{variant}_{kind}_cpu8", "fnit", kind, 8, "bbr", "bbr", options)

    for function in ("cli_contract", "wrapper_parity"):
        for kind in ("baseline", "candidate_v4"):
            add(f"mc180_{function}_{kind}_cpu8", "fnit", kind, 8, function, "mc180", [])

    source_manifest = json.loads(args.source_manifest.read_text())
    manifest = {
        "status": "prepared_not_executed", "baseline_commit": baseline_revision,
        "source_revision": args.candidate_revision, "cpu_group": group, "cpu_lock": str(args.cpu_lock),
        "default_slice_timing": False, "jobs": jobs,
        "adapter_sha256": digest(Path(__file__).with_name("benchmark_baseline.py")),
        "feature_worker_sha256": digest(Path(__file__).with_name("feature_checks.py")),
        "candidate_source_sha256": source_manifest["optimized_source_sha256"],
        "comparison_contract": "Complete matrices, parameter convention, image grid/dtype/header and all output values; native exec/exit/SIGCHLD proof required. API-only modes compare frozen baseline. Functional repeats do not replace matched process-scope default speed gates.",
        "execution_contract": "Coordinator confirms finished default v4 gates and free assigned CPU group before launch. Controller must not hold the worker CPU lock. Frozen input/source paths and prior results are retained.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Choose a fresh prepared manifest")
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "prepared_not_executed", "jobs": len(jobs), "official_jobs": sum(j["backend"] == "official" for j in jobs)}))


if __name__ == "__main__":
    main()
