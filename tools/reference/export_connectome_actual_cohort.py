"""Export completed actual ten-case tables, execution order and CPU figures.

No MRI computation, interpolation, input mutation, FNIT/torch import or GPU
query is performed. Partial cohorts remain waiting; their final figures are
never rendered. Saved FA undefined values are retained and marked in figures.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

try:
    from . import summarize_connectome_actual_cohort as tables
except ImportError:
    import summarize_connectome_actual_cohort as tables

compare = tables.compare
ARMS = ("baseline", "candidate")
LIMIT = 20_000_000_000


def complete_ten(summary):
    """Coverage gate; equality and MRtrix acceptance are separate conclusions."""
    compare.check(summary.get("status") == "complete_actual_ten_case_tables" and
                  summary.get("requested_cases") == summary.get("completed_pairs") == 10 and
                  summary.get("ready_for_ten_case_render") is True, "ten actual paired cases are not complete")
    cases = summary.get("case_rows", [])
    ids = [row["case_id"] for row in cases]
    compare.check(len(ids) == len(set(ids)) == 10 and all(row.get("paired_status") == "completed_comparison" and
                  row.get("anatomy_status") == "completed" for row in cases), "ten actual independent FS comparisons required")
    anatomy = summary.get("anatomy_rows", [])
    matrices = summary.get("matrix_rows", [])
    sources = summary.get("source_rows", [])
    expected_fs = {name for _, names in compare.anatomy.SCIENTIFIC_GROUPS for name in names}
    atlases = summary.get("atlas_names", [])
    compare.check(len(atlases) == len(set(atlases)) == 8, "eight actual declared atlases required")
    compare.check(len(anatomy) == 130 and {(row["case_id"], row["file"]) for row in anatomy} ==
                  {(case, name) for case in ids for name in expected_fs} and
                  all(row.get("status") == "completed" for row in anatomy), "13 actual FS files per case required")
    compare.check(len(matrices) == 320 and {(row["case_id"], row["atlas"], row["kind"]) for row in matrices} ==
                  {(case, atlas, kind) for case in ids for atlas in atlases for kind in compare.MATRICES} and
                  all(row.get("status") == "completed" and isinstance(row.get("node_count"), int) and row["node_count"] > 0
                      and row.get("nodes_semantics_equal") is True for row in matrices), "32 actual case-specific matrices required")
    compare.check(len(sources) == 20 and {(row["case_id"], row["arm"]) for row in sources} ==
                  {(case, arm) for case in ids for arm in ARMS} and
                  all(row.get("status") == "completed_case_verified" for row in sources), "20 actual executed source ledgers required")
    return ids


def number(value, label, *, positive=False):
    compare.check(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and
                  (value > 0 if positive else value >= 0), f"invalid actual {label}")
    return value


def utc_value(value):
    compare.check(isinstance(value, str), "missing actual execution UTC")
    result = datetime.fromisoformat(value)
    compare.check(result.utcoffset() is not None, "execution UTC requires timezone")
    return result


def memory_record(GPU):
    """Independently inspect original sampler/allocator, including interval gaps."""
    memory = GPU.get("gpu_memory", {})
    process, allocator = memory.get("process", {}), memory.get("allocator", {})
    compare.check(process.get("status") == "measured" and not process.get("errors") and
                  process.get("failed_samples") == process.get("unresolved_device_samples") == 0 and
                  isinstance(process.get("samples"), int) and process["samples"] > 0, "actual NVML process measurement incomplete")
    peaks = {"process_tree_bytes": process.get("peak_process_tree_bytes"),
             "allocated_bytes": allocator.get("allocated_bytes"), "reserved_bytes": allocator.get("reserved_bytes")}
    for key, value in peaks.items():
        number(value, key)
        compare.check(value < LIMIT, "actual GPU peak is outside strict 20 GB budget")
    budget = GPU.get("memory_budget", {})
    expected = {"process_tree": peaks["process_tree_bytes"], "allocated_bytes": peaks["allocated_bytes"],
                "reserved_bytes": peaks["reserved_bytes"]}
    compare.check(budget.get("status") == "observed_below_budget" and not budget.get("monitor_issues") and
                  budget.get("limit_bytes") == LIMIT and budget.get("measurements") == expected,
                  "original independent peak measurements disagree with qualification")
    return {**peaks, "sample_interval_seconds": number(process.get("sample_interval_seconds"), "sample interval", positive=True),
            "max_observed_interval_seconds": number(process.get("max_observed_interval_seconds"), "sample gap", positive=True),
            "observed_span_seconds": number(process.get("observed_span_seconds"), "sampled span", positive=True),
            "samples": process["samples"], "gpu_uuid": process.get("gpu_uuid"),
            "process_scope": process.get("scope"), "allocator_scope": allocator.get("scope"),
            "continuous_bound": False}


def execution_records(summary, configuration):
    """Read actual immutable reports; list UTC order only on their common host."""
    complete_ten(summary)
    sources = {(row["case_id"], row["arm"]): row for row in summary["source_rows"]}
    records = []
    for case in summary["case_rows"]:
        case_id = case["case_id"]
        for arm in ARMS:
            source = sources[case_id, arm]
            root = source.get("actual_GPU_root") or configuration[arm + "_root"]
            job = Path(root) / arm / case_id
            GPU, gpu_identity = tables.bounded_json(job / "gpu_report.json")
            wall, wall_identity = tables.bounded_json(job / "raw_bids_wall.json")
            compare.check(gpu_identity["sha256"] == source["GPU_report_sha256"] and wall_identity["sha256"] == source["wall_report_sha256"] and
                          GPU.get("case_id") == case_id and GPU.get("version") == arm and GPU.get("status") == "completed" and GPU.get("exit_code") == 0 and
                          wall.get("status") == "completed" and wall.get("exit_code") == 0 and wall.get("outputs", {}).get("status") == "complete",
                          "actual completed GPU/CLI identity changed")
            start, end = wall.get("start_utc"), wall.get("end_utc")
            compare.check(utc_value(end) >= utc_value(start), "actual same-host CLI end precedes start")
            record = {"case_id": case_id, "arm": arm, "host": GPU["identity"]["hostname"],
                      "actual_GPU_root": root, "GPU_origin_binding": source.get("GPU_origin_binding"),
                      "worker_start_utc": GPU.get("start_utc"), "worker_end_utc": GPU.get("end_utc"),
                      "CLI_start_utc": start, "CLI_end_utc": end, "command": GPU["command"],
                      "source_fingerprint": source["source_fingerprint"], "GPU_report": gpu_identity, "wall_report": wall_identity,
                      "actual_eddy_gp_seeds": wall.get("actual_eddy_gp_seeds"),
                      "official_recon_command_seconds": number(case[arm + "_official_recon_command_seconds"], "official reconstruction time"),
                      "full_timing_scope": case[arm + "_full_timing_scope"], "memory": memory_record(GPU)}
            for field in tables.TIME_FIELDS:
                actual = wall.get("total_runtime_seconds") if field == "raw_dwi_cli_total_runtime_seconds" else GPU.get(field)
                compare.check(actual == case[arm + "_" + field], "actual timer differs from completed comparison table")
                record[field] = number(actual, field)
            records.append(record)
    compare.check(len({row["host"] for row in records}) == 1, "global UTC execution order needs a common actual GPU host")
    records.sort(key=lambda row: (utc_value(row["CLI_start_utc"]), utc_value(row["CLI_end_utc"]), row["arm"], row["case_id"]))
    for index, row in enumerate(records, 1): row["actual_CLI_start_order"] = index
    return records


def summarize_memory(records):
    result = {}
    for arm in (*ARMS, "all"):
        selected = [row["memory"] for row in records if arm == "all" or row["arm"] == arm]
        compare.check(selected, "no actual memory observations")
        result[arm] = {"actual_runs": len(selected),
            **{key: max(row[key] for row in selected) for key in ("allocated_bytes", "reserved_bytes", "process_tree_bytes", "max_observed_interval_seconds")},
            "requested_sample_intervals_seconds": sorted({row["sample_interval_seconds"] for row in selected}),
            "sampled_spans_seconds": [row["observed_span_seconds"] for row in selected],
            "samples": sum(row["samples"] for row in selected), "limit_bytes": LIMIT,
            "scope": "actual per-run sampler maxima plus allocator ledger; sample intervals and largest measured gap retained; not a continuous mathematical bound"}
    return result


def scientific_modules():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    return matplotlib, plt, nib, np


def file_identity(path):
    path = Path(path)
    return {"path": str(path), "sha256": compare.anatomy.sha(path)}


def load_plot_inputs(output, atlas, ledger, expected_K):
    """Read full native-grid arrays. Axis permutations/flips are not interpolation."""
    _, _, nib, np = scientific_modules()
    names = ["fa_dwi.nii.gz", "brain_mask_dwi.nii.gz", f"atlases/{atlas}/atlas_dwi.nii.gz", f"atlases/{atlas}/nodes.tsv",
             *[f"atlases/{atlas}/connectome_{kind}.csv" for kind in compare.MATRICES]]
    identities = {}
    for name in names:
        path = Path(output) / name
        original = ledger[name]
        compare.check(path.is_file() and not path.is_symlink() and original.get("exists") is True and original.get("path") == str(path) and
                      compare.anatomy.sha(path) == original["sha256"], "actual figure input changed")
        identities[name] = file_identity(path)
    _, nodes = compare.node_semantics(Path(output) / f"atlases/{atlas}/nodes.tsv")
    compare.check(len(nodes) == expected_K, "actual per-case figure nodes differ")
    images = [nib.as_closest_canonical(nib.load(Path(output) / name)) for name in names[:3]]
    compare.check(len(images[0].shape) == 3 and all(image.shape == images[0].shape and np.array_equal(image.affine, images[0].affine)
                  for image in images), "actual FA/mask/atlas grids differ; figure generator never resamples")
    fa, mask, labels = [np.asanyarray(image.dataobj) for image in images]
    compare.check(np.isfinite(mask).all() and np.isfinite(labels).all() and np.array_equal(labels, np.rint(labels)) and
                  np.all((labels >= 0) & (labels <= expected_K)), "invalid actual figure mask or node labels")
    matrices = {}
    for kind in compare.MATRICES:
        data = np.loadtxt(Path(output) / f"atlases/{atlas}/connectome_{kind}.csv", delimiter=",", ndmin=2)
        compare.check(data.shape == (expected_K, expected_K) and np.isfinite(data).all() and np.array_equal(data, data.T),
                      "actual figure matrix/node shape, symmetry or finite gate invalid")
        if kind == "count": compare.check(np.all(data >= 0) and np.array_equal(data, np.rint(data)), "invalid actual count matrix")
        matrices[kind] = data
    compare.check(np.any(mask > 0), "empty actual brain mask")
    center = [int(np.median(indices)) for indices in np.where(mask > 0)]
    undefined = ~np.isfinite(fa)
    nan_positions = np.argwhere(undefined)
    undefined_z = int(nan_positions[0, 2]) if len(nan_positions) else center[2]
    return {"fa": fa, "mask": mask, "labels": labels, "matrices": matrices, "nodes": nodes,
            "center": center, "undefined_z": undefined_z, "identities": identities,
            "canonical_affine": images[0].affine.tolist(), "shape": list(fa.shape),
            "undefined_counts": {"NaN": int(np.isnan(fa).sum()), "positive_Inf": int(np.isposinf(fa).sum()), "negative_Inf": int(np.isneginf(fa).sum())}}


def save_figure(figure, path, plt):
    compare.check(not path.exists() and not path.is_symlink(), "fresh figure output required")
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)
    return file_identity(path)


def render_case(case_id, atlas, records, summary, configuration, report_dir):
    _, plt, _, np = scientific_modules()
    K = next(row["node_count"] for row in summary["matrix_rows"] if row["case_id"] == case_id and row["atlas"] == atlas)
    data, wall_ledgers = {}, {}
    for arm in ARMS:
        record = next(row for row in records if row["case_id"] == case_id and row["arm"] == arm)
        wall, identity = tables.bounded_json(record["wall_report"]["path"])
        compare.check(identity == record["wall_report"], "actual figure wall changed")
        wall_ledgers[arm] = wall["outputs"]["files"]
        data[arm] = load_plot_inputs(Path(record["actual_GPU_root"]) / arm / case_id / "connectome", atlas, wall_ledgers[arm], K)
    compare.check(data["baseline"]["nodes"] == data["candidate"]["nodes"], "actual figure node semantics differ")
    finite = np.concatenate([value["fa"][np.isfinite(value["fa"]) & (value["mask"] > 0)] for value in data.values()])
    maximum = float(np.percentile(finite, 99)) if finite.size else 1.
    maximum = maximum if maximum > 0 else 1.
    cmap = plt.colormaps["gray"].copy(); cmap.set_bad("magenta")
    figure, axes = plt.subplots(2, 4, figsize=(13, 7), facecolor="white")
    for row, arm in enumerate(ARMS):
        value = data[arm]; x, y, z = value["center"]
        views = ((0, x, "sagittal"), (1, y, "coronal"), (2, z, "axial"), (2, value["undefined_z"], "undefined-value slice"))
        for axis, (dim, index, title) in zip(axes[row], views):
            fa = np.take(value["fa"], index, axis=dim).T
            labels = np.take(value["labels"], index, axis=dim).T
            axis.imshow(np.ma.masked_invalid(fa), origin="lower", cmap=cmap, vmin=0, vmax=maximum, interpolation="none")
            axis.imshow(np.ma.masked_where(labels == 0, labels), origin="lower", cmap="turbo", vmin=1, vmax=K,
                        alpha=.32, interpolation="none")
            axis.set_title(f"{arm}: {title}, index={index}", fontsize=9); axis.axis("off")
        counts = value["undefined_counts"]
        axes[row, 0].text(0, -.08, f"NaN={counts['NaN']}, +Inf={counts['positive_Inf']}, -Inf={counts['negative_Inf']}", transform=axes[row, 0].transAxes, fontsize=9)
    figure.suptitle(f"{case_id} | {atlas}, K={K} | saved FA + DWI atlas\nNative voxels; canonical axis flips/permutations only; undefined FA magenta", fontsize=12)
    brain = save_figure(figure, report_dir / f"{case_id}.{atlas}.brain.png", plt)
    figure, axes = plt.subplots(3, 4, figsize=(14, 10), facecolor="white")
    for col, kind in enumerate(compare.MATRICES):
        a, b = [data[arm]["matrices"][kind] for arm in ARMS]
        high = max(float(a.max()), float(b.max()), 1e-12)
        low = min(float(a.min()), float(b.min()), 0.)
        for row, (arm, matrix) in enumerate(zip(ARMS, (a, b))):
            image = axes[row, col].imshow(matrix, origin="lower", cmap="viridis", vmin=low, vmax=high,
                                          extent=(.5, K + .5, .5, K + .5), interpolation="none")
            axes[row, col].set_title(f"{arm} {kind}, K={K}", fontsize=10)
            figure.colorbar(image, ax=axes[row, col], shrink=.7)
        delta = np.abs(b - a); high_delta = float(delta.max())
        image = axes[2, col].imshow(delta, origin="lower", cmap="magma", vmin=0, vmax=high_delta or 1.,
                                   extent=(.5, K + .5, .5, K + .5), interpolation="none")
        axes[2, col].set_title(f"absolute difference; max={high_delta:.3g}", fontsize=10)
        figure.colorbar(image, ax=axes[2, col], shrink=.7)
    figure.suptitle(f"{case_id} | {atlas} | all persisted matrix entries, original ordered nodes\nCSV decoded float64; common per-kind color scale; no trimming/padding", fontsize=12)
    matrix = save_figure(figure, report_dir / f"{case_id}.{atlas}.matrices.png", plt)
    for arm in ARMS:
        for identity in data[arm]["identities"].values():
            compare.check(compare.anatomy.sha(identity["path"]) == identity["sha256"], "actual input changed during figure generation")
    return {"case_id": case_id, "atlas": atlas, "node_count": K, "brain_figure": brain, "matrix_figure": matrix,
            "inputs": {arm: data[arm]["identities"] for arm in ARMS},
            "geometry": {arm: {key: data[arm][key] for key in ("shape", "canonical_affine", "center", "undefined_z", "undefined_counts")} for arm in ARMS},
            "scope": "actual saved MRI/matrix CPU display; no interpolation, approximation or MRI modifications; FA undefined values retained and marked"}


def render_timings(records, report_dir):
    _, plt, _, np = scientific_modules()
    figure, axes = plt.subplots(3, 1, figsize=(14, 11))
    positions = np.arange(len(records)); colors = ["#3374a2" if row["arm"] == "baseline" else "#cf783c" for row in records]
    labels = [f"{row['actual_CLI_start_order']} {row['case_id']}\n{row['arm']}" for row in records]
    axes[0].bar(positions, [row["raw_dwi_cli_total_runtime_seconds"] for row in records], color=colors)
    axes[0].set_ylabel("raw DWI CLI seconds"); axes[0].set_title("Actual CLI start order on the same GPU host; includes import, full raw-DWI computation and output writes")
    axes[1].bar(positions, [row["gpu_lock_queue_seconds"] for row in records], color=colors)
    axes[1].set_ylabel("GPU lock queue seconds"); axes[1].set_title("Measured queue shown separately; never summed stage medians or inferred cold pipeline time")
    for key, color in (("allocated_bytes", "#3877ad"), ("reserved_bytes", "#e5a34a"), ("process_tree_bytes", "#8d69b4")):
        offset = (list(("allocated_bytes", "reserved_bytes", "process_tree_bytes")).index(key) - 1) * .24
        axes[2].bar(positions + offset, [row["memory"][key] / 1e9 for row in records], width=.24, color=color, label=key)
    axes[2].axhline(20, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("GB, decimal"); axes[2].set_title("Allocator maxima and sampled process-tree peak; sampled maximum is not a continuous bound"); axes[2].legend(fontsize=9)
    for axis in axes:
        axis.set_xticks(positions, labels, fontsize=7); axis.grid(axis="y", alpha=.2)
    return save_figure(figure, report_dir / "actual_order_timing_memory.png", plt)


def write_csv(path, records):
    fields = ("actual_CLI_start_order", "case_id", "arm", "host", "CLI_start_utc", "CLI_end_utc", "worker_start_utc", "worker_end_utc",
              "official_recon_command_seconds", *tables.TIME_FIELDS, "allocated_bytes", "reserved_bytes", "process_tree_bytes",
              "sample_interval_seconds", "max_observed_interval_seconds", "observed_span_seconds", "samples", "source_fingerprint")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore"); writer.writeheader()
        writer.writerows({**row, **row["memory"]} for row in records)
    temporary.replace(path)


def protected_export_namespaces(state, comparison_root, summary_root=None):
    configuration = state["configuration"]
    manifest, identity = tables.bounded_json(configuration["manifest"])
    compare.check(identity == state["manifest"], "actual raw manifest changed before final export")
    cases = compare.manifest_cases(manifest)
    origins, _, _ = compare.load_origins(configuration["prep_bindings"], cases)
    protected = [comparison_root, *[configuration[key] for key in ("baseline_anatomy_root", "baseline_root", "candidate_root")],
        *[Path(configuration[key]).parent for key in ("manifest", "prep_bindings", "baseline_driver", "candidate_driver")],
        *[origin[key] for origin in origins for key in ("root", "driver_dir")],
        *[Path(item["path"]).parent for case in cases for item in case["input_files"]]]
    if summary_root is not None: protected.append(summary_root)
    for arm, filename in (("baseline", "cohort_config.json"), ("candidate", "staged_gpu_config.json")):
        config, _ = tables.bounded_json(Path(configuration[arm + "_root"]) / filename)
        protected.extend(source["directory"] for source in config["frozen_sources"].values())
    for binding in tables.configuration_GPU_origins(configuration, cases).values():
        declaration = binding["declaration"]
        protected += [declaration["replacement"]["root"], Path(declaration["replacement"]["driver_status"]).parent,
                      Path(declaration["replacement"]["runtime_preflight"]["path"]).parent, Path(declaration["original"]["driver_snapshot"]["path"]).parent]
    if state.get("prior_comparison_binding"): protected.append(Path(state["prior_comparison_binding"]["original_status"]["path"]).parent)
    return protected


def export(comparison_root, report_dir, representatives, atlas, *, candidate_source_label=None):
    started = time.perf_counter()
    report_dir = Path(report_dir)
    state, state_identity = tables.bounded_json(Path(comparison_root) / "status.json")
    compare.check(report_dir.is_absolute() and report_dir.is_dir() and not report_dir.is_symlink(), "new explicit final export directory required")
    compare.check_report_namespace(report_dir, protected_export_namespaces(state, comparison_root))
    compare.check(all(path.name == "status.json" and path.is_file() and not path.is_symlink() for path in report_dir.iterdir()), "final export never overwrites prior tables or figures")
    summary = tables.build_summary(comparison_root, candidate_source_label=candidate_source_label)
    ids = complete_ten(summary)
    compare.check(representatives and len(set(representatives)) == len(representatives) and all(case in ids for case in representatives) and
                  atlas in summary["atlas_names"], "representative case or atlas is outside the actual cohort")
    compare.check(state_identity["sha256"] == summary["comparison_status_sha256"], "completed comparison state changed")
    configuration = state["configuration"]
    records = execution_records(summary, configuration)
    memory = summarize_memory(records)
    tables.write_tables(report_dir, summary); write_csv(report_dir / "actual_execution_order.csv", records)
    figures = [render_case(case, atlas, records, summary, configuration, report_dir) for case in representatives]
    timing_figure = render_timings(records, report_dir)
    matplotlib, _, nib, np = scientific_modules()
    protocol = {"schema_version": 1, "status": "completed_actual_ten_case_export", "observed_utc": compare.anatomy.utc(),
        "completed_pairs": 10, "completed_independent_anatomy_cases": 10, "compared_matrices": 320, "independent_anatomy_files": 130,
        "comparison_status": state_identity, "summary_sha256": compare.anatomy.sha(report_dir / "summary.json"),
        "actual_execution_order": records, "memory": memory, "figures": figures, "timing_memory_figure": timing_figure,
        "scientific_equal": {"independent_FS_all": all(row["anatomy_exact"] for row in summary["case_rows"]),
            "count_all": all(row["count_exact_all_atlases"] for row in summary["case_rows"]),
            "all_matrices": all(row["matrix_numeric_exact_all_atlases"] for row in summary["case_rows"]),
            "shared_images_all": all(values["strict_scientific_equal"] for row in summary["case_rows"] for values in row["images"].values()),
            "numeric_files_all": all(values["exact_scientific_array_equal"] for row in summary["case_rows"] for values in row["numeric_files"].values())},
        "runtime": {"python": sys.executable, "python_sha256": compare.anatomy.sha(sys.executable), "python_version": sys.version,
            "host": platform.node(), "matplotlib": matplotlib.__version__, "numpy": np.__version__, "nibabel": nib.__version__,
            "tool": file_identity(__file__)},
        "scope": "actual ten-pair completed outputs; actual common-GPU-host CLI UTC start order; every original FS/CLI/worker/queue/gap scope retained separately, not a synthetic continuous cold run; no MRtrix acceptance or stable speedup inferred",
        "MRI_source_modified": False, "GPU_used": False, "CPU_export_wall_seconds": time.perf_counter() - started}
    for record in records:
        for key in ("GPU_report", "wall_report"):
            compare.check(compare.anatomy.sha(record[key]["path"]) == record[key]["sha256"], "actual executed report changed during export")
    compare.check(compare.anatomy.sha(state_identity["path"]) == state_identity["sha256"], "actual completed comparison changed during export")
    compare.anatomy.atomic_json(report_dir / "final_report.json", protocol)
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path, required=True, help="actual existing summary observer; only complete ten-pair status permits export")
    parser.add_argument("--report-dir", type=Path, required=True, help="fresh final-export namespace")
    parser.add_argument("--representative-cases", nargs="+", default=["sub-CON01", "sub-CON09"])
    parser.add_argument("--figure-atlas", default="fs-aparc")
    parser.add_argument("--candidate-source-label")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=72)
    options = parser.parse_args(argv)
    compare.check(all(path.is_absolute() for path in (options.comparison_root, options.summary_root, options.report_dir)) and
                  1 <= options.poll_seconds <= 60 and 0 < options.timeout_hours <= 168, "invalid paths or observation interval")
    state, _ = tables.bounded_json(options.comparison_root / "status.json")
    compare.check_report_namespace(options.report_dir, protected_export_namespaces(state, options.comparison_root, options.summary_root))
    options.report_dir.parent.mkdir(parents=True, exist_ok=True); options.report_dir.mkdir(exist_ok=False)
    started = time.perf_counter()
    try:
        while True:
            actual, identity = tables.bounded_json(options.summary_root / "summary.json")
            if actual.get("status", "").startswith("failed"):
                raise ValueError("actual summary observer failed; no final rendering")
            if actual.get("ready_for_ten_case_render") is True:
                complete_ten(actual)
                _, current_state = tables.bounded_json(options.comparison_root / "status.json")
                compare.check(actual.get("comparison_status_sha256") == current_state["sha256"], "ready summary is not bound to current actual comparison")
                result = export(options.comparison_root, options.report_dir, options.representative_cases, options.figure_atlas,
                                candidate_source_label=options.candidate_source_label)
                compare.anatomy.atomic_json(options.report_dir / "status.json", {"status": result["status"], "completed_pairs": 10,
                    "completed_independent_anatomy_cases": 10, "summary_observation": identity, "GPU_used": False})
                break
            waiting = {"status": "waiting_actual_ten_case_outputs", "observed_utc": compare.anatomy.utc(),
                "summary_observation": identity, "completed_pairs": actual.get("completed_pairs"),
                "completed_independent_anatomy_cases": sum(row.get("anatomy_status") == "completed" for row in actual.get("case_rows", [])),
                "figures_created": False, "GPU_used": False}
            compare.anatomy.atomic_json(options.report_dir / "status.json", waiting)
            print(json.dumps(waiting), flush=True)
            if not options.watch or (options.report_dir / "STOP_OBSERVATION").exists() or time.perf_counter() - started >= options.timeout_hours * 3600: break
            time.sleep(options.poll_seconds)
    except Exception as error:
        compare.anatomy.atomic_json(options.report_dir / "status.json", {"status": "failed_actual_ten_case_export", "observed_utc": compare.anatomy.utc(),
            "error": {"type": type(error).__name__, "message": str(error)}, "GPU_used": False, "MRI_source_modified": False})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
