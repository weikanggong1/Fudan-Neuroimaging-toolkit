"""冻结真实首试步的完整有序接受 ABBA；不重新计算 MRI 或借用参考接受状态。"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import time
import traceback

import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(a, b):
    return {
        "different_coordinate_elements": int(np.count_nonzero(a[0] != b[0])),
        "max_coordinate_difference_mm": float(np.max(np.abs(a[0].astype(np.float64) - b[0]))),
        "different_order_elements": int(np.count_nonzero(a[1] != b[1])) if a[1].shape == b[1].shape else None,
        "same_order": bool(np.array_equal(a[1], b[1])),
        "different_accepted_offset_elements": int(np.count_nonzero(a[2] != b[2])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--historical-reference", type=Path)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compare-grid-cells", action="store_true", help="保持GPU有序接受，配对完整2³/3³桶索引")
    parser.add_argument("--compare-retained-mht", action="store_true",
                        help="读真实拒绝重试检查点，原桶树循环与显式编译循环ABBA")
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if args.compare_grid_cells and args.compare_retained_mht:
        parser.error("grid tuning and retained MHT must be measured separately")
    if args.output_directory.exists():
        raise FileExistsError(args.output_directory)
    args.output_directory.mkdir(parents=True)
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.candidate_directory.resolve()))
    from fnit.recon_all import place_surface_collision as collision
    from fnit.recon_all.place_surface_candidates_torch import conservative_face_candidates_torch
    from fnit.recon_all.place_surface_snapshot import snapshot_ordered_step
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("complete trial GPU benchmark requires an explicitly indexed CUDA device")
    report = {
        "scope": "complete_frozen_real_first_collision_trial_ordered_acceptance_only",
        "hostname": platform.node(), "threads": args.threads, "device": str(device),
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "code_commit": args.code_commit, "input_sha256": sha256(args.input),
        "script_sha256": sha256(__file__), "source_sha256": {},
        "tolerance_before_run": {"coordinate_mm": 0, "acceptance_offsets": 0, "order": 0},
        "tf32_matmul": True, "tf32_cudnn": True, "half_precision": False,
        "strategy": "GPU conservative widephase followed by unchanged live ordered Numba narrowphase",
        "reference": "same-host current tree implementation; saved H100 reference reported separately",
        "full_pial": "not_run", "whole_recon_all": "not_run", "overall_metric_equivalence": "not_assessed",
        "process_memory": "not_measured; only target-device allocator counters",
        "runs": [],
    }
    for function in (collision.asynchronous_first_step, conservative_face_candidates_torch, snapshot_ordered_step):
        source = Path(inspect.getfile(inspect.unwrap(function)))
        report["source_sha256"][source.name] = sha256(source)
    with np.load(args.input) as data:
        vertices, faces, proposal, ripped = (data[name] for name in ("vertices", "faces", "proposal", "ripped"))
        if args.compare_retained_mht:
            momentum,offsets,neighbors,valid,stale=(data[name] for name in
                ("accepted_offsets_initial","offsets","neighbors","neighbor_valid","stale_mht_trial"))
            checkpoint_expected=tuple(data[name] for name in
                ("expected_coordinates","expected_order","expected_accepted_offsets"))
            report["scope"]="complete_frozen_real_retained_MHT_trial_ordered_acceptance_only"
            report["reference"]="same-host source tree loop and saved real stage trial; not official geometry"
        else:
            momentum, offsets, neighbors, valid = (data[name] for name in ("momentum", "offsets", "neighbors", "valid"))
            stale=None
    report["vertices"], report["faces"] = len(vertices), len(faces)

    def save(status):
        report["execution_status"] = status
        temp = args.output_directory / "report.json.tmp"
        temp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        temp.replace(args.output_directory / "report.json")

    save("started")
    context_started = time.perf_counter()
    try:
        torch.cuda.synchronize(device)
        report["cuda_context_setup_seconds"] = time.perf_counter() - context_started
        report["gpu"] = torch.cuda.get_device_name(device)
        cold_results, paired_results = {}, []
        control_name,candidate_name=(("retained_tree","retained_compiled") if args.compare_retained_mht
            else (("torch_snapshot_grid2", "torch_snapshot_grid3") if args.compare_grid_cells
                  else ("tree", "torch_snapshot")))
        sequence = (("cold", control_name), ("cold", candidate_name),
                    ("paired", control_name), ("paired", candidate_name),
                    ("paired", candidate_name), ("paired", control_name))
        report["paired_scope"]=("retained source bucket tree versus compiled live loop; complete grid3"
            if args.compare_retained_mht else ("same GPU complete grid2 versus grid3" if args.compare_grid_cells else "tree versus GPU"))
        for kind, backend in sequence:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            accepted = momentum.copy()
            diagnostics = {}
            coordinates, order = collision.asynchronous_first_step(
                vertices, faces, proposal, ripped, offsets=offsets,
                accepted_offsets=accepted, ordered_neighbors=(neighbors, valid),
                candidate_backend="torch_snapshot" if args.compare_grid_cells or args.compare_retained_mht else backend,
                candidate_grid_cells_per_axis=3 if backend == "torch_snapshot_grid3" or args.compare_retained_mht else 2,
                retained_mht_backend="compiled" if backend=="retained_compiled" else "tree",
                stale_mht_trial=stale,
                candidate_device=str(device) if backend != "tree" else None,
                candidate_diagnostics=diagnostics,
            )
            torch.cuda.synchronize(device)
            result = (coordinates, order, accepted)
            row = {
                "kind": kind, "backend": backend, "seconds": time.perf_counter() - started,
                "diagnostics": diagnostics,
                "coordinate_sha256": hashlib.sha256(coordinates.tobytes()).hexdigest(),
                "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
                "accepted_offset_sha256": hashlib.sha256(accepted.tobytes()).hexdigest(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
            if kind == "cold":
                cold_results[backend] = result
                np.savez(args.output_directory / f"result-{backend}.npz",
                         coordinates=coordinates, order=order, accepted_offsets=accepted)
            else:
                paired_results.append((backend, result))
            report["runs"].append(row)
            save("running")
        report["cold_comparison"] = compare(cold_results[control_name], cold_results[candidate_name])
        report["paired_comparison_to_cold_tree"] = [
            {"backend": backend, **compare(cold_results[control_name], result)} for backend, result in paired_results]
        comparisons = [report["cold_comparison"], *report["paired_comparison_to_cold_tree"]]
        if args.compare_retained_mht:
            report["comparison_to_saved_live_trial"]=compare(cold_results[control_name],checkpoint_expected)
            comparisons.append(report["comparison_to_saved_live_trial"])
        report["strict_same_input_reproduction"] = "passed" if all(
            row["different_coordinate_elements"] == 0 and row["same_order"]
            and row["different_accepted_offset_elements"] == 0 for row in comparisons) else "failed"
        report["new_degradation_under_declared_exact_trial_gate"] = (
            "none_detected" if report["strict_same_input_reproduction"] == "passed" else "detected")
        medians = {backend: statistics.median(row["seconds"] for row in report["runs"]
            if row["kind"] == "paired" and row["backend"] == backend) for backend in (control_name, candidate_name)}
        report["paired_median_seconds"] = medians
        report["speed_ratio_control_over_candidate"] = medians[control_name] / medians[candidate_name]
        if not args.compare_grid_cells and not args.compare_retained_mht:
            report["speed_ratio_tree_over_torch_snapshot"] = medians[control_name] / medians[candidate_name]
        if args.historical_reference is not None:
            with np.load(args.historical_reference) as data:
                historical = (data["coordinates"], data["order"], data["accepted_offsets"])
            report["historical_reference_sha256"] = sha256(args.historical_reference)
            report["cross_environment_comparison_to_historical"] = compare(cold_results[control_name], historical)
        if sha256(args.input) != report["input_sha256"]:
            raise RuntimeError("frozen collision input changed during replay")
        save("complete")
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        save("failed")
        raise


if __name__ == "__main__":
    main()
