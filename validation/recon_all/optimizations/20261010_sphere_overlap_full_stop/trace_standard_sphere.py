"""透明记录完整自产标准球面的首步与各尺度；只用于同输入诊断。"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import nibabel.freesurfer.io as fsio
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--inflated", type=Path, required=True)
    parser.add_argument("--smoothwm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--averaging-device", default="cpu")
    parser.add_argument("--finish-device", default="cpu")
    parser.add_argument("--bbox-mode", choices=("legacy_integer", "source_float"), default=None,
                        help="optional explicit initialization mode, only when the frozen API supports it")
    parser.add_argument("--coordinate-checkpoints", type=int, default=8,
                        help="save full arrays for first N updates; SHA and scalar trace cover every update")
    parser.add_argument("--diagnostic-native-checkpoints", type=Path, default=None,
                        help="isolated diagnosis only: compare reference snapshots without feeding them to the algorithm")
    parser.add_argument("--capture-update", type=int, action="append", default=[],
                        help="additional zero-based updates to archive before/after arrays")
    parser.add_argument("--stop-after-updates", type=int, default=None,
                        help="diagnostic prefix only: stop after N updates without writing a final sphere")
    args = parser.parse_args()
    if args.coordinate_checkpoints < 1:
        raise ValueError("coordinate-checkpoints must be positive")
    if any(index < 0 for index in args.capture_update) or (args.stop_after_updates is not None and args.stop_after_updates < 1):
        raise ValueError("capture indices must be nonnegative and prefix limit positive")
    started = time.perf_counter()
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    sys.path.insert(0, str(args.source_directory))
    import torch
    from fnit.recon_all import sphere_standard_run as operation

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    faces = fsio.read_geometry(args.inflated)[1].astype(np.int32)
    kwargs = dict(finish_device=args.finish_device, averaging_device=args.averaging_device)
    if args.bbox_mode is not None:
        if "bbox_mode" not in inspect.signature(operation.run_standard_sphere).parameters:
            raise ValueError("frozen run_standard_sphere has no explicit bbox mode")
        kwargs["bbox_mode"] = args.bbox_mode
    if "overlap_diagnostics" in inspect.signature(operation.run_standard_sphere).parameters:
        kwargs["overlap_diagnostics"] = True
    rows = []
    report = {"status": "running", "scope": "transparent full same-input sphere trace; no native inputs consumed",
              "script_sha256": sha(Path(__file__)), "source_files_sha256": {},
              "inputs_sha256": {"inflated": sha(args.inflated), "smoothwm": sha(args.smoothwm)},
              "threads": torch.get_num_threads(), "affinity": sorted(os.sched_getaffinity(0)),
              "kwargs": kwargs, "precision": {"cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                                                  "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                                                  "half_precision_requested": False},
              "allocator_environment": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
              "allocator_environment_presence_disables_caching": "PYTORCH_NO_CUDA_MEMORY_CACHING" in os.environ,
              "trace": rows, "coordinate_checkpoints": args.coordinate_checkpoints,
              "additional_capture_updates": args.capture_update, "diagnostic_prefix_limit": args.stop_after_updates}
    if args.diagnostic_native_checkpoints is not None:
        report["scope"] = "isolated full-stage diagnostic observer; reference snapshots compared only, never fed back"
        report["diagnostic_native_checkpoint_directory"] = str(args.diagnostic_native_checkpoints)
    original_first = operation.first_epoch_line_search
    original_nonlinear = operation.nonlinear_epoch_line_search
    original_project = operation.project_radially
    original_finish = operation.finish_standard_sphere
    original_gradient = operation.first_epoch_gradient
    current = None

    class DiagnosticPrefixComplete(Exception):
        pass

    def compare_snapshot(value, index):
        if args.diagnostic_native_checkpoints is None:
            return None
        hemisphere = args.inflated.name.split(".")[0]
        reference = args.diagnostic_native_checkpoints / f"{hemisphere}.sphere{index:04d}"
        if not reference.exists():
            return {"status": "reference_checkpoint_missing", "index": index}
        coordinates, reference_faces = fsio.read_geometry(reference)
        if not np.array_equal(faces, reference_faces) or value.shape != coordinates.shape:
            return {"status": "vertex_correspondence_failed", "index": index}
        delta = value.astype(np.float64) - coordinates.astype(np.float64)
        lengths = np.linalg.norm(delta, axis=1)
        return {"status": "compared", "index": index, "reference_sha256": sha(reference),
                "different_coordinates": int(np.count_nonzero(delta)),
                "max_abs_coordinate_mm": float(np.abs(delta).max(initial=0)),
                "max_distance_mm": float(lengths.max(initial=0)),
                "p99_distance_mm": float(np.percentile(lengths, 99)),
                "mean_distance_mm": float(lengths.mean())}

    def save():
        temporary = args.output / "trace.tmp"
        temporary.write_text(json.dumps(json_value(report), indent=2) + "\n")
        temporary.replace(args.output / "trace.json")

    def observe_search(original, kind, *positional, **keywords):
        nonlocal current
        xyz, gradient = positional[:2]
        result = original(*positional, **keywords)
        row = {"index": len(rows), "kind": kind, "coordinates_before_sha256": array_sha(xyz),
               "gradient_after_average_sha256": array_sha(gradient), "search": json_value(result)}
        caller = inspect.currentframe().f_back.f_back
        row["integration_state"] = {key: json_value(caller.f_locals[key])
                                    for key in ["stage", "weight", "averages", "steps_at_scale"]
                                    if key in caller.f_locals}
        del caller
        row["diagnostic_native_before"] = compare_snapshot(xyz, len(rows))
        if len(rows) < args.coordinate_checkpoints or len(rows) in args.capture_update:
            path = args.output / f"step-{len(rows):04d}.before.npz"
            np.savez(path, coordinates=np.asarray(xyz), gradient=np.asarray(gradient), faces=faces)
            row["before_checkpoint"] = {"name": path.name, "sha256": sha(path)}
        rows.append(row)
        current = row
        return result

    def first(*positional, **keywords):
        return observe_search(original_first, "area-distance", *positional, **keywords)

    def gradient(*positional, **keywords):
        value = original_gradient(*positional, **keywords)
        if len(rows) in args.capture_update:
            path = args.output / f"step-{len(rows):04d}.forces.npz"
            np.savez(path, coordinates=positional[0], distance=value[0],
                     area=value[1], combined=value[2], original_distances=positional[5],
                     offsets=positional[3], neighbors=positional[4])
            report.setdefault("force_checkpoints", {})[str(len(rows))] = {
                "name": path.name, "sha256": sha(path), "metric": json_value(value[3])}
        return value

    def nonlinear(*positional, **keywords):
        return observe_search(original_nonlinear, "nonlinear-area-distance", *positional, **keywords)

    def project(*positional, **keywords):
        value = original_project(*positional, **keywords)
        if current is not None and "coordinates_after_sha256" not in current:
            current["coordinates_after_sha256"] = array_sha(value)
            compared = compare_snapshot(value, current["index"] + 1)
            current["diagnostic_native_after"] = compared
            # Diagnostic archive only; this threshold selects useful checkpoints,
            # does not accept a mesh or alter a force/step/stop condition.
            if compared and compared.get("max_distance_mm", 0) > 1e-4 and "first_nontrivial_difference" not in report:
                report["first_nontrivial_difference"] = compared
                np.savez(args.output / "first-nontrivial-after.npz", coordinates=value, faces=faces)
            if current["index"] < args.coordinate_checkpoints or current["index"] in args.capture_update:
                path = args.output / f"step-{current['index']:04d}.after.npz"
                np.savez(path, coordinates=value, faces=faces)
                current["after_checkpoint"] = {"name": path.name, "sha256": sha(path)}
            save()
            if args.stop_after_updates is not None and len(rows) >= args.stop_after_updates:
                raise DiagnosticPrefixComplete()
        elif current is None and "initial_projection_sha256" not in report:
            report["initial_projection_sha256"] = array_sha(value)
            np.savez(args.output / "initial-projection.npz", coordinates=value, faces=faces)
            save()
        return value

    def finish(*positional, **keywords):
        value = np.asarray(positional[0])
        np.savez(args.output / "actual-cleanup-input.npz", coordinates=value, faces=faces)
        report["actual_cleanup_input_sha256"] = array_sha(value)
        save()
        return original_finish(*positional, **keywords)

    operation.first_epoch_line_search = first
    operation.first_epoch_gradient = gradient
    operation.nonlinear_epoch_line_search = nonlinear
    operation.project_radially = project
    operation.finish_standard_sphere = finish
    try:
        hemisphere = args.inflated.name.split(".")[0]
        api = operation.run_standard_sphere(args.inflated, args.smoothwm, args.output / (hemisphere + ".sphere"), **kwargs)
        report.update(status="complete", api=api, total_trace_wall_seconds=time.perf_counter() - started)
        if torch.device(args.averaging_device).type == "cuda":
            torch.cuda.synchronize(args.averaging_device)
            report["torch_target_memory"] = {"device": args.averaging_device,
                "allocated_peak_bytes": torch.cuda.max_memory_allocated(args.averaging_device),
                "reserved_peak_bytes": torch.cuda.max_memory_reserved(args.averaging_device)}
        for index, row in enumerate(rows):
            row["api_schedule"] = {key: value for key, value in api["updates"][index].items() if key != "seconds"}
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name.startswith("fnit.recon_all") and path is not None and Path(path).suffix == ".py":
                report["source_files_sha256"][name] = sha(Path(path))
        save()
    except DiagnosticPrefixComplete:
        report.update(status="diagnostic_prefix_complete", total_trace_wall_seconds=time.perf_counter() - started,
                      final_sphere_written=False)
        save()
    except BaseException as error:
        report.update(status="failed_execution", exception_type=type(error).__name__, message=str(error))
        save()
        raise
    finally:
        operation.first_epoch_line_search = original_first
        operation.nonlinear_epoch_line_search = original_nonlinear
        operation.project_radially = original_project
        operation.finish_standard_sphere = original_finish
        operation.first_epoch_gradient = original_gradient
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name.startswith("fnit.recon_all") and path is not None and Path(path).suffix == ".py":
                report["source_files_sha256"][name] = sha(Path(path))
        save()


if __name__ == "__main__":
    main()
