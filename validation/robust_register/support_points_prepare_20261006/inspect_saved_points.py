"""Inspect two saved-geometry support differences, with no registration calls.

The unchanged frozen B sampler evaluates four complete target grids. Only
after exact persisted scalar reproduction may four one-point evaluations
observe its boundary branches. Atlas values and coordinates stay private.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

from common import check_bindings, digest, save_exclusive, validate_plan


def flags():
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only diagnostic initialized CUDA")
    if "fnit.robust_register" in sys.modules:
        raise RuntimeError("production robust namespace was loaded")
    if any(name == "fnit.gems" or name.startswith("fnit.gems.")
           for name in sys.modules):
        raise RuntimeError("GEMS namespace was loaded")
    return {
        "CPU_affinity": [int(value) for value in sorted(os.sched_getaffinity(0))],
        "Torch_threads": int(torch.get_num_threads()),
        "Torch_interop_threads": int(torch.get_num_interop_threads()),
        "Torch_version": str(torch.__version__), "CUDA_initialized": False,
        "TF32_CPU_unused": bool(torch.backends.cuda.matmul.allow_tf32),
        "cuDNN_enabled": bool(torch.backends.cudnn.enabled),
        "cuDNN_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "MKLDNN_enabled": bool(torch.backends.mkldnn.enabled),
        "Torch_grad_enabled": bool(torch.is_grad_enabled()),
        "Torch_default_dtype": str(torch.get_default_dtype()),
        "autocast_CPU_enabled": bool(torch.is_autocast_cpu_enabled()),
        "address_space_cap_bytes": int(resource.getrlimit(resource.RLIMIT_AS)[0]),
    }


def configure(plan):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "8":
            raise ValueError("the CPU8 environment differs: " + name)
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH",
                 "OPENBLAS_CORETYPE", "FS_SetVoxToRasXform_Change_VoxSize"):
        if name in os.environ:
            raise ValueError("environment injection is prohibited: " + name)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CUDA must be hidden")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("frozen sources must remain cache-free")
    os.sched_setaffinity(0, plan["physical_cores"])
    cap = int(plan["limits"]["address_space_bytes"])
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    return flags()


def load_sampler(plan):
    """Reuse the old six-file package in an independent namespace."""
    sys.path.insert(0, str(Path(plan["source_directory"]) / "src"))
    import fnit
    expected = Path(plan["source_directory"]) / "src/fnit/__init__.py"
    if Path(fnit.__file__).resolve() != expected.resolve():
        raise ValueError("fnit did not load from the frozen source tree")
    source = Path(plan["legacy_candidate_directory"])
    namespace = "fnit._robust_support_saved_legacy_20261006"
    if namespace in sys.modules:
        raise ValueError("a fresh diagnostic process is required")
    specification = importlib.util.spec_from_file_location(
        namespace, source / "__init__.py", submodule_search_locations=[str(source)])
    if specification is None or specification.loader is None:
        raise ImportError("cannot load the unchanged B sampler package")
    package = importlib.util.module_from_spec(specification)
    sys.modules[namespace] = package
    try:
        specification.loader.exec_module(package)
    except BaseException:
        sys.modules.pop(namespace, None)
        raise
    sampling = importlib.import_module(namespace + "._sampling")
    geometry = importlib.import_module(namespace + ".registration")
    return sampling, geometry._compose_native


def header_geometry(image, compose):
    import numpy as np
    return compose(image.shape, np.asarray(image.header["delta"], np.float32),
                   np.asarray(image.header["Mdc"], np.float32).T,
                   np.asarray(image.header["Pxyz_c"], np.float32))


def loaded_sources():
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name.startswith("fnit._robust_support_saved_legacy_20261006"):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename)
                result[name] = {"path": str(path),
                                "bytes": int(path.stat().st_size),
                                "sha256": digest(path)}
    return result


def inspect_point(values, target_index, pull, observed_value, sampling, report):
    """Record the old sampler's single-point state using its exact FP32 grid."""
    import numpy as np
    import torch
    matrix = torch.as_tensor(np.asarray(pull, np.float32))
    grid = [torch.tensor([int(value)], dtype=torch.float32)
            for value in target_index]
    coordinates = []
    for axis in range(3):
        coordinate = torch.zeros_like(grid[0])
        for column in range(3):
            coordinate = coordinate + matrix[axis, column] * grid[column]
        coordinates.append(coordinate + matrix[axis, 3])
    q32 = torch.stack(coordinates)
    q = q32.double()
    rounded = q.round()
    support = sampling._inside_round_support(q, values.shape)
    truncated = torch.trunc(q)
    integer_distance = (q - truncated).abs()
    near_axes = integer_distance < np.finfo(np.float32).eps
    near = bool(near_axes.all())
    low, high, up, clamped = [], [], [], []
    for axis, size in enumerate(values.shape):
        coordinate = q[axis].clamp(0, int(size) - 1)
        lower = coordinate.long()
        low.append(lower)
        high.append((lower + 1).clamp_max(int(size) - 1))
        up.append(coordinate - lower.double())
        clamped.append(float(coordinate.item()))
    corners = []
    for x in range(2):
        for y in range(2):
            for z in range(2):
                bits = (x, y, z)
                indices = [high[a] if bits[a] else low[a] for a in range(3)]
                weights = [up[a] if bits[a] else 1 - up[a] for a in range(3)]
                scalar = float(values[tuple(indices)].item())
                corners.append({
                    "bits": [int(value) for value in bits],
                    "indices": [int(value.item()) for value in indices],
                    "value": scalar, "nonzero": bool(scalar != 0),
                    "axis_weights_double": [float(value.item()) for value in weights],
                    "weight_double": float((weights[0] * weights[1] * weights[2]).item()),
                })
    nearest = [q[a].round().long().clamp(0, int(size) - 1)
               for a, size in enumerate(values.shape)]
    nearest_value = float(values[tuple(nearest)].item())
    report["point_only_original_linear_calls"] += 1
    evaluated = sampling._linear(values, q32)[0]
    evaluated_bits = int(evaluated.numpy().view(np.uint32).item())
    observed_bits = int(observed_value.numpy().view(np.uint32).item())
    if evaluated_bits != observed_bits:
        raise RuntimeError("the one-point original sampler did not reproduce baseline bits")
    return {
        "pull_FP32": np.asarray(pull, np.float32).tolist(),
        "coordinates_FP32": [float(value.item()) for value in q32[:, 0]],
        "coordinate_uint32": [int(value) for value in
                              q32[:, 0].numpy().view(np.uint32).tolist()],
        "rint_ties_even": [int(value.item()) for value in rounded[:, 0]],
        "round_support_axes": [bool((rounded[a, 0] >= 0)
                                    and (rounded[a, 0] < int(values.shape[a])))
                               for a in range(3)],
        "round_support_valid": bool(support.item()),
        "clamped_coordinates_double": clamped,
        "clamp_changed_axes": [bool(q[a, 0].item() != clamped[a]) for a in range(3)],
        "FEQUAL_epsilon": float(np.finfo(np.float32).eps),
        "FEQUAL_truncated_indices": [int(value.item()) for value in truncated[:, 0]],
        "FEQUAL_distance_to_trunc_double": [float(value.item())
                                           for value in integer_distance[:, 0]],
        "FEQUAL_per_axis_near": [bool(value.item()) for value in near_axes[:, 0]],
        "FEQUAL_all_axes_integer_shortcut": near,
        "nearest_indices": [int(value.item()) for value in nearest],
        "nearest_value": nearest_value, "corners_x_y_z_order": corners,
        "observed_FP32_value": float(observed_value.item()),
        "observed_uint32": observed_bits, "one_point_original_sampler_bit_exact": True,
    }


def classify(official, fnit):
    """Describe observed branch states without assigning optimizer causality."""
    if official["round_support_valid"] != fnit["round_support_valid"]:
        return "half_voxel_round_support_switch"
    if (official["FEQUAL_all_axes_integer_shortcut"]
            != fnit["FEQUAL_all_axes_integer_shortcut"]):
        return "FEQUAL_integer_shortcut_switch"
    old = [(item["indices"], item["nonzero"])
           for item in official["corners_x_y_z_order"]]
    new = [(item["indices"], item["nonzero"])
           for item in fnit["corners_x_y_z_order"]]
    if old != new:
        return "interpolation_stencil_or_nonzero_corner_switch"
    if official["FEQUAL_all_axes_integer_shortcut"]:
        return "same_integer_shortcut_with_different_selected_value"
    return "same_domain_and_stencil_continuous_interpolation_zero_crossing"


def evaluate(plan, report):
    import nibabel as nib
    import numpy as np
    import torch
    expected_record = json.loads(Path(plan["baseline_score"]).read_text())
    baseline_flags = expected_record["actual_flags_before"]
    current = flags()
    for key in ("CPU_affinity", "Torch_threads", "Torch_interop_threads",
                "Torch_version", "CUDA_initialized", "TF32_CPU_unused",
                "address_space_cap_bytes"):
        if current[key] != baseline_flags[key]:
            raise ValueError("saved-score CPU resource flags differ: " + key)
    sampling, compose = load_sampler(plan)
    original, target = nib.load(plan["moving"]), nib.load(plan["fixed"])
    original_shape = [int(value) for value in original.shape]
    target_shape = [int(value) for value in target.shape]
    if original_shape != [131, 241, 99] or target_shape != plan["expected_target_shape"]:
        raise ValueError("the frozen moving/target shapes differ")
    original_values = np.asanyarray(original.dataobj)
    if original_values.dtype != np.uint8 or not np.isfinite(original_values).all():
        raise ValueError("finite saved uint8 atlas values required")
    values = torch.from_numpy(original_values.astype(np.float32))
    target_geometry = header_geometry(target, compose)
    pairs, baselines = {}, {}
    # Complete both saved-geometry baselines before inspecting any corner state.
    for mode in ("rigid", "affine"):
        pair = {}
        for label, directory in (("official", plan["official_directory"]),
                                 ("fnit", plan["saved_output_directory"])):
            image = nib.load(str(Path(directory) / (mode + ".header.mgz")))
            if [int(value) for value in image.shape] != original_shape:
                raise ValueError("saved moving grid differs")
            pull = sampling.native_matmul(
                sampling.native_inverse(header_geometry(image, compose)), target_geometry)
            report["baseline_full_grid_resample_calls"] += 1
            warped = sampling.resample(values, tuple(target_shape), pull,
                                      chunk_size=plan["parameters"]["spatial_chunk_size"])
            pair[label] = {"pull": pull, "warped": warped}
        difference = (pair["fnit"]["warped"] - pair["official"]["warped"]).double()
        reference = pair["official"]["warped"].double()
        denominator = float(torch.linalg.vector_norm(reference))
        if denominator == 0:
            raise ValueError("saved official geometry has empty warped support")
        # Identical norm/division expression to the original saved-output score.
        relative = float(torch.linalg.vector_norm(difference)
                         / torch.linalg.vector_norm(reference))
        support = (pair["fnit"]["warped"] != 0) ^ (pair["official"]["warped"] != 0)
        support_count = int(torch.count_nonzero(support))
        actual = {"warp_relative_L2": relative,
                  "warp_nonzero_support_difference_voxels": support_count}
        old = expected_record["result"]["stages"][mode]
        if actual != plan["expected_baseline"][mode] or any(
                actual[key] != old[key] for key in actual):
            raise RuntimeError("persisted baseline did not reproduce exactly: " + mode)
        baselines[mode] = actual
        pairs[mode] = (pair, support)
    report["both_baselines_scalar_exact_before_point_inspection"] = True
    points, public_stages = {}, {}
    for mode in ("rigid", "affine"):
        pair, support = pairs[mode]
        indices = torch.nonzero(support, as_tuple=False)
        if tuple(int(value) for value in indices.shape) != (1, 3):
            raise RuntimeError("each persisted support difference must have one point")
        index = tuple(int(value) for value in indices[0].tolist())
        states = {}
        for label in ("official", "fnit"):
            states[label] = inspect_point(values, index, pair[label]["pull"],
                                          pair[label]["warped"][index], sampling, report)
        category = classify(states["official"], states["fnit"])
        points[mode] = {"target_index_private": [int(value) for value in index],
                        "states_private": states, "observed_branch_classification": category}
        public_stages[mode] = {
            "support_difference_points": int(indices.shape[0]),
            "classification_counts": {category: 1},
            "persisted_baseline_exact": True,
            "classification_is_observed_state_not_optimizer_root_cause": True,
        }
    report["private_points"] = points
    report["baseline_scalars_private"] = baselines
    report["source_array_shape_private"] = original_shape
    report["target_array_shape_private"] = target_shape
    report["public_summary"] = {
        "schema": 1, "status": "saved_geometry_baseline_reproduced_points_observed",
        "stages": public_stages,
        "original_formal_acceptance_unchanged": {"total": 20, "persisted_passed": 17,
                                                  "persisted_failed": 3},
        "affine_inherits_own_different_rigid_MGH": True,
        "new_registration_calls": 0, "new_official_commands": 0,
        "new_GEMS_calls": 0, "new_GPU_calls": 0,
        "baseline_full_grid_resample_calls": 4,
        "point_only_original_linear_calls": 4, "production_adopted": False,
        "coordinates_atlas_values_pulls_images_published": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--approved-plan-sha", required=True)
    parser.add_argument("--cpu-lock-fd", type=int, required=True)
    args = parser.parse_args()
    if digest(args.plan) != args.approved_plan_sha:
        raise ValueError("the explicit approved PLAN SHA differs")
    plan = json.loads(Path(args.plan).read_text())
    validate_plan(plan)
    # The parent owns this same open-file description throughout the child.
    if args.cpu_lock_fd < 3:
        raise ValueError("the controller's inherited CPU lock descriptor is required")
    inherited = os.fstat(args.cpu_lock_fd)
    expected_lock = Path(plan["common_CPU_lock"]).stat()
    if (inherited.st_dev, inherited.st_ino, inherited.st_uid) != (
            expected_lock.st_dev, expected_lock.st_ino, os.getuid()):
        raise ValueError("inherited descriptor does not match the common CPU lock")
    fcntl.flock(args.cpu_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.umask(0o077)
    directory = Path(plan["run_directory"]) / "worker"
    directory.mkdir(mode=0o700, exist_ok=False)
    started, code = time.monotonic(), 1
    report = {"status": "started", "PID": int(os.getpid()),
              "plan_sha256": args.approved_plan_sha,
              "baseline_full_grid_resample_calls": 0,
              "point_only_original_linear_calls": 0,
              "new_registration_calls": 0, "new_official_commands": 0,
              "new_GEMS_calls": 0, "new_GPU_calls": 0}
    try:
        report["bindings_before"] = check_bindings(plan["bindings"])
        report["harness_before"] = check_bindings(plan["harness_bindings"])
        if Path(__file__).resolve() != Path(plan["worker"]).resolve():
            raise ValueError("the actually invoked worker is not the frozen worker")
        report["actual_flags_before"] = configure(plan)
        evaluate(plan, report)
        report["status"] = "completed_saved_geometry_diagnostic"
        code = 0
    except BaseException as error:
        report["status"] = "failed_first_guard_or_diagnostic"
        report["exception"] = {"type": type(error).__name__, "message": str(error)}
        (directory / "exception.private.txt").write_text(traceback.format_exc())
    finally:
        report["wall_seconds"] = float(time.monotonic() - started)
        report["maxrss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        try:
            report["bindings_after"] = check_bindings(plan["bindings"])
            report["harness_after"] = check_bindings(plan["harness_bindings"])
            report["actual_flags_after"] = flags()
            if report.get("actual_flags_before") != report["actual_flags_after"]:
                raise RuntimeError("precision/resource flags changed during the diagnostic")
            report["precision_resource_flags_before_after_exact"] = True
            report["actual_legacy_modules"] = loaded_sources()
        except BaseException as error:
            report["postcheck_exception"] = {"type": type(error).__name__, "message": str(error)}
            report["status"] = "failed_postcheck"
            report.pop("public_summary", None)
            code = 1
        report["scientific_returncode"] = int(code)
        save_exclusive(directory / "report.private.json", report)
        if code == 0:
            save_exclusive(directory / "classification.public.json", report["public_summary"])
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
