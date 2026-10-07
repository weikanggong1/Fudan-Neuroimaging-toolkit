"""ONE saved FP32 moving smoothing bridge; no registration or native calls."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

from smoothing_io import (bound, check_bindings, check_freeze, check_headers,
                          git_head, input_paths, nifti_header_metadata, write_json)
from smoothing_adapter import classify_header, smooth_original_storage


def flags(torch):
    return {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "grad_enabled": torch.is_grad_enabled(),
            "cpu_autocast": torch.is_autocast_enabled("cpu"),
            "cuda_autocast": torch.is_autocast_enabled("cuda"),
            "cuda_initialized": torch.cuda.is_initialized()}


def fsl_maps_snapshot():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        token = line.split()[-1]
        if "/FSL/" in token or any(s in token.lower() for s in ("libfsl", "libnewimage", "libmiscmaths", "libnewmat")):
            paths.add(Path(token).name)
    return sorted(paths)


def array_meta(value, np):
    result = value.detach().numpy() if hasattr(value, "detach") else value
    return {"shape": list(result.shape), "dtype": str(result.dtype),
            "byte_strides": list(result.strides),
            "logical_c_sha256": hashlib.sha256(result.tobytes(order="C")).hexdigest()}


def metrics(value, reference, np):
    if value.shape != reference.shape or value.dtype != np.dtype("float32") or reference.dtype != np.dtype("float32"):
        raise ValueError("full-image comparison requires same-shape FP32")
    a = value.reshape(-1, order="C")
    b = reference.reshape(-1, order="C")
    different, max_abs, error2, ref2 = 0, 0.0, 0.0, 0.0
    for start in range(0, a.size, 1000000):
        aa, bb = a[start:start + 1000000], b[start:start + 1000000]
        if not np.isfinite(aa).all() or not np.isfinite(bb).all():
            raise ValueError("nonfinite image comparison")
        different += int(np.count_nonzero(aa.view(np.uint32) != bb.view(np.uint32)))
        delta = aa.astype(np.float64) - bb.astype(np.float64)
        max_abs = max(max_abs, float(np.max(np.abs(delta), initial=0.0)))
        error2 += float(np.sum(delta * delta, dtype=np.float64))
        rb = bb.astype(np.float64)
        ref2 += float(np.sum(rb * rb, dtype=np.float64))
    return {"shape": list(value.shape), "dtype": "float32", "values": int(a.size),
            "different_bits": different, "all_value_bits_exact": different == 0,
            "max_abs": max_abs, "rmse": float(np.sqrt(error2 / a.size)),
            "relative_l2": float(np.sqrt(error2 / ref2)) if ref2 > 0 else (0.0 if error2 == 0 else None),
            "statistics_scope": "full logical C order; FP64 chunk statistics, not PCG reductions"}


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approved-smoothing-bridge", action="store_true", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    resource.setrlimit(resource.RLIMIT_AS, (20000000000, 20000000000))
    started = time.perf_counter()
    report = {"status": "smoothing_started_not_accepted", "accepted": False,
              "scope": "current compiled CPU plain bridge, then private header adapter; saved moving only",
              "host": socket.gethostname(), "affinity": sorted(os.sched_getaffinity(0)),
              "calls": {"mature_blur": 0, "compiled_cpu_helper": 0, "adapter": 0,
                        "normalization": 0, "RHS": 0, "H": 0, "PCG": 0,
                        "evaluate": 0, "linearize": 0, "native": 0, "GPU": 0},
              "gates": {}, "arms": {}, "production_changed": False,
              "interpretation": "preprocessing identity only; old full orientation candidate was rejected; no full precision improvement",
              "read_reference_policy": "native image values decoded only after adapter result is computed"}
    torch = np = numba = None
    operands, operands_before = {}, {}
    before_flags = None
    helper_module = original_helper = None
    error = error_tb = None
    complete = False
    expected = json.loads((args.workspace / "expected.public.json").read_text())
    try:
        before, failed = check_bindings(args.root, expected)
        frozen, freeze_failed = check_freeze(args.workspace)
        headers, header_failed = check_headers(args.root, expected)
        report.update(bindings_before=before, harness_before=frozen, headers_before=headers,
                      main_head_at_read=git_head(args.root / "repo"))
        if failed or freeze_failed or header_failed:
            raise RuntimeError("first source/input/header/freeze mismatch")
        required = {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                    "NUMBA_NUM_THREADS": "8", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
        if any(os.environ.get(k) != v for k, v in required.items()):
            raise RuntimeError("declared environment mismatch")
        if report["affinity"] != expected["cpu_affinity"]:
            raise RuntimeError("physical CPU affinity mismatch")
        if any(os.environ.get(k) for k in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "NUMBA_DISABLE_JIT")):
            raise RuntimeError("unexpected loader/Python/JIT override")
        prefix = (args.root / "envs/default").resolve()
        if (Path(sys.prefix).resolve() != prefix
                or hashlib.sha256(str(prefix).encode()).hexdigest() != expected["prefix_path_sha256"]
                or bound(Path(sys.executable).resolve()) != expected["python_interpreter"]):
            raise RuntimeError("runtime interpreter/prefix mismatch")
        cache_directory = (args.output.parent / "numba_cache").resolve()
        if Path(os.environ.get("NUMBA_CACHE_DIR", "")).resolve() != cache_directory:
            raise RuntimeError("Numba cache must be private run-local")
        if cache_directory.exists() and any(cache_directory.iterdir()):
            raise RuntimeError("Numba cache must start empty; no stale compiled context")
        report["numba_cache_empty_before_numerical_import"] = True
        sys.path.insert(0, str(args.root / "repo/src"))
        import numpy as np
        import torch
        import numba
        import nibabel as nib
        import llvmlite
        from fnit.fnirt import registration
        from fnit.fnirt import _smoothing_cpu as helper_module
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        versions = {"python": sys.version.split()[0], "torch": torch.__version__,
                    "numpy": np.__version__, "numba": numba.__version__, "nibabel": nib.__version__}
        if versions != expected["runtime_versions"]:
            raise RuntimeError("declared runtime versions changed")
        before_flags = flags(torch)
        report.update(runtime={"versions": versions, "llvmlite": llvmlite.__version__,
                               "prefix_basename": prefix.name,
                               "prefix_path_sha256": hashlib.sha256(str(prefix).encode()).hexdigest(),
                               "canonical_default_resolves_to_actual_prefix": True,
                               "interpreter": bound(Path(sys.executable).resolve()),
                               "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                               "numba_threads": numba.get_num_threads(), "numba_config_threads": numba.config.NUMBA_NUM_THREADS,
                               "environment": {k: os.environ.get(k) for k in required},
                               "numba_cache_is_new_run_local": True}, flags_before=before_flags,
                      fsl_dso_snapshot_before=fsl_maps_snapshot())
        if torch.get_num_threads() != 8 or numba.get_num_threads() != 8 or numba.config.NUMBA_NUM_THREADS != 8:
            raise RuntimeError("CPU thread budget mismatch")
        if before_flags["cuda_initialized"] or before_flags["cpu_autocast"] or before_flags["cuda_autocast"] or report["fsl_dso_snapshot_before"]:
            raise RuntimeError("unexpected CUDA/autocast/FSL state")
        # Verify every actually imported project module against declared bytes.
        imported = {}
        source_paths = {str((args.root / "repo" / name).resolve()): item
                        for section in ("production_source", "source_extra")
                        for name, item in expected[section].items()}
        for name, module in list(sys.modules.items()):
            if name == "fnit" or name.startswith("fnit."):
                filename = getattr(module, "__file__", None)
                if filename is None:
                    continue
                path = str(Path(filename).resolve())
                if path not in source_paths or bound(path) != source_paths[path]:
                    raise RuntimeError("undeclared imported FNIT source: " + name)
                imported[name] = source_paths[path]
        report["actually_imported_project_sources"] = imported
        paths = input_paths(args.root, expected)
        producer = json.loads(paths["old_reference_report"].read_text())
        if producer["source"]["registration"] != expected["legacy_registration_sha256"] or producer["inputs"]["gm"] != expected["files"]["original_gm_header_only"]["sha256"]:
            raise RuntimeError("saved normalized producer identity mismatch")
        raw, header_info = nifti_header_metadata(paths["original_gm_header_only"])
        header = nib.Nifti1Header.from_fileobj(io.BytesIO(raw), check=True)
        resolved = classify_header(header, np)
        report["resolved_header"] = resolved
        report["gates"]["declared_header_orientation_and_zoom"] = (
            resolved["orientation"] == expected["orientation"] and resolved["pixdim"] == expected["pixdim"])
        if not report["gates"]["declared_header_orientation_and_zoom"]:
            raise RuntimeError("first orientation/zoom mismatch")
        normalized = np.load(paths["normalized_moving"], allow_pickle=False)
        old_smoothed = np.load(paths["old_smoothed_moving"], allow_pickle=False)
        for name, array in (("normalized_moving", normalized), ("old_smoothed_moving", old_smoothed)):
            if array.dtype != np.dtype("float32") or list(array.shape) != expected["shape"] or not np.isfinite(array).all():
                raise RuntimeError("saved FP32 array contract mismatch: " + name)
            operands[name] = array
            operands_before[name] = array_meta(array, np)
        volume = torch.from_numpy(normalized)[None, None]
        operands["input_tensor"] = volume
        operands_before["input_tensor"] = array_meta(volume, np)
        original_helper = helper_module.gaussian_blur_cpu

        def counted_helper(value, kernels):
            report["calls"]["compiled_cpu_helper"] += 1
            if report["calls"]["compiled_cpu_helper"] > 2:
                raise RuntimeError("compiled helper call limit")
            if value.device.type != "cpu" or value.dtype != torch.float32 or value.requires_grad:
                raise RuntimeError("compiled helper dtype/device/grad guard")
            return original_helper(value, kernels)

        helper_module.gaussian_blur_cpu = counted_helper

        def mature_blur(*values, **options):
            report["calls"]["mature_blur"] += 1
            if report["calls"]["mature_blur"] > 2:
                raise RuntimeError("mature blur call limit")
            return registration._fsl_masked_gaussian_blur(*values, **options)

        report["operands_before"] = operands_before
        report["clock_seconds_import_bind_header_restore"] = time.perf_counter() - started
        with torch.no_grad():
            arm_start = time.perf_counter()
            plain = mature_blur(volume, expected["fwhm_mm"], tuple(resolved["pixdim"]), None, execution="optimized")
            report["clock_seconds_plain_cold"] = time.perf_counter() - arm_start
            if report["calls"]["compiled_cpu_helper"] != 1 or plain.dtype != torch.float32 or plain.device.type != "cpu" or plain.requires_grad:
                raise RuntimeError("plain arm bypassed declared compiled FP32 helper")
            plain_comparison = metrics(plain[0, 0].numpy(), old_smoothed, np)
            report["arms"]["plain_current_vs_legacy_saved"] = plain_comparison
            report["gates"]["plain_current_matches_legacy_saved_bits"] = plain_comparison["all_value_bits_exact"]
            if not report["gates"]["plain_current_matches_legacy_saved_bits"]:
                raise RuntimeError("FIRST plain provenance mismatch; adapter not run")
            report["calls"]["adapter"] += 1
            arm_start = time.perf_counter()
            adapted = smooth_original_storage(volume, expected["fwhm_mm"], tuple(resolved["pixdim"]),
                                              flip_x=resolved["flip_x"], blur=mature_blur)
            report["clock_seconds_adapter_warm"] = time.perf_counter() - arm_start
            if report["calls"]["compiled_cpu_helper"] != 2 or adapted.dtype != torch.float32 or adapted.device.type != "cpu" or adapted.requires_grad:
                raise RuntimeError("adapter arm bypassed declared compiled FP32 helper")
        # Reference image values cannot select or modify the algorithm above.
        official = nib.load(paths["official_smoothed_moving"])
        native = np.asanyarray(official.dataobj)
        if (native.dtype != np.dtype("float32") or list(native.shape) != expected["shape"]
                or list(official.header.get_zooms()[:3]) != expected["pixdim"]):
            raise RuntimeError("official saved grid/zoom mismatch")
        report["native_reference_decode"] = {"actual_dtype": str(native.dtype),
                                             "proxy_slope": float(official.dataobj.slope),
                                             "proxy_intercept": float(official.dataobj.inter),
                                             "reference_cast_performed": False}
        operands["native_reference_decoded_after_adapter"] = native
        operands_before["native_reference_decoded_after_adapter"] = array_meta(native, np)
        comparison = metrics(adapted[0, 0].numpy(), native, np)
        report["arms"]["adapter_current_vs_saved_native"] = comparison
        report["gates"]["adapter_matches_saved_native_bits"] = comparison["all_value_bits_exact"]
        report["numba_dispatcher_context"] = {
            "selected_by_current_helper": "parallel" if 1 < numba.get_num_threads() <= torch.get_num_threads() else "serial",
            "serial_signatures": [str(s) for s in helper_module._serial.signatures],
            "parallel_signatures": [str(s) for s in helper_module._parallel.signatures],
            "threading_layer": numba.threading_layer(),
            "assembly_assessed": False, "per_axis_native_runtime_arrays_available": False,
            "compiled_helper_contains_three_source_bound_axes": True,
            "axis_runtime_count_not_separately_instrumented": True}
        if not report["gates"]["adapter_matches_saved_native_bits"]:
            raise RuntimeError("FIRST adapter image mismatch; no numeric retry")
        if report["calls"]["mature_blur"] != 2 or report["calls"]["compiled_cpu_helper"] != 2:
            raise RuntimeError("unexpected final call count")
        complete = True
    except BaseException as exc:
        error, error_tb = exc, exc.__traceback__
        report.update(status="smoothing_failed_stopped_no_retry", failure_class=type(exc).__name__, failure=str(exc))
    finally:
        if helper_module is not None and original_helper is not None:
            helper_module.gaussian_blur_cpu = original_helper
        try:
            after, after_failures = check_bindings(args.root, expected)
            after_frozen, after_freeze_failures = check_freeze(args.workspace)
            headers_after, after_header_failures = check_headers(args.root, expected)
            report.update(bindings_after=after, harness_after=after_frozen, headers_after=headers_after,
                          source_inputs_and_harness_unchanged=(not after_failures and not after_freeze_failures and not after_header_failures))
            if np is not None:
                actual = {name: array_meta(value, np) for name, value in operands.items()}
                report.update(operands_after=actual, all_available_operands_unchanged=(actual == operands_before))
            if torch is not None:
                report["flags_after"] = flags(torch)
                report["flags_unchanged"] = before_flags is not None and before_flags == report["flags_after"]
                report["fsl_dso_snapshot_after"] = fsl_maps_snapshot()
                report["no_fsl_dso_in_recorded_snapshots"] = not report.get("fsl_dso_snapshot_before", []) and not report["fsl_dso_snapshot_after"]
            post = (report.get("source_inputs_and_harness_unchanged", False)
                    and report.get("all_available_operands_unchanged", False)
                    and report.get("flags_unchanged", False)
                    and report.get("no_fsl_dso_in_recorded_snapshots", False)
                    and not report.get("flags_after", {}).get("cuda_initialized", True))
            report["accepted"] = complete and error is None and post
            if complete and error is None and not post:
                report.update(status="smoothing_postcondition_failed_not_accepted")
            elif report["accepted"]:
                report["status"] = "saved_current_smoothing_bridge_passed_no_registration"
        except BaseException as post_error:
            report.update(postcondition_failure_class=type(post_error).__name__,
                          postcondition_failure=str(post_error), accepted=False)
            if error is None:
                error, error_tb = post_error, post_error.__traceback__
        report["clock_seconds_worker_including_postchecks"] = time.perf_counter() - started
        report["peak_rss_KiB_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["time_scope"] = "one cold plain then warm adapter; not a speed ratio or full registration"
        write_json(args.output / "summary.public.json", report)
    if error is not None:
        raise error.with_traceback(error_tb)
    if not report["accepted"]:
        raise RuntimeError("final postconditions failed; not accepted")
    print(json.dumps({"status": report["status"], "accepted": True, "calls": report["calls"]}))


if __name__ == "__main__":
    main()
