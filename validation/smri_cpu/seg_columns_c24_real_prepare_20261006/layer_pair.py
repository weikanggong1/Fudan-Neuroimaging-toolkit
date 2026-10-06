"""Prepared original/flipped C24 preELU pair; no tail, full CNN or new compiler."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

from real_bindings import (atomic_json, capture_bindings, check_bindings, check_environment,
                           check_runtime, flags, identity, load_helper, require, value_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--approved-real-layer", action="store_true", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    plan = json.loads((args.workspace / "PLAN.json").read_text())
    plan["PLAN_identity"] = identity(args.workspace / "PLAN.json")
    modes = {"A1_baseline": "baseline", "B1_candidate": "candidate", "B2_candidate": "candidate", "A2_baseline": "baseline"}
    require(args.workspace.resolve() == (args.root / plan["workspace"]).resolve() and
            args.run.resolve() == (args.root / plan["runs"]).resolve() and
            args.output.parent.resolve() == args.run.resolve() and modes.get(args.output.name) == args.mode,
            "declared canonical arm required")
    check_environment(plan)
    resource.setrlimit(resource.RLIMIT_AS, (32_000_000_000, 32_000_000_000))
    args.output.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    report = {"schema": "fnit_C24_two_pass_real_layer/v1", "status": "started_not_accepted",
              "mode": args.mode, "arm": args.output.name, "PLAN": plan["PLAN_identity"], "passes": [],
              "conv1_calls": 0, "candidate_copy_calls": 0, "candidate_SGEMM_calls": 0,
              "model_forward_calls": 0, "MRI_decode_calls": 0, "native_calls": 0, "GPU_calls": 0,
              "new_compile_calls": 0, "completed": False, "postcondition_errors": {}}
    torch = None
    initial = before = runtime_before = capture_before = None
    helper = None
    original_copy = original_gemm = None
    reference_before = None
    try:
        before = check_bindings(args.root, args.workspace, plan)
        report["bindings_before"] = before
        capture_meta, capture_before = capture_bindings(args.run, plan)
        report["capture_before"] = capture_before
        if args.output.name != "A1_baseline":
            reference_report_path = args.run / "A1_baseline/report.json"
            reference_report = json.loads(reference_report_path.read_text())
            require(reference_report["valid_layer_arm"] and reference_report["status"] == "two_baseline_references_saved" and
                    reference_report["PLAN"] == report["PLAN"] and reference_report["capture_before"] == capture_before,
                    "valid A1 same-input reference required")
            reference_before = {"report": identity(reference_report_path)}
            for row in reference_report["passes"]:
                name = row["pass"]
                require(name in ("original", "flipped"), "A1 pass names differ")
                reference_before[name] = identity(args.run / "A1_baseline" / (name + ".preELU.private.npy"))
                require(reference_before[name] == row["reference_saved"], "A1 reference changed")
            report["reference_before"] = reference_before
        sys.path.insert(0, str(args.root / "repo/src"))
        import torch as loaded_torch
        import numpy as np
        torch = loaded_torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(8)
        initial = flags(torch)
        report["flags_before"] = initial
        require(not initial["CUDA_initialized"] and not initial["CPU_autocast"], "CPU-only FP32 scope required")
        from fnit.synthseg_parc.segment import SegmentUNet
        from fnit.synthseg_parc.cpu_conv import CPUInferenceConv3d
        runtime_before = check_runtime(torch, plan)
        report["runtime_before"] = runtime_before
        model = SegmentUNet().load_h5(args.root / plan["assets"]["segmentation_weight"]["fnit_relative"]).eval()
        layer = model.down[0].conv1
        require(type(layer) is CPUInferenceConv3d and not getattr(layer, "_fnit_columns_reuse", False),
                "real C24 layer must remain mature unmarked type")
        require(value_sha(layer.weight) == plan["layer"]["kernel_value_sha256"] and
                value_sha(layer.bias) == plan["layer"]["bias_value_sha256"], "actual C24 parameter bits differ")
        weight_sha, bias_sha = value_sha(layer.weight), value_sha(layer.bias)
        if args.mode == "candidate":
            helper = load_helper(args.root, plan)
            helper._provider_still_matches()
            original_copy, original_gemm = helper.copy, helper.gemm
            # Finite local counters keep their dispatch overhead in the candidate clock.
            def copy_counter(*values):
                report["candidate_copy_calls"] += 1
                return original_copy(*values)
            def gemm_counter(*values):
                report["candidate_SGEMM_calls"] += 1
                return original_gemm(*values)
            helper.copy, helper.gemm = copy_counter, gemm_counter
        for name in ("original", "flipped"):
            input_path = args.run / "capture" / (name + ".private.npy")
            # Private copy-on-write mapping, never mutate the captured file.
            input_array = np.load(input_path, mmap_mode="c", allow_pickle=False)
            image = torch.from_numpy(input_array)
            require(list(image.shape) == plan["layer_input_shape"] and image.dtype == torch.float32 and image.is_contiguous(),
                    "real captured C24 layout differs")
            require(value_sha(image) == capture_meta["features"][name]["value_sha256"], "capture value bits differ")
            row = {"pass": name, "input_value_sha256": value_sha(image), "comparison_executed": args.output.name != "A1_baseline"}
            report["passes"].append(row)
            atomic_json(args.output / "report.json", report)
            with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
                row["active_flags"] = flags(torch)
                tick = time.monotonic()
                report["conv1_calls"] += 1
                actual = layer(image) if args.mode == "baseline" else helper.forward(layer, image)
                row["layer_seconds_including_candidate_qualification_counters"] = time.monotonic() - tick
            require(list(actual.shape) == plan["layer_input_shape"] and actual.dtype == torch.float32 and actual.is_contiguous(),
                    "preELU output geometry changed")
            require(actual.untyped_storage().data_ptr() != image.untyped_storage().data_ptr(), "output aliases input")
            require(value_sha(image) == row["input_value_sha256"], "real input changed")
            require(value_sha(layer.weight) == weight_sha and value_sha(layer.bias) == bias_sha, "real layer parameters changed")
            row["input_parameters_unchanged"] = True
            row["output"] = {"value_sha256": value_sha(actual), "shape": [int(value) for value in actual.shape],
                               "stride": [int(value) for value in actual.stride()], "dtype": str(actual.dtype)}
            array = actual.detach().numpy()
            reference = None
            if row["comparison_executed"]:
                reference = np.load(args.run / "A1_baseline" / (name + ".preELU.private.npy"), mmap_mode="r", allow_pickle=False)
                require(array.shape == reference.shape and array.dtype == reference.dtype and array.strides == reference.strides,
                        "A1 output shape/stride/dtype differs")
            different, maximum, finite = 0, 0.0, True
            for channel in range(24):
                for depth in range(0, 192, 8):
                    part = array[:, channel, depth:depth + 8]
                    finite = finite and bool(np.isfinite(part).all())
                    if reference is not None:
                        expected = reference[:, channel, depth:depth + 8]
                        different += int(np.count_nonzero(part.view(np.uint32) != expected.view(np.uint32)))
                        if finite:
                            maximum = max(maximum, float(np.max(np.abs(part - expected))))
            row["bit_gate"] = {"different_bits": different if reference is not None else None,
                                "max_abs": maximum if reference is not None and finite else None,
                                "finite": finite, "shape_stride_dtype_equal": True, "output_independent": True}
            if reference is None:
                path = args.output / (name + ".preELU.private.npy")
                np.save(path, array, allow_pickle=False)
                row["reference_saved"] = identity(path)
                require(row["reference_saved"]["bytes"] <= plan["single_array_max_bytes"], "A1 reference disk bound exceeded")
            atomic_json(args.output / "report.json", report)
            require(finite and different == 0, "first real preELU bit difference; no next pass/arm")
            del reference, array, actual, image, input_array
        if helper is not None:
            helper._provider_still_matches()
        expected = 12 if args.mode == "candidate" else 0
        require(report["conv1_calls"] == 2 and report["candidate_copy_calls"] == report["candidate_SGEMM_calls"] == expected,
                "declared real32 slab call count differs")
        report["completed"] = True
        report["status"] = "two_baseline_references_saved" if args.output.name == "A1_baseline" else "two_real_passes_bit_exact"
    except BaseException as error:
        report["status"] = "real_layer_first_failure_stopped"
        report["error_type"], report["error"] = type(error).__name__, str(error)
        raise
    finally:
        if helper is not None and original_copy is not None:
            helper.copy, helper.gemm = original_copy, original_gemm
        for name, callback, expected in (("bindings", lambda: check_bindings(args.root, args.workspace, plan), before),
                                         ("runtime", lambda: check_runtime(torch, plan), runtime_before),
                                         ("flags", lambda: flags(torch), initial),
                                         ("capture", lambda: capture_bindings(args.run, plan)[1], capture_before)):
            if name in ("runtime", "flags") and torch is None:
                report["postcondition_errors"][name] = "Torch not imported"
                continue
            try:
                report[name + "_after"] = callback()
                report[name + "_unchanged"] = report[name + "_after"] == expected
            except Exception as error:
                report["postcondition_errors"][name] = str(error)
        if reference_before is not None:
            try:
                observed = {"report": identity(args.run / "A1_baseline/report.json")}
                for name in ("original", "flipped"):
                    observed[name] = identity(args.run / "A1_baseline" / (name + ".preELU.private.npy"))
                report["references_unchanged"] = observed == reference_before
            except Exception as error:
                report["references_unchanged"] = False
                report["postcondition_errors"]["references"] = str(error)
        else:
            report["references_unchanged"] = args.output.name == "A1_baseline"
        report["maximum_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["worker_seconds_with_import_IO_hash_comparison"] = time.monotonic() - started
        report["valid_layer_arm"] = bool(report["completed"] and len(report["passes"]) == 2 and not report["postcondition_errors"]
                                         and all(report.get(name + "_unchanged") for name in ("bindings", "runtime", "flags", "capture"))
                                         and report["references_unchanged"] and not report.get("flags_after", {}).get("CUDA_initialized", True)
                                         and torch.get_num_threads() == torch.get_num_interop_threads() == 8
                                         and report["maximum_RSS_bytes"] <= plan["limits"]["AS_and_RSS_bytes"])
        atomic_json(args.output / "report.json", report)
    require(report["valid_layer_arm"], "first real layer postcondition failure; no later arm")
    print(json.dumps({"status": report["status"], "valid_layer_arm": True, "passes": 2, "conv1_calls": 2}))


if __name__ == "__main__":
    main()
