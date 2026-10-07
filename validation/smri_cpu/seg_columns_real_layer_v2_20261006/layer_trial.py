"""Prepared one up3.conv0 call on saved real input; approval required, no tail."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

from trial_bindings import check_bindings, check_runtime, flags, identity, require, value_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--approved-real-layer", action="store_true", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    plan_identity = identity(args.workspace / "PLAN.json")
    plan = json.loads((args.workspace / "PLAN.json").read_text())
    modes = {"A1_baseline": "baseline", "B1_candidate": "candidate", "B2_candidate": "candidate", "A2_baseline": "baseline"}
    require(args.output.parent.resolve() == (args.root / plan["runs"]).resolve()
            and modes.get(args.output.name) == args.mode, "declared canonical arm output required")
    require((args.reference is None) == (args.output.name == "A1_baseline"), "A1 alone omits a reference")
    if args.reference is not None:
        require(args.reference.resolve() == (args.root / plan["runs"] / "A1_baseline/baseline_preELU.private.npy").resolve(), "only the A1 file is a reference")
    require(sorted(os.sched_getaffinity(0)) == plan["affinity"], "same eight physical cores required")
    require(all(os.environ.get(name) == "8" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")), "CPU8 thread environment required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and not any(name in os.environ for name in ("PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "OPENBLAS_CORETYPE")), "CPU-only route without overrides required")
    resource.setrlimit(resource.RLIMIT_AS, (32_000_000_000, 32_000_000_000))
    started = time.monotonic()
    bound_before = check_bindings(args.root, args.workspace, plan)
    args.output.mkdir(parents=False, exist_ok=False, mode=0o700)
    source = args.root / "repo/src"
    sys.path.insert(0, str(source))
    import torch
    import numpy as np
    import fnit
    from fnit.synthseg_parc.segment import SegmentUNet
    from fnit.synthseg_parc.cpu_join import cpu_join_allowed, join_nearest_cpu
    require(Path(fnit.__file__).resolve() == (source / "fnit/__init__.py").resolve(), "canonical source import required")
    require(torch.__version__ == "2.5.1", "target Torch2.5.1 required")
    runtime_before = check_runtime(torch, plan)
    # The same five-file v2 helper is imported by exact path, not copied/modified.
    helper_path = args.root / plan["dependencies"]["v2_prototype.py"]["fnit_relative_path"]
    specification = importlib.util.spec_from_file_location("fnit_private_columns_v2", helper_path)
    private_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(private_module)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    initial_flags = flags(torch)
    require(not initial_flags["CUDA_initialized"] and not initial_flags["CPU_autocast"], "no CUDA or autocast initialization")
    report = {"schema": "fnit_columns_saved_real_layer_arm/v1", "status": "started_not_accepted", "mode": args.mode,
              "scope": "One saved real decoder up3.conv0 pre-ELU call; no tail/full CNN/native/GPU",
              "PLAN": plan_identity, "bindings_before": bound_before,
              "runtime_bindings_before": runtime_before,
              "flags_before": initial_flags, "affinity": sorted(os.sched_getaffinity(0)),
              "loadavg_before": list(os.getloadavg()),
              "Torch": torch.__version__, "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "reference_generation_only": args.reference is None, "comparison_executed": False,
              "candidate_copy_calls": 0, "candidate_SGEMM_calls": 0, "direct_layer_calls": 0,
              "whole_model_calls": 0, "ELU_BN_head_softmax_calls": 0, "native_calls": 0, "GPU_calls": 0,
              "new_compilation_calls": 0, "whole_map_CSV_assessed": False, "production_changed": False,
              "completed": False, "postcondition_failures": {}}
    reference_identity = None
    helper = None
    real_copy = real_gemm = None
    try:
        if args.reference is None:
            require(args.mode == "baseline", "A1 baseline alone generates the reference")
        else:
            reference_meta = json.loads((args.reference.parent / "report.json").read_text())
            require(reference_meta["status"] == "baseline_reference_saved" and reference_meta["valid_real_layer_arm"]
                    and reference_meta["reference_generation_only"] and not reference_meta["comparison_executed"], "valid A1 reference required")
            require(reference_meta["PLAN"] == report["PLAN"] and reference_meta["bindings_before"] == bound_before, "reference came from a different frozen state")
            reference_identity = identity(args.reference)
            require(reference_identity == reference_meta["saved_reference"]["file"], "A1 saved reference changed")
            report["reference"] = {"file": reference_identity, "value_sha256": reference_meta["output"]["value_sha256"]}
        weight = args.root / plan["dependencies"]["weight"]["fnit_relative_path"]
        model = SegmentUNet().load_h5(weight).eval()
        layer = model.up[3].conv0
        checkpoint = args.root / plan["checkpoint"]
        skip = torch.from_numpy(np.load(checkpoint / "skip.npy", allow_pickle=False))
        value = torch.from_numpy(np.load(checkpoint / "value.npy", allow_pickle=False))
        require(list(skip.shape) == [1, 24, 192, 224, 256] and list(value.shape) == [1, 48, 96, 112, 128]
                and skip.dtype == value.dtype == torch.float32, "actual decoder checkpoint shape/dtype required")
        helper = private_module.ColumnsReuse(args.root / plan["dependencies"]["v1_binary"]["fnit_relative_path"],
                                            plan["provider_sha256"], allow_compute=True, allow_bounded_contracts=False)
        helper._provider_still_matches()
        real_copy, real_gemm = helper.copy, helper.gemm
        # Private counters include their dispatch overhead in the candidate clock.
        # They do not install Module hooks or alter C++/SGEMM inputs.
        def counted_copy(*values):
            report["candidate_copy_calls"] += 1
            return real_copy(*values)
        def counted_gemm(*values):
            report["candidate_SGEMM_calls"] += 1
            return real_gemm(*values)
        helper.copy, helper.gemm = counted_copy, counted_gemm
        with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
            require(cpu_join_allowed(model, skip, value), "mature CPU join guard must remain eligible")
            image = join_nearest_cpu(skip, value)
            del skip, value
            require(list(image.shape) == plan["input_shape"] and image.is_contiguous(), "joined geometry differs")
            joined_hash = value_sha(image)
            require(joined_hash == plan["joined_input_value_sha256"], "saved joined input bits differ")
            require(list(layer.weight.shape) == [24, 72, 3, 3, 3] and list(layer.bias.shape) == [24], "actual layer parameters differ")
            weight_hash = value_sha(layer.weight.view(1, *layer.weight.shape))
            bias_hash = value_sha(layer.bias.view(1, 24, 1, 1, 1))
            report["operation_scope_flags"] = flags(torch)
            usage_before = resource.getrusage(resource.RUSAGE_SELF)
            tick = time.monotonic()
            report["direct_layer_calls"] += 1
            actual = layer(image) if args.mode == "baseline" else helper.forward(layer, image)
            report["operation_seconds_including_candidate_guards_counters"] = time.monotonic() - tick
            usage_after = resource.getrusage(resource.RUSAGE_SELF)
            report["operation_usage"] = {"user_seconds": usage_after.ru_utime - usage_before.ru_utime,
                                         "system_seconds": usage_after.ru_stime - usage_before.ru_stime,
                                         "minor_faults": usage_after.ru_minflt - usage_before.ru_minflt,
                                         "major_faults": usage_after.ru_majflt - usage_before.ru_majflt}
            require(value_sha(image) == joined_hash, "joined input changed")
            require(value_sha(layer.weight.view(1, *layer.weight.shape)) == weight_hash
                    and value_sha(layer.bias.view(1, 24, 1, 1, 1)) == bias_hash, "weights/bias changed")
            report["joined_input_unchanged"] = report["parameter_values_unchanged"] = True
            report["joined_input_value_sha256"] = joined_hash
            report["weight_value_sha256"], report["bias_value_sha256"] = weight_hash, bias_hash
        require(list(actual.shape) == plan["output_shape"] and actual.dtype == torch.float32
                and actual.device.type == "cpu" and actual.is_contiguous(), "output shape/dtype/stride/device differs")
        require(actual.untyped_storage().data_ptr() != image.untyped_storage().data_ptr(), "output aliases input")
        expected_calls = 14 if args.mode == "candidate" else 0
        require(report["candidate_copy_calls"] == report["candidate_SGEMM_calls"] == expected_calls, "real14 slabs not consumed as declared")
        helper._provider_still_matches()
        del image, model, layer
        report["same_public_provider_gate_passed"] = True
        array = actual.numpy()
        reference = np.load(args.reference, mmap_mode="r", allow_pickle=False) if args.reference else None
        if reference is not None:
            require(array.shape == reference.shape and array.dtype == reference.dtype and array.strides == reference.strides, "A1 output layout differs")
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
        report["output"] = {"value_sha256": value_sha(actual), "shape": list(actual.shape),
                            "stride": list(actual.stride()), "dtype": str(actual.dtype)}
        report["comparison_executed"] = reference is not None
        report["bit_gate"] = {"different_bits": different, "max_abs": maximum if finite else None,
                              "finite": finite, "shape_stride_dtype_equal": True, "output_independent": True}
        if reference is None:
            path = args.output / "baseline_preELU.private.npy"
            np.save(path, array, allow_pickle=False)
            report["saved_reference"] = {"file": identity(path), "value_sha256": report["output"]["value_sha256"]}
            require(report["saved_reference"]["file"]["bytes"] <= plan["limits"]["maximum_new_private_array_bytes_with_header"], "new private array disk bound exceeded")
            report["status"] = "baseline_reference_saved" if finite else "baseline_nonfinite_failed"
        else:
            require(identity(args.reference) == reference_identity, "A1 reference file changed during comparison")
            report["reference_file_unchanged"] = True
            report["status"] = "complete_real_layer_bits_exact" if finite and different == 0 else "real_layer_first_nonexact_stopped"
        report["completed"] = True
    except Exception as error:
        report["status"] = "real_layer_error_stopped"
        report["error_type"], report["error"] = type(error).__name__, str(error)
        raise
    finally:
        if helper is not None and real_copy is not None:
            helper.copy, helper.gemm = real_copy, real_gemm
        for label, callback in (("bindings", lambda: check_bindings(args.root, args.workspace, plan)),
                                ("runtime_bindings", lambda: check_runtime(torch, plan)),
                                ("PLAN", lambda: identity(args.workspace / "PLAN.json")),
                                ("flags", lambda: flags(torch))):
            try:
                report[label + "_after"] = callback()
            except Exception as error:
                report["postcondition_failures"][label] = {"error_type": type(error).__name__, "error": str(error)}
        report["bindings_unchanged"] = report.get("bindings_after") == bound_before
        report["runtime_bindings_unchanged"] = report.get("runtime_bindings_after") == runtime_before
        report["PLAN_unchanged"] = report.get("PLAN_after") == plan_identity
        report["flags_unchanged"] = report.get("flags_after") == initial_flags
        report["maximum_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["loadavg_after"] = list(os.getloadavg())
        report["worker_observation_seconds_with_IO_hash_comparison"] = time.monotonic() - started
        report["valid_real_layer_arm"] = bool(report["completed"] and not report["postcondition_failures"]
            and report["bindings_unchanged"] and report["runtime_bindings_unchanged"] and report["PLAN_unchanged"] and report["flags_unchanged"] and not report["flags_after"]["CUDA_initialized"]
            and report["maximum_RSS_bytes"] <= 32_000_000_000 and report["status"] in ("baseline_reference_saved", "complete_real_layer_bits_exact"))
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    require(report["valid_real_layer_arm"], "first nonexact or failed postcondition; no later arms allowed")
    print(json.dumps({"status": report["status"], "comparison_executed": report["comparison_executed"],
                      "mode": args.mode, "operation_seconds": report["operation_seconds_including_candidate_guards_counters"]}))


if __name__ == "__main__":
    main()
