"""Prepared one MRI preprocess plus two conv0+ELU prefixes; explicit approval required."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

from real_bindings import (atomic_json, check_bindings, check_environment, check_runtime,
                           finite_tensor, flags, identity, require, value_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "run"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-real-layer", action="store_true", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    plan = json.loads((args.workspace / "PLAN.json").read_text())
    require(args.workspace.resolve() == (args.root / plan["workspace"]).resolve() and
            args.run.resolve() == (args.root / plan["runs"]).resolve(), "declared workspace/run required")
    check_environment(plan)
    resource.setrlimit(resource.RLIMIT_AS, (32_000_000_000, 32_000_000_000))
    output = args.run / "capture"
    output.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    report = {"schema": "fnit_C24_real_prefix_capture/v1", "status": "started_not_accepted",
              "PLAN": identity(args.workspace / "PLAN.json"), "MRI_decode_calls": 0,
              "preprocess_calls": 0, "conv0_calls": 0, "conv1_calls": 0, "model_forward_calls": 0,
              "native_calls": 0, "GPU_calls": 0, "new_compile_calls": 0, "features": {},
              "completed": False, "postcondition_errors": {}}
    torch = None
    initial = None
    before = runtime_before = None
    try:
        before = check_bindings(args.root, args.workspace, plan)
        report["bindings_before"] = before
        sys.path.insert(0, str(args.root / "repo/src"))
        import torch as loaded_torch
        import numpy as np
        torch = loaded_torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(8)
        initial = flags(torch)
        report["flags_before"] = initial
        require(not initial["CUDA_initialized"] and not initial["CPU_autocast"], "no CUDA or CPU autocast allowed")
        from fnit.synthseg_parc.preprocess import preprocess_t1
        from fnit.synthseg_parc.segment import SegmentUNet
        from torch.nn import functional as F
        runtime_before = check_runtime(torch, plan)
        report["runtime_before"] = runtime_before
        model = SegmentUNet().load_h5(args.root / plan["assets"]["segmentation_weight"]["fnit_relative"]).eval()
        layer = model.down[0].conv0
        require(not getattr(layer, "_fnit_columns_reuse", False), "conv0 must use mature path")
        for name in ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks"):
            require(not getattr(layer, name), "no layer hooks in prefix capture")
        raw_input = args.root / plan["assets"]["raw_t1"]["fnit_relative"]
        report["MRI_decode_calls"] += 1
        report["preprocess_calls"] += 1
        tick = time.monotonic()
        with torch.inference_mode():
            prepared = preprocess_t1(raw_input, device="cpu", min_pad=128)
        report["preprocess_seconds"] = time.monotonic() - tick
        require(list(prepared.image.shape) == [192, 224, 256] and prepared.image.dtype == torch.float32,
                "actual mature preprocessed input geometry differs")
        with torch.inference_mode():
            image = prepared.image.to(device="cpu", dtype=torch.float32)[None, None]
        require(image.is_contiguous(), "original network input must be contiguous")
        input_sha = value_sha(image)
        parameter_shas = [value_sha(value) for value in model.parameters()]
        # Exact prefix of _Block.forward, fed by the posterior's own x/flip(x,(2,)).
        # Calling the two original operations directly avoids hooks or any later layer.
        with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
            report["active_flags"] = flags(torch)
            for name in ("original", "flipped"):
                prefix_input = image if name == "original" else torch.flip(image, (2,))
                report["conv0_calls"] += 1
                tick = time.monotonic()
                feature = F.elu(model.down[0].conv0(prefix_input))
                seconds = time.monotonic() - tick
                require(list(feature.shape) == plan["layer_input_shape"] and feature.dtype == torch.float32
                        and feature.is_contiguous() and finite_tensor(feature, np), "captured feature shape/dtype/finite changed")
                require(feature.untyped_storage().data_ptr() != image.untyped_storage().data_ptr(), "feature aliases original input")
                feature_sha = value_sha(feature)
                path = output / (name + ".private.npy")
                np.save(path, feature.detach().numpy(), allow_pickle=False)
                require(path.stat().st_size <= plan["single_array_max_bytes"], "captured array disk bound exceeded")
                report["features"][name] = {"file": identity(path), "value_sha256": feature_sha,
                                            "shape": [int(value) for value in feature.shape],
                                            "stride": [int(value) for value in feature.stride()],
                                            "dtype": str(feature.dtype), "prefix_seconds": seconds}
                del feature, prefix_input
                atomic_json(output / "report.json", report)
        require(value_sha(image) == input_sha, "original network input changed")
        require([value_sha(value) for value in model.parameters()] == parameter_shas, "model parameters changed")
        report["network_input_value_sha256"] = input_sha
        report["network_input_and_parameters_unchanged"] = True
        report["new_private_array_bytes"] = sum(entry["file"]["bytes"] for entry in report["features"].values())
        require(report["new_private_array_bytes"] <= 2 * plan["single_array_max_bytes"], "capture disk budget changed")
        report["completed"] = True
        report["status"] = "two_prefix_features_saved_no_conv1"
    except BaseException as error:
        report["status"] = "prefix_first_failure_stopped"
        report["error_type"], report["error"] = type(error).__name__, str(error)
        raise
    finally:
        for name, callback, expected in (("bindings", lambda: check_bindings(args.root, args.workspace, plan), before),
                                         ("runtime", lambda: check_runtime(torch, plan), runtime_before),
                                         ("flags", lambda: flags(torch), initial)):
            if name != "bindings" and torch is None:
                report["postcondition_errors"][name] = "Torch not imported"
                continue
            try:
                report[name + "_after"] = callback()
                report[name + "_unchanged"] = report[name + "_after"] == expected
            except Exception as error:
                report["postcondition_errors"][name] = str(error)
        report["maximum_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["worker_seconds_including_import_IO_and_hash"] = time.monotonic() - started
        report["valid_capture"] = bool(report["completed"] and report["conv0_calls"] == 2 and len(report["features"]) == 2
                                        and not report["postcondition_errors"] and report.get("bindings_unchanged")
                                        and report.get("runtime_unchanged") and report.get("flags_unchanged")
                                        and not report.get("flags_after", {}).get("CUDA_initialized", True)
                                        and torch.get_num_threads() == torch.get_num_interop_threads() == 8
                                        and report["maximum_RSS_bytes"] <= plan["limits"]["AS_and_RSS_bytes"])
        atomic_json(output / "report.json", report)
    require(report["valid_capture"], "prefix capture postcondition failed; no real layer dispatch")
    print(json.dumps({"status": report["status"], "valid_capture": True, "conv0_calls": 2, "conv1_calls": 0}))


if __name__ == "__main__":
    main()
