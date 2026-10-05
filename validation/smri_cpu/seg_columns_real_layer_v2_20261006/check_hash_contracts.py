"""Pure byte-hash preflight only; no convolution, im2col, BLAS or MRI values."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys

from trial_bindings import check_bindings, check_runtime, flags, identity, require, value_sha


def legacy5_sha(tensor):
    """The unchanged v1 channel byte loop, only valid for image/output5D."""
    array = tensor.detach().numpy()
    require(array.ndim == 5 and array.shape[0] == 1 and array.flags.c_contiguous,
            "legacy contiguous singleton-batch5D required")
    digest = hashlib.sha256()
    for channel in range(array.shape[1]):
        digest.update(memoryview(array[0, channel]).cast("B"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    plan_identity = identity(args.workspace / "PLAN.json")
    plan = json.loads((args.workspace / "PLAN.json").read_text())
    require(args.output.resolve() == (args.root / plan["runs"] / "HASH_CONTRACTS.json").resolve()
            and not args.output.exists(), "one canonical new hash receipt required")
    require(sorted(os.sched_getaffinity(0)) == plan["affinity"], "fixed CPU8 affinity required")
    require(all(os.environ.get(name) == "8" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")), "CPU8 budget required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not any(name in os.environ
            for name in ("PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "OPENBLAS_CORETYPE")),
            "CPU only without overrides required")
    resource.setrlimit(resource.RLIMIT_AS, (32_000_000_000, 32_000_000_000))
    bound_before = check_bindings(args.root, args.workspace, plan)
    sys.path.insert(0, str(args.root / "repo/src"))
    import torch
    import numpy as np
    from fnit.synthseg_parc.segment import SegmentUNet
    require(torch.__version__ == "2.5.1", "target Torch2.5.1 required")
    runtime_before = check_runtime(torch, plan)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    before = flags(torch)
    require(not before["CUDA_initialized"] and not before["CPU_autocast"], "no CUDA/autocast required")
    report = {"schema": "fnit_real_layer_hash_contract/v1", "PLAN": plan_identity,
              "status": "started_not_accepted", "bindings_before": bound_before,
              "runtime_before": runtime_before, "flags_before": before,
              "Torch": torch.__version__, "affinity": sorted(os.sched_getaffinity(0)),
              "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "rows": [], "negative_guards": [], "postcondition_failures": {}, "completed": False,
              "convolution_calls": 0, "candidate_copy_calls": 0, "candidate_SGEMM_calls": 0,
              "MRI_array_loads": 0, "whole_model_forward_calls": 0, "GPU_calls": 0,
              "numeric_convolution_result_assessed": False}

    def check(name, tensor):
        array = tensor.detach().numpy()
        require(array.flags.c_contiguous, "C-contiguous contract value required")
        # Direct whole raw bytes is independent of the channel-hash traversal.
        raw = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
        observed = value_sha(tensor)
        require(observed == raw, "first hash vs raw-byte mismatch: " + name)
        old = legacy5_sha(tensor) if tensor.ndim == 5 else None
        require(old is None or old == observed, "old5D byte hash changed")
        require(hashlib.sha256(memoryview(array).cast("B")).hexdigest() == raw, "hash input bytes changed")
        report["rows"].append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                               "raw_C_bytes_sha256": raw, "channel_sha256": observed,
                               "legacy5_sha256": old, "exact": True, "input_bytes_unchanged": True})

    try:
        # Raw uint32 bit patterns include signed zero, infinities and a NaN payload.
        # They are copied/viewed only; no floating arithmetic or random generation.
        bit_pattern = bytes.fromhex("00000000000000800000803f000080bf0000807f000080ff3412c07f01000000")
        for shape in plan["hash_contracts"]["synthetic_bit_only_shapes"]:
            count = 1
            for size in shape:
                count *= size
            raw = (bit_pattern * ((count * 4 + len(bit_pattern) - 1) // len(bit_pattern)))[:count * 4]
            array = np.frombuffer(raw, dtype=np.uint32).copy().view(np.float32).reshape(shape)
            check("pattern_" + str(len(shape)) + "D", torch.from_numpy(array))
        # Load only the already-bound weight file; no MRI or any model forward.
        model = SegmentUNet().load_h5(args.root / plan["dependencies"]["weight"]["fnit_relative_path"]).eval()
        tensor = model.up[3].conv0.weight.view(1, *model.up[3].conv0.weight.shape)
        require(list(tensor.shape) == plan["hash_contracts"]["actual_weight_shape"], "actual weight6D shape differs")
        check("actual_weight6D", tensor)
        model = tensor = None
        values = torch.from_numpy(np.frombuffer(bit_pattern, dtype=np.uint32).copy().view(np.float32))
        invalid = [("rank1", values), ("batch2", values.reshape(2, 4)),
                   ("noncontiguous", values.reshape(1, 2, 2, 2).transpose(2, 3))]
        for name, tensor in invalid:
            try:
                value_sha(tensor)
            except RuntimeError as error:
                require(str(error) == "contiguous singleton-batch value hash required", "unexpected guard error")
                report["negative_guards"].append({"name": name, "expected_rejection": True})
            else:
                raise RuntimeError("missing hash guard: " + name)
        require(len(report["rows"]) == 4 and len(report["negative_guards"]) == 3, "hash contract counts differ")
        report["completed"] = True
        report["status"] = "pure_hash_contracts_passed_no_convolution"
    except Exception as error:
        report["status"] = "pure_hash_contract_failed_no_layer_dispatched"
        report["error_type"], report["error"] = type(error).__name__, str(error)
        raise
    finally:
        for name, callback in (("bindings", lambda: check_bindings(args.root, args.workspace, plan)),
                               ("runtime", lambda: check_runtime(torch, plan)),
                               ("PLAN", lambda: identity(args.workspace / "PLAN.json")),
                               ("flags", lambda: flags(torch))):
            try:
                report[name + "_after"] = callback()
            except Exception as error:
                report["postcondition_failures"][name] = {"type": type(error).__name__, "error": str(error)}
        report["bindings_unchanged"] = report.get("bindings_after") == bound_before
        report["runtime_unchanged"] = report.get("runtime_after") == runtime_before
        report["PLAN_unchanged"] = report.get("PLAN_after") == plan_identity
        report["flags_unchanged"] = report.get("flags_after") == before
        report["maximum_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["valid_hash_contracts"] = (report["completed"] and not report["postcondition_failures"]
            and all(report[name] for name in ("bindings_unchanged", "runtime_unchanged", "PLAN_unchanged", "flags_unchanged"))
            and report["maximum_RSS_bytes"] <= 32_000_000_000)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    require(report["valid_hash_contracts"], "pure hash postconditions failed; no layer arm permitted")
    print(json.dumps({"status": report["status"], "rows": 4, "negative_guards": 3, "convolution_calls": 0}))


if __name__ == "__main__":
    main()
