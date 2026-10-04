"""Isolated real-image diagnostic of scalar libm versus Torch FAST CPU math.

This script patches only its own benchmark process. It is not imported by
FNIT and does not change the public CPU or CUDA algorithms.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--reference-prefix", required=True)
    parser.add_argument("--previous-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--math", choices=("stock", "libm"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import nibabel as nib
    import numpy as np
    import torch
    from numba import njit, prange, set_num_threads
    from fnit.fast import TorchFAST, algorithm

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    set_num_threads(8)
    report = {"scope": "real_full_brain_cpu_math_diagnostic", "variant": args.math,
              "input_sha256": digest(args.image), "mask_sha256": digest(args.mask),
              "threads": 8, "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "production_changed": False, "trace": [], "files": {},
              "source_sha256": {name: digest(Path(algorithm.__file__).with_name(name))
                                for name in ("algorithm.py", "_fsl_cpu.py", "_fsl_scan.py")}}

    def array_digest(tensor):
        return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()

    def trace_function(name):
        original = getattr(algorithm, name)

        def traced(*values, **keywords):
            result = original(*values, **keywords)
            tensors = result if isinstance(result, tuple) else (result,)
            report["trace"].append({"function": name,
                                    "input": array_digest(values[0]),
                                    "output": [array_digest(x) for x in tensors]})
            return result

        setattr(algorithm, name, traced)

    for name in ("_fsl_moments", "_fsl_bias", "_fsl_partial_volumes"):
        trace_function(name)

    old_exp, old_log = torch.exp, torch.log
    if args.math == "libm":
        library = ctypes.CDLL(None)
        exp32, exp64 = library.expf, library.exp
        log32, log64 = library.logf, library.log
        for function in (exp32, log32):
            function.argtypes, function.restype = [ctypes.c_float], ctypes.c_float
        for function in (exp64, log64):
            function.argtypes, function.restype = [ctypes.c_double], ctypes.c_double

        @njit(parallel=True, fastmath=False)
        def evaluate32(values, logarithm):
            output = np.empty_like(values)
            for index in prange(values.size):
                output[index] = log32(values[index]) if logarithm else exp32(values[index])
            return output

        @njit(parallel=True, fastmath=False)
        def evaluate64(values, logarithm):
            output = np.empty_like(values)
            for index in prange(values.size):
                output[index] = log64(values[index]) if logarithm else exp64(values[index])
            return output

        def evaluate(tensor, logarithm):
            if tensor.is_cuda or tensor.dtype not in (torch.float32, torch.float64):
                raise RuntimeError("This diagnostic supports only CPU FP32/FP64")
            values = tensor.detach().contiguous().numpy().reshape(-1)
            function = evaluate32 if tensor.dtype == torch.float32 else evaluate64
            output = function(values, logarithm)
            return torch.from_numpy(output.reshape(tensor.shape))

        torch.exp = lambda tensor: evaluate(tensor, False)
        torch.log = lambda tensor: evaluate(tensor, True)

    started = time.perf_counter()
    try:
        result = TorchFAST(device="cpu", threads=8, execution="fsl")(args.image, mask=args.mask)
    finally:
        torch.exp, torch.log = old_exp, old_log
    report["api_seconds_with_trace"] = time.perf_counter() - started
    fields = {"pve_0": "pve_csf", "pve_1": "pve_gm", "pve_2": "pve_wm",
              "seg": "hard_segmentation", "pveseg": "pve_segmentation",
              "mixeltype": "mixel_type", "bias": "bias_field", "restore": "restored"}
    for suffix, field in fields.items():
        nib.save(getattr(result, field), args.output / ("fast_" + suffix + ".nii.gz"))
    for path in sorted(args.output.glob("fast_*.nii.gz")):
        suffix = path.name[len("fast_"):]
        image = nib.load(path)
        values = np.asarray(image.dataobj)
        entry = {"sha256": digest(path), "shape": list(values.shape), "dtype": str(values.dtype),
                 "finite": bool(np.isfinite(values).all())}
        for key, prefix in (("official", args.reference_prefix), ("previous", args.previous_prefix)):
            reference = nib.load(prefix + "_" + suffix)
            other = np.asarray(reference.dataobj)
            error = values.astype(np.float64) - other.astype(np.float64)
            entry[key] = {"different_voxels": int(np.count_nonzero(values != other)),
                          "maximum_absolute_error": float(np.max(np.abs(error))),
                          "rmse": float(np.sqrt(np.mean(error * error))),
                          "affine_equal": bool(np.array_equal(image.affine, reference.affine))}
        report["files"][suffix] = entry
    (args.output / "record.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"variant": args.math, "api_seconds_with_trace": report["api_seconds_with_trace"],
                      "files": report["files"]}))


if __name__ == "__main__":
    main()
