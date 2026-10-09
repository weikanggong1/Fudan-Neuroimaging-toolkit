"""完全相同最终 lattice/logfield/expfield 的 CPU/CUDA 分段数学比较。

候选仅读取明确冻结的阶段输入；参考值在每次候选计算结束后用于比较。
这是诊断，不能用于生产流程，不能视作完整 N4 的精度或提速结论。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from fnit.recon_all import n4_bspline_torch, n4_itk_torch_experimental


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_raw(path, shape):
    values = np.fromfile(path, np.float32)
    if values.size != np.prod(shape):
        raise ValueError("frozen raw shape mismatch")
    return values.reshape(tuple(shape), order="F")


def compare(value, reference):
    delta = value.astype(np.float64) - reference.astype(np.float64)
    error = np.abs(delta)
    return {"different": int(np.count_nonzero(value != reference)), "max_abs": float(error.max()),
            "p99_abs": float(np.quantile(error, .99)), "mean_signed": float(delta.mean()),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "negative": int(np.count_nonzero(delta < 0)), "positive": int(np.count_nonzero(delta > 0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True, help="final diagnostic profile prefix")
    parser.add_argument("--input-raw", type=Path, required=True)
    parser.add_argument("--corrected-reference-raw", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    prefix = str(args.prefix)
    lattice_meta = json.loads(Path(prefix + ".final_lattice.json").read_text())
    field_meta = json.loads(Path(prefix + ".logfield.json").read_text())
    lattice = read_raw(prefix + ".final_lattice.raw", lattice_meta["shape"])
    logfield = read_raw(prefix + ".logfield.raw", field_meta["shape"])
    expfield = read_raw(prefix + ".expfield.raw", field_meta["shape"])
    image = read_raw(args.input_raw, field_meta["shape"])
    reference = read_raw(args.corrected_reference_raw, field_meta["shape"])
    files = [args.input_raw, args.corrected_reference_raw,
             Path(prefix + ".final_lattice.raw"), Path(prefix + ".logfield.raw"), Path(prefix + ".expfield.raw")]
    report = {"kind": "fixed_lattice_and_fixed_field_diagnostic_not_complete_n4", "host": platform.node(),
              "device": str(device), "threads": args.threads, "process_affinity": sorted(os.sched_getaffinity(0)),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_tf32": torch.backends.cudnn.allow_tf32, "half_precision": False,
              "sources_sha256": {str(path): sha(path) for path in [Path(__file__), Path(n4_bspline_torch.__file__),
                                      Path(n4_itk_torch_experimental.__file__)]},
              "inputs_sha256": {str(path): sha(path) for path in files}, "comparisons": {},
              "production_default_changed": False, "whole_n4_equivalence": "not evaluated"}
    if device.type == "cuda":
        torch.empty(1, device=device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        report["gpu_name"] = torch.cuda.get_device_name(device)
    sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    def measured(label, function, ref):
        sync()
        start = time.perf_counter()
        output = function()
        sync()
        compute_seconds = time.perf_counter() - start
        start = time.perf_counter()
        array = output.cpu().numpy()
        transfer_seconds = time.perf_counter() - start
        report["comparisons"][label] = {**compare(array, ref), "compute_seconds": compute_seconds,
                                          "d2h_seconds": transfer_seconds}
        return output
    # Transfer is measured separately from compute; every comparison shares exact frozen source tensors.
    sync()
    start = time.perf_counter()
    lattice_tensor = torch.from_numpy(lattice.copy()).to(device)
    log_tensor = torch.from_numpy(logfield.copy()).to(device)
    exp_tensor = torch.from_numpy(expfield.copy()).to(device)
    image_tensor = torch.from_numpy(image.copy()).to(device)
    sync()
    report["load_tensor_h2d_seconds"] = time.perf_counter() - start
    sync()
    start = time.perf_counter()
    reconstruct = n4_itk_torch_experimental.N4CubicReconstruction(control_shape=lattice_meta["shape"],
        output_shape=field_meta["shape"], spacing=field_meta["spacing_mm"], device=device)
    sync()
    report["reconstruction_geometry_cache_seconds"] = time.perf_counter() - start
    measured("identical_lattice_to_logfield", lambda: reconstruct(lattice_tensor), logfield)
    measured("identical_logfield_exp", lambda: torch.exp(log_tensor), expfield)
    measured("identical_expfield_divide", lambda: image_tensor / exp_tensor, reference)
    measured("identical_logfield_exp_divide", lambda: image_tensor / torch.exp(log_tensor), reference)
    if device.type == "cuda":
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
