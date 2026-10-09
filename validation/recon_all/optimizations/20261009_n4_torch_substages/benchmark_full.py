"""原始固定真实 N4 输入 → 200 次自产反馈 → 完整浮点/uint8 输出。"""
import time
_PROCESS_STARTED = time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import threading

import numpy as np
import torch

from fnit.recon_all.n4_itk_torch_experimental import correct_tensor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(a, b):
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"different": int(np.count_nonzero(a != b)), "max_abs": float(error.max()),
            "p99_abs": float(np.quantile(error, .99)), "rmse": float(np.sqrt(np.mean(error*error)))}


def quantize(a):
    return np.floor(np.clip(a, 0, 255) + .5).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-raw", type=Path, required=True)
    parser.add_argument("--shape", nargs=3, type=int, default=(256, 256, 256))
    parser.add_argument("--spacing", nargs=3, type=float, default=(1, 1, 1))
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--official-frozen-raw", type=Path)
    parser.add_argument("--frozen-prefix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    report = {"kind": "complete_single_input_n4_feedback_not_recon_all",
              "host": platform.node(), "threads": args.threads, "interop_threads": 1,
              "device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda,
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_tf32": torch.backends.cudnn.allow_tf32, "half_precision": False,
              "profile_synchronizes_each_phase": args.profile,
              "recipe": {"shrink": 4, "levels": 4, "max_iterations": [50]*4,
                         "convergence_threshold": 0, "bins": 200, "fwhm": .15,
                         "wiener_noise": .01, "spline_order": 3, "mask": "all ones"},
              "input_path": str(args.input_raw), "input_sha256": sha(args.input_raw),
              "source_sha256": {}, "whole_recon_all_acceleration": "not measured"}
    import fnit.recon_all.n4_bspline_torch as fit_module
    import fnit.recon_all.n4_itk_torch_experimental as full_module
    for source in (Path(__file__), Path(fit_module.__file__), Path(full_module.__file__)):
        report["source_sha256"][str(source.resolve())] = sha(source)
    samples = []; stop = threading.Event()
    if device.type == "cuda":
        torch.empty(1, dtype=torch.float32, device=device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["external_gpu_state"] = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True).stdout
        def monitor():
            while not stop.is_set():
                rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
                                      capture_output=True, text=True).stdout.splitlines()
                for row in rows:
                    fields = [s.strip() for s in row.split(",")]
                    if fields and fields[0] == str(os.getpid()):
                        samples.append({"elapsed_seconds": time.perf_counter()-_PROCESS_STARTED,
                                        "gpu_uuid": fields[1], "process_bytes": int(fields[2].split()[0])*1024**2})
                stop.wait(.1)
        thread = threading.Thread(target=monitor, daemon=True); thread.start()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    array = np.fromfile(args.input_raw, np.float32).reshape(tuple(args.shape), order="F")
    image = torch.from_numpy(array.copy()).to(device)
    trace = []
    def capture(level, iteration, residual, phi):
        if iteration != 0:
            return
        entry = {"level": level, "iteration": iteration, "control_shape": list(phi.shape)}
        current_residual = residual.cpu().numpy()
        current_phi = phi.cpu().numpy()
        np.save(args.output.parent / f"full_level{level}_phi.npy", current_phi)
        if args.frozen_prefix is not None:
            stem = str(args.frozen_prefix) + f".cp{phi.shape[0]}"
            reference_residual = np.fromfile(stem + ".residual.raw", np.float32).reshape(residual.shape, order="F")
            reference_phi = np.fromfile(stem + ".phi.raw", np.float32).reshape(phi.shape, order="F")
            entry["produced_residual_vs_itk"] = compare(current_residual, reference_residual)
            entry["produced_phi_vs_itk"] = compare(current_phi, reference_phi)
            entry["kind"] = "continuous_upstream_not_frozen_same_input_operator"
        trace.append(entry)
    result = correct_tensor(image=image, spacing=args.spacing, profile=args.profile, callback=capture)
    floating = result.corrected.cpu().numpy()
    destination = args.output.parent / "full_candidate.final.raw"
    floating.ravel(order="F").tofile(destination)
    report["total_api_load_transfer_compute_write_seconds"] = time.perf_counter() - started
    report["iterations"] = result.iteration_count; report["timings"] = result.timings
    report["convergence_values"] = result.convergence
    report["first_iteration_each_level"] = trace
    report["output_sha256"] = sha(destination)
    # 参考在候选完整计算之后才读取，不作为 correct_tensor 的输入。
    reference = np.fromfile(args.reference_raw, np.float32).reshape(tuple(args.shape), order="F")
    report["reference_path"] = str(args.reference_raw); report["reference_sha256"] = sha(args.reference_raw)
    report["float_vs_current_itk"] = compare(floating, reference)
    report["uint8_vs_current_itk"] = compare(quantize(floating), quantize(reference))
    report["strict_reproduction"] = report["float_vs_current_itk"]["different"] == 0
    report["production_default_changed"] = False
    report["full_n4_acceptance"] = "not established by one input; do not infer from local phi gate"
    if args.official_frozen_raw is not None:
        official = np.fromfile(args.official_frozen_raw, np.float32).reshape(tuple(args.shape), order="F")
        report["official_frozen_sha256"] = sha(args.official_frozen_raw)
        report["official_reference_kind"] = "existing same-host fixed-input ITK4.8 GCC repeat-verified record; not a fresh official run"
        report["float_vs_official_frozen"] = compare(floating, official)
        report["uint8_vs_official_frozen"] = compare(quantize(floating), quantize(official))
        report["current_itk_vs_official_frozen"] = compare(reference, official)
    if device.type == "cuda":
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        stop.set(); thread.join(timeout=5)
        report["process_gpu_samples"] = samples
        report["process_gpu_sampling_requested_seconds"] = .1
        report["process_gpu_sample_peak_bytes"] = max([s["process_bytes"] for s in samples], default=None)
    report["script_wall_including_imports_and_comparison_seconds"] = time.perf_counter()-_PROCESS_STARTED
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
