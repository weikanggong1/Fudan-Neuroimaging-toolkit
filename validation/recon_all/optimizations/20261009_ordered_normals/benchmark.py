"""Same-input ordered vertex-normal CPU/CUDA regression and timing."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numba
import numpy as np
import torch
from fnit.recon_all.place_surface_normals import FaceNormalTopology, TorchFaceNormalTopology
import fnit.recon_all.place_surface_normals as normal_module

MAX_COMPONENT_ERROR = 5.0e-7  # Fixed before the real mesh run, unit vector dimensionless.


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(function, repeats, device=None):
    samples = []
    result = None
    for _ in range(repeats):
        if device is not None:
            torch.cuda.synchronize(device)
        started = perf_counter()
        result = function()
        if device is not None:
            torch.cuda.synchronize(device)
        samples.append(perf_counter() - started)
    return result, {"seconds": samples, "median_seconds": float(np.median(samples))}


def measure_pair(cpu_function, torch_function, repeats, device):
    samples = {"numba": [], "torch": []}
    sequence = []
    results = {}
    for round_number in range(repeats):
        order = ("numba", "torch", "torch", "numba") if round_number % 2 == 0 else ("torch", "numba", "numba", "torch")
        for backend in order:
            function = cpu_function if backend == "numba" else torch_function
            result, timing = measure(function, 1, device)
            results[backend] = result
            samples[backend].append(timing["seconds"][0])
            sequence.append(backend)
    summary = {key: {"seconds": value, "median_seconds": float(np.median(value))}
               for key, value in samples.items()}
    return results, summary, sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--code-version", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    numba.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or an explicit CUDA device")
    synchronize_device = device if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    report = {
        "code_version": args.code_version,
        "source_sha256": sha256(normal_module.__file__),
        "hostname": platform.node(), "platform": platform.platform(),
        "torch_version": torch.__version__, "numba_version": numba.__version__,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "threads": args.threads, "affinity": sorted(os.sched_getaffinity(0)),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32, "dtype": "float32",
        "autocast": False, "max_component_error_threshold": MAX_COMPONENT_ERROR,
        "official_program_comparison": "not_run_no_independent_cli",
        "whole_pipeline_comparison": "not_run",
        "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout,
        "process_memory_sampling": "not_measured_kernel_only_pytorch_peaks",
        "meshes": [],
    }
    for path in args.input:
        started = perf_counter()
        xyz, faces = fsio.read_geometry(str(path))
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        read_seconds = perf_counter() - started
        started = perf_counter()
        cpu = FaceNormalTopology(triangles=faces, nvertices=len(xyz))
        cpu_setup = perf_counter() - started
        expected, cold_cpu = measure(lambda: cpu.evaluate(vertices=xyz), 1)
        cpu.evaluate(vertices=xyz)
        started = perf_counter()
        candidate = TorchFaceNormalTopology(triangles=faces, nvertices=len(xyz), device=str(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gpu_setup = perf_counter() - started
        resident = torch.as_tensor(xyz, device=device)
        measure(lambda: candidate.evaluate_tensor(vertices=resident), 1, synchronize_device)
        result, kernel_time = measure(lambda: candidate.evaluate_tensor(vertices=resident), args.repeats, synchronize_device)
        pair_results, paired, sequence = measure_pair(
            lambda: cpu.evaluate(vertices=xyz), lambda: candidate.evaluate(vertices=xyz),
            args.repeats, synchronize_device)
        expected, observed = pair_results["numba"], pair_results["torch"]
        cpu_time, api_time = paired["numba"], paired["torch"]
        error = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
        component = {"different_components": int(np.count_nonzero(observed != expected)),
                     "max_abs": float(error.max(initial=0)), "p99_abs": float(np.percentile(error, 99)) if error.size else 0.0,
                     "finite": bool(np.isfinite(observed).all()), "passed": bool(error.max(initial=0) <= MAX_COMPONENT_ERROR)}
        report["meshes"].append({"input": str(path), "input_sha256": sha256(path),
                                 "vertices": len(xyz), "faces": len(faces), "max_degree": candidate.max_degree,
                                 "read_seconds": read_seconds, "cpu_setup_seconds": cpu_setup,
                                 "cpu_first_call_including_jit": cold_cpu, "torch_setup_seconds": gpu_setup,
                                 "cpu_hot": cpu_time, "torch_resident_kernel": kernel_time,
                                 "torch_numpy_api_including_transfers": api_time, "pair_sequence": sequence, "precision": component})
        del resident, result, candidate
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    report["all_meshes_passed"] = all(item["precision"]["passed"] for item in report["meshes"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"all_meshes_passed": report["all_meshes_passed"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
