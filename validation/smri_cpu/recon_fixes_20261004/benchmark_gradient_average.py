"""Old/new ordered averaging on a complete real cortical surface.

Only averaging is timed. Real sphere and smoothwm generate the distance/area
gradient first; this diagnostic is not an end-to-end registration benchmark.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_averager(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sphere", required=True)
    parser.add_argument("--smoothwm", required=True)
    parser.add_argument("--baseline-source", required=True)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import nibabel as nib
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == "cuda":
        torch.cuda.init()
        properties = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(20e9 / properties.total_memory, 0)
        torch.cuda.reset_peak_memory_stats(0)
    from fnit.recon_all.mris_register_nonlinear import (
        first_area_gradient, first_distance_gradient,
        prepare_registration_force_cache,
    )
    sphere, faces = nib.freesurfer.read_geometry(args.sphere)
    original, original_faces = nib.freesurfer.read_geometry(args.smoothwm)
    if sphere.shape != original.shape or not np.array_equal(faces, original_faces):
        raise ValueError("sphere and smoothwm must have identical vertex/face correspondence")
    positions = torch.from_numpy(sphere.copy()).float()
    smoothwm = torch.from_numpy(original.copy()).float()
    topology = torch.from_numpy(faces.astype(np.int64, copy=True))
    cache = prepare_registration_force_cache(positions, smoothwm, topology)
    gradient = first_distance_gradient(positions, smoothwm, positions, topology, cache=cache)
    gradient = first_area_gradient(positions, smoothwm, positions, topology, gradient, cache=cache)
    if not bool(torch.isfinite(gradient).all()) or not bool(torch.any(gradient != 0)):
        raise ValueError("real gradient must be finite and nontrivial")
    modules = {
        "baseline": import_averager(args.baseline_source, "fnit.recon_all._average_baseline_bench"),
        "candidate": import_averager(args.candidate_source, "fnit.recon_all._average_candidate_bench"),
    }
    averagers = {key: module.RegistrationGradientAverager(cache.neighbors, cache.degrees,
                                                       device=args.device)
                 for key, module in modules.items()}

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize(0)

    def measure(key, iterations):
        synchronize()
        started = time.perf_counter()
        values = averagers[key](gradient, iterations)
        synchronize()
        return values, time.perf_counter() - started

    report = {"scope": "complete_real_surface_gradient_averaging_only",
              "hostname": socket.gethostname(), "device": args.device,
              "threads": args.threads, "affinity": sorted(os.sched_getaffinity(0)),
              "input_sha256": {"sphere": digest(args.sphere), "smoothwm": digest(args.smoothwm)},
              "source_sha256": {"baseline": digest(args.baseline_source),
                                "candidate": digest(args.candidate_source)},
              "vertices": len(sphere), "faces": len(faces),
              "gradient_sha256": hashlib.sha256(gradient.numpy().tobytes()).hexdigest(),
              "records": []}
    initial = gradient.clone()
    cold = {}
    for key in averagers:
        _, cold[key] = measure(key, 1)
    report["first_call_seconds_including_jit"] = cold
    for iterations in (1, 16, 256):
        values = {}
        records = []
        for key in ("baseline", "candidate", "candidate", "baseline"):
            output, seconds = measure(key, iterations)
            if key in values and not torch.equal(output, values[key]):
                raise AssertionError("same-backend repeat differs")
            values[key] = output
            records.append({"variant": key, "seconds": seconds})
        equal = bool(torch.equal(values["baseline"], values["candidate"]))
        if not equal or not torch.equal(gradient, initial):
            raise AssertionError("old/new output or original input changed")
        report["records"].append({"iterations": iterations, "output_equal": equal,
                                  "output_sha256": hashlib.sha256(values["candidate"].numpy().tobytes()).hexdigest(),
                                  "abba": records})
    if args.device == "cuda":
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        report["gpu_name"] = torch.cuda.get_device_name(0)
    report["gate_passed"] = True
    (args.output / "record.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
