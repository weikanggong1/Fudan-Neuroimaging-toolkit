"""Frozen-input standard sphere pairing; only the normals backend changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numba
import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(first, second):
    a, fa = fsio.read_geometry(str(first))
    b, fb = fsio.read_geometry(str(second))
    same_faces = np.array_equal(fa, fb)
    correspondence = same_faces and a.shape == b.shape
    output = {"same_vertex_count": len(a) == len(b), "same_face_count": len(fa) == len(fb),
              "ordered_faces_equal": same_faces, "vertex_correspondence": correspondence,
              "first_sha256": sha256(first), "second_sha256": sha256(second)}
    if correspondence:
        distance = np.linalg.norm(a - b, axis=1)
        output.update({"coordinates_equal": bool(np.array_equal(a, b)),
                       "coordinates_bitwise_equal": bool(np.array_equal(a.astype(np.float32).view(np.uint32),
                                                                          b.astype(np.float32).view(np.uint32))),
                       "different_vertices": int(np.count_nonzero(np.any(a != b, axis=1))),
                       "mean_mm": float(distance.mean()), "p99_mm": float(np.percentile(distance, 99)),
                       "max_mm": float(distance.max(initial=0))})
    for name, xyz, faces in (("first", a, fa), ("second", b, fb)):
        corners = xyz[faces]
        signed = np.einsum("ij,ij->i", np.cross(corners[:, 1] - corners[:, 0],
                                                corners[:, 2] - corners[:, 0]), corners.mean(axis=1))
        output[name + "_negative_faces"] = int(np.count_nonzero(signed < 0))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "candidate", "compare"), required=True)
    parser.add_argument("--inflated", type=Path, required=True)
    parser.add_argument("--smoothwm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--official-sphere", type=Path)
    parser.add_argument("--normals-device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "compare":
        result = {"candidate_vs_baseline": compare(args.output / "baseline.sphere", args.output / "candidate.sphere"),
                  "official_comparison": "not_provided", "whole_pipeline": "not_run"}
        if args.official_sphere:
            result["official_comparison"] = "frozen_sphere_only_input_provenance_requires_separate_check"
            result["baseline_vs_official"] = compare(args.output / "baseline.sphere", args.official_sphere)
            result["candidate_vs_official"] = compare(args.output / "candidate.sphere", args.official_sphere)
        (args.output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False))
        return
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    numba.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from fnit.recon_all.sphere_standard_run import run_standard_sphere
    device = torch.device(args.normals_device) if args.mode == "candidate" else None
    if device is not None:
        torch.cuda.set_device(device)
        torch.empty(1, device=device)  # Real allocation, no OOM skip or fallback.
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    result = run_standard_sphere(inflated=args.inflated, smoothwm=args.smoothwm,
                                output=args.output / (args.mode + ".sphere"),
                                finish_device="cpu", averaging_device="cpu",
                                normals_device=str(device) if device is not None else None)
    if device is not None:
        torch.cuda.synchronize(device)
    total = perf_counter() - started
    source_hashes = {name: sha256(mod.__file__) for name, mod in sorted(sys.modules.items())
                     if name.startswith("fnit.recon_all.") and getattr(mod, "__file__", None)
                     and Path(mod.__file__).suffix == ".py"}
    report = {"code_version": args.code_version, "script_sha256": sha256(__file__),
              "source_hashes": source_hashes, "host": platform.node(), "platform": platform.platform(),
              "mode": args.mode, "threads": args.threads, "torch_interop_threads": 1,
              "affinity": sorted(os.sched_getaffinity(0)), "torch_version": torch.__version__,
              "numba_version": numba.__version__, "gpu_device": str(device) if device else None,
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "tf32_cudnn": torch.backends.cudnn.allow_tf32, "autocast": False,
              "input_sha256": {"inflated": sha256(args.inflated), "smoothwm": sha256(args.smoothwm)},
              "wrapper_seconds_including_stage_io": total, "stage_report": result,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device else None,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device else None,
              "process_memory_sampling": "not_measured", "whole_pipeline": "not_run",
              "shared_load": "Concurrent FNIT CPU and GPU validations; observation, not stable throughput",
              "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                             "--format=csv,noheader"], capture_output=True, text=True).stdout}
    (args.output / (args.mode + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"mode": args.mode, "seconds": total, "output": result["output"]}))


if __name__ == "__main__":
    main()
