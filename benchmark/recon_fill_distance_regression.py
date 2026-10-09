"""真实aseg完整CC距离场：旧有序Python与Torch初始化/Numba反馈比较。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import nibabel as nib
import numpy as np
import torch


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def difference(actual, expected):
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"different_elements": int(np.count_nonzero(delta)),
            "max_abs": float(delta.max()), "p99_abs": float(np.percentile(delta, 99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aseg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--module-dir", type=Path)
    args = parser.parse_args()
    if args.module_dir is not None:
        import fnit.recon_all
        fnit.recon_all.__path__.insert(0, str(args.module_dir.resolve()))
    from fnit.recon_all import fill_aseg_python as fill
    from fnit.recon_all import fill_boundary_torch as boundary
    from fnit.recon_all import fill_marching_numba as marching
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("GPU test requires explicit device index")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.synchronize(device)
    image = nib.load(args.aseg)
    seg = np.asarray(image.dataobj, dtype=np.int32)
    cc = (seg >= 251) & (seg <= 255)
    report = {"scope": "complete_frozen_same_input_CC_distance_fields; no full filled or raw T1 timing",
        "code_commit": args.code_commit, "modules_sha256": {
            Path(module.__file__).name: sha(module.__file__) for module in (fill, boundary, marching)},
        "script_sha256": sha(__file__), "input_sha256": sha(args.aseg),
        "input_shape": [int(value) for value in image.shape], "input_dtype": str(image.get_data_dtype()),
        "input_affine_mm": image.affine.tolist(), "host": platform.node(), "device": str(device),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": args.threads,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "source_algorithm": "current FNIT source-order fast marching; Conda reference complete filled comparison separate",
        "records": [], "precision": "float32 state updates, explicit float64 sqrt/solution, no fastmath/no half"}
    old_fields, new_fields = [], []
    for label in (2, 41):
        old_profile, new_profile = {}, {}
        tick = time.perf_counter()
        old = fill._cc_outside_distance(seg, cc, label, profile=old_profile)
        old_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        new = fill._cc_outside_distance(seg, cc, label, boundary_backend="torch",
            marching_backend="numba", device=str(device), profile=new_profile)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        new_seconds = time.perf_counter() - tick
        report["records"].append({"label": label, "old_seconds": old_seconds,
            "new_seconds": new_seconds, "new_timing_includes_first_signature_JIT_if_missing": True,
            "old_profile": old_profile, "new_profile": new_profile,
            "full_field": difference(new, old), "CC_query_field": difference(new[cc], old[cc])})
        old_fields.append(old[cc]); new_fields.append(new[cc])
    report["CC_left_right_choice_differences"] = int(np.count_nonzero(
        (old_fields[0] < old_fields[1]) != (new_fields[0] < new_fields[1])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
