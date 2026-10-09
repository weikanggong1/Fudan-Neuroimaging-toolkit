"""真实原生首试步：同梯度限幅表达式及完整有序碰撞回归。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import nibabel.freesurfer.io as fs
import numpy as np
import torch

from diagnose_pial_first_difference import compare, native_state, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-diagnostic-directory", type=Path, required=True)
    p.add_argument("--python-diagnostic-directory", type=Path, required=True)
    p.add_argument("--subject", type=Path, required=True)
    p.add_argument("--candidate-directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.candidate_directory))
    from fnit.recon_all import place_surface_step as step
    from fnit.recon_all import place_surface_collision as collision
    from fnit.recon_all.place_surface_smoothing import _ordered_neighbors
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    x, faces = fs.read_geometry(str(args.subject / "surf/lh.white"))
    prefix = args.native_diagnostic_directory / "native_gradient"
    clear = native_state(Path(f"{prefix}.step01.clear"), len(x))
    target = native_state(Path(f"{prefix}.step01.after_collision"), len(x))
    native_gradient = native_state(Path(f"{prefix}.step01.tangential_spring"), len(x))["floats"][:, 6:9].copy()
    current = clear["floats"][:, :3].copy()
    ripped = np.asarray(clear["flags"][:, 0], dtype=np.bool_)
    with np.load(args.python_diagnostic_directory / "python_step01.npz") as py:
        old = py["accepted"].copy()
        python_gradient = np.float32(np.float32(np.float32(py["averaged"] + py["normal"]) +
            np.float32(py["curvature"][:, None] * py["normals"])) + py["tangent"])
    if not np.array_equal(native_gradient, python_gradient):
        raise ValueError("first native/Python gradients differ; cannot isolate norm rounding")
    neighbors, valid, _ = _ordered_neighbors(faces.astype(np.int32), len(x))
    expression = np.fromfile(args.native_diagnostic_directory.parent / "norm_expression_output.bin",
        dtype=np.dtype([("offset", "<f4", (3,)), ("magnitude", "<f8")]))["offset"][:len(x)]
    native_accepted = target["floats"][:, :3]
    report = {"scope": "real_complete_first_pial_trial_norm_and_ordered_collision_only",
        "hostname": platform.node(), "threads": args.threads, "device": args.device,
        "half_precision": False, "tf32_matmul": True, "tf32_cudnn": True,
        "required_native_coordinate_tolerance_mm": 0,
        "native_report_sha256": sha(args.native_diagnostic_directory / "report.json"),
        "python_report_sha256": sha(args.python_diagnostic_directory / "report.json"),
        "source_sha256": {Path(step.__file__).name: sha(step.__file__),
            Path(collision.__file__).name: sha(collision.__file__)},
        "old_acceptance_vs_native": compare(old, native_accepted), "runs": []}
    for backend in ("tree", "torch_snapshot"):
        device = torch.device(args.device)
        if backend != "tree":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        proposal, offsets = step.unconstrained_step_with_offsets(current, native_gradient, ripped, dt=0.5, max_mm=0.3)
        native_expression = expression.copy()
        native_expression[ripped] = 0
        if not np.array_equal(offsets, native_expression):
            raise ValueError("new displacement differs from same-compiler scalar expression")
        accepted_offsets = native_gradient.copy()
        actual, order = collision.asynchronous_first_step(current, faces, proposal, ripped,
            fast=True, offsets=offsets, accepted_offsets=accepted_offsets,
            ordered_neighbors=(neighbors, valid), candidate_backend=backend,
            candidate_grid_cells_per_axis=3 if backend == "torch_snapshot" else 2,
            retained_mht_backend="compiled" if backend == "torch_snapshot" else "tree",
            candidate_device=args.device if backend == "torch_snapshot" else None)
        if backend != "tree":
            torch.cuda.synchronize(device)
        row = {"backend": backend, "complete_trial_seconds": time.perf_counter()-started,
            "expression_offsets_exact": True, "acceptance_vs_native": compare(actual, native_accepted),
            "coordinate_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
            "peak_allocated_bytes": None if backend == "tree" else torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": None if backend == "tree" else torch.cuda.max_memory_reserved(device)}
        report["runs"].append(row)
    report["passed"] = all(r["acceptance_vs_native"]["different_elements"] == 0 for r in report["runs"])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
