"""用真实 T1 球面输入测量源顺序梯度平滑的 Numba 线程开销。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
from numba import get_num_threads, set_num_threads

from fnit.recon_all.sphere_python import project_radially
from fnit.recon_all.sphere_standard_average import average_standard_gradient
from fnit.recon_all.sphere_standard_metric import (
    average_standard_metric, sample_standard_metric_matrix)
from fnit.recon_all.sphere_standard_python import project_before_standard_unfold
from fnit.recon_all.sphere_standard_unfold import first_epoch_gradient


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inflated", type=Path, required=True)
    parser.add_argument("--smoothwm", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--rounds", type=int, default=1024)
    args = parser.parse_args()
    xyz_input, faces = fsio.read_geometry(str(args.inflated))
    metric_input, metric_faces = fsio.read_geometry(str(args.smoothwm))
    if not np.array_equal(faces, metric_faces):
        raise ValueError("inflated and smoothwm face order differs")
    faces = np.asarray(faces, np.int32)
    xyz = project_radially(project_before_standard_unfold(xyz_input), already_sphere=True)
    offsets, neighbors, raw, _ = sample_standard_metric_matrix(metric_input, faces)
    distances, _, _ = average_standard_metric(offsets, neighbors, raw)
    _, _, gradient, _ = first_epoch_gradient(
        xyz, faces, metric_input, offsets, neighbors, distances, 0.1)
    average_standard_gradient(gradient, offsets, neighbors, 1)
    cases = []
    for threads in args.threads:
        set_num_threads(threads)
        start = time.perf_counter()
        result = average_standard_gradient(gradient, offsets, neighbors, args.rounds)
        cases.append({"threads": get_num_threads(), "seconds": time.perf_counter()-start,
                      "sha256": hashlib.sha256(result.tobytes()).hexdigest()})
    report = {"code_commit": args.code_commit, "host": platform.node(),
              "inflated_sha256": _sha256(args.inflated),
              "smoothwm_sha256": _sha256(args.smoothwm),
              "vertices": len(xyz), "rounds": args.rounds,
              "cases": cases}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["cases"], indent=2))


if __name__ == "__main__":
    main()
