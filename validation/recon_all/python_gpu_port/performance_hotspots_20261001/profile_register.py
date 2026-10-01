"""用冻结真实表面剖析完整球面配准；只写新的诊断目录。"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numba
import numpy as np
import torch

from fnit.recon_all.mris_register_run import run_register_sphere


def main() -> None:
    """读取 surface RAS/mm 网格、同序 sulc 和 TIFF 图谱，输出 pstats/JSON/网格。

    --subject 是已有 FNIT 被试目录，--atlas 是对应半球图谱；--hemi
    选择 lh/rh，--threads 固定 Torch/Numba 线程预算（默认 4）。输出目录
    必须不存在。--code-commit 绑定实际被测版本；同时保存模块和输入哈希。
    cProfile 含记录开销，不能将该耗时视为普通阶段配对的提速结果。
    输入缺失、网格不对应、优化未收敛或输出已存在时直接抛异常。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    numba.set_num_threads(args.threads)
    surface = args.subject / "surf"
    inputs = {name: surface / f"{args.hemi}.{name}"
              for name in ("sphere", "smoothwm", "sulc", "sphere.reg")}
    inputs["atlas"] = args.atlas
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in inputs.items()}
    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    result = run_register_sphere(
        sphere=inputs["sphere"], smoothwm=inputs["smoothwm"],
        sulc=inputs["sulc"], atlas=inputs["atlas"],
        output=args.output / "sphere.reg", overlap_device="cpu")
    profile.disable()
    measured_seconds = time.perf_counter() - start
    profile.dump_stats(str(args.output / "register.pstats"))
    with (args.output / "profile.txt").open("w") as stream:
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(70)
    rows = [{"file": key[0], "line": key[1], "function": key[2],
             "primitive_calls": val[0], "calls": val[1],
             "self_seconds": val[2], "cumulative_seconds": val[3]}
            for key, val in pstats.Stats(profile).stats.items()]
    reference, faces = fsio.read_geometry(str(inputs["sphere.reg"]))
    candidate, got_faces = fsio.read_geometry(str(args.output / "sphere.reg"))
    comparable = reference.shape == candidate.shape and np.array_equal(faces, got_faces)
    comparison = {"ordered_faces_equal": bool(np.array_equal(faces, got_faces)),
                  "same_coordinate_shape": reference.shape == candidate.shape}
    if comparable:
        error = np.abs(reference - candidate)
        comparison.update(different_coordinate_elements=int(np.count_nonzero(error)),
                          max_coordinate_error_mm=float(error.max(initial=0)),
                          p99_coordinate_error_mm=float(np.quantile(error, .99)))
    report = {"scope": "frozen FNIT inputs; cProfile overhead included; no official rerun",
              "code_commit": args.code_commit, "host": platform.node(),
              "threads": {"torch": torch.get_num_threads(), "numba": numba.get_num_threads()},
              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "input_sha256": hashes, "input_paths": {k: str(v) for k, v in inputs.items()},
              "profile_wall_seconds": measured_seconds, "registration": result,
              "versus_saved_same_input_registration": comparison,
              "functions": sorted(rows, key=lambda x: x["cumulative_seconds"], reverse=True),
              "source_sha256": {str(Path(m.__file__)): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                for n, m in __import__("sys").modules.items()
                                if n.startswith("fnit.recon_all.mris_register") and getattr(m, "__file__", None)},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"profile_wall_seconds": measured_seconds, "comparison": comparison}))


if __name__ == "__main__":
    main()
