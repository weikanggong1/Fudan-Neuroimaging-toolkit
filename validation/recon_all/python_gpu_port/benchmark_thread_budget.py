"""真实冻结球面或 CA normalize 的 Numba 预算回归，不替代整例。

--subject 为已有 FNIT 自产目录；--assets 为声明资产；--stage 为 sphere/ca；
--hemi 默认 lh；--masks 默认 128 4；--torch-threads 默认 4。
输出须为新目录，写每个掩码的阶段文件和 report.json，包含读写墙钟、输入
及源码 SHA-256、有效预算和数组/几何差异。球面为 surface RAS/mm；
norm/ctrl 为 conform 网格。首轮 JIT 未排除，不据此宣称线程提速。
失败保存已完成结果并退出非零。官方流程对应 mris_sphere / mri_ca_normalize，
此包装器没有独立官方命令；仅改变 Numba 工作掩码，保持 Torch 预算。
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import nibabel as nib
from nibabel.freesurfer.io import read_geometry
import numba
import numpy as np
import torch

from fnit.recon_all.thread_budget import thread_budget


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_volume(a_path, b_path):
    a_img, b_img = nib.load(str(a_path)), nib.load(str(b_path))
    a, b = np.asanyarray(a_img.dataobj), np.asanyarray(b_img.dataobj)
    if a.shape != b.shape:
        return {"pass": False, "shape_equal": False}
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    row = {"shape_equal": True, "dtype_equal": a.dtype == b.dtype,
           "different_elements": int(np.count_nonzero(a != b)),
           "max_error": float(delta.max()), "p99_error": float(np.quantile(delta, .99)),
           "affine_max_error_mm": float(np.abs(a_img.affine - b_img.affine).max())}
    row["pass"] = row["dtype_equal"] and row["different_elements"] == 0 and row["affine_max_error_mm"] == 0
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("sphere", "ca"), required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), default="lh")
    parser.add_argument("--masks", type=int, nargs="+", default=[128, 4])
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    if len(args.masks) < 2 or min(args.masks) < 1 or max(args.masks) > numba.config.NUMBA_NUM_THREADS:
        raise ValueError("需要至少两个有效 Numba 掩码")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.torch_threads)
    initial_mask = numba.get_num_threads()
    if args.stage == "sphere":
        from fnit.recon_all import sphere_standard_run as stage
        inputs = [args.subject / "surf" / f"{args.hemi}.{name}" for name in ("inflated", "smoothwm")]
    else:
        from fnit.recon_all import ca_normalize_python as stage
        inputs = [args.subject / "mri" / name for name in ("nu.mgz", "brainmask.mgz", "transforms/talairach.lta")]
        inputs.append(args.assets / "average/RB_all_2020-01-02.gca")
    report = {"scope": "frozen_same_input_thread_mask_regression", "host": platform.node(),
              "code_commit": args.code_commit, "stage": args.stage,
              "torch_version": torch.__version__, "numba_version": numba.__version__,
              "torch_threads": args.torch_threads, "numba_initial_mask": initial_mask,
              "numba_capacity": numba.config.NUMBA_NUM_THREADS,
              "timing_scope": "includes IO and first JIT; no speedup claim",
              "input_sha256": {str(p): digest(p) for p in inputs},
              "source_sha256": {str(p.relative_to(Path(stage.__file__).parent)): digest(p)
                                 for p in Path(stage.__file__).parent.rglob("*.py")},
              "script_sha256": digest(__file__), "runs": []}
    try:
        for mask in args.masks:
            folder = args.output / str(mask)
            folder.mkdir()
            numba.set_num_threads(mask)
            started = time.perf_counter()
            if args.stage == "sphere":
                value = stage.run_standard_sphere(inflated=inputs[0], smoothwm=inputs[1],
                                                  output=folder / "sphere", finish_device="cpu")
            else:
                value = stage.run_ca_normalize(nu_path=inputs[0], mask_path=inputs[1],
                    lta_path=inputs[2], gca_path=inputs[3],
                    norm_path=folder / "norm.mgz", ctrl_path=folder / "ctrl_pts.mgz")
            report["runs"].append({"numba_mask": numba.get_num_threads(),
                                  "torch_threads": torch.get_num_threads(),
                                  "wall_seconds_including_io": time.perf_counter() - started,
                                  "stage_report": value})
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        first = args.output / str(args.masks[0])
        comparisons = {}
        for mask in args.masks[1:]:
            second = args.output / str(mask)
            if args.stage == "sphere":
                a, af = read_geometry(str(first / "sphere"))
                b, bf = read_geometry(str(second / "sphere"))
                same_shape = a.shape == b.shape
                row = {"same_vertex_count": same_shape, "ordered_faces_equal": bool(np.array_equal(af, bf))}
                if same_shape:
                    delta = np.abs(a - b)
                    row.update(different_coordinates=int(np.count_nonzero(a != b)),
                               max_coordinate_error_mm=float(delta.max()),
                               p99_coordinate_error_mm=float(np.quantile(delta, .99)))
                row["pass"] = same_shape and row["ordered_faces_equal"] and row["different_coordinates"] == 0
                comparisons[str(mask)] = row
            else:
                comparisons[str(mask)] = {name: compare_volume(first / name, second / name)
                                         for name in ("norm.mgz", "ctrl_pts.mgz")}
        report["comparisons"] = comparisons
        report["pass"] = all(row["pass"] if args.stage == "sphere" else all(v["pass"] for v in row.values())
                             for row in comparisons.values())
    finally:
        numba.set_num_threads(initial_mask)
        report["numba_restored_mask"] = numba.get_num_threads()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

