"""冻结真实 white/pial，比较原密集厚度与完整空间候选厚度。

输入为相同有序网格的 white/pial（surface RAS，mm），原实现源码路径，
新空输出目录和明确代码版本。默认 cuda:0、四个 PyTorch 线程、两轮；
GPU 前后同步计时，包含函数加载影像、传输、计算及写出，不含导入。
写出每轮两张 float32 morph 厚度及 report.json，含输入/源码 SHA-256、
秒数、峰值统计和差异。优化回归预先使用绝对 1e-6 mm、相对 0；
可另传同输入 reference-map，沿用既有 0.005+0.001*abs(ref) 厚度门槛。
未通过时仍保存报告并退出 1。官方参考只能在隔离 benchmark 路径产生。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import time

import nibabel.freesurfer as fs
import numpy as np
import torch

from fnit.recon_all import surface_thickness_gpu as candidate


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparison(reference, result, absolute, relative):
    delta = np.abs(reference.astype(np.float64) - result.astype(np.float64))
    failed = ~np.isfinite(delta) | (delta > absolute + relative * np.abs(reference))
    return {"pass": not bool(failed.any()), "different_values": int(np.count_nonzero(delta)),
            "outliers": int(failed.sum()), "max_abs_mm": float(delta.max()),
            "p99_abs_mm": float(np.quantile(delta, .99)), "mae_mm": float(delta.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--white", type=Path, required=True)
    parser.add_argument("--pial", type=Path, required=True)
    parser.add_argument("--baseline-source-file", type=Path, required=True)
    parser.add_argument("--reference-map", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats 必须至少为 1")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    gpu = torch.device(args.device).type == "cuda"
    if gpu:
        torch.cuda.init()
        torch.cuda.synchronize(args.device)
    spec = importlib.util.spec_from_file_location("frozen_dense_thickness", args.baseline_source_file)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    vertices, faces = fs.read_geometry(str(args.white))
    pial, pfaces = fs.read_geometry(str(args.pial))
    if vertices.shape != pial.shape or not np.array_equal(faces, pfaces):
        raise ValueError("white/pial 必须有对应顶点和相同有序面")
    reference = fs.read_morph_data(str(args.reference_map)) if args.reference_map else None
    if reference is not None and len(reference) != len(vertices):
        raise ValueError("reference-map 必须来自完全相同的冻结网格")
    cache = os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") != "1"
    report = {"code_commit": args.code_commit, "baseline_commit": "3d9856c9dc659a50b685dbc8c0b9b6c695461fe7",
              "host": socket.gethostname(), "device": args.device, "threads": args.threads,
              "precision": {"dtype": "float32", "matmul_tf32": True, "cudnn_tf32": True,
                            "float16_or_bfloat16": False},
              "vertices": len(vertices), "faces": len(faces),
              "input_sha256": {"white": sha256(args.white), "pial": sha256(args.pial)},
              "source_sha256": {"baseline": sha256(args.baseline_source_file),
                                "candidate": sha256(candidate.__file__), "script": sha256(__file__)},
              "thresholds": {"optimization": {"absolute_mm": 1e-6, "relative": 0},
                             "existing_reference": {"absolute_mm": .005, "relative": .001}},
              "timing_scope": "同步目标 GPU；包含函数影像加载、传输、计算及写出，排除脚本导入/上下文初始化",
              "cuda_allocator_cache_enabled": cache, "runs": [],
              "whole_pipeline_speedup": "not_measured"}
    if args.reference_map:
        report["reference_map_sha256"] = sha256(args.reference_map)
    for repeat in range(args.repeats):
        values, row = {}, {"repeat": repeat, "order": [], "implementations": {}}
        order = ("dense", "indexed") if repeat % 2 == 0 else ("indexed", "dense")
        for name in order:
            row["order"].append(name)
            output = args.output / f"{repeat}.{name}.thickness"
            if gpu:
                torch.cuda.synchronize(args.device)
                if cache:
                    torch.cuda.reset_peak_memory_stats(args.device)
            started_utc = datetime.now(timezone.utc).isoformat()
            tick = time.perf_counter()
            function = baseline.thickness_map if name == "dense" else candidate.thickness_map_indexed
            details = function(white_file=args.white, pial_file=args.pial,
                               output_file=output, device=args.device)
            if gpu:
                torch.cuda.synchronize(args.device)
            seconds = time.perf_counter() - tick
            row["implementations"][name] = {"seconds_including_io": seconds,
                    "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
                    "function_report": details, "output_sha256": sha256(output),
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if gpu and cache else None,
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device) if gpu and cache else None}
            values[name] = fs.read_morph_data(str(output))
        row["optimization_comparison"] = comparison(values["dense"], values["indexed"], 1e-6, 0)
        if reference is not None:
            row["reference_comparisons"] = {name: comparison(reference, data, .005, .001)
                                           for name, data in values.items()}
        report["runs"].append(row)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    passed = all(row["optimization_comparison"]["pass"] and
                 all(value["pass"] for value in row.get("reference_comparisons", {}).values())
                 for row in report["runs"])
    report["pass"] = passed
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
