"""冻结真实 white/pial，只运行一次新厚度，验证 KDTree 线程预算修正。

--candidate-source-file 指向独立诊断目录的候选源码；--expected-source-sha256
要求其 SHA-256 完全一致。--pair-report 为此前旧/新同输入阶段 JSON，
--white、--pial 必须符合其中记录的 SHA-256（surface RAS、mm）；读取该
报告旁的 0.indexed.thickness 作为冻结旧图，不能复制到生产路径。
--conda-map 为相同输入、20/5 参数的 Conda 源码构建厚度；--conda-binary
只记录其版本哈希，不执行。--output 是新的空诊断目录。
--code-version 绑定候选代码标识，--gpu-uuid 明确物理目标，--device 默认
cuda:0，--threads 默认 4。缓存须在进程启动前关闭，FP32/TF32、不用半精度。

读取原图、初始化 CUDA 及导入不计入函数时间；GPU 前后同步，函数时间
包含其输入读取、传输、计算及写出。输出新 float32 morph（mm）与 report.json，
要求与冻结图零差异，并沿用 Conda 0.005+0.001*abs(reference) 门槛。
失败保存报告并退出 1。只验证本阶段，不能判定整例提速或整体指标等效。
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
import sys
import time

import nibabel.freesurfer as fs
import numpy as np
import torch


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
    for name in ("candidate-source-file", "pair-report", "white", "pial", "conda-map",
                 "conda-binary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-source-sha256", "code-version", "gpu-uuid"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or torch.device(args.device).type != "cuda":
        parser.error("此真实 GPU 回归需要 CUDA 和正线程预算")
    if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") != "1":
        raise ValueError("进程启动前须显式设置 PYTORCH_NO_CUDA_MEMORY_CACHING=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid or args.device != "cuda:0":
        raise ValueError("CUDA_VISIBLE_DEVICES 须只映射指定 UUID 到逻辑 cuda:0")
    source_sha = sha256(args.candidate_source_file)
    if source_sha != args.expected_source_sha256:
        raise ValueError("候选源码 SHA-256 不符合冻结版本")
    pair = json.loads(args.pair_report.read_text())
    inputs = {"white": sha256(args.white), "pial": sha256(args.pial)}
    if inputs != pair["input_sha256"]:
        raise ValueError("white/pial SHA-256 不符合冻结配对报告")
    if pair["thresholds"] != {"optimization": {"absolute_mm": 1e-6, "relative": 0},
                              "existing_reference": {"absolute_mm": .005, "relative": .001}}:
        raise ValueError("既有阶段门槛发生变化")
    frozen_run = pair["runs"][0]
    frozen_map = args.pair_report.parent / f"{frozen_run['repeat']}.indexed.thickness"
    frozen_sha = sha256(frozen_map)
    if frozen_sha != frozen_run["implementations"]["indexed"]["output_sha256"]:
        raise ValueError("冻结旧厚度图 SHA-256 不符合报告")
    conda_sha = sha256(args.conda_map)
    if conda_sha != pair["reference_map_sha256"]:
        raise ValueError("Conda 图 SHA-256 不符合冻结同输入参考")
    frozen = fs.read_morph_data(str(frozen_map))
    conda = fs.read_morph_data(str(args.conda_map))
    white, faces = fs.read_geometry(str(args.white))
    pial, pfaces = fs.read_geometry(str(args.pial))
    if white.shape != pial.shape or not np.array_equal(faces, pfaces) or \
            white.shape != (pair["vertices"], 3) or len(faces) != pair["faces"] or \
            frozen.shape != (len(white),) or conda.shape != frozen.shape or \
            not np.isfinite(frozen).all() or not np.isfinite(conda).all():
        raise ValueError("网格或同序顶点图结构不符合冻结输入")
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("fnit_thickness_final_workers", args.candidate_source_file)
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate
    spec.loader.exec_module(candidate)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(args.device)
    actual_uuid = str(getattr(properties, "uuid", "unavailable"))
    if actual_uuid != "unavailable" and actual_uuid.removeprefix("GPU-") != args.gpu_uuid.removeprefix("GPU-"):
        raise ValueError("CUDA 设备属性 UUID 与明确目标不符")
    torch.cuda.synchronize(args.device)
    started = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()
    output = args.output / "lh.thickness"
    details = candidate.thickness_map(white_file=args.white, pial_file=args.pial,
                                     output_file=output, device=args.device)
    torch.cuda.synchronize(args.device)
    seconds = time.perf_counter() - tick
    ended = datetime.now(timezone.utc).isoformat()
    value = fs.read_morph_data(str(output))
    if value.shape != frozen.shape:
        raise ValueError("新厚度图长度不符合冻结网格")
    regression = comparison(frozen, value, 1e-6, 0)
    conda_comparison = comparison(conda, value, .005, .001)
    report = {"scope": "sub-01 LH frozen FNIT white/pial; one candidate function call",
              "code_version": args.code_version, "host": socket.gethostname(),
              "device": args.device, "gpu_uuid_requested": args.gpu_uuid,
              "gpu_uuid_from_torch": actual_uuid, "gpu_name": properties.name,
              "threads": torch.get_num_threads(),
              "thread_environment": {key: os.environ.get(key) for key in
                                     ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                      "NUMEXPR_NUM_THREADS")},
              "precision": {"dtype": "float32", "matmul_tf32": True, "cudnn_tf32": True,
                            "float16_or_bfloat16": False},
              "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cuda_allocator_cache_enabled": False,
              "gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None,
              "vertices": len(white), "faces": len(faces), "input_sha256": inputs,
              "source_sha256": {"candidate": source_sha, "script": sha256(__file__)},
              "frozen_pair_report": {"path": str(args.pair_report), "sha256": sha256(args.pair_report),
                                     "code_commit": pair["code_commit"],
                                     "candidate_source_sha256": pair["source_sha256"]["candidate"]},
              "frozen_map": {"path": str(frozen_map), "sha256": frozen_sha},
              "conda_map": {"path": str(args.conda_map), "sha256": conda_sha},
              "conda_program": {"path": str(args.conda_binary), "sha256": sha256(args.conda_binary),
                                "execution": "not run; existing same-input map"},
              "output_sha256": sha256(output), "function_report": details,
              "started_utc": started, "ended_utc": ended,
              "seconds_including_io": seconds,
              "timing_scope": "synchronized target GPU; function input I/O, transfers, computation, output I/O; excludes imports/CUDA init/diagnostic hashing",
              "thresholds": pair["thresholds"], "optimization_comparison": regression,
              "strict_frozen_reproduction": regression["different_values"] == 0,
              "conda_comparison": conda_comparison,
              "new_official_validation": "not_run", "whole_pipeline_speedup": "not_measured",
              "overall_equivalence": "not_assessed"}
    report["pass"] = (report["strict_frozen_reproduction"] and regression["pass"] and
                      conda_comparison["pass"] and details["kdtree_workers"] == min(4, args.threads))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
