"""用原始单幅 T1 验证已初始化 CUDA 的 API；不生成官方参考。

命令行输入为 t1（原始 NIfTI 路径）、subject（不存在或为空的输出目录）、
--weights-dir/--assets-dir（已校验的权重/资产目录）、--device（默认 cuda:0，
必须为 CUDA）和 --threads（默认 4，先于 CUDA 初始化设置 PyTorch 线程）。
输出为标准重建目录及 run-api-invocation.json，后者记录设备、线程、调用全程
秒数、allocator 环境和脚本 SHA-256；surface RAS/mm 等数据约定沿用重建入口。
CUDA 初始化或任何阶段失败时抛异常，侧车仅在整例成功后写出。此验证包装器
没有官方等价 CLI；所验证标准流程对应 recon-all 的单 T1 -all profile。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from fnit.recon_all.native_free import run_recon_all_python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("t1", type=Path)
    parser.add_argument("subject", type=Path)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if torch.device(args.device).type != "cuda":
        raise ValueError("this validation requires a CUDA device")
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    retained = torch.ones(1, dtype=torch.float32, device=args.device)
    torch.cuda.synchronize(args.device)
    assert torch.cuda.is_initialized()
    device_uuid = str(torch.cuda.get_device_properties(args.device).uuid)
    print(json.dumps({"validation_device_uuid": device_uuid}), flush=True)
    result = run_recon_all_python(
        t1=args.t1, subject_dir=args.subject,
        weights_dir=args.weights_dir, assets_dir=args.assets_dir,
        device=args.device, threads=args.threads, native_bin_dir=None)
    torch.cuda.synchronize(args.device)
    metadata = {"invocation": "initialized CUDA Python API",
                "cuda_initialized_before_api": True,
                "device": args.device, "threads": args.threads,
                "thread_budget_set_before_cuda": True,
                "device_uuid": device_uuid,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "PYTORCH_NO_CUDA_MEMORY_CACHING":
                os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
                "seconds_including_context_and_run": time.perf_counter() - started,
                "retained_tensor_bytes": retained.numel() * retained.element_size(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.subject / "run-api-invocation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
