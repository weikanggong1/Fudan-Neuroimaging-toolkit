"""从原始 T1 运行已初始化 CUDA 的 Python API，单独记录调用方式。"""

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
    retained = torch.ones(1, dtype=torch.float32, device=args.device)
    torch.cuda.synchronize(args.device)
    assert torch.cuda.is_initialized()
    result = run_recon_all_python(
        t1=args.t1, subject_dir=args.subject,
        weights_dir=args.weights_dir, assets_dir=args.assets_dir,
        device=args.device, threads=args.threads, native_bin_dir=None)
    torch.cuda.synchronize(args.device)
    metadata = {"invocation": "initialized CUDA Python API",
                "cuda_initialized_before_api": True,
                "device": args.device, "threads": args.threads,
                "PYTORCH_NO_CUDA_MEMORY_CACHING":
                os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
                "seconds_including_context_and_run": time.perf_counter() - started,
                "retained_tensor_bytes": retained.numel() * retained.element_size(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.subject / "run-api-invocation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
