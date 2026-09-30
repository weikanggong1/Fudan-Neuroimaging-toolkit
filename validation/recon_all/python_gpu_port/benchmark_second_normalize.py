"""同输入真实 T1 的第二轮归一化 CPU/CUDA 配对计时，包含 MGH 读写和数据传输。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import torch

from fnit.recon_all.normalization import aseg_pipeline
from fnit.recon_all.normalization.aseg_pipeline import normalize_t1_aseg


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("norm", "aseg", "brainmask", "output", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cuda = torch.device(args.device).type == "cuda"
    cache_disabled = os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1"
    if cuda:
        torch.cuda.synchronize(args.device)
        if not cache_disabled:
            torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    result = normalize_t1_aseg(
        norm_file=args.norm, aseg_file=args.aseg,
        brainmask_file=args.brainmask, output_file=args.output,
        device=args.device, three_d_iterations=2)
    if cuda:
        torch.cuda.synchronize(args.device)
    seconds = time.perf_counter() - started
    report = {
        "code_commit": args.code_commit, "host": platform.node(),
        "stage_source_sha256": _sha256(Path(aseg_pipeline.__file__)),
        "benchmark_script_sha256": _sha256(Path(__file__)),
        "device": args.device, "threads": args.threads,
        "tf32_default": True, "float16_or_bfloat16": False,
        "input_sha256": {name: _sha256(getattr(args, name))
                         for name in ("norm", "aseg", "brainmask")},
        "output_sha256": _sha256(args.output), "seconds_including_io": seconds,
        "function_report": result,
        "gpu_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(args.device) if cuda and
            not cache_disabled else None),
        "gpu_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(args.device) if cuda and
            not cache_disabled else None),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.device}: {seconds:.3f} s")


if __name__ == "__main__":
    main()
