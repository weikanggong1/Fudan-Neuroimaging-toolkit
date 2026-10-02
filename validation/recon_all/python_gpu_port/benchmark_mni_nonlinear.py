"""同输入真实 T1 的 MNI 非线性链 CPU/CUDA 配对计时，包含模型加载、原生转换与 I/O。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import torch

from fnit.recon_all import mni_nonlinear_chain
from fnit.recon_all.mni_nonlinear_chain import run_mni_nonlinear_chain


INPUTS = ("orig.mgz", "transforms/synthmorph.1.0mm.1.0mm/aff.lta",
          "transforms/synthmorph.1.0mm.1.0mm/invol.crop.nii.gz")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source_subject", "output_subject", "weights", "assets",
                 "native_bin", "report"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    if args.output_subject.exists():
        raise FileExistsError(args.output_subject)
    input_hashes = {}
    for name in INPUTS:
        source = args.source_subject / "mri" / name
        target = args.output_subject / "mri" / name
        input_hashes[name] = _sha256(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
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
    result = run_mni_nonlinear_chain(
        subject_dir=args.output_subject, weights_dir=args.weights,
        assets_dir=args.assets,
        warp_convert=args.native_bin / "mri_warp_convert",
        ca_register=args.native_bin / "mri_ca_register",
        mri_convert=args.native_bin / "mri_convert",
        device=args.device, threads=args.threads)
    if cuda:
        torch.cuda.synchronize(args.device)
    seconds = time.perf_counter() - started
    outputs = {key: {"sha256": _sha256(Path(path)),
                     "size_bytes": Path(path).stat().st_size}
               for key, path in result.items() if key in ("forward", "inverse", "check")}
    report = {
        "code_commit": args.code_commit, "host": platform.node(),
        "stage_source_sha256": _sha256(Path(mni_nonlinear_chain.__file__)),
        "benchmark_script_sha256": _sha256(Path(__file__)),
        "device": args.device, "threads": args.threads,
        "precision": {"matmul_tf32_after_stage": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_tf32_after_stage": torch.backends.cudnn.allow_tf32,
                      "float16_or_bfloat16": False},
        "input_sha256": input_hashes,
        "native_binary_sha256": {name: _sha256(args.native_bin / name) for name in
                                 ("mri_warp_convert", "mri_ca_register", "mri_convert")},
        "seconds_including_model_io_conversion": seconds,
        "stage_timings_seconds": result["timings_seconds"],
        "outputs": outputs,
        "gpu_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(args.device) if cuda and not cache_disabled
            else None),
        "gpu_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(args.device) if cuda and not cache_disabled
            else None),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.device}: {seconds:.3f} s")


if __name__ == "__main__":
    main()
