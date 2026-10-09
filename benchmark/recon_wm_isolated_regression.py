"""两种父CUDA状态的完整WM exec回归，复用公共显存采样器。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time

import nibabel as nib
import numpy as np
import torch


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reference", "output-dir", "module-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--parent-mode", choices=("fresh", "initialized"), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") != "1":
        raise ValueError("parent must start with cache disabled")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.module_dir.resolve()))
    from fnit.recon_all.wm_torch_worker import run_isolated_segmentation
    from fnit.recon_all.profiling import ProcessTreeDeviceSampler
    # False仅用于验证子阶段不改父精度，不能作为生产全局设置。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent_tensor = None
    if args.parent_mode == "initialized":
        parent_tensor = torch.zeros((64_000_000,), dtype=torch.uint8, device=args.device)
        torch.cuda.synchronize(args.device)
    before = {"cuda_initialized": torch.cuda.is_initialized(),
              "allocator_environment": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_tf32": torch.backends.cudnn.allow_tf32,
              "torch_threads": torch.get_num_threads()}
    sampler = ProcessTreeDeviceSampler(device=args.device, parent_pid=os.getpid(), interval=.5)
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            sampler.sample_if_due(force=True)
            stop.wait(.5)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    tick = time.perf_counter()
    try:
        worker = run_isolated_segmentation(
            source_path=args.source, output_path=args.output_dir / "wm.seg.mgz",
            report_path=args.output_dir / "worker.json", device=args.device,
            threads=args.threads, histogram_batch_size=2048, planar_batch_size=256,
            profile_stages=False, code_version=args.code_commit)
        wall = time.perf_counter() - tick
    finally:
        stop.set()
        monitor.join(timeout=10)
    sampler.sample_if_due(force=True)
    after = {"cuda_initialized": torch.cuda.is_initialized(),
             "allocator_environment": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
             "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
             "cudnn_tf32": torch.backends.cudnn.allow_tf32,
             "torch_threads": torch.get_num_threads()}
    reference, output = [nib.load(str(path)) for path in
                         (args.reference, args.output_dir / "wm.seg.mgz")]
    lhs, rhs = np.asarray(reference.dataobj), np.asarray(output.dataobj)
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        raise ValueError("reference/output grids or dtype differ")
    error = np.abs(lhs.astype(np.int16) - rhs.astype(np.int16))
    dice = {}
    for label in np.union1d(np.unique(lhs), np.unique(rhs)):
        first, second = lhs == label, rhs == label
        dice[str(int(label))] = 2 * int(np.count_nonzero(first & second)) / int(first.sum() + second.sum())
    report = {
        "scope": "single complete WM fresh-exec or initialized-parent API; not ABBA or raw T1 whole recon",
        "code_commit": args.code_commit, "code_commit_role": "baseline plus worker executed module SHA overlay",
        "script_sha256": sha(__file__), "host": platform.node(), "parent_pid": os.getpid(),
        "cpu_model": next((line.split(":", 1)[1].strip() for line in
            Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unavailable"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "parent_mode": args.parent_mode,
        "parent_state_before": before, "parent_state_after": after,
        "parent_policy_preserved": before == after,
        "parent_live_tensor_bytes": parent_tensor.numel() if parent_tensor is not None else 0,
        "full_exec_API_seconds": wall, "worker": worker,
        "timing_includes": "parent wrapper validation plus child exec/import/init/hash/input/transfer/calculation/output/exit; no synthetic stage omitted",
        "input_sha256": sha(args.source), "reference_sha256": sha(args.reference),
        "output_sha256": sha(args.output_dir / "wm.seg.mgz"),
        "reference_kind": "existing same-input cached FNIT WM, comparison only",
        "different_voxels": int(np.count_nonzero(error)), "max_abs": int(error.max()),
        "p99_abs": float(np.percentile(error, 99)), "label_dice": dice,
        "affine_equal_reference": bool(np.array_equal(output.affine, reference.affine)),
        "mgh_header_equal_reference": output.header.binaryblock == reference.header.binaryblock,
        "shape": list(rhs.shape), "dtype": str(rhs.dtype),
        "process_tree_device_memory": sampler.report(),
        "overall_metric_equivalence": "not_assessed", "whole_recon_speedup": "not_measured",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
        ("full_exec_API_seconds", "different_voxels", "parent_policy_preserved")}), flush=True)


if __name__ == "__main__":
    main()
