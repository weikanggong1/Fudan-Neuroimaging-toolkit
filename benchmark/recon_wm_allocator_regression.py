"""完整缓存WM文件API的分配器回归：只读已验证输出作为对照，不修补候选。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

import nibabel as nib
import numpy as np
import torch


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    """三维uint8源图→完整WM与JSON；网格毫米affine不变，失败传播。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="FNIT自产三维uint8 WM强度图，XYZ体素网格")
    parser.add_argument("--reference", type=Path, required=True,
                        help="仅比较的同输入缓存开启WM输出，不作为候选输入")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="新目录，保存完整wm.seg.mgz与report.json")
    parser.add_argument("--module-dir", type=Path, required=True,
                        help="已冻结的独立recon模块目录；其余包保持原版本")
    parser.add_argument("--device", default="cuda:1", help="明确目标CUDA设备")
    parser.add_argument("--threads", type=int, default=4, help="总Torch CPU线程预算")
    parser.add_argument("--code-commit", required=True, help="冻结源码基线commit")
    args = parser.parse_args()
    if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") != "1":
        raise ValueError("set PYTORCH_NO_CUDA_MEMORY_CACHING=1 before Python startup")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None or args.threads < 1:
        raise ValueError("require explicit CUDA index and positive thread count")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.module_dir.resolve()))
    from fnit.recon_all import mri_segment as wm
    from fnit.recon_all import mri_segment_histogram_torch as histogram
    from fnit.recon_all import mri_segment_planar_torch as planar
    rows, errors = [], []
    stop = threading.Event()
    sampled_start = time.perf_counter()

    def sample():
        while not stop.is_set():
            try:
                free, total = torch.cuda.mem_get_info(device)
                rows.append({"t_seconds": time.perf_counter() - sampled_start,
                             "used_bytes": int(total - free), "total_bytes": int(total)})
            except RuntimeError as error:
                errors.append(str(error))
            stop.wait(.25)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    torch.cuda.reset_peak_memory_stats(device)
    output_file = args.output_dir / "wm.seg.mgz"
    try:
        started = time.perf_counter()
        api = wm.segment_white_matter_mgz(
            source_path=args.source, output_path=output_file, device=device,
            histogram_backend="torch", histogram_batch_size=2048,
            planar_backend="cached", planar_batch_size=256)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
    finally:
        stop.set()
        worker.join(timeout=5)
    source, reference, output = [nib.load(str(path)) for path in
                                 (args.source, args.reference, output_file)]
    lhs, rhs = np.asarray(reference.dataobj), np.asarray(output.dataobj)
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        raise ValueError("reference/output grid or dtype mismatch")
    difference = np.abs(lhs.astype(np.int16) - rhs.astype(np.int16))
    dice = {}
    for label in np.union1d(np.unique(lhs), np.unique(rhs)):
        first, second = lhs == label, rhs == label
        dice[str(int(label))] = 2 * int(np.count_nonzero(first & second)) / int(first.sum() + second.sum())
    gpu_uuid = subprocess.check_output(["nvidia-smi", f"--id={device.index}",
        "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    report = {
        "scope": "single complete same-input cached WM API allocator regression; not ABBA or whole recon",
        "code_commit": args.code_commit, "code_commit_role": "baseline plus executed module SHA",
        "modules_sha256": {Path(module.__file__).name: sha(module.__file__)
                           for module in (wm, histogram, planar)},
        "script_sha256": sha(__file__), "input_sha256": sha(args.source),
        "reference_kind": "same-input validated cache-enabled FNIT cached WM; comparison only",
        "reference_sha256": sha(args.reference), "output_sha256": sha(output_file),
        "host": platform.node(), "cpu_model": next((line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")), "unavailable"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": args.threads,
        "actual_torch_threads": torch.get_num_threads(),
        "thread_environment": {name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "allocator_environment": {name: os.environ.get(name)
            for name in ("PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")},
        "device": str(device), "gpu_uuid": gpu_uuid, "gpu": torch.cuda.get_device_name(device),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "nibabel": nib.__version__, "torch_cuda": torch.version.cuda,
        "precision": {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_tf32": torch.backends.cudnn.allow_tf32, "no_half": True},
        "full_API_seconds": seconds, "api": api,
        "timing_includes": "first file API including parameter checks, loading, H2D/D2H, ordered stages, JIT/cache loading and compressed write; target CUDA sync",
        "timing_excludes": "interpreter/import/CUDA initialization; not cold whole recon",
        "shape": list(rhs.shape), "dtype": str(rhs.dtype),
        "affine_equal_source": bool(np.array_equal(output.affine, source.affine)),
        "affine_equal_reference": bool(np.array_equal(output.affine, reference.affine)),
        "mgh_header_equal_reference": output.header.binaryblock == reference.header.binaryblock,
        "different_voxels": int(np.count_nonzero(difference)),
        "max_abs": int(difference.max()), "p99_abs": float(np.percentile(difference, 99)),
        "label_dice": dice,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "torch_counter_scope": "allocator cache disabled; zero counters do not represent process GPU memory",
        "process_tree_GPU_memory": "not_measured",
        "target_card_memory_upper_bound": {
            "scope": "explicit target CUDA total-free; includes all processes and driver",
            "requested_interval_seconds": .25,
            "max_interval_seconds": max((b["t_seconds"] - a["t_seconds"]
                for a, b in zip(rows, rows[1:])), default=None),
            "peak_card_used_bytes": max((row["used_bytes"] for row in rows), default=None),
            "samples": rows, "errors": errors},
        "whole_metric_equivalence": "not_assessed", "whole_recon_speedup": "not_measured",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("full_API_seconds", "different_voxels",
        "torch_peak_allocated_bytes", "torch_peak_reserved_bytes")}), flush=True)


if __name__ == "__main__":
    main()
