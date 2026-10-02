"""完整真实 BOLD 的 MCFLIRT spline float32 位级采样验证。

输入为四维 BOLD、同网格三维 FEAT reference 及完整 float64 逐帧矩阵。
矩阵可为 N×4×4 .npy，也可为按帧号排列的 MAT_#### 目录。
逐帧比较原 sample_motion_frame 和 CUDA graph 的未截断 float32 输出；
只保留当前帧 CPU 临时数组，不保存影像或逐帧结果。
JSON 仅包含匿名汇总、文件/源码哈希及验证壁钟，不作为速度 benchmark。

例：
python validation/mcflirt/benchmark_sampling_exact.py \\
  --bold "$RAW_BOLD" --reference "$FROZEN_FEAT_REFERENCE" \\
  --matrices "$FROZEN_FLOAT64_MOTION_MATRICES" \\
  --output "$ANONYMOUS_SAMPLING_REPORT" --device cuda:0 --threads 8 \\
  --source-revision "$TESTED_SOURCE_REVISION"
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import time

import nibabel as nib
import numpy as np
import torch


def file_identity(path):
    """只记录大小与 SHA-256，不记录文件名或服务器路径。"""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def source_identities():
    identities = {
        name: file_identity(importlib.import_module(name).__file__)
        for name in ("fnit.mcflirt.sampling", "fnit.mcflirt._sampling_cuda",
                     "fnit.flirt.core")
    }
    identities["benchmark_driver"] = file_identity(__file__)
    return identities


def matrix_identity(path, frame_count):
    """读取完整矩阵；目录文件字节按帧号串接计算 SHA-256。"""
    if path.is_dir():
        files = sorted(path.glob("MAT_[0-9][0-9][0-9][0-9]"))
        if len(files) != frame_count:
            raise ValueError("matrix directory must contain every input frame")
        files = [path / f"MAT_{frame:04d}" for frame in range(frame_count)]
        digest = hashlib.sha256()
        size_bytes = 0
        matrices = []
        for file in files:
            content = file.read_bytes()
            digest.update(content)
            size_bytes += len(content)
            matrices.append(np.loadtxt(file, dtype=np.float64))
        matrices = np.stack(matrices)
        identity = {"kind": "MAT_directory", "file_count": frame_count,
                    "size_bytes": size_bytes, "sha256": digest.hexdigest()}
    else:
        matrices = np.load(path, allow_pickle=False)
        if matrices.dtype != np.dtype(np.float64):
            raise ValueError("matrix .npy must retain float64 values")
        identity = {"kind": "float64_npy", **file_identity(path)}
    if matrices.shape != (frame_count, 4, 4) or not np.isfinite(matrices).all():
        raise ValueError("matrices must contain one finite 4x4 affine per frame")
    if not np.allclose(matrices[:, 3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError("matrices must be homogeneous affine transforms")
    if np.any(np.abs(np.linalg.det(matrices[:, :3, :3])) < 1e-10):
        raise ValueError("matrices must be invertible")
    identity["decoded_float64_sha256"] = hashlib.sha256(
        np.ascontiguousarray(matrices).tobytes()
    ).hexdigest()
    return matrices, identity


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bold", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--matrices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--source-revision", required=True)
    return parser


def main(argv=None):
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", args.source_revision):
        parser.error("source-revision must contain 7-40 hexadecimal characters")
    if args.output.exists():
        parser.error("choose a fresh anonymous output report")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA is required for the graph comparison")

    from fnit.mcflirt._sampling_cuda import CudaMotionFrameSampler
    from fnit.mcflirt.sampling import sample_motion_frame

    torch.set_num_threads(args.threads)
    torch.cuda.set_device(device)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    source_before = source_identities()
    input_before = {"bold": file_identity(args.bold),
                    "reference": file_identity(args.reference)}
    image = nib.load(str(args.bold))
    reference = nib.load(str(args.reference))
    if image.ndim != 4 or image.shape[3] < 1 or reference.ndim != 3:
        parser.error("BOLD must be 4D and reference must be 3D")
    if image.shape[:3] != reference.shape or not np.allclose(
        image.affine, reference.affine, atol=1e-4, rtol=0
    ):
        parser.error("BOLD and reference must have the same voxel grid")
    frame_count = image.shape[3]
    matrices, matrices_before = matrix_identity(args.matrices, frame_count)
    input_before["matrices"] = matrices_before
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(data).all():
        parser.error("BOLD must contain finite values")
    if min(image.shape[:3]) < 2:
        parser.error("spatial axes must contain at least two voxels")

    sampler = CudaMotionFrameSampler(
        image, reference, device=device, interpolation="spline"
    )
    expected_digest = hashlib.sha256()
    actual_digest = hashlib.sha256()
    total_samples = 0
    unequal_bits = 0
    unequal_frames = 0
    sum_squared_error = 0.0
    max_abs_error = 0.0
    finite_outputs = True
    for frame in range(frame_count):
        expected = sample_motion_frame(
            data[..., frame], image, reference, matrices[frame],
            device=device, interpolation="spline",
        ).cpu().numpy()
        actual = sampler.sample_numpy(data[..., frame], matrices[frame])
        expected = np.ascontiguousarray(expected, dtype=np.float32)
        actual = np.ascontiguousarray(actual, dtype=np.float32)
        if actual.shape != reference.shape or expected.shape != reference.shape:
            raise RuntimeError("sampling output grid differs from reference")
        expected_digest.update(expected.tobytes())
        actual_digest.update(actual.tobytes())
        different = int(np.count_nonzero(
            expected.view(np.uint32) != actual.view(np.uint32)
        ))
        unequal_bits += different
        unequal_frames += int(different > 0)
        total_samples += actual.size
        finite_outputs &= bool(np.isfinite(expected).all() and np.isfinite(actual).all())
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        sum_squared_error += float(np.square(difference).sum(dtype=np.float64))
        max_abs_error = max(max_abs_error, float(np.max(np.abs(difference))))
        if (frame + 1) % 50 == 0 or frame + 1 == frame_count:
            print("frames_compared", frame + 1, flush=True)
    torch.cuda.synchronize(device)
    source_after = source_identities()
    input_after = {"bold": file_identity(args.bold),
                   "reference": file_identity(args.reference)}
    _, matrices_after = matrix_identity(args.matrices, frame_count)
    input_after["matrices"] = matrices_after
    source_unchanged = source_before == source_after
    inputs_unchanged = input_before == input_after
    report = {
        "scope": "完整真实 BOLD；仅比较原 eager 与 CUDA graph spline 的未截断 float32 采样",
        "source_revision": args.source_revision,
        "source_before": source_before,
        "source_after": source_after,
        "source_unchanged": source_unchanged,
        "input_before": input_before,
        "input_after": input_after,
        "inputs_unchanged": inputs_unchanged,
        "frames": frame_count,
        "reference_shape": list(reference.shape),
        "total_float32_samples": total_samples,
        "unequal_float32_bit_patterns": unequal_bits,
        "bit_equal_fraction": 1.0 - unequal_bits / total_samples,
        "frames_with_unequal_bits": unequal_frames,
        "rmse": float(np.sqrt(sum_squared_error / total_samples)),
        "max_abs_error": max_abs_error,
        "finite_outputs": finite_outputs,
        "decoded_stream": {
            "layout": "按帧号串接；每帧 XYZ C-order float32 原始字节",
            "byte_order": sys.byteorder,
            "eager_sha256": expected_digest.hexdigest(),
            "cuda_graph_sha256": actual_digest.hexdigest(),
        },
        "validation_wall_seconds": time.perf_counter() - started,
        "timing_scope": "包含影像读取、双份采样、逐位比较及哈希；不作为采样速度 benchmark",
        "device": str(device),
        "threads": args.threads,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "nibabel": nib.__version__,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "accepted": bool(unequal_bits == 0 and finite_outputs
                         and source_unchanged and inputs_unchanged),
        "privacy": "不保存或公开影像、全数组、逐帧数值和临床路径；公开匿名汇总与哈希",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
