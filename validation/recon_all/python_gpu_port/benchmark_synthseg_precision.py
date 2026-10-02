"""真实 conform T1 上诊断 SynthSeg 实际精度和 allocator；不等同整例。

输入 --input/--weights/--output 为 FNIT orig、固定权重目录和新诊断目录；
--cudnn-tf32 true/false 为实际卷积策略，--allocator disabled/enabled
在 CUDA 初始化前设置，--initialized-api 保留一个 float32 元素再调用模型。
--baseline-seg 可提供同输入分割，仅作数据/dtype/几何和逐标签 Dice 诊断。
输出分割、体积 CSV、actual-forward.json；墙钟包含模型加载、传输、推理、
后处理和这两份数据写出；GPU 在计时边界同步。精度修复可能改变标签，
不得把本试验声称为整例严格复现。保留 TF32 matmul，不启用 autocast。
对应 mri_synthseg --i orig.mgz --o segmentation.mgz --vol volumes.csv。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import nibabel as nib
import numpy as np
import torch

from fnit.recon_all.profiling import configure_cuda_allocator, StageProfiler
from fnit.synthseg_parc import SynthSeg


def digest(path):
    """逐块 SHA-256；仅处理本次实际读取的影像、声明权重和源码。"""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-seg", type=Path)
    parser.add_argument("--cudnn-tf32", choices=("true", "false"), required=True)
    parser.add_argument("--allocator", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--initialized-api", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--code-version", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    allocator = configure_cuda_allocator(device=args.device, policy=args.allocator)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    retained = torch.ones(1, device=args.device) if args.initialized_api else None
    if retained is not None:
        torch.cuda.synchronize(args.device)
    profiler = StageProfiler(device=args.device, synchronize=True, allocator=allocator)

    def run():
        result = SynthSeg(weights=args.weights, device=args.device, threads=args.threads,
                          cudnn_tf32=args.cudnn_tf32 == "true")(args.input, keep_geometry=True)
        result.segmentation.save(str(args.output / "segmentation.mgz"))
        result.write_volumes_csv(source=args.input, path=args.output / "volumes.csv")
        return result

    started = time.perf_counter()
    result = profiler.run("SynthSeg_with_data_writes", run)
    report = {"scope": "frozen_same_input_precision_and_allocator_diagnostic",
              "code_version": args.code_version, "host": platform.node(),
              "device": args.device, "device_uuid": str(torch.cuda.get_device_properties(args.device).uuid),
              "threads": args.threads, "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "allocator": allocator, "initialized_api": args.initialized_api,
              "wall_seconds_including_data_io": time.perf_counter() - started,
              "stage": profiler.last_row, "actual_forward": result.precision,
              "input_sha256": digest(args.input),
              "weights_sha256": {p.name: digest(p) for p in args.weights.glob("synthseg*") if p.is_file()},
              "source_sha256": {}, "torch_version": torch.__version__,
              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "total_intracranial_mm3": result.total_intracranial_mm3}
    import fnit._dmri, fnit.synthseg_parc.segment, fnit.synthseg_parc.synthseg
    for module in (fnit._dmri, fnit.synthseg_parc.segment, fnit.synthseg_parc.synthseg):
        report["source_sha256"][module.__name__] = digest(module.__file__)
    if args.baseline_seg:
        reference = nib.load(str(args.baseline_seg))
        # 比较落盘数据；内存 Volume 的 dtype 与 MGH 的存储 dtype 可不同。
        candidate = nib.load(str(args.output / "segmentation.mgz"))
        left = np.asanyarray(reference.dataobj)
        right = np.asanyarray(candidate.dataobj)
        comparison = {"reference_sha256": digest(args.baseline_seg),
                      "same_shape": left.shape == right.shape,
                      "same_dtype": left.dtype == right.dtype,
                      "affine_max_absolute_mm": float(np.max(np.abs(reference.affine - candidate.affine)))}
        if left.shape == right.shape:
            comparison["different_voxels"] = int(np.count_nonzero(left != right))
            comparison["dice_by_label"] = {str(int(label)): float(
                2 * np.count_nonzero((left == label) & (right == label)) /
                (np.count_nonzero(left == label) + np.count_nonzero(right == label)))
                for label in np.union1d(left, right)}
        report["baseline_comparison"] = comparison
    (args.output / "actual-forward.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"seconds": report["wall_seconds_including_data_io"],
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
