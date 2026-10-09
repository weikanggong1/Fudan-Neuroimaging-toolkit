"""从原始单T1连续生成conform、SynthStrip、Talairach与N4/nu。"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

from .input_talairach_chain import run_input_talairach_chain
from .n4_itk import correct_volume
from .n4_wrapper import make_nu


def run_input_n4_chain(t1: str | Path, subject_dir: str | Path,
                       weights_dir: str | Path, assets_dir: str | Path,
                       *, n4_binary: str | Path | None = None, device: str = "cpu",
                       threads: int = 4,
                       n4_backend: str = "native", profile: bool = False) -> dict:
    """原始3D T1→conformed网格uint8 orig/nu0/nu及RAS/mm变换。

    t1为三维NIfTI-1路径（既有导入器限制）；subject_dir须不存在或为空。weights_dir/assets_dir
    为已校验SynthStrip/Talairach权重及MNI305模板；全部输出在subject_dir。
    device默认cpu，控制神经推理及显式Torch N4；threads默认4，为前段CPU
    预算。native默认复用Conda ITK，n4_binary必须提供，拟合/重建均1线程；
    torch显式实验复用完整固定ITK5.4.7配方，不使用旧blur残差近似，不读取
    参考。profile默认False；True记录原生细分或同步Torch子段。
    返回前段原字段、nu0/nu路径、后端/实际算法、N4/包装器耗时、scale、
    bins及含输入校验/模型加载/搬运/IO的total_seconds。N4输出均为同网格
    uint8；不是旧近似的float32。GPU完整N4有已记录系统尾差，整体等效
    未判，故不改变生产native默认、TF32策略或启用半精度。参数、资源、
    上游、N4/包装器失败抛异常，可能留部分产物，不能当作完成输出。
    """
    wall_started = time.perf_counter()
    # Reject invalid backend/binary before loading models or creating outputs.
    if n4_backend not in {"native", "torch"}:
        raise ValueError("n4_backend must be 'native' or 'torch'")
    if n4_backend == "native" and n4_binary is None:
        raise ValueError("n4_binary is required when n4_backend='native'")
    result = run_input_talairach_chain(
        t1, subject_dir, weights_dir, assets_dir,
        device=device, threads=threads)
    mri = Path(subject_dir) / "mri"
    scratch = mri / "tmp"
    scratch.mkdir(exist_ok=True)
    nu0 = scratch / "nu0.mgz"
    nu = mri / "nu.mgz"
    started = time.perf_counter()
    if n4_backend == "native":
        profile_path = Path(subject_dir) / "scripts/n4.profile.json" if profile else None
        correct_volume(input_file=mri / "orig.mgz", output_file=nu0, binary=n4_binary,
                       reconstruction_threads=1, profile_path=profile_path)
        n4_details = {"backend": "native_conda_itk", "fitting_threads": 1,
                      "requested_reconstruction_threads": 1}
        if profile_path is not None:
            n4_details["profile"] = json.loads(profile_path.read_text())
    else:
        # The mature full feedback implementation replaces only this pipeline
        # binding. The separate n4_gpu approximation API is not this algorithm.
        from .n4_itk_torch_experimental import correct_volume as correct_volume_torch

        n4_details = correct_volume_torch(input_path=mri / "orig.mgz", output_path=nu0,
                                         device=device, profile=profile)
    n4_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scale, histogram_bins = make_nu(
        mri / "orig.mgz", nu0, result["talairach_xfm"], nu)
    wrapper_seconds = time.perf_counter() - started
    return {**result, "nu0": str(nu0), "nu": str(nu),
            "n4_backend": n4_backend,
            "n4_details": n4_details,
            "n4_seconds": n4_seconds, "n4_wrapper_seconds": wrapper_seconds,
            "n4_global_mean_scale": scale, "n4_histogram_bins": histogram_bins,
            "total_seconds": time.perf_counter() - wall_started,
            "production_default_changed": False}


def main() -> None:
    """显式单T1 CLI，默认原生；TF32仅设置本进程并保留前段FP32例外。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--n4-binary", type=Path)
    parser.add_argument("--n4-backend", choices=("native", "torch"), default="native")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.n4_backend == "native" and args.n4_binary is None:
        parser.error("--n4-binary is required for native backend")
    import torch
    from numba import set_num_threads
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    initialized = torch.cuda.is_initialized()
    report = run_input_n4_chain(t1=args.t1, subject_dir=args.subject_dir,
        weights_dir=args.weights_dir, assets_dir=args.assets_dir, n4_binary=args.n4_binary,
        n4_backend=args.n4_backend, device=args.device, threads=args.threads, profile=args.profile)
    report["cuda_initialized_before_api"] = initialized
    report["caller_matmul_tf32"] = torch.backends.cuda.matmul.allow_tf32
    report["caller_cudnn_tf32"] = torch.backends.cudnn.allow_tf32
    report["half_precision"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
