"""从原始单T1连续生成conform、SynthStrip、Talairach与N4/nu。"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

from .input_talairach_chain import run_input_talairach_chain
from .n4_itk import correct_volume
from .n4_wrapper import make_nu


def validate_n4_execution(*, n4_backend: str, n4_execution: str, device: str) -> None:
    """创建输出前检查N4路由，不加载影像、模型或初始化CUDA。

    n4_backend为native/torch，n4_execution为in-process/isolated，device
    沿用调用者逻辑设备。isolated只允许完整Torch及显式cuda:N；其余
    非法组合抛ValueError。无输出/坐标单位，属于调度内部参数检查。
    """
    if n4_backend not in {"native", "torch"}:
        raise ValueError("n4_backend must be 'native' or 'torch'")
    if n4_execution not in {"in-process", "isolated"}:
        raise ValueError("n4_execution must be in-process or isolated")
    if n4_execution == "isolated":
        import torch
        try:
            target = torch.device(device)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError("isolated N4 requires explicit CUDA device cuda:N") from error
        if n4_backend != "torch" or target.type != "cuda" or target.index is None:
            raise ValueError("isolated N4 requires n4_backend='torch' and explicit CUDA device cuda:N")


def run_n4_stage(*, input_path: str | Path, output_path: str | Path,
                 n4_backend: str = "native", n4_execution: str = "in-process",
                 native_binary: str | Path | None = None, device: str = "cpu",
                 threads: int = 4, profile: bool = False,
                 report_path: str | Path | None = None) -> dict:
    """在自产orig网格运行既有完整N4，供输入链/整例共用。

    input_path/output_path为三维影像路径，输出uint8、同affine/mm。
    native默认独立Conda ITK，native_binary必填，拟合/重建各1线程。
    torch默认in-process沿用父allocator；isolated复用新exec缓存worker，
    必须cuda:N及新report_path。threads默认4；profile默认False同步
    Torch细分，native在report_path非None时写原生profile。返回实际
    N4 report，isolated完整200轮详情在api，SHA/precision/allocated/
    reserved及父策略完整保留。失败传播，不复制参考或静默回退。
    """
    validate_n4_execution(n4_backend=n4_backend, n4_execution=n4_execution, device=device)
    if n4_backend == "native":
        if native_binary is None:
            raise ValueError("native_binary is required for native N4")
        correct_volume(input_file=input_path, output_file=output_path, binary=native_binary,
                       reconstruction_threads=1, profile_path=report_path)
        details = {"backend": "native_conda_itk", "execution": n4_execution,
                   "fitting_threads": 1, "requested_reconstruction_threads": 1}
        if report_path is not None:
            profile_report = json.loads(Path(report_path).read_text())
            details = {**profile_report, **details, "profile": profile_report}
        return details
    if n4_execution == "isolated":
        if report_path is None:
            raise ValueError("report_path is required for isolated N4")
        from .n4_torch_worker import run_isolated_n4
        return run_isolated_n4(input_path=input_path, output_path=output_path,
                               report_path=report_path, device=device, threads=threads,
                               profile=profile)
    from .n4_itk_torch_experimental import correct_volume as correct_volume_torch
    return correct_volume_torch(input_path=input_path, output_path=output_path,
                                device=device, profile=profile)


def run_input_n4_chain(t1: str | Path, subject_dir: str | Path,
                       weights_dir: str | Path, assets_dir: str | Path,
                       *, n4_binary: str | Path | None = None, device: str = "cpu",
                       threads: int = 4,
                       n4_backend: str = "native", n4_execution: str = "in-process",
                       profile: bool = False) -> dict:
    """原始3D T1→conformed网格uint8 orig/nu0/nu及RAS/mm变换。

    t1为三维NIfTI-1路径（既有导入器限制）；subject_dir须不存在或为空。weights_dir/assets_dir
    为已校验SynthStrip/Talairach权重及MNI305模板；全部输出在subject_dir。
    device默认cpu，控制神经推理及显式Torch N4；threads默认4，为前段CPU
    预算。native默认复用Conda ITK，n4_binary必须提供，拟合/重建均1线程；
    torch显式实验复用完整固定ITK5.4.7配方，不使用旧blur残差近似，不读取
    参考。n4_execution默认in-process保留父allocator；isolated只允许
    Torch/cuda:N，在新exec中局部缓存并保留完整阶段收据/父CUDA状态。
    profile默认False；True记录原生细分或同步Torch子段。
    返回前段原字段、nu0/nu路径、后端/实际算法、N4/包装器耗时、scale、
    bins及含输入校验/模型加载/搬运/IO的total_seconds。N4输出均为同网格
    uint8；不是旧近似的float32。GPU完整N4有已记录系统尾差，整体等效
    未判，故不改变生产native默认、TF32策略或启用半精度。参数、资源、
    上游、N4/包装器失败抛异常，可能留部分产物，不能当作完成输出。
    """
    wall_started = time.perf_counter()
    # Reject invalid backend/binary before loading models or creating outputs.
    validate_n4_execution(n4_backend=n4_backend, n4_execution=n4_execution, device=device)
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
    report_path = (Path(subject_dir) / "scripts/n4-isolated.json" if n4_execution == "isolated"
                   else Path(subject_dir) / "scripts/n4.profile.json" if profile and n4_backend == "native"
                   else None)
    n4_details = run_n4_stage(input_path=mri / "orig.mgz", output_path=nu0,
                             n4_backend=n4_backend, n4_execution=n4_execution,
                             native_binary=n4_binary, device=device, threads=threads,
                             profile=profile, report_path=report_path)
    n4_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scale, histogram_bins = make_nu(
        mri / "orig.mgz", nu0, result["talairach_xfm"], nu)
    wrapper_seconds = time.perf_counter() - started
    return {**result, "nu0": str(nu0), "nu": str(nu),
            "n4_backend": n4_backend,
            "n4_execution": n4_execution,
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
    parser.add_argument("--n4-execution", choices=("in-process", "isolated"), default="in-process")
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
        n4_backend=args.n4_backend, n4_execution=args.n4_execution,
        device=args.device, threads=args.threads, profile=args.profile)
    report["cuda_initialized_before_api"] = initialized
    report["caller_matmul_tf32"] = torch.backends.cuda.matmul.allow_tf32
    report["caller_cudnn_tf32"] = torch.backends.cudnn.allow_tf32
    report["half_precision"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
