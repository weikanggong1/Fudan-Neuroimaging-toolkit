r"""冻结真实 sulc 配准首轮梯度，比较有序 CPU/GPU 平均；不是整例 benchmark。

输入 subject/surf/<hemi>.sphere、smoothwm、sulc 和声明的半球 TIFF atlas。
表面坐标为 surface RAS/mm；捕获的 gradient 为 (N,3) float32 配准力，
neighbors 为 (N,K) int64 有序邻接，degrees 为 (N,) int64 有效列数。
在首次平均入口停止，不写 sphere.reg，也不读官方输出作为生产输入。
实际首轮须为 16384 次；另测 0/1/64 次，数值绝对/相对容差预先固定为 0。
首次 CPU JIT、GPU 构造/首次 kernel 与暖运行分别报告；暖运行 AB/BA 两轮。
GPU 计时同步显式目标设备，包含每次 H2D/D2H、分配及全部平均轮次。
输出目录必须不存在：report.json 和 captured_average_input.pt 只存诊断目录，
不要提交捕获的真实数组。函数对应 mris_register 的内部 MRISaverageGradients，
没有独立官方 CLI；不据此判定整例提速或整体指标等效。

具名示例（每个变量在 shell 中明确赋值，按授权数据位置替换）：
    subject=/absolute/path/FNIT_subject  # FNIT 自产同序表面目录
    hemi=lh  # 本轮真实半球；另一半球另建新目录运行
    atlas=/absolute/path/lh.folding.atlas.tif  # 声明的固定半球图谱
    output=/absolute/path/diagnostics/lh_average_new  # 必须是新目录
    device=cuda:0  # CUDA_VISIBLE_DEVICES 映射后的明确逻辑 GPU
    threads=4  # Torch/Numba 预算及本进程库初始化线程环境
    python benchmark_register_gpu_average.py --subject "$subject" --hemi "$hemi" \
        --atlas "$atlas" --output "$output" --device "$device" --threads "$threads"
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "NUMBA_NUM_THREADS",
)


class _StopAfterCapture(Exception):
    """仅在首轮梯度已复制到 CPU 后退出；不能用来跳过实际平均回归。"""


def _sha256(path: Path) -> str:
    """分块读取 path 文件并返回 SHA-256；路径不可读时传播 I/O 异常。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_manifest(tensor) -> dict:
    """返回 tensor 的 shape/dtype/device 和连续 CPU 数据 SHA，不改变其数值。"""
    array = tensor.detach().cpu().contiguous().numpy()
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "device": str(tensor.device),
            "data_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def _source_manifest() -> dict:
    """绑定实际导入的 recon_all 源目录和脚本；Git 缺失时仅报告可核验源码 SHA。"""
    from fnit.recon_all import mris_register_sulc_run
    directory = Path(mris_register_sulc_run.__file__).resolve().parent
    repository = next((parent for parent in directory.parents
                       if (parent / ".git").exists()), None)
    git = {"head": None, "worktree_status": None, "repository": None}
    if repository is not None:
        git["repository"] = str(repository)
        for key, arguments in (("head", ["rev-parse", "HEAD"]),
                               ("worktree_status", ["status", "--short"])):
            result = subprocess.run(["git", "-C", str(repository), *arguments],
                                    capture_output=True, text=True, check=False)
            if result.returncode == 0:
                git[key] = result.stdout.strip()
    return {"git": git, "directory": str(directory),
            "recon_all_py_sha256": {str(path.relative_to(directory)): _sha256(path)
                                     for path in sorted(directory.rglob("*.py"))},
            "script_sha256": _sha256(Path(__file__).resolve())}


def _capture_first_average(*, inputs: dict[str, Path], output: Path) -> tuple[dict, dict]:
    """运行真实 sulc 前段并冻结首轮平均输入；无首轮或轮数改变时抛异常。

    inputs 含 sphere/smoothwm/sulc/atlas 路径；output 是新诊断目录。
    返回 CPU tensor 字典和含秒数、16384 轮及数据 SHA 的描述；finally
    恢复类方法，私有 Stop 以外的配准/影像异常原样传播，不生成替代表面。
    """
    from fnit.recon_all.mris_register_average_numba import RegistrationGradientAverager
    from fnit.recon_all.mris_register_sulc_run import run_register_sulc
    captured = {}
    original_call = RegistrationGradientAverager.__call__

    def capture(averager, gradient, iterations):
        """复制本次真实梯度与静态拓扑后停止，不执行或更改平均算法。"""
        captured.update(gradient=gradient.detach().cpu().contiguous().clone(),
                        neighbors=averager.neighbors.detach().cpu().contiguous().clone(),
                        degrees=averager.degrees.detach().cpu().contiguous().clone(),
                        iterations=int(iterations))
        raise _StopAfterCapture()

    started = time.perf_counter()
    RegistrationGradientAverager.__call__ = capture
    try:
        run_register_sulc(sphere=inputs["sphere"], smoothwm=inputs["smoothwm"],
                          sulc=inputs["sulc"], atlas_file=inputs["atlas"],
                          output=output / "unused_sulc_seed", max_updates=1,
                          averaging_device="cpu")
    except _StopAfterCapture:
        pass
    finally:
        RegistrationGradientAverager.__call__ = original_call
    if not captured or captured["iterations"] != 16384:
        raise RuntimeError("本脚本要求实际首轮平均为 16384 次；未捕获或调度已改变")
    import torch
    if not torch.isfinite(captured["gradient"]).all():
        raise ValueError("真实捕获梯度有非有限值，不能据此判定数值回归通过")
    description = {"seconds_including_real_input_io_and_capture": time.perf_counter() - started,
                   "actual_first_iterations": captured["iterations"],
                   "tensors": {key: _tensor_manifest(captured[key])
                               for key in ("gradient", "neighbors", "degrees")}}
    return captured, description


def _memory(*, device, stats_valid: bool) -> dict:
    """返回显式 GPU 的 allocated/reserved 及峰值字节；禁用缓存时写 None 而非零。

    stats_valid 为本次启动环境是否启用 PyTorch caching allocator；这里的
    统计只覆盖 PyTorch allocator，不等于 NVML 整进程或整例总显存。
    """
    import torch
    names = ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes")
    if not stats_valid:
        return {"status": "unavailable_when_caching_allocator_disabled",
                **dict.fromkeys(names, None)}
    return {"status": "available_pytorch_allocator_only",
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}


def _measure(*, implementation: str, function, device, stats_valid: bool) -> tuple[object, dict]:
    """同步目标 GPU 后调用无参数 function；返回 CPU 结果及完整算子墙钟/显存。

    计时从前同步结束开始，包含函数分配、CPU/GPU 运算、往返传输和后同步；
    拓扑一次构造、数组比较与报告写出另记。异常原样传播，不缓存近似结果。
    """
    import torch
    torch.cuda.synchronize(device)
    if stats_valid:
        torch.cuda.reset_peak_memory_stats(device)
    before = _memory(device=device, stats_valid=stats_valid)
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError("平均输出必须为 CPU float32，与真实配准下一步接口一致")
    row = {"implementation": implementation, "seconds_including_transfer_and_allocation": seconds,
           "memory_before": before, "memory_after": _memory(device=device, stats_valid=stats_valid),
           "output": _tensor_manifest(value)}
    return value, row


def _compare(*, reference, candidate) -> dict:
    """同序 float32 数组比较，返回不同元素/max/P99；容差固定零，非有限值失败。

    输入应同 shape/dtype；坐标与标签不参与此算子测试。另记原始位差以诊断
    正负零，数值验收由零绝对/相对容差决定，不将 signed-zero 当新增数值误差。
    """
    import numpy as np
    left, right = reference.numpy(), candidate.numpy()
    if left.shape != right.shape or left.dtype != right.dtype:
        return {"pass": False, "reason": "shape_or_dtype_mismatch"}
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    error = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"pass": finite and bool(np.array_equal(left, right)), "finite": finite,
            "different_elements": int(np.count_nonzero(left != right)),
            "bitwise_different_elements": int(np.count_nonzero(left.view(np.uint32) != right.view(np.uint32))),
            "max_absolute_error": float(error.max()) if finite and error.size else None,
            "p99_absolute_error": float(np.quantile(error, .99)) if finite and error.size else None}


def _runtime(*, device) -> dict:
    """记录主机、CPU、显式 GPU UUID、版本和精度状态；不读取其他环境或许可证。"""
    import torch
    import numba
    properties = torch.cuda.get_device_properties(device)
    cpu_model = platform.processor()
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    versions = {}
    for distribution in ("numpy", "torch", "numba", "triton", "nibabel", "tifffile", "fnit"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    uuid = getattr(properties, "uuid", None)
    return {"host": platform.node(), "platform": platform.platform(), "cpu_model": cpu_model,
            "cpu_logical_count": os.cpu_count(), "python": sys.version, "versions": versions,
            "torch_cuda_version": torch.version.cuda, "device": str(device),
            "gpu_name": properties.name, "gpu_uuid": str(uuid) if uuid is not None else None,
            "gpu_uuid_status": "torch_device_properties" if uuid is not None else "unavailable",
            "gpu_total_memory_bytes": int(properties.total_memory),
            "precision": {"gradient_dtype": "float32", "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                          "cudnn_tf32": torch.backends.cudnn.allow_tf32, "autocast_cuda": torch.is_autocast_enabled(),
                          "half_precision_enabled_by_script": False, "kernel_fp_fusion": False},
            "torch_intraop_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "numba_threads": numba.get_num_threads(), "numba_initial_capacity": numba.config.NUMBA_NUM_THREADS,
            "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
            "persistent_triton_cache_reuse": "possible_not_cleared_by_benchmark"}


def main(argv: list[str] | None = None) -> int:
    """解析具名 CLI，运行真实同输入回归；失败写报告并返回非零，旧目录不覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", type=Path, required=True, help="FNIT 自产 subject 目录")
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True, help="与图谱对应的半球")
    parser.add_argument("--atlas", type=Path, required=True, help="声明的半球 TIFF 配准图谱")
    parser.add_argument("--output", type=Path, required=True, help="不存在的新诊断目录")
    parser.add_argument("--device", default="cuda:0", help="明确逻辑 GPU，默认 cuda:0，不能只写 cuda")
    parser.add_argument("--threads", type=int, default=4, help="CPU 线程预算，默认 4")
    parser.add_argument("--code-version", help="可选声明版本；实际 Git HEAD 和源码 SHA 单独记录")
    args = parser.parse_args(argv)
    if args.threads < 1 or not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        parser.error("threads 必须为正整数，device 必须是明确 cuda:N")
    inputs = {name: (args.subject / "surf" / f"{args.hemi}.{name}").resolve()
              for name in ("sphere", "smoothwm", "sulc")}
    inputs["atlas"] = args.atlas.resolve()
    for path in inputs.values():
        if not path.is_file():
            parser.error(f"真实输入不存在或不可读：{path}")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    whole_started = time.perf_counter()
    before_environment = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    # 此脚本是独立进程，在 NumPy/Torch/Numba 导入前设置库初始化预算。
    for name in THREAD_VARIABLES:
        os.environ[name] = str(args.threads)
    report = {"status": "running", "scope": "frozen_real_first_sulc_gradient_operator_only",
              "started_utc": datetime.now(timezone.utc).isoformat(), "declared_code_version": args.code_version,
              "thresholds_predeclared": {"absolute_tolerance": 0.0, "relative_tolerance": 0.0,
                                          "require_finite": True, "bitwise_difference_is_diagnostic": True},
              "input_paths": {key: str(path) for key, path in inputs.items()},
              "thread_environment_before": before_environment,
              "whole_pipeline_speedup": None, "whole_metric_equivalence": "not_assessed",
              "cold": {}, "warm_pairs": []}
    exit_code = 1
    try:
        report["input_sha256"] = {key: _sha256(path) for key, path in inputs.items()}
        import torch
        from fnit.recon_all.thread_budget import thread_budget
        from fnit.recon_all.mris_register_average_numba import (
            RegistrationGradientAverager, average_gradients_exact_cpu,
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        report["source_before"] = _source_manifest()
        device = torch.device(args.device)
        stats_valid = "PYTORCH_NO_CUDA_MEMORY_CACHING" not in os.environ
        report["allocator"] = {"disabled_by_environment_presence": not stats_valid,
                               "environment_value": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
                               "pytorch_statistics_known_valid": stats_valid,
                               "policy_changed_by_script": False}
        with thread_budget(threads=args.threads) as thread_record:
            report["thread_budget"] = thread_record
            captured, report["capture"] = _capture_first_average(inputs=inputs, output=args.output)
            checkpoint = args.output / "captured_average_input.pt"
            torch.save(captured, checkpoint)
            report["capture"].update(checkpoint_path=str(checkpoint), checkpoint_sha256=_sha256(checkpoint))
            init_started = time.perf_counter()
            torch.cuda.init()
            torch.cuda.synchronize(device)
            report["gpu_context_setup_seconds_excluded_from_operator_runs"] = time.perf_counter() - init_started
            report["runtime"] = _runtime(device=device)
            gradient, neighbors, degrees = (captured[key] for key in ("gradient", "neighbors", "degrees"))
            iterations = captured["iterations"]
            cpu = lambda count: average_gradients_exact_cpu(gradient=gradient, neighbors=neighbors,
                                                          degrees=degrees, iterations=count)
            cpu_first, report["cold"]["cpu_first_call"] = _measure(
                implementation="cpu_numba_source_order", function=lambda: cpu(iterations),
                device=device, stats_valid=stats_valid)
            torch.cuda.synchronize(device)
            construct_started = time.perf_counter()
            averager = RegistrationGradientAverager(neighbors=neighbors, degrees=degrees, device=str(device))
            torch.cuda.synchronize(device)
            constructor_seconds = time.perf_counter() - construct_started
            gpu_first, report["cold"]["gpu_first_call"] = _measure(
                implementation="gpu_triton_source_order", function=lambda: averager(gradient=gradient, iterations=iterations),
                device=device, stats_valid=stats_valid)
            report["cold"].update(iterations=iterations, gpu_constructor_seconds_including_static_transfer=constructor_seconds,
                                  gpu_constructor_plus_first_call_seconds=constructor_seconds + report["cold"]["gpu_first_call"]["seconds_including_transfer_and_allocation"],
                                  comparison=_compare(reference=cpu_first, candidate=gpu_first),
                                  scope="本进程首次调用；Numba specialization/Triton 编译或磁盘缓存加载均计入；CUDA context 另记")
            references = {iterations: cpu_first}
            all_pass = report["cold"]["comparison"]["pass"]
            for count in (iterations, 0, 1, 64):
                if count not in references:
                    references[count] = cpu(count)
                for repeat in (1, 2):
                    for order in ("AB", "BA"):
                        pair = {"iterations": count, "repeat": repeat, "order": order,
                                "A": "cpu_numba_source_order", "B": "gpu_triton_source_order", "runs": {}}
                        outputs = {}
                        for letter in order:
                            function = (lambda count=count: cpu(count)) if letter == "A" else (
                                lambda count=count: averager(gradient=gradient, iterations=count))
                            outputs[letter], pair["runs"][letter] = _measure(
                                implementation=pair[letter], function=function, device=device, stats_valid=stats_valid)
                            pair["runs"][letter]["comparison_to_frozen_cpu"] = _compare(
                                reference=references[count], candidate=outputs[letter])
                            all_pass &= pair["runs"][letter]["comparison_to_frozen_cpu"]["pass"]
                        pair["comparison_AB"] = _compare(reference=outputs["A"], candidate=outputs["B"])
                        all_pass &= pair["comparison_AB"]["pass"]
                        pair["cpu_seconds_over_gpu_seconds"] = (
                            pair["runs"]["A"]["seconds_including_transfer_and_allocation"] /
                            pair["runs"]["B"]["seconds_including_transfer_and_allocation"])
                        report["warm_pairs"].append(pair)
                        print(json.dumps({"iterations": count, "repeat": repeat, "order": order,
                                          "pass": pair["comparison_AB"]["pass"]}), flush=True)
            report["warm_summary"] = {}
            for count in references:
                pairs = [pair for pair in report["warm_pairs"] if pair["iterations"] == count]
                times = {letter: [pair["runs"][letter]["seconds_including_transfer_and_allocation"]
                                  for pair in pairs] for letter in "AB"}
                report["warm_summary"][str(count)] = {
                    "calls_per_implementation": len(pairs),
                    "cpu_median_seconds": statistics.median(times["A"]),
                    "gpu_median_seconds_including_transfer": statistics.median(times["B"]),
                    "ratio_of_medians_cpu_over_gpu": statistics.median(times["A"]) / statistics.median(times["B"]),
                    "all_numeric_comparisons_pass": all(pair["comparison_AB"]["pass"] for pair in pairs)}
            unchanged = all(_tensor_manifest(captured[key]) == report["capture"]["tensors"][key]
                            for key in ("gradient", "neighbors", "degrees"))
            report["captured_tensors_unchanged"] = unchanged
            all_pass &= unchanged
        report["source_after"] = _source_manifest()
        report["source_content_unchanged"] = (
            report["source_before"]["recon_all_py_sha256"] == report["source_after"]["recon_all_py_sha256"]
            and report["source_before"]["script_sha256"] == report["source_after"]["script_sha256"])
        all_pass &= report["source_content_unchanged"]
        report["input_sha256_after"] = {key: _sha256(path) for key, path in inputs.items()}
        report["input_files_unchanged"] = report["input_sha256"] == report["input_sha256_after"]
        all_pass &= report["input_files_unchanged"]
        report.update(status="complete", strict_numeric_reproduction_pass=bool(all_pass))
        exit_code = 0 if all_pass else 2
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    except KeyboardInterrupt:
        report.update(status="interrupted", error={"type": "KeyboardInterrupt", "message": "用户中断"})
        exit_code = 130
    finally:
        report["script_seconds_including_capture_context_and_report_preparation"] = time.perf_counter() - whole_started
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        report["exit_code"] = exit_code
        report["timing_scope"] = ("算子墙钟包括分配、全部轮次、H2D/D2H 和目标 GPU 后同步；"
                                   "静态拓扑构造单列；首轮真实配准前段、CUDA context、比较和诊断写出不属于算子提速。"
                                   "script_seconds 截止最终 report.json 写出前，不等于外部完整命令或 recon-all 墙钟。")
        (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "exit_code": exit_code,
                      "report": str(args.output / "report.json")}, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
