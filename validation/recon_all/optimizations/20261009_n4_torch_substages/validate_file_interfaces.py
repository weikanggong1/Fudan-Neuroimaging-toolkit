"""两例真实 MGZ 的已初始化 CUDA API 和冷 CLI 回归；不评价整例等效。

--pair-directory 必须已有完成的两例 raw N4 配对。文件 API 和 CLI 不读取
参考；候选写出以后才与同版张量输出比较，另列与当前 native 的差异。
--output 拒绝覆盖；该脚本不安装依赖，不修改生产默认。
"""
import argparse
import hashlib
import importlib.util
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

from fnit.recon_all import n4_bspline_torch, n4_itk_torch_experimental


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantize(array):
    return np.floor(np.clip(array, 0, 255) + .5).astype(np.uint8)


def monitor_family(sampler, stop):
    """复用 FNIT 进程树采样器；所有 PID 未决项保留为 null。"""
    while not stop.is_set():
        sampler.sample_if_due(force=True)
        stop.wait(.1)


def comparison(output_path, original, tensor_expected, native_expected):
    saved = nib.load(str(output_path))
    actual = np.asarray(saved.dataobj)
    difference = actual.astype(np.int16) - native_expected.astype(np.int16)
    return {
        "shape": list(actual.shape), "dtype": str(actual.dtype),
        "same_shape": actual.shape == original.shape,
        "same_affine": bool(np.array_equal(saved.affine, original.affine)),
        "same_zooms": list(saved.header.get_zooms()) == list(original.header.get_zooms()),
        "same_image_class": type(saved) is type(original),
        "same_tensor_uint8": bool(np.array_equal(actual, tensor_expected)),
        "tensor_uint8_different": int(np.count_nonzero(actual != tensor_expected)),
        "native_uint8_different": int(np.count_nonzero(difference)),
        "native_uint8_max_abs": int(np.abs(difference).max()),
        "native_signed_uint8_counts": {str(int(value)): int(count)
            for value, count in zip(*np.unique(difference, return_counts=True))},
        "output_sha256": sha(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="冻结公开输入 manifest 所在目录")
    parser.add_argument("--pair-directory", type=Path, required=True, help="完整 raw N4 配对输出")
    parser.add_argument("--cli-source", type=Path, required=True, help="已冻结实验 CLI 源文件")
    parser.add_argument("--profiling-source", type=Path, required=True, help="冻结 FNIT profiling.py；复用进程树与 namespace 采样器")
    parser.add_argument("--output", type=Path, required=True, help="新目录，拒绝覆盖")
    parser.add_argument("--device", default="cuda:0", help="显式目标 GPU")
    parser.add_argument("--threads", type=int, default=1, help="固定 Torch/BLAS 线程预算")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paired = json.loads((args.pair_directory / "summary.json").read_text())
    if paired["status"] != "complete_stage_pair":
        raise ValueError("complete raw stage pair must finish before file-interface comparison")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this regression explicitly covers initialized CUDA API and GPU CLI")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    args.output.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("fnit_n4_frozen_profiling", args.profiling_source)
    profiling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiling)
    files = (Path(__file__), args.cli_source, args.profiling_source, Path(n4_bspline_torch.__file__),
             Path(n4_itk_torch_experimental.__file__))
    report = {"kind": "real_mgz_initialized_cuda_api_and_cold_cli_not_recon_all", "host": platform.node(),
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "cuda": torch.version.cuda, "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "cuda_initialized_before_api": torch.cuda.is_initialized(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "half_precision": False, "source_sha256": {str(path): sha(path) for path in files},
        "cuda_allocator_disable_cache_env": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
        "cuda_allocator_conf_env": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "data_manifest_sha256": sha(args.data / "manifest.json"),
        "pair_summary_sha256": sha(args.pair_directory / "summary.json"),
        "production_default_changed": False, "whole_recon_all_acceleration": "not measured", "cases": []}
    dataset = json.loads((args.data / "manifest.json").read_text())
    for case in dataset["cases"]:
        directory = args.output / case["id"]
        directory.mkdir()
        original = nib.load(str(args.data / case["input_image"]))
        pair = args.pair_directory / case["id"]
        tensor_expected = quantize(np.fromfile(pair / "v1_gpu_2/full_candidate.final.raw", np.float32)
            .reshape(original.shape, order="F"))
        native_expected = quantize(np.fromfile(pair / "native_1.final.raw", np.float32)
            .reshape(original.shape, order="F"))
        output_path = directory / "api.mgz"
        torch.cuda.reset_peak_memory_stats(device)
        api = n4_itk_torch_experimental.correct_volume(input_path=args.data / case["input_image"],
            output_path=output_path, device=str(device), profile=False)
        api["geometry_and_voxels"] = comparison(output_path, original, tensor_expected, native_expected)
        api["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        api["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        api["memory_scope"] = "API same-process PyTorch counters; not parent-child process occupancy"
        # No model/tensor is kept live by the completed API; context/cache persist.
        cli_output = directory / "cli.mgz"
        cli_report = directory / "cli.json"
        command = [os.sys.executable, str(args.cli_source), "--input", str(args.data / case["input_image"]),
            "--output", str(cli_output), "--report", str(cli_report), "--device", str(device),
            "--threads", str(args.threads)]
        environment = os.environ.copy()
        environment.update({"OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
            "OPENBLAS_NUM_THREADS": str(args.threads)})
        started = time.perf_counter()
        sampler = profiling.ProcessTreeDeviceSampler(device=str(device), parent_pid=os.getpid(), interval=.1)
        stop = threading.Event()
        with (directory / "cli.log").open("w") as stream:
            process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            sampler.add_worker(process.pid)
            monitor = threading.Thread(target=monitor_family, args=(sampler, stop), daemon=True)
            monitor.start()
            returncode = process.wait()
            cold_wall = time.perf_counter() - started
            stop.set(); monitor.join(timeout=5)
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        cli = {"cold_process_wall_seconds": cold_wall,
            "geometry_and_voxels": comparison(cli_output, original, tensor_expected, native_expected),
            "reported_api": json.loads(cli_report.read_text()), "command": command,
            "memory_parent_pid": os.getpid(), "memory_cli_pid": process.pid,
            "same_time_parent_child_sampling": sampler.report()}
        report["cases"].append({"case": case["id"], "input_sha256": sha(args.data / case["input_image"]),
            "api": api, "cli": cli})
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    contract = ("same_shape", "same_affine", "same_zooms", "same_image_class", "same_tensor_uint8")
    report["interface_contract_pass"] = all(row[kind]["geometry_and_voxels"][field]
        for row in report["cases"] for kind in ("api", "cli") for field in contract)
    report["status"] = "complete" if report["interface_contract_pass"] else "failed_contract"
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["interface_contract_pass"]:
        raise RuntimeError("real file-interface contract changed; see preserved report")


if __name__ == "__main__":
    main()
