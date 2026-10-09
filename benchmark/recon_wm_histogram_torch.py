"""真实T1冻结同输入WM直方图：原生中间参考、旧CPU和新Torch比较。

诊断参考只供测试输入/输出比较，候选函数不读取参考结果。这里不运行
完整recon-all，也不把带诊断写出的原生命令墙钟当作单个直方图时间。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import nibabel as nib
import numpy as np
import torch


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--diagnostic-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--reference-binary", required=True, type=Path)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--reference-source-root", required=True, type=Path)
    args = parser.parse_args()
    if sha(args.reference_binary) != args.reference_sha256:
        raise ValueError("reference command hash differs from declared binary")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    total_started = time.perf_counter()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this paired benchmark requires an explicit CUDA device")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Explicit context initialization precedes package imports. Runtime failures
    # propagate; there is no automatic retry or hidden CPU fallback.
    tick = time.perf_counter()
    probe = torch.zeros(1, dtype=torch.uint8, device=device)
    torch.cuda.synchronize(device)
    context_seconds = time.perf_counter() - tick
    del probe
    from fnit.recon_all import mri_segment as cpu
    from fnit.recon_all import mri_segment_histogram_torch as gpu
    # Reuse the already reviewed process sampler and exact discrete comparator.
    helper_path = Path(__file__).with_name("recon_wm_aseg_torch.py")
    spec = importlib.util.spec_from_file_location("wm_report_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    gpu_uuid = subprocess.check_output(["nvidia-smi", f"--id={device.index}",
        "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    sampler = helper.ProcessMemorySampler(gpu_uuid)
    sampler.start()
    torch.cuda.reset_peak_memory_stats(device)
    source_image = nib.load(args.source)
    source = np.array(source_image.dataobj, dtype=np.uint8, copy=True)
    if source.ndim != 3 or source_image.get_data_dtype() != np.dtype(np.uint8):
        raise ValueError("expected original 3D uint8 WM segmentation intensity input")
    image_cpu = torch.from_numpy(source)
    image_gpu = image_cpu.to(device)
    native_histo1_file = args.diagnostic_dir / "wmseg.histo.1.mgz"
    native_histo1 = np.array(nib.load(native_histo1_file).dataobj, copy=True)
    stats = cpu.detect_intensity_thresholds(image_cpu, torch.from_numpy(native_histo1))
    parameters = ({"wm_low": 79., "wm_hi": 125., "gray_hi": 99.},
                  {"wm_low": stats.wm_low, "wm_hi": 125., "gray_hi": stats.gray_hi})
    reference_sources = ("mri_segment/mri_segment.cpp", "utils/mrihisto.cpp", "utils/histo.cpp")
    report = {"scope": "frozen_same_input_two_histogram_passes; not_whole_WM_or_recon_all",
        "code_commit": args.code_commit, "code_commit_role": "baseline plus modules bound by executed SHA-256",
        "benchmark_sha256": sha(__file__), "report_helper_sha256": sha(helper_path),
        "source_modules": {Path(m.__file__).name: sha(m.__file__) for m in (cpu, gpu)},
        "host": platform.node(), "pid": os.getpid(), "threads": args.threads,
        "actual_torch_threads": torch.get_num_threads(),
        "thread_environment": {key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "device": str(device), "gpu_uuid": gpu_uuid,
        "gpu": torch.cuda.get_device_name(device), "context_seconds": context_seconds,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "precision": {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "arithmetic": "int32 histogram counts; ordered float32 smoothing; C-double MIN_STD comparison; no half"},
        "reference_program_sha256": sha(args.reference_binary),
        "reference_sources_sha256": {path: sha(args.reference_source_root / path) for path in reference_sources},
        "native_diagnostic_start_utc": (args.diagnostic_dir / "start_utc.txt").read_text().strip(),
        "native_diagnostic_end_utc": (args.diagnostic_dir / "end_utc.txt").read_text().strip(),
        "native_diagnostic_timing_use": "reference generation only; concurrent root whole run; not an algorithm time",
        "inputs": {"source": {"sha256": sha(args.source), "shape": list(source.shape),
            "dtype": str(source.dtype), "affine_mm": source_image.affine.tolist()}},
        "derived_statistics": vars(stats), "passes": {}, "whole_recon_speedup": "not_measured",
        "whole_metric_equivalence": "not_assessed", "isolated_deployment": "not_verified"}
    for pass_index, params in enumerate(parameters, 1):
        label_path = args.diagnostic_dir / f"wmseg.int.{pass_index}.mgz"
        expected_path = args.diagnostic_dir / f"wmseg.histo.{pass_index}.mgz"
        label_image, expected_image = nib.load(label_path), nib.load(expected_path)
        if any(image.shape != source_image.shape or not np.array_equal(image.affine, source_image.affine)
               for image in (label_image, expected_image)):
            raise ValueError("native diagnostic inputs/outputs do not share the exact source grid")
        label_array = np.array(label_image.dataobj, dtype=np.uint8, copy=True)
        expected = np.asarray(expected_image.dataobj)
        labels_cpu = torch.from_numpy(label_array)
        labels_gpu = labels_cpu.to(device)
        intensity = cpu.intensity_segmentation(image_cpu, **params)
        row = {"parameters_float": params,
            "parameters_native_int": {key: int(value) for key, value in params.items()},
            "candidate_count": int(np.count_nonzero(label_array == 128)),
            "intensity_label_vs_native": helper.compare(intensity.numpy(), label_array),
            "input_labels_sha256": sha(label_path), "native_output_sha256": sha(expected_path),
            "measurement": "complete histogram array stage; all candidates; preloaded arrays; excludes file IO/H2D/D2H",
            "order": ["cpu", "cuda", "cuda", "cpu"], "samples_seconds": {"cpu": [], "cuda": []},
            "results": {}}
        for index, backend in enumerate(row["order"]):
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            output = (cpu.histogram_segmentation(image_cpu, labels_cpu, **params)
                      if backend == "cpu" else gpu.histogram_segmentation_torch(
                          image_gpu, labels_gpu, **params, batch_size=2048))
            torch.cuda.synchronize(device)
            row["samples_seconds"][backend].append(time.perf_counter() - tick)
            array = output.cpu().numpy()
            row["results"][f"{index}_{backend}"] = helper.compare(array, expected)
            if backend == "cuda":
                nib.save(nib.MGHImage(array, source_image.affine, header=label_image.header.copy()),
                         args.output_dir / f"histo-{pass_index}-{index}-cuda.mgz")
        row["medians_seconds"] = {key: float(np.median(values)) for key, values in row["samples_seconds"].items()}
        row["speedup_vs_existing_python"] = row["medians_seconds"]["cpu"] / row["medians_seconds"]["cuda"]
        report["passes"][str(pass_index)] = row
        print("pass", pass_index, json.dumps(row), flush=True)
    memory = sampler.finish()
    (args.output_dir / "process_memory.json").write_text(json.dumps(memory, indent=2))
    report["process_memory"] = {key: value for key, value in memory.items() if key != "samples"}
    report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    report["benchmark_wall_seconds"] = time.perf_counter() - total_started
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
