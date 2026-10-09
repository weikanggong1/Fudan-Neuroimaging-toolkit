"""真实冻结输入完整WM分割：固定Conda源码、旧CPU histogram、新Torch配对。"""
from __future__ import annotations

import argparse
import functools
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--reference-binary", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--reference-source-root", type=Path, required=True)
    parser.add_argument("--reference-assets", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.reference_binary) != args.reference_sha256:
        raise ValueError("declared reference executable hash mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("require an explicit CUDA device")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    from fnit.recon_all import mri_segment as wm
    from fnit.recon_all import mri_segment_histogram_torch as histogram
    helper_path = Path(__file__).with_name("recon_wm_aseg_torch.py")
    spec = importlib.util.spec_from_file_location("wm_report_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    uuid = subprocess.check_output(["nvidia-smi", f"--id={device.index}",
        "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    sampler = helper.ProcessMemorySampler(uuid)
    sampler.start()
    torch.cuda.reset_peak_memory_stats(device)
    source_image = nib.load(args.source)
    report = {
        "scope": "complete_frozen_same_input_WM_file_API; not_raw_T1_whole_recon",
        "code_commit": args.code_commit,
        "code_commit_role": "baseline plus executed file SHA-256 overlay",
        "source_modules": {Path(module.__file__).name: sha(module.__file__)
                           for module in (wm, histogram)},
        "benchmark_sha256": sha(__file__), "helper_sha256": sha(helper_path),
        "reference_program_sha256": sha(args.reference_binary),
        "reference_sources_sha256": {path: sha(args.reference_source_root / path)
            for path in ("mri_segment/mri_segment.cpp", "utils/mrihisto.cpp", "utils/histo.cpp")},
        "input": {"sha256": sha(args.source), "shape": [int(axis) for axis in source_image.shape],
                  "dtype": str(source_image.get_data_dtype()), "affine_mm": source_image.affine.tolist()},
        "host": platform.node(), "pid": os.getpid(), "cpu_model": next(
            (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
             if line.startswith("model name")), "unavailable"),
        "threads": args.threads, "actual_torch_threads": torch.get_num_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "allocator_environment": {name: os.environ.get(name)
            for name in ("PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")},
        "device": str(device), "gpu": torch.cuda.get_device_name(device), "gpu_uuid": uuid,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "nibabel": nib.__version__, "torch_cuda": torch.version.cuda,
        "precision": {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_tf32": torch.backends.cudnn.allow_tf32, "no_half": True},
        "measurement": "whole file API/command including load, H2D/D2H, ordered scans and compressed write; target CUDA synchronized; per-stage sync only in benchmark",
        "timing_excludes": "interpreter startup, top-level module imports and CUDA context initialization; not whole recon-all timing",
        "cold_scope": "first complete Python file API in this process; filesystem cache not flushed; compilation/cache state recorded by separate cold call",
        "order": ["native", "cpu", "torch", "torch", "cpu", "native"],
        "records": [], "whole_recon_speedup": "not_measured", "whole_metric_equivalence": "not_assessed",
        "isolated_deployment": "not_verified",
    }
    current_profile = []
    names = ("intensity_segmentation", "histogram_segmentation", "detect_intensity_thresholds",
             "median_curve_segmentation", "reclassify_border", "mask_white_labels",
             "recover_bright_white", "remove_wrong_direction", "remove_1d_structures",
             "_closed_strand_volume", "thin_strand_candidates", "_strand_segments",
             "_dilate_strand_segments", "_thicken_strands_core", "remove_bright_nonwhite",
             "filter_diagonal_morphology")

    def timed(function, name):
        @functools.wraps(function)
        def wrapped(*positional, **keywords):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = function(*positional, **keywords)
            torch.cuda.synchronize(device)
            current_profile.append({"function": name, "seconds": time.perf_counter() - started})
            return output
        return wrapped

    for name in names:
        setattr(wm, name, timed(getattr(wm, name), name))
    histogram.histogram_segmentation_torch = timed(histogram.histogram_segmentation_torch,
                                                 "histogram_segmentation_torch")

    def call(backend, path):
        if backend == "native":
            command = [str(args.reference_binary), "-wsizemm", "13", "-mprage",
                       str(args.source.resolve()), str(path.resolve())]
            with path.with_suffix(".log").open("w") as stream:
                subprocess.run(command, env=dict(os.environ, FREESURFER_HOME=str(args.reference_assets)),
                               check=True, stdout=stream, stderr=subprocess.STDOUT)
            return {"command": [args.reference_binary.name, "-wsizemm", "13", "-mprage",
                                "source.mgz", "output.mgz"], "diag_write": False}
        return wm.segment_white_matter_mgz(
            source_path=args.source, output_path=path, device=device,
            histogram_backend=backend, histogram_batch_size=2048)

    # The cold full API populates any existing ordered NumPy/Numba caches and
    # is reported separately rather than mixed into the paired warm median.
    cold_path = args.output_dir / "cold-cpu.mgz"
    tick = time.perf_counter()
    cold_result = call("cpu", cold_path)
    torch.cuda.synchronize(device)
    report["cold_python"] = {"seconds": time.perf_counter() - tick,
                             "api": cold_result, "stage_profile": list(current_profile)}
    current_profile.clear()
    native_reference = None
    for index, backend in enumerate(report["order"]):
        path = args.output_dir / f"{index}-{backend}.mgz"
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        api = call(backend, path)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - tick
        image = nib.load(path)
        array = np.asarray(image.dataobj)
        if native_reference is None:
            native_reference = array.copy()
            report["cold_python"]["vs_reference"] = helper.compare(
                np.asarray(nib.load(cold_path).dataobj), native_reference)
        row = {"backend": backend, "seconds": seconds, "api": api,
               "file_sha256": sha(path), "vs_native": helper.compare(array, native_reference),
               "dtype": str(image.get_data_dtype()), "shape": [int(axis) for axis in image.shape],
               "affine_equal_source": bool(np.array_equal(image.affine, source_image.affine)),
               "mgh_header_equal_source": image.header.binaryblock == source_image.header.binaryblock,
               "stage_profile": list(current_profile)}
        current_profile.clear()
        report["records"].append(row)
        print(json.dumps(row), flush=True)
    report["median_seconds"] = {backend: float(np.median([row["seconds"] for row in report["records"]
        if row["backend"] == backend])) for backend in ("native", "cpu", "torch")}
    report["speedup_vs_existing_python"] = report["median_seconds"]["cpu"] / report["median_seconds"]["torch"]
    report["speedup_vs_source_built_native"] = report["median_seconds"]["native"] / report["median_seconds"]["torch"]
    memory = sampler.finish()
    (args.output_dir / "process_memory.json").write_text(json.dumps(memory, indent=2))
    report["process_memory"] = {key: value for key, value in memory.items() if key != "samples"}
    report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
