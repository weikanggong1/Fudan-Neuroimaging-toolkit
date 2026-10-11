"""真实同输入WM逐阶段诊断；参考中间结果只用于比较，不进入候选计算。"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output-directory", "reference-binary", "reference-assets",
                 "reference-source-root", "module-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--histogram-backend", choices=("cpu", "torch"), default="torch")
    parser.add_argument("--planar-backend", choices=("python", "cached"), default="cached")
    args = parser.parse_args()
    if file_sha(args.reference_binary) != args.reference_sha256:
        raise ValueError("reference executable SHA differs")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    native = args.output_directory / "native"
    candidate = args.output_directory / "candidate"
    native.mkdir(); candidate.mkdir()
    import numpy as np
    import nibabel as nib
    import torch
    import numba
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.module_directory.resolve()))
    from fnit.recon_all import mri_segment as wm
    from fnit.recon_all import mri_segment_histogram_torch as hist
    from fnit.recon_all import mri_segment_planar_torch as planar
    torch.set_num_threads(args.threads)
    numba.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    image = nib.load(str(args.source))
    report = {"status": "running", "scope": "same-input complete WM with extra diagnostic writes; not production timing",
        "code_version": args.code_version, "driver_sha256": file_sha(__file__),
        "source_sha256": file_sha(args.source), "reference_binary_sha256": file_sha(args.reference_binary),
        "reference_sources_sha256": {n: file_sha(args.reference_source_root / n) for n in
            ("mri_segment/mri_segment.cpp", "utils/mrihisto.cpp", "utils/histo.cpp")},
        "modules_sha256": {Path(m.__file__).name: file_sha(m.__file__) for m in (wm, hist, planar)},
        "input": {"shape": [int(n) for n in image.shape], "dtype": str(image.get_data_dtype()),
                  "affine_mm": image.affine.tolist()},
        "host": platform.node(), "device": args.device, "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "captures": {}, "comparisons": {}}
    command = [str(args.reference_binary.resolve()), "-diag-write", "-diag-verbose", "-wsizemm", "13",
               "-mprage", str(args.source.resolve()), str((native / "output.mgz").resolve())]
    tick = time.perf_counter()
    with (native / "command.log").open("w") as stream:
        subprocess.run(command, cwd=native, env=dict(os.environ,
            FREESURFER_HOME=str(args.reference_assets.resolve())), stdout=stream, stderr=subprocess.STDOUT, check=True)
    report["native_with_diagnostic_writes_seconds"] = time.perf_counter() - tick
    counts = {}; originals = []
    mapping = {"intensity_segmentation": "int", "histogram_segmentation": "histo",
        "median_curve_segmentation": "medcurv", "reclassify_border": "reclassified",
        "recover_bright_white": "recover-brightwm", "remove_wrong_direction": "wrong-dir",
        "remove_1d_structures": "rm1d", "_thicken_strands_core": "thicken",
        "remove_bright_nonwhite": "bright-nonwm", "filter_diagonal_morphology": "filter"}
    def wrapper(owner, name, stem):
        original = getattr(owner, name); originals.append((owner, name, original))
        @functools.wraps(original)
        def wrapped(*a, **kw):
            value = original(*a, **kw)
            volume = value[0] if isinstance(value, tuple) else value
            array = volume.detach().cpu().numpy() if isinstance(volume, torch.Tensor) else np.asarray(volume)
            counts[stem] = counts.get(stem, 0) + 1
            suffix = f".{counts[stem]}" if stem in ("int", "histo") else ""
            filename = "wmseg." + stem + suffix + ".mgz"
            nib.save(nib.MGHImage(array, image.affine, header=image.header.copy()), str(candidate / filename))
            report["captures"][filename] = file_sha(candidate / filename)
            return value
        setattr(owner, name, wrapped)
    for name, stem in mapping.items():
        wrapper(wm, name, stem)
    wrapper(hist, "histogram_segmentation_torch", "histo")
    original_threshold = wm.detect_intensity_thresholds
    originals.append((wm, "detect_intensity_thresholds", original_threshold))
    def thresholds(*a, **kw):
        result = original_threshold(*a, **kw)
        report["thresholds"] = dataclasses.asdict(result)
        return result
    wm.detect_intensity_thresholds = thresholds
    tick = time.perf_counter()
    try:
        wm.segment_white_matter_mgz(source_path=args.source, output_path=candidate / "output.mgz",
            device=args.device, histogram_backend=args.histogram_backend, histogram_batch_size=2048,
            planar_backend=args.planar_backend, planar_batch_size=256)
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(args.device))
        report["torch_allocated_peak_bytes"] = torch.cuda.max_memory_allocated(torch.device(args.device))
        report["torch_reserved_peak_bytes"] = torch.cuda.max_memory_reserved(torch.device(args.device))
    report["candidate_with_diagnostic_writes_seconds"] = time.perf_counter() - tick
    report["captures"]["output.mgz"] = file_sha(candidate / "output.mgz")
    for name in report["captures"]:
        reference = nib.load(str(native / name)); output = nib.load(str(candidate / name))
        a, b = np.asarray(reference.dataobj), np.asarray(output.dataobj)
        delta = b.astype(np.float64) - a.astype(np.float64)
        difference = np.abs(delta)
        labels = np.union1d(np.unique(a), np.unique(b))
        report["comparisons"][name] = {"different_voxels": int(np.count_nonzero(delta)),
            "maximum_absolute_error": float(difference.max()), "p99_absolute_error": float(np.quantile(difference, .99)),
            "dtype_reference": str(a.dtype), "dtype_candidate": str(b.dtype),
            "affine_exact": bool(np.array_equal(reference.affine, output.affine)),
            "per_label_dice": {str(int(label)): float(2 * np.count_nonzero((a == label) & (b == label)) /
                (np.count_nonzero(a == label) + np.count_nonzero(b == label))) for label in labels},
            "reference_sha256": file_sha(native / name), "candidate_sha256": file_sha(candidate / name)}
    report["status"] = "complete"
    (args.output_directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v["different_voxels"] for k, v in report["comparisons"].items()}))


if __name__ == "__main__":
    main()
