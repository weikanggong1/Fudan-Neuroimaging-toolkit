"""同输入完整 WM 编辑回归与独立 source benchmark；不执行 recon-all。

参考程序必须显式给出路径及SHA；生产 API 不使用它。外层1线程采样，
fresh exec 子进程3线程计算，默认总预算4。MRI、许可证及原程序不导出。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare(candidate, reference):
    import nibabel as nib
    import numpy as np
    a, b = (nib.load(str(path)) for path in (candidate, reference))
    x, y = np.asarray(a.dataobj), np.asarray(b.dataobj)
    if x.shape != y.shape:
        raise ValueError("output shapes differ")
    difference = np.abs(x.astype(np.int16) - y.astype(np.int16))
    labels = np.union1d(np.unique(x), np.unique(y))
    counts = {str(int(label)): [int(np.count_nonzero(x == label)),
        int(np.count_nonzero(y == label)), int(np.count_nonzero((x == label) & (y == label)))]
        for label in labels}
    foreground_x, foreground_y = x >= 5, y >= 5
    foreground_denominator = int(foreground_x.sum() + foreground_y.sum())
    return {"different_voxels": int(np.count_nonzero(difference)),
        "maximum_absolute_error": int(difference.max()),
        "p99_absolute_error": float(np.percentile(difference, 99)),
        "per_label_counts": counts,
        "per_label_dice": {label: 2 * row[2] / (row[0] + row[1]) for label, row in counts.items()},
        "wm_foreground_definition": "uint8 intensity >=5",
        "wm_foreground_dice": (2 * int(np.count_nonzero(foreground_x & foreground_y)) /
                               foreground_denominator if foreground_denominator else 1.0),
        "shape_equal": x.shape == y.shape, "dtype_equal": a.get_data_dtype() == b.get_data_dtype(),
        "affine_exact": bool(np.array_equal(a.affine, b.affine)),
        "vox2ras_tkr_exact": bool(np.array_equal(a.header.get_vox2ras_tkr(), b.header.get_vox2ras_tkr())),
        "candidate_sha256": sha(candidate), "reference_sha256": sha(reference),
        "raw_bytes_equal": Path(candidate).read_bytes() == Path(reference).read_bytes()}


def one(args):
    started = time.perf_counter()
    import_start = time.perf_counter()
    import nibabel as nib
    import numpy as np
    import numba
    import torch
    from fnit.recon_all import edit_wm_aseg_core_python as cpu
    from fnit.recon_all import edit_wm_aseg_torch as hybrid
    from fnit.recon_all.wm_remove_paths import remove_mtl_paths
    import_seconds = time.perf_counter() - import_start
    numba.set_num_threads(args.threads)
    torch.set_num_threads(args.threads)
    files = {name: args.input_directory / (name + ".mgz")
             for name in ("wm.seg", "brain", "aseg.presurf", "entowm")}
    input_hashes = {name: sha(path) for name, path in files.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "output.mgz"
    report = {"status": "running", "case": args.case, "backend": args.backend,
        "scope": "frozen same-input complete WM file stage, not raw-T1 whole pipeline",
        "input_producer": args.input_producer, "code_version": args.code_version,
        "producer_receipt_sha256": sha(args.producer_receipt),
        "hostname_private": platform.node(), "cpu_architecture": platform.machine(),
        "input_sha256": input_hashes, "script_sha256": sha(__file__),
        "module_sha256": {Path(module.__file__).name: sha(module.__file__) for module in (cpu, hybrid)},
        "path_module_sha256": sha(sys.modules[remove_mtl_paths.__module__].__file__),
        "source_manifest_sha256": sha(args.source_manifest),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "computation_threads": args.threads,
        "import_seconds": import_seconds, "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_allocator_environment": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
        "original_storage_dtype": str(nib.load(str(files["aseg.presurf"])).get_data_dtype()),
        "overall_metric_equivalence": "not_assessed", "used_reference_inputs_in_production": False}
    if args.backend == "native":
        if not args.reference_program or sha(args.reference_program) != args.reference_program_sha256:
            raise ValueError("explicit fixed source benchmark program SHA mismatch")
        command = [str(args.reference_program), "-keep-in", "-fix-ento-wm", str(files["entowm"]),
            "3", "255", "255", "-fix-acj", str(files["aseg.presurf"]), "255", "255", "-fill-seg-wm",
            "-fix-scm-ha", "1", str(files["wm.seg"]), str(files["brain"]), str(files["aseg.presurf"]), str(output)]
        tick = time.perf_counter()
        with (args.output / "reference.log").open("w") as stream:
            child = subprocess.run(command, cwd=args.output, stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=args.reference_timeout, check=False)
        report.update(file_api_wall_seconds=time.perf_counter() - tick,
                      reference_program_sha256=sha(args.reference_program),
                      reference_receipt_sha256=sha(args.reference_receipt), reference_exit_code=child.returncode)
        count = re.search(r"(\d+) voxels added to wm to prevent paths from MTL structures to cortex",
                          (args.output / "reference.log").read_text(errors="replace"))
        report["source_path_filled_count"] = int(count.group(1)) if count else None
        if child.returncode:
            save(args.output / "report.json", dict(report, status="failed_reference"))
            raise RuntimeError("isolated reference failed; log retained")
    else:
        api = cpu.write_wm_asegedit_profile
        if args.backend == "baseline":
            spec = importlib.util.spec_from_file_location("fnit.recon_all.wm_paths_frozen_baseline", args.baseline_module)
            module = importlib.util.module_from_spec(spec)
            # Numba 磁盘缓存需要可导入的稳定模块名；注册后才执行基线模块。
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            api = module.write_wm_asegedit_profile
            report["baseline_module_sha256"] = sha(args.baseline_module)
        kwargs = dict(wm_file=files["wm.seg"], brain_file=files["brain"],
            aseg_file=files["aseg.presurf"], entowm_file=files["entowm"], output_file=output,
            fill_seg_wm=True)
        target = None
        if args.backend == "hybrid":
            api = hybrid.write_wm_asegedit_hybrid_diagnostic
            target = torch.device(args.device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.synchronize(target)
            torch.cuda.reset_peak_memory_stats(target)
            kwargs["device"] = args.device
        tick = time.perf_counter()
        details = api(**kwargs)
        if target is not None:
            torch.cuda.synchronize(target)
        report.update(file_api_wall_seconds=time.perf_counter() - tick, api=details)
        if target is not None:
            report["cuda"] = {"name": torch.cuda.get_device_name(target),
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "autocast_enabled": torch.is_autocast_enabled("cuda"),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(target),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(target),
                "allocator_stats_zero_is_zero_memory_claim": False}
        if args.backend == "cpu":
            wm = np.asarray(nib.load(str(files["wm.seg"])).dataobj)
            brain = np.asarray(nib.load(str(files["brain"])).dataobj)
            image = nib.load(str(files["aseg.presurf"]))
            aseg = np.asarray(image.dataobj).astype(np.int32)
            _, report["paths_diagnostic"] = remove_mtl_paths(wm=wm, brain=brain, aseg=aseg,
                aseg_storage_dtype=image.get_data_dtype())
    report.update(status="complete", output_sha256=sha(output),
                  input_sha256_after={name: sha(path) for name, path in files.items()},
                  one_process_seconds=time.perf_counter() - started)
    save(args.output / "report.json", report)


def pair(args):
    from fnit.recon_all.profiling import ProcessTreeDeviceSampler
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    outputs = []
    for index, backend in enumerate(args.order.split(",")):
        directory = args.output / f"{index}-{backend}"
        command = [sys.executable, "-X", "faulthandler", "-u", str(Path(__file__).resolve()),
            "--mode", "one", "--case", args.case, "--backend", backend,
            "--input-directory", str(args.input_directory), "--output", str(directory),
            "--source-manifest", str(args.source_manifest), "--code-version", args.code_version,
            "--input-producer", args.input_producer, "--device", args.device,
            "--producer-receipt", str(args.producer_receipt),
            "--threads", str(args.threads), "--baseline-module", str(args.baseline_module)]
        if args.reference_program:
            command.extend(["--reference-program", str(args.reference_program),
                "--reference-program-sha256", args.reference_program_sha256,
                "--reference-receipt", str(args.reference_receipt),
                "--reference-timeout", str(args.reference_timeout)])
        tick = time.perf_counter()
        with (args.output / f"{index}-{backend}.log").open("w") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            sampler = ProcessTreeDeviceSampler(device=args.device if backend == "hybrid" else "cpu",
                parent_pid=child.pid, interval=.5)
            while child.poll() is None:
                sampler.sample_if_due()
                time.sleep(.1)
            sampler.sample_if_due(force=True)
            sampler.close()
        record = {"order_index": index, "backend": backend, "exit_code": child.returncode,
            "fresh_CLI_wall_seconds": time.perf_counter() - tick, "memory_samples": sampler.report()}
        records.append(record)
        save(args.output / "process_records.json", records)
        if child.returncode:
            raise RuntimeError("stage process failed; all outputs and logs retained")
        outputs.append(directory / "output.mgz")
    reference_index = next((i for i, row in enumerate(records) if row["backend"] == "native"), 0)
    comparisons = {str(index): compare(output, outputs[reference_index]) for index, output in enumerate(outputs)}
    report = {"status": "complete", "scope": "frozen same-input WM stage regression, not whole reconstruction",
        "case": args.case, "source_manifest_sha256": sha(args.source_manifest),
        "records": records, "comparisons": comparisons, "reference_index": reference_index,
        "total_cpu_budget": args.threads + 1, "sample_interval_seconds": .5,
        "overall_metric_equivalence": "not_assessed"}
    api_records = [json.loads((args.output / f"{i}-{row['backend']}" / "report.json").read_text())
                   for i, row in enumerate(records)]
    cpu_record = next((row for row in api_records if row["backend"] == "cpu"), None)
    report["active_path_real_benchmark"] = (any(not side["proof_unreachable"] for side in
        cpu_record["paths_diagnostic"]["hemispheres"]) if cpu_record else None)
    report["path_fill_count_comparison"] = [{"backend": row["backend"], "filled":
        row.get("source_path_filled_count") if row["backend"] == "native" else
        row.get("api", {}).get("paths_filled", row.get("api", {}).get("path_proof_changed"))}
        for row in api_records]
    save(args.output / "COMPLETE.json", report)
    if any(row["different_voxels"] or not row["affine_exact"] or not row["dtype_equal"] or
           not row["vox2ras_tkr_exact"] for row in comparisons.values()):
        raise RuntimeError("decoded full WM stage regression differs; report retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("one", "pair"), default="pair")
    parser.add_argument("--backend", choices=("native", "baseline", "cpu", "hybrid"), default="cpu")
    parser.add_argument("--case", required=True)
    for name in ("input-directory", "output", "source-manifest", "baseline-module", "producer-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--input-producer", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--order", default="hybrid,cpu,baseline,native,native,cpu")
    parser.add_argument("--reference-program", type=Path)
    parser.add_argument("--reference-program-sha256")
    parser.add_argument("--reference-receipt", type=Path)
    parser.add_argument("--reference-timeout", type=float, default=300)
    args = parser.parse_args()
    if args.threads < 1 or args.reference_program and (
            not args.reference_program_sha256 or not args.reference_receipt):
        parser.error("positive threads and explicit fixed reference SHA/receipt are required")
    one(args) if args.mode == "one" else pair(args)


if __name__ == "__main__":
    main()
