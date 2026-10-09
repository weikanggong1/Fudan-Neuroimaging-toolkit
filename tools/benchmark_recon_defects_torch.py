"""同输入双侧 defects 原生/PyTorch 完整读写配对，不写入被试目录。"""
from __future__ import annotations
import argparse
import hashlib
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

from fnit.recon_all.defects_label_volume_torch import defects_to_volume
from fnit.recon_all.profiling import ProcessTreeDeviceSampler


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("benchmark output must be new")
    args.output.mkdir(parents=True)
    native_environment = dict(os.environ, FREESURFER_HOME=str(args.assets.resolve()),
                              SUBJECTS_DIR=str(args.subject.resolve().parent),
                              OMP_NUM_THREADS=str(args.threads))
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    report = {"host": platform.node(), "code_commit": args.code_commit,
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "device": args.device, "threads": args.threads,
              "precision": {"image_dtype": "float32", "centroid_dtype": "float64",
                            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                            "autocast": False, "half_precision": False,
                            "note": "projection uses ordered scalar products, not a TF32 matrix multiply"},
              "thread_environment": {key: os.environ.get(key) for key in
                                     ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "timing": "warm process API includes load/transfer/write; validation and first context initialization separate",
              "scope": "same-input complete two-hemisphere projection, including read/write; not raw-T1 end-to-end",
              "source_sha256": {"defects_label_volume_torch.py": sha(__import__("fnit.recon_all.defects_label_volume_torch", fromlist=["__file__"]).__file__),
                                "benchmark": sha(__file__)},
              "binary_sha256": sha(args.binary), "inputs": {}, "trials": [],
              "overall_equivalence": "not_assessed"}
    surf, mri, label = (args.subject / name for name in ("surf", "mri", "label"))
    for path in [mri / "orig.mgz", *(surf / f"{h}.{n}" for h in ("lh", "rh") for n in ("orig.nofix", "defect_labels")),
                 *(label / f"{h}.nofix.cortex.label" for h in ("lh", "rh"))]:
        report["inputs"][str(path)] = sha(path)
    devices = ("cpu", args.device) if args.device != "cpu" else ("cpu",)
    order = ["native", *devices, *reversed(devices), "native"]
    sampler = ProcessTreeDeviceSampler(device=args.device, parent_pid=os.getpid(), interval=.1)
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            sampler.sample_if_due()
            stop.wait(.05)
    monitor = threading.Thread(target=sample, daemon=True)
    try:
        if args.device.startswith("cuda"):
            tick = time.perf_counter()
            # 先在主线程初始化明确目标，再启动仅用于benchmark的监测线程。
            # 初始化失败仍写失败报告，不隐式重试或回退。
            torch.cuda.init()
            torch.cuda.set_device(args.device)
            torch.cuda.synchronize(args.device)
            report["cuda_initialization_seconds"] = time.perf_counter() - tick
        monitor.start()
        for trial, backend in enumerate(order):
            output = args.output / f"trial_{trial}_{backend.replace(':','_')}.mgz"
            if backend.startswith("cuda"):
                torch.cuda.synchronize(backend)
                torch.cuda.reset_peak_memory_stats(backend)
            tick = time.perf_counter()
            for hemi, offset in (("lh", 1000), ("rh", 2000)):
                template = mri / "orig.mgz" if hemi == "lh" else output
                if backend == "native":
                    command = [str(args.binary), "--defects", str(surf / f"{hemi}.orig.nofix"),
                               str(surf / f"{hemi}.defect_labels"), str(template), str(offset),
                               str(int(hemi == "rh")), str(output), str(label / f"{hemi}.nofix.cortex.label")]
                    with (args.output / f"trial_{trial}_{hemi}.log").open("w") as log:
                        subprocess.run(command, check=True, env=native_environment,
                                       stdout=log, stderr=subprocess.STDOUT)
                else:
                    defects_to_volume(surface_file=surf / f"{hemi}.orig.nofix",
                                      defect_file=surf / f"{hemi}.defect_labels", template_file=template,
                                      output_file=output, offset=offset, merge=hemi == "rh",
                                      cortex_file=label / f"{hemi}.nofix.cortex.label", device=backend)
            if backend.startswith("cuda"):
                torch.cuda.synchronize(backend)
            row = {"backend": backend, "seconds": time.perf_counter() - tick, "output_sha256": sha(output)}
            if backend.startswith("cuda"):
                row.update(allocated_peak_bytes=torch.cuda.max_memory_allocated(backend),
                           reserved_peak_bytes=torch.cuda.max_memory_reserved(backend))
            image = nib.load(str(output))
            data = np.asarray(image.dataobj)
            if trial == 0:
                reference, geometry, dtype = data.copy(), image.affine.copy(), image.get_data_dtype()
            else:
                diff = np.abs(data.astype(np.int64) - reference.astype(np.int64))
                row.update(different_voxels=int(np.count_nonzero(diff)), max_abs=int(diff.max()),
                           p99=float(np.percentile(diff, 99)), affine_max_abs=float(np.abs(image.affine - geometry).max()),
                           dtype_equal=image.get_data_dtype() == dtype,
                           label_dice={str(int(v)): float(2 * np.count_nonzero((data == v) & (reference == v)) /
                                                       (np.count_nonzero(data == v) + np.count_nonzero(reference == v)))
                                       for v in np.union1d(np.unique(reference), np.unique(data)) if v != 0})
            report["trials"].append(row)
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        stop.set()
        if monitor.ident is not None:
            monitor.join(timeout=5)
        report["process_memory"] = sampler.report()
        (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
