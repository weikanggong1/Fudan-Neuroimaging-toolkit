"""冻结真实smoothwm→离散八图同输入回归；不修改原被试，不假称完整curv.stats。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

import fnit.recon_all.discrete_curvature_torch as module
from fnit.recon_all.discrete_curvature_torch import DiscreteCurvatureTopology, write_discrete_curvature

# 在首个真实输入测试前声明，仅工程探索门；不是整体等效标准。
MAX_ALLOWED_ABS = 1e-5
P99_ALLOWED_ABS = 1e-6
NAMES = ("K", "H", "K1", "K2", "BE", "C", "FI", "S")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(first, second):
    if first.shape != second.shape:
        raise ValueError("reference and candidate vertex correspondence differs")
    delta = np.abs(first.astype(np.float64) - second.astype(np.float64))
    maximum, p99 = float(delta.max(initial=0)), float(np.percentile(delta, 99))
    return {"different_elements": int(np.count_nonzero(first != second)), "max_abs": maximum,
            "p99_abs": p99, "mean_abs": float(delta.mean()), "finite": bool(np.isfinite(first).all()),
            "within_exploratory_tolerance": bool(maximum <= MAX_ALLOWED_ABS and p99 <= P99_ALLOWED_ABS)}


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reference-program", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--code-version", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(args.device)
        torch.cuda.synchronize(args.device)
    report = {"scope": "frozen surface, complete discrete principal plus eight raw maps; not curv.stats or end-to-end",
              "host": platform.node(), "code_version": args.code_version,
              "source_sha256": {"module": sha(module.__file__), "script": sha(__file__)},
              "threads": args.threads, "device": args.device, "torch": torch.__version__,
              "precision": "float32 with source-required double sqrt and final expressions; TF32 default; no half precision",
              "threshold_before_test": {"max_abs": MAX_ALLOWED_ABS, "p99_abs": P99_ALLOWED_ABS},
              "reference_program_sha256": sha(args.reference_program) if args.reference_program else None,
              "process_memory": "not_measured", "whole_metric_equivalence": "not_assessed", "cases": []}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    for index, subject in enumerate(args.subject):
        for hemi in ("lh", "rh"):
            surf = subject / "surf" / f"{hemi}.smoothwm"
            vertices, faces = fsio.read_geometry(str(surf))
            output = args.output / f"case_{index}_{hemi}"
            output.mkdir()
            row = {"subject": str(subject), "hemi": hemi, "surface_sha256": sha(surf),
                   "vertices": len(vertices), "faces": len(faces), "trials": [], "reference": {}}
            report["cases"].append(row)
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            if args.reference_program:
                if args.assets is None:
                    raise ValueError("--assets is required with --reference-program")
                frozen = output / "reference_subject"
                for folder in ("surf", "stats", "scripts"):
                    (frozen / folder).mkdir(parents=True)
                for name in ("smoothwm", "curv", "sulc"):
                    shutil.copyfile(subject / "surf" / f"{hemi}.{name}", frozen / "surf" / f"{hemi}.{name}")
                command = [str(args.reference_program.resolve()), "-m", "--writeCurvatureFiles", "-G", "-o",
                           str((frozen / "stats" / f"{hemi}.curv.stats").resolve()), "-F", "smoothwm", frozen.name,
                           hemi, "curv", "sulc"]
                environment = dict(os.environ, SUBJECTS_DIR=str(output.resolve()),
                                   FREESURFER_HOME=str(args.assets.resolve()), OMP_NUM_THREADS=str(args.threads))
                row["native_seconds"] = []
                for repeat in range(2):
                    tick = time.perf_counter()
                    with (output / f"reference_{repeat}.log").open("w") as stream:
                        subprocess.run(command, cwd=frozen / "scripts", env=environment, check=True,
                                       stdout=stream, stderr=subprocess.STDOUT)
                    row["native_seconds"].append(time.perf_counter() - tick)
                    current = {name: fsio.read_morph_data(str(frozen / "surf" / f"{hemi}.smoothwm.{name}.crv")) for name in NAMES}
                    if repeat == 0:
                        reference = current
                    else:
                        row["reference_repeat"] = {name: metric(current[name], reference[name]) for name in NAMES}
            else:
                reference = {name: fsio.read_morph_data(str(subject / "surf" / f"{hemi}.smoothwm.{name}.crv")) for name in NAMES}
                row["reference_repeat"] = "not_retested_frozen_reference"
            for name in NAMES:
                row["reference"][name] = sha((frozen if args.reference_program else subject) / "surf" / f"{hemi}.smoothwm.{name}.crv")
            for backend in ("cpu", args.device, args.device, "cpu") if args.device != "cpu" else ("cpu", "cpu"):
                if backend.startswith("cuda"):
                    torch.cuda.synchronize(backend)
                    torch.cuda.reset_peak_memory_stats(backend)
                prefix = output / f"trial_{len(row['trials'])}_{backend.replace(':', '_')}"
                tick = time.perf_counter()
                candidate = write_discrete_curvature(surface_file=surf, output_prefix=prefix, device=backend)
                if backend.startswith("cuda"):
                    torch.cuda.synchronize(backend)
                trial = {"backend": backend, "api_seconds": time.perf_counter() - tick,
                         "maps": {name: metric(fsio.read_morph_data(candidate["outputs"][name]), reference[name]) for name in NAMES}}
                if backend.startswith("cuda"):
                    trial["allocated_peak_bytes"] = torch.cuda.max_memory_allocated(backend)
                    trial["reserved_peak_bytes"] = torch.cuda.max_memory_reserved(backend)
                row["trials"].append(trial)
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
    report["status"] = "complete"
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


def main():
    try:
        run()
    except Exception as error:
        # 保留当前输入、程序/源码身份及已完成配对；失败不补跑或改变阈值。
        if "--output" in sys.argv:
            path = Path(sys.argv[sys.argv.index("--output") + 1]) / "report.json"
            if path.exists():
                report = json.loads(path.read_text())
                report.update(status="failed", error=repr(error))
                path.write_text(json.dumps(report, indent=2))
        raise


if __name__ == "__main__":
    main()
