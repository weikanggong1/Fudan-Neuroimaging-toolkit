"""从公开原始配对 T1w+BOLD 连续调用完整 surface API，保留失败与连续墙钟。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

import nibabel as nib
import numpy as np
import torch

from fnit.fmri import fMRISurface_pipeline, locate_bids_inputs
from fnit.fmri.derivatives import fmri_derivative_paths


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes(root):
    return {path.relative_to(root).as_posix(): sha256(path)
            for path in sorted((root / "src/fnit").rglob("*.py"))}


def process_tree():
    rows = subprocess.run(["ps", "-e", "-o", "pid=,ppid="], capture_output=True,
                          text=True, check=True).stdout.splitlines()
    edges = [tuple(map(int, row.split())) for row in rows]
    pids = {os.getpid()}
    while True:
        expanded = pids | {pid for pid, parent in edges if parent in pids}
        if expanded == pids:
            return pids
        pids = expanded


class Monitor:
    """记录本进程树同时显存及共享卡负载，不改变其他任务。"""

    def __init__(self, path):
        self.path = path
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.peak = 0
        self.errors = []

    def sample(self):
        with self.path.open("x") as stream:
            while not self.stop.is_set():
                try:
                    pids = process_tree()
                    processes = subprocess.run([
                        "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
                    owned = []
                    for line in processes.stdout.splitlines():
                        fields = [item.strip() for item in line.split(",")]
                        if len(fields) == 3 and fields[0].isdigit() and int(fields[0]) in pids:
                            owned.append({"pid": int(fields[0]), "memory_mib": int(fields[1]),
                                          "gpu_uuid": fields[2]})
                    simultaneous = sum(item["memory_mib"] for item in owned) * 1024**2
                    self.peak = max(self.peak, simultaneous)
                    load = subprocess.run([
                        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
                    stream.write(json.dumps({"monotonic_seconds": time.monotonic(),
                                             "owned_processes": owned,
                                             "owned_simultaneous_bytes": simultaneous,
                                             "shared_gpu_load": load.stdout.splitlines()}) + "\n")
                    stream.flush()
                except Exception as error:
                    self.errors.append(f"{type(error).__name__}: {error}")
                self.stop.wait(2)


def image_check(path, frames, tr):
    image = nib.load(str(path), keep_file_open=True)
    if image.ndim == 4:
        if image.shape[3] != frames or image.header.get_xyzt_units()[1] != "sec" or not np.isclose(image.header.get_zooms()[3], tr):
            raise ValueError("final volume frame count/TR differ from the entire raw run")
        for first in range(0, frames, 8):
            if not np.isfinite(np.asanyarray(image.dataobj[..., first:first + 8])).all():
                raise ValueError("final volume contains nonfinite values")
    elif isinstance(image, nib.Cifti2Image):
        series, brain = image.header.get_axis(0), image.header.get_axis(1)
        if (image.shape != (frames, 91282) or not np.isclose(series.step, tr)
                or series.unit != "SECOND" or len(brain) != 91282
                or not np.isfinite(np.asanyarray(image.dataobj)).all()):
            raise ValueError("final CIFTI axes/frames/TR/data are invalid")
    else:
        raise ValueError("unexpected final image format")
    return {"shape": list(image.shape), "sha256": sha256(path), "all_finite": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="私有 JSON：本例所有具名 API 参数")
    parser.add_argument("--output", type=Path, required=True, help="新建报告目录")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    output = args.output
    before = source_hashes(args.source_root)
    write(output / "source.private.json", before)
    report = {"subject": config["subject"], "status": "initializing",
              "source_revision": args.source_revision, "source_sha256": sha256(output / "source.private.json"),
              "scope": "Fresh raw paired T1w and complete BOLD through automatic volume, complete reconstruction, MSMSulc, GIFTI/CIFTI, QC and final publication in one API call; excludes import, resource installation, queue and posthoc comparison.",
              "driver_sha256": sha256(__file__), "failure": None}
    write(output / "report.public.json", report)
    monitor = Monitor(output / "gpu_load.public.jsonl")
    started = None
    try:
        inputs = locate_bids_inputs(config["bids_root"], subject=config["subject"],
                                    session=config.get("session"), task=config.get("task", "rest"))
        if len(inputs.t1w_images) != 1:
            raise ValueError("public paired benchmark requires exactly one T1w")
        raw = nib.load(str(inputs.bold))
        frames = raw.shape[3]
        paths = fmri_derivative_paths(inputs, inputs.t1w_images[0], config["derivatives_root"], signal="preproc")
        if paths.root.exists() or Path(config["recon_all_output_dir"]).exists():
            raise FileExistsError("whole benchmark must start with fresh empty derivatives and reconstruction destinations")
        if config.get("recon_all") is not None or config.get("registered_spheres") is not None:
            raise ValueError("whole benchmark cannot supply cached reconstruction or registered spheres")
        config.setdefault("volume_options", {})["reuse_anatomical"] = False
        report.update(input_sha256={"t1w": sha256(inputs.t1w_images[0]), "bold": sha256(inputs.bold)},
                      frames=frames, repetition_time=inputs.tr, status="running",
                      backend=config.get("recon_all_backend", "fnit"))
        write(output / "report.public.json", report)
        device = torch.device(config.get("device", "cuda:0"))
        torch.set_num_threads(config.get("cpu_threads", 8))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.init()
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, 20e9 / torch.cuda.get_device_properties(device).total_memory), device)
        torch.cuda.synchronize(device)
        monitor.thread.start()
        started = time.perf_counter()
        result = fMRISurface_pipeline(**config)
        torch.cuda.synchronize(device)
        whole = time.perf_counter() - started
        if not result.volume_executed:
            raise RuntimeError("fresh whole benchmark did not execute volume")
        checks = {key: image_check(path, frames, inputs.tr) for key, path in {
            "preproc_t1w": paths.preproc_t1w, "preproc_mni": paths.preproc_mni,
            "clean_native": paths.clean_native, "clean_mni": paths.clean_mni,
            "dtseries": result.dtseries}.items()}
        files = {key: str(path) for key, path in {
            "preproc_t1w": paths.preproc_t1w, "preproc_mni": paths.preproc_mni,
            "clean_native": paths.clean_native, "clean_mni": paths.clean_mni,
            "dtseries": result.dtseries, "left": result.left, "right": result.right,
            "recon_all": result.recon_all, "metadata": result.metadata,
            "qc_report": result.qc_report}.items()}
        write(output / "files.private.json", files)
        report.update(status="complete", continuous_api_wall_seconds=whole,
                      stage_seconds=result.timing_seconds, output_checks=checks,
                      volume_executed=result.volume_executed,
                      reconstruction=json.loads(result.metadata.read_text())["FNIT"]["Reconstruction"])
    except Exception as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)},
                      failed_attempt_wall_seconds=time.perf_counter() - started if started is not None else None)
        (output / "failure.private.txt").write_text(traceback.format_exc())
    finally:
        monitor.stop.set()
        if monitor.thread.ident is not None:
            monitor.thread.join(timeout=10)
        report.update(source_unchanged_during_run=before == source_hashes(args.source_root),
                      owned_tree_peak_bytes=monitor.peak,
                      owned_tree_under_20gb=monitor.peak <= 20_000_000_000,
                      memory_sampling_errors=monitor.errors)
        # Reconstruction provenance can contain local source paths/command
        # arguments. Keep the full record private; public metrics are anonymous.
        reconstruction = report.pop("reconstruction", None)
        if reconstruction is not None:
            write(output / "reconstruction.private.json", reconstruction)
        if report.get("failure"):
            write(output / "failure.private.json", report["failure"])
            report["failure"] = {"type": report["failure"]["type"], "details": "failure.private.json"}
        write(output / "report.public.json", report)
        print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if report["status"] == "complete" and report["source_unchanged_during_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
