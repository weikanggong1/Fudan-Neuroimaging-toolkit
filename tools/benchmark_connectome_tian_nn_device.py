"""真实新下载 T1 上固定 SynthMorph warp 的 CPU/CUDA NN 逐值回归。

这里只验证 NN 设备迁移；不替代完整 raw pipeline 或官方 atlas 对照。
配准使用未改变的 FNIT joint 默认设置，随后 S1/S4 同 warp AB/BA；
GPU 锁等待在外层调度计时。--t1 可为本轮 raw T1 或自产 recon brain。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import nibabel as nib
import numpy as np
import torch

from fnit.synthmorph import SynthMorph, apply_transform
from fnit.weights import WEIGHT_FILES, verify_file


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("t1", "mni", "atlas-dir", "weights", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    verified_weights = {}
    for name in ("synthmorph.affine.2.h5", "synthmorph.deform.3.h5"):
        _, size, digest = WEIGHT_FILES[name]
        if not verify_file(a.weights/name, size, digest): raise ValueError("weight SHA mismatch: " + name)
        verified_weights[name] = {"size": size, "sha256": digest}
    # Independent NVML sampling does not reset when a subfunction resets torch peaks.
    samples=[]; failures=[]; stopped=threading.Event()
    def sample():
        while not stopped.is_set():
            try:
                rows = subprocess.check_output(["nvidia-smi",
                    "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                    text=True, timeout=2).splitlines()
                own = [row.split(",") for row in rows if int(row.split(",")[0].strip()) == os.getpid()]
                memory = sum(int(row[2].strip()) * 1024**2 for row in own)
                samples.append((time.monotonic(), memory, [row[1].strip() for row in own]))
            except Exception as e:failures.append(type(e).__name__)
            stopped.wait(.2)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.empty(1, device=a.device)  # Initialize the explicit CUDA device before resetting peaks.
    torch.cuda.reset_peak_memory_stats(a.device)
    report={"scope":"newly downloaded raw T1 NN component only; not recon/raw end-to-end",
            "declared_tolerance":{"label_neq":0}, "inputs":{"t1":sha256(a.t1),"mni":sha256(a.mni)},
            "source_sha256":{"synthmorph_pipeline":sha256(Path(__file__).resolve().parents[1]/"src/fnit/synthmorph/pipeline.py"),
                             "benchmark_tool":sha256(__file__)},
            "runs":[], "differences":{}, "verified_weights":verified_weights,
            "torch_version":torch.__version__, "threads":torch.get_num_threads(),
            "allocator_no_cache_environment":os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
            "torch_memory_stats_valid":os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") is None,
            "timing_note":"Weight SHA verification precedes registration timing; model load is included. Lock wait excluded."}
    print("verified weights; starting unchanged joint registration", flush=True)
    started=time.perf_counter()
    model=SynthMorph(weights=a.weights,device=a.device,model="joint")
    precision=[]
    transform=model(moving=nib.load(a.mni),fixed=nib.load(a.t1),precision_report=precision).transform
    torch.cuda.synchronize(a.device)
    report["registration_seconds"]=time.perf_counter()-started;report["precision"]=precision
    transform.save(a.output_dir/"mni_to_t1.nii.gz")
    print("joint registration finished; starting fixed-warp NN AB/BA", flush=True)
    for level in (1,4):
        path=a.atlas_dir/f"Tian_Subcortex_S{level}_3T.nii.gz";atlas=nib.load(path)
        report["inputs"][f"tian_s{level}"]=sha256(path);results={}
        for order in (("cpu",a.device),(a.device,"cpu")):
            for device in order:
                torch.cuda.synchronize(a.device);started=time.perf_counter()
                output=apply_transform(image=atlas,transformation=transform,method="nearest",dtype="int16",device=device)
                torch.cuda.synchronize(a.device);wall=time.perf_counter()-started
                data=np.asarray(output.dataobj)
                results.setdefault(device,data.copy())
                report["runs"].append({"level":level,"device":device,"wall_seconds":wall})
        cpu,gpu=results["cpu"],results[a.device]
        # Keep labels on the server for a separate figure environment; no raw T1 export.
        t1_affine = nib.load(a.t1).affine
        nib.save(nib.Nifti1Image(cpu, t1_affine), a.output_dir/f"tian_s{level}_cpu.nii.gz")
        nib.save(nib.Nifti1Image(gpu, t1_affine), a.output_dir/f"tian_s{level}_cuda.nii.gz")
        report["differences"][f"s{level}"]={"neq":int(np.count_nonzero(cpu!=gpu)),"voxels":cpu.size,
                                           "shape":list(cpu.shape),"dtype":str(cpu.dtype),
                                           "foreground_voxels":int(np.count_nonzero(cpu)),
                                           "positive_labels":np.unique(cpu[cpu>0]).astype(int).tolist()}
    report["torch_peak_bytes"]={"allocated":torch.cuda.max_memory_allocated(a.device),"reserved":torch.cuda.max_memory_reserved(a.device)}
    stopped.set();thread.join()
    report["nvml"]={"peak_process_bytes":max((x[1] for x in samples),default=None),"samples":len(samples),"sampler":"nvidia-smi NVML process query across GPUs; MiB resolution",
                     "gpu_uuids":sorted({uuid for sample in samples for uuid in sample[2]}),
                     "max_interval_seconds":max((samples[i][0]-samples[i-1][0] for i in range(1,len(samples))),default=None),"failures":failures}
    report["label_parity_passed"]=all(x["neq"]==0 for x in report["differences"].values())
    peak = report["nvml"]["peak_process_bytes"]
    report["memory_budget_passed"] = (peak is not None and 0 < peak < 20_000_000_000
        and len(report["nvml"]["gpu_uuids"]) == 1
        and max(report["torch_peak_bytes"].values()) < 20_000_000_000)
    report["passed"] = report["label_parity_passed"] and report["memory_budget_passed"]
    (a.output_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    if not report["label_parity_passed"]:raise SystemExit("NN label mismatch")
    if not report["memory_budget_passed"]:raise SystemExit("memory budget not verified")


if __name__=="__main__":main()
