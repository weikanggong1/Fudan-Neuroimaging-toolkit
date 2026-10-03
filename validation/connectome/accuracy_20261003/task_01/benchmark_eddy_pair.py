"""Same-input EDDY baseline/candidate subprocess pairs on private real data.

Official files are consumed only by this isolated benchmark. Production never
reads oracle data. All scientific outputs remain under a fresh private root.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(args):
    import torch
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(20e9/torch.cuda.get_device_properties(0).total_memory)
    from fnit.eddy import TorchEDDY
    from fnit.eddy.fsl2111_strict import geometry
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    result = TorchEDDY(device="cuda:0").run(
        imain=args.reference/"raw/AP.nii.gz", mask=args.reference/"mask/nodif_brain_mask.nii.gz",
        acqp=args.reference/"topup/acqparams.txt", index=args.reference/"eddy/eddy_index.txt",
        bvecs=args.reference/"raw/AP.bvec", bvals=args.reference/"raw/AP.bval",
        topup=args.reference/"topup/fieldmap_out", ref_scan_no=args.ref_scan_no,
        gp_seed=12345,out=args.output/"data")
    torch.cuda.synchronize()
    (args.output/"worker_receipt.json").write_text(json.dumps({
        "wall_seconds":time.perf_counter()-start,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
        "geometry_source_sha256":sha(geometry.__file__),
        "TorchEDDY_source_sha256":sha(sys.modules[TorchEDDY.__module__].__file__),
        "tf32_matmul_restored":torch.backends.cuda.matmul.allow_tf32,
        "torch_version":torch.__version__,"pid":os.getpid(),
        "config":"default niter=8, default fwhm and GP budget, gp_seed=12345",
    },indent=2)+"\n")


def monitor(pid, stopped, rows):
    while not stopped.is_set():
        row={"time":time.monotonic()}
        try:
            # Worker is a single Python process; it creates no GPU children.
            lines=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory",
                    "--format=csv,noheader,nounits"],text=True)
            row["process_tree_bytes"]=sum(int(x.split(", ")[2])*1024**2
                         for x in lines.splitlines() if int(x.split(", ")[0])==pid)
            row["GPU_inventory"]=subprocess.check_output(["nvidia-smi",
                    "--query-gpu=uuid,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).splitlines()
        except Exception as exc:
            row["error"]=repr(exc)
        rows.append(row)
        stopped.wait(.5)


def compare(candidate, reference, maskpath):
    import nibabel as nib
    import numpy as np
    mask=np.asarray(nib.load(str(maskpath)).dataobj)>0
    left=np.asarray(nib.load(str(candidate/"data.nii.gz")).dataobj,dtype=np.float32)
    right=np.asarray(nib.load(str(reference/"data.nii.gz")).dataobj,dtype=np.float32)
    delta=(left[mask].astype(np.float64)-right[mask].astype(np.float64)).reshape(-1)
    bg=np.loadtxt(candidate/"data.eddy_rotated_bvecs").T
    br=np.loadtxt(reference/"data.eddy_rotated_bvecs").T
    good=(np.linalg.norm(bg,axis=1)>0)&(np.linalg.norm(br,axis=1)>0)
    bg=bg[good]/np.linalg.norm(bg[good],axis=1,keepdims=True)
    br=br[good]/np.linalg.norm(br[good],axis=1,keepdims=True)
    angle=np.degrees(np.arccos(np.clip(np.sum(bg*br,axis=1),-1,1)))
    # EDDY maps have an un-commented textual header.
    ol=np.loadtxt(candidate/"data.eddy_outlier_map",skiprows=1)
    ort=np.loadtxt(reference/"data.eddy_outlier_map",skiprows=1)
    return {"brain_RMSE":float(np.sqrt(np.mean(delta*delta))),
            "brain_P99_abs":float(np.percentile(np.abs(delta),99)),
            "brain_max_abs":float(np.max(np.abs(delta))),
            "brain_values":int(delta.size),
            "bvec_max_angle_degrees":float(angle.max()),
            "bvec_RMS_angle_degrees":float(np.sqrt(np.mean(angle*angle))),
            "outlier_map_neq":int(np.count_nonzero(ol!=ort)),
            "corrected_image_SHA256":sha(candidate/"data.nii.gz"),
            "rotated_bvec_SHA256":sha(candidate/"data.eddy_rotated_bvecs")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source",type=Path)
    parser.add_argument("--candidate-source",type=Path)
    parser.add_argument("--reference-root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--subjects",nargs="+",default=["CON01","CON03"])
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--reference",type=Path)
    parser.add_argument("--ref-scan-no",type=int)
    args=parser.parse_args()
    if args.worker:
        worker(args);return
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    results={"scope":"matched EDDY stage including load/save; official field/mask, same GP seed, no budget changes",
             "host":os.uname().nodename,"gpu_uuid":os.environ.get("CUDA_VISIBLE_DEVICES"),
             "shared_GPU_load":True,"stable_speedup_assessed":False,
             "threads":8,"script_SHA256":sha(__file__),"cases":{}}
    lock=open("/tmp/fnit-recon-five-20261002-gongwk.gpu.lock","a+")
    for subject in args.subjects:
        reference=args.reference_root/f"sub-{subject}"
        results["cases"][subject]={"runs":[]}
        inputs=[reference/"raw/AP.nii.gz",reference/"raw/AP.bvec",reference/"raw/AP.bval",
                reference/"topup/acqparams.txt",reference/"topup/fieldmap_out_fieldcoef.nii.gz",
                reference/"topup/fieldmap_out_movpar.txt",reference/"mask/nodif_brain_mask.nii.gz",
                reference/"eddy/eddy_index.txt",reference/"eddy/data.nii.gz"]
        results["cases"][subject]["input_sha256"]={str(p):sha(p) for p in inputs}
        for label,source in [("baseline",args.baseline_source),("candidate",args.candidate_source)]:
            destination=args.output/subject/label;destination.mkdir(parents=True)
            start=time.perf_counter();fcntl.flock(lock,fcntl.LOCK_EX)
            wait=time.perf_counter()-start
            env=dict(os.environ,PYTHONPATH=str(source/"src"),OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",
                     OPENBLAS_NUM_THREADS="8",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            command=[sys.executable,__file__,"--worker","--reference",str(reference),
                     "--output",str(destination),"--ref-scan-no",str(76 if subject=="CON01" else 0)]
            with (destination/"driver.log").open("w") as log:
                start=time.perf_counter();proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                stopped=threading.Event();samples=[]
                thread=threading.Thread(target=monitor,args=(proc.pid,stopped,samples),daemon=True);thread.start()
                rc=proc.wait();elapsed=time.perf_counter()-start;stopped.set();thread.join()
            fcntl.flock(lock,fcntl.LOCK_UN)
            row={"label":label,"process_wall_seconds":elapsed,"lock_wait_seconds":wait,"returncode":rc,
                 "GPU_monitor":{"interval_seconds":.5,"samples":samples,
                   "process_tree_peak_bytes":max((s.get("process_tree_bytes",0) for s in samples),default=None),
                   "max_gap_seconds":max((b["time"]-a["time"] for a,b in zip(samples,samples[1:])),default=None),
                   "failed_samples":sum("error" in s for s in samples)}}
            if rc==0:
                row["worker"]=json.loads((destination/"worker_receipt.json").read_text())
                row["official_comparison"]=compare(destination,reference/"eddy",reference/"mask/nodif_brain_mask.nii.gz")
            results["cases"][subject]["runs"].append(row)
            (args.output/"summary.json").write_text(json.dumps(results,indent=2)+"\n")
            print(subject,label,rc,row.get("official_comparison"),flush=True)
            if rc:
                raise RuntimeError(f"{subject} {label} failed")


if __name__=="__main__":
    main()
