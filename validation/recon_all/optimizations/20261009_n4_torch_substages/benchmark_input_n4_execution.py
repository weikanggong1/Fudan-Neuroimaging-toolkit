"""两例原始T1新空目录的isolated N4输入链，对照已测完整in-process链。

--previous-raw-pair只在候选完成后用于体素/几何/LTA诊断；计算只读
原始T1、已声明权重/资产和自产中间。--config绑定原始SHA，--output新目录。
缓存关闭的父CUDA可已初始化；完整阶段包含exec、传输、压缩写出及退出。
复用既有比较器，不将旧组负载下的时间当成同刻配对吞吐或完整recon-all。
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import sys
import threading
import time

import nibabel as nib
import numpy as np
import torch

from benchmark_input_chain import metrics,lta_matrix_comparison
from fnit.recon_all.input_n4_chain import run_input_n4_chain,validate_n4_execution


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","previous-raw-pair","weights","assets","native","output","profiling-module"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",default="cuda:0");parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--seed",type=int,default=1729);parser.add_argument("--code-version",required=True)
    args=parser.parse_args()
    validate_n4_execution(n4_backend="torch",n4_execution="isolated",device=args.device)
    if args.output.exists():raise FileExistsError(args.output)
    old=json.loads((args.previous_raw_pair/"summary.json").read_text())
    if old["status"]!="complete_raw_input_to_nu_pair":raise ValueError("previous raw pair is incomplete")
    entries=json.loads(args.config.read_text())
    for entry in entries:
        if sha(entry["input"])!=entry["sha256"]:raise ValueError("raw input SHA changed")
        prior=next(row for row in old["rows"] if row["case"]==entry["case"])
        if prior["sha256"]!=entry["sha256"]:raise ValueError("paired original T1 differs")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    from numba import set_num_threads
    set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    spec=importlib.util.spec_from_file_location("n4_execution_sampler",args.profiling_module)
    profiling=importlib.util.module_from_spec(spec);spec.loader.exec_module(profiling)
    sampler=profiling.ProcessTreeDeviceSampler(device=args.device,parent_pid=os.getpid(),interval=.5)
    stop=threading.Event()
    def sample():
        while not stop.is_set():sampler.sample_if_due(force=True);stop.wait(.5)
    background=threading.Thread(target=sample,daemon=True);background.start()
    report={"scope":"raw_T1_empty_directory_isolated_N4_to_nu_not_whole_recon_all",
        "status":"running","rows":[],"host":socket.gethostname(),"code_version":args.code_version,
        "device":args.device,"threads":args.threads,"cpu_affinity":sorted(os.sched_getaffinity(0)),"seed":args.seed,
        "previous_raw_report_sha256":sha(args.previous_raw_pair/"summary.json"),
        "source_sha256":{str(Path(__file__)):sha(__file__),str(args.profiling_module):sha(args.profiling_module)},
        "resource_sha256":{str(path):sha(path) for path in (args.native,args.weights/"synthstrip.1.pt",
            args.weights/"synthmorph.affine.2.h5",args.assets/"average/mni305.cor.stripped.mgz")},
        "parent_allocator_environment":{name:os.environ.get(name) for name in
            ("CUDA_VISIBLE_DEVICES","PYTORCH_NO_CUDA_MEMORY_CACHING","PYTORCH_CUDA_ALLOC_CONF")},
        "precision_policy":"TF32 default; existing SynthStrip cuDNN/Talairach FP32 exceptions; no half",
        "production_default_changed":False,"whole_recon_all_acceleration":"not measured",
        "whole_metrics_equivalence":"not assessed","timing_limit":"new shared-load observation vs earlier same-hardware/thread run, not contemporaneous ABBA"}
    def save():
        path=args.output/"summary.json";pending=path.with_suffix(".pending")
        pending.write_text(json.dumps(report,indent=2)+"\n");pending.replace(path)
    try:
        for entry in entries:
            random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
            subject=args.output/entry["case"]/"subject"
            if subject.exists():raise FileExistsError(subject)
            torch.cuda.synchronize(args.device)
            begin=time.perf_counter()
            result=run_input_n4_chain(t1=entry["input"],subject_dir=subject,weights_dir=args.weights,assets_dir=args.assets,
                device=args.device,threads=args.threads,n4_backend="torch",n4_execution="isolated",profile=False)
            torch.cuda.synchronize(args.device)
            row={**entry,"wall_seconds_including_load_transfer_exec_io":time.perf_counter()-begin,"report":result,
                 "output_sha256":{str(path.relative_to(subject)):sha(path) for path in subject.rglob("*") if path.is_file()}}
            report["rows"].append(row);save()
            # Only after the candidate writes all outputs, load previous outputs for diagnostic comparison.
            reference=args.previous_raw_pair/entry["case"]/"torch/subject"
            native=args.previous_raw_pair/entry["case"]/"native/subject"
            brain=np.asarray(nib.load(str(reference/"mri/synthstrip.mgz")).dataobj)>0
            paths=("mri/rawavg.mgz","mri/orig.mgz","mri/synthstrip.mgz","mri/tmp/nu0.mgz","mri/nu.mgz")
            row["vs_prior_complete_inprocess"]={}
            for path in paths:
                # rawavg remains in the original imported grid. A conform-grid
                # SynthStrip mask is invalid there; report whole-grid values only.
                mask=np.ones(nib.load(str(subject/path)).shape,bool) if path=="mri/rawavg.mgz" else brain
                value=metrics(subject/path,reference/path,mask)
                if path=="mri/rawavg.mgz":
                    value.pop("brain",None);value.pop("outside_brain",None)
                    value["brain_definition"]="not assessed: raw imported grid; whole-grid comparison only"
                else:value["brain_definition"]="previous self-produced complete Torch chain SynthStrip image>0, not atlas labels"
                row["vs_prior_complete_inprocess"][path]=value
            row["vs_prior_native"]={path:metrics(subject/path,native/path,brain) for path in ("mri/tmp/nu0.mgz","mri/nu.mgz")}
            for value in row["vs_prior_native"].values():
                value["brain_definition"]="previous self-produced complete Torch chain SynthStrip image>0; previously matched native strip exactly"
            row["talairach_xfm_text_identical"]=(subject/"mri/transforms/talairach.xfm").read_bytes()==(reference/"mri/transforms/talairach.xfm").read_bytes()
            row["talairach_lta"]={name:lta_matrix_comparison(subject/name,reference/name) for name in
                ("mri/transforms/synthmorph.mni305/aff.lta","mri/transforms/talairach.xfm.lta")}
            old_row=next(item for item in old["rows"] if item["case"]==entry["case"])
            old_torch=next(item for item in old_row["runs"] if item["backend"]=="torch")
            row["prior_complete_inprocess_wall_seconds"]=old_torch["wall_seconds_including_load_transfer_and_io"]
            row["prior_n4_seconds"]=old_torch["report"]["n4_seconds"]
            save();print("DONE",entry["case"],row["wall_seconds_including_load_transfer_exec_io"],flush=True)
        report["status"]="complete_two_raw_t1_isolated_input_chains"
    except Exception as error:
        report.update(status="failed",error=repr(error));raise
    finally:
        stop.set();background.join(timeout=10);sampler.sample_if_due(force=True)
        report["gpu_process_sampler"]=sampler.report()
        report["experiment_wall_seconds_including_validation_context_diagnostics"]=time.perf_counter()-started
        report["each_api_timing_boundary"]="GPU sync before/after API, initialized parent CUDA; Python imports excluded, child exec imports included"
        for name,module in tuple(sys.modules.items()):
            path=getattr(module,"__file__",None)
            if name.startswith("fnit.") and path and str(path).endswith(".py"):report["source_sha256"][str(path)]=sha(path)
        save()


if __name__=="__main__":main()
