"""outside1mm 精度修复的完整 white.preaparc 真输入回归。

候选仅读取自己复制的原始阶段输入；独立参考与逐轮探针只用于运行后比较。
不改变生产调度，固定四轮/有序碰撞/源相交清理，时间包含加载与写出。
"""
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

import nibabel.freesurfer.io as fs
import numpy as np
import torch

from diagnose_pial_first_difference import compare,native_state
from diagnose_preaparc_current_input import mesh,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject",type=Path,required=True)
    p.add_argument("--hemisphere",choices=("lh","rh"),required=True)
    p.add_argument("--candidate-directory",type=Path,required=True)
    p.add_argument("--native-binary",type=Path,required=True)
    p.add_argument("--assets-directory",type=Path,required=True)
    p.add_argument("--previous-reference-surface",type=Path,required=True)
    p.add_argument("--output-directory",type=Path,required=True)
    p.add_argument("--reuse-reference-report",type=Path)
    p.add_argument("--native-gradient-prefix",type=Path)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--code-base-commit",required=True)
    a=p.parse_args();h=a.hemisphere;out=a.output_directory
    if out.exists():raise FileExistsError(out)
    out.mkdir(mode=0o700,parents=True)
    inputs=[f"surf/{h}.{n}" for n in ("orig","orig.premesh")]
    inputs += [f"mri/{n}.mgz" for n in ("brain.finalsurfs","wm","aseg.presurf")]
    input_sha={v:sha(a.subject/v) for v in inputs}
    report=dict(scope="full_same_input_preaparc_precision_fix_not_raw_T1",
        code_base_commit=a.code_base_commit,script_sha256=sha(__file__),
        hostname=platform.node(),cpu_affinity=sorted(os.sched_getaffinity(0)),threads=a.threads,
        device=a.device,input_sha256=input_sha,native_program_sha256=sha(a.native_binary),
        reference={},candidate={},status="started",overall_metric_equivalence="not_assessed")
    def save():
        temp=out/"report.pending";temp.write_text(json.dumps(report,indent=2)+"\n");temp.replace(out/"report.json")
    def copy_subject(name):
        subject=out/name
        for rel in inputs:
            dest=subject/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.subject/rel,dest)
        return subject
    save()
    if a.reuse_reference_report:
        d=json.loads(a.reuse_reference_report.read_text())
        if d["status"]!="complete" or d["input_sha256"]!=input_sha or d["native_program_sha256"]!=sha(a.native_binary):
            raise RuntimeError("reused reference receipt input/program mismatch")
        row=d["reference"]["native_repeat"]
        if row["vs_previous_reference"]["coordinates"]["different_elements"] or not row["vs_previous_reference"]["ordered_faces_equal"]:
            raise RuntimeError("reference repetition not stable")
        report["reference"]=dict(reused_receipt_sha256=sha(a.reuse_reference_report),
            native_repeat=row,reference_reproducibility="same_host_same_input_repetition_exact")
    else:
        subject=copy_subject("native_repeat");started=time.perf_counter()
        with (out/"native.private.log").open("w") as stream:
            proc=subprocess.run([sys.executable,"-m","fnit.recon_all.white_preaparc_conda",str(subject),h,
                "--binary",str(a.native_binary),"--assets-dir",str(a.assets_directory),"--threads",str(a.threads)],
                stdout=stream,stderr=stream)
        row=dict(returncode=proc.returncode,seconds_including_IO=time.perf_counter()-started)
        if proc.returncode:
            report.update(reference=row,status="failed_reference");save();raise RuntimeError("native repeat failed")
        row["vs_previous_reference"]=mesh(subject/f"surf/{h}.white.preaparc",a.previous_reference_surface)
        if row["vs_previous_reference"]["coordinates"]["different_elements"] or not row["vs_previous_reference"]["ordered_faces_equal"]:
            report.update(reference=row,status="failed_reference_stability");save();raise RuntimeError("reference unstable")
        report["reference"]=dict(native_repeat=row,reference_reproducibility="same_host_same_input_repetition_exact")
    save()
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(a.candidate_directory))
    from fnit.recon_all import place_white_preaparc_python as stage
    from fnit.recon_all.autodet_gwstats_python import write_autodet_stats
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    subject=copy_subject("candidate");report["candidate"]=dict(status="running",trace=[])
    save();native_xyz,_=fs.read_geometry(str(a.previous_reference_surface))
    def callback(step,outer,xyz,details):
        row=dict(step=step,pass_index=outer,coordinates_sha256=hashlib.sha256(xyz.tobytes()).hexdigest(),details=details)
        if a.native_gradient_prefix:
            path=Path(f"{a.native_gradient_prefix}.step{step:02d}.after_collision")
            if not path.is_file():raise FileNotFoundError(path)
            row["vs_admitted_native_accepted_coordinates"]=compare(native_state(path,len(xyz))["floats"][:,:3],xyz)
        report["candidate"]["trace"].append(row);save()
    torch.cuda.synchronize(a.device);torch.cuda.reset_peak_memory_stats(a.device)
    started=time.perf_counter()
    try:
        stats=subject/f"surf/autodet.gw.stats.{h}.dat"
        write_autodet_stats(brain_path=subject/"mri/brain.finalsurfs.mgz",
            wm_path=subject/"mri/wm.mgz",surface_path=subject/f"surf/{h}.orig.premesh",output_path=stats,hemisphere=h)
        result=stage.place_white_preaparc(subject_dir=subject,hemi=h,output=subject/f"surf/{h}.white.preaparc",
            output_volume=subject/"mri/mrisps.wpa.mgz",max_steps=400,
            regularization_backend="cpu",sampling_backend="cpu",candidate_backend="torch_snapshot",
            candidate_grid_cells_per_axis=3,retained_mht_backend="compiled",
            cleanup_marking_backend="source_torch",cleanup_candidate_grid_cells_per_axis=3,
            device=a.device,trace_callback=callback)
        torch.cuda.synchronize(a.device)
    except Exception as exc:
        report["candidate"].update(status="failed",exception_type=type(exc).__name__)
        report["status"]="failed_candidate";save();raise
    report["candidate"].update(status="complete",seconds_including_stats_IO=time.perf_counter()-started,
        result=result,final=mesh(subject/f"surf/{h}.white.preaparc",a.previous_reference_surface),
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device),
        process_tree_gpu_memory="not_measured; torch allocator is not process tree memory")
    report["precision"]=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,autocast=False,half=False)
    report["allocator_environment"]={k:os.environ.get(k) for k in
        ("PYTORCH_NO_CUDA_MEMORY_CACHING","PYTORCH_CUDA_ALLOC_CONF","PYTORCH_ALLOC_CONF")}
    report["candidate_source_sha256"]={name:sha(m.__file__) for name,m in list(sys.modules.items())
        if name.startswith("fnit.recon_all.place_") and getattr(m,"__file__",None)}
    report["input_sha256_after"]={v:sha(a.subject/v) for v in inputs}
    report["status"]="complete" if report["input_sha256_after"]==input_sha else "input_changed"
    save()


if __name__=="__main__":main()
