"""自产 orig/MRI 的完整 preaparc 首差诊断，固定参考重复及透明逐步回放。

该工具只用于隔离 benchmark。参考及候选都复制同一组自产输入；候选
不读取参考结果。逐步快照只增加诊断读写，墙钟不能当作生产性能。
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

from diagnose_pial_first_difference import compare, native_state


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mesh(a, b):
    x, f = fs.read_geometry(str(a)); y, g = fs.read_geometry(str(b))
    return {"ordered_faces_equal": bool(np.array_equal(f, g)),
            "coordinates": compare(x, y), "first_sha256": sha(a), "second_sha256": sha(b)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=Path, required=True)
    p.add_argument("--candidate-directory", type=Path, required=True)
    p.add_argument("--native-probe", type=Path, required=True)
    p.add_argument("--native-binary", type=Path, required=True)
    p.add_argument("--previous-reference-surface", type=Path, required=True)
    p.add_argument("--assets-directory", type=Path, required=True)
    p.add_argument("--output-directory", type=Path, required=True)
    p.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--code-base-commit", required=True)
    p.add_argument("--candidate-backends", nargs="+", choices=("torch_snapshot", "snapshot"),
                   default=("torch_snapshot", "snapshot"))
    a = p.parse_args()
    out, h = a.output_directory, a.hemisphere
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True, mode=0o700)
    inputs = [Path(f"surf/{h}.{n}") for n in ("orig", "orig.premesh")]
    inputs += [Path(f"mri/{n}.mgz") for n in ("brain.finalsurfs", "wm", "aseg.presurf")]
    report = dict(scope="frozen_self_generated_input_full_preaparc_diagnostic_not_raw_T1",
        hostname=platform.node(), cpu_affinity=sorted(os.sched_getaffinity(0)), threads=a.threads,
        device=a.device, code_base_commit=a.code_base_commit, script_sha256=sha(__file__),
        input_sha256={str(v): sha(a.subject/v) for v in inputs},
        native_program_sha256=sha(a.native_binary), native_probe_sha256=sha(a.native_probe),
        timing_scope="full diagnostics with extra snapshots; not production performance",
        reference={}, candidate={}, status="started", overall_metric_equivalence="not_assessed")

    def save():
        temp=out/"report.pending";temp.write_text(json.dumps(report,indent=2)+"\n");temp.replace(out/"report.json")

    def copy_subject(name):
        subject=out/name
        for rel in inputs:
            dest=subject/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.subject/rel,dest)
        return subject

    save()
    prefix=out/"native_gradient"
    # 未插桩参考先重复，透明 probe 只有通过完整几何门后可用作状态参考。
    for name, binary in (("native_repeat", a.native_binary), ("native_probe", a.native_probe)):
        subject=copy_subject(name)
        env=dict(os.environ)
        if name=="native_probe":
            env.update(PLACE_GRAD_PREFIX=str(prefix), PLACE_VALUES_PREFIX=str(out/"native_targets"))
        started=time.perf_counter()
        with (out/f"{name}.private.log").open("w") as stream:
            proc=subprocess.run([sys.executable,"-m","fnit.recon_all.white_preaparc_conda",str(subject),h,
                "--binary",str(binary),"--assets-dir",str(a.assets_directory),"--threads",str(a.threads)],
                env=env,stdout=stream,stderr=stream)
        row=dict(returncode=proc.returncode,seconds_including_IO=time.perf_counter()-started)
        report["reference"][name]=row
        if proc.returncode:
            report["status"]="failed_reference";save();raise RuntimeError(name+" failed")
        row["vs_previous_reference"]=mesh(subject/f"surf/{h}.white.preaparc",a.previous_reference_surface)
        if row["vs_previous_reference"]["coordinates"].get("different_elements"):
            report["status"]="failed_reference_stability_or_probe_admission";save()
            raise RuntimeError("reference geometry differs; keep all evidence")
        save()
    native_subject=out/"native_probe"
    native_xyz,native_faces=fs.read_geometry(str(native_subject/f"surf/{h}.white.preaparc"))
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(a.candidate_directory))
    from fnit.recon_all import place_white_preaparc_python as stage
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    report["precision"]=dict(dtype="float32 plus declared float64 geometry",matmul_tf32=True,
        cudnn_tf32=True,autocast=False)
    report["allocator_environment"]={k:os.environ.get(k) for k in
        ("PYTORCH_NO_CUDA_MEMORY_CACHING","PYTORCH_CUDA_ALLOC_CONF","PYTORCH_ALLOC_CONF")}
    for backend in a.candidate_backends:
        subject=copy_subject(backend)
        # 从候选自产 MRI/premesh 独立计算阈值，不读取参考结果。
        stats=f"surf/autodet.gw.stats.{h}.dat"
        from fnit.recon_all.autodet_gwstats_python import write_autodet_stats
        write_autodet_stats(subject/"mri/brain.finalsurfs.mgz",subject/"mri/wm.mgz",
                           subject/f"surf/{h}.orig.premesh",subject/stats,h)
        row=dict(status="running",trace=[],input_sha256={str(v):sha(subject/v) for v in inputs},
                 threshold_sha256=sha(subject/stats))
        report["candidate"][backend]=row;save()
        state={"step":0,"part":{}}
        originals={}
        def wrap(name,recorder):
            original=getattr(stage,name);originals[name]=original
            def call(*args,**kwargs):
                result=original(*args,**kwargs);recorder(args,kwargs,result);return result
            setattr(stage,name,call)
        def intensity(args,kwargs,result):
            state["step"]+=1
            state["part"]={"current":args[1].copy(),"normals":args[2].copy(),"ripped":args[3].copy(),
                "targets":args[4].copy(),"sigmas":args[5].copy(),"intensity":result.copy(),"trials":[]}
        wrap("intensity_gradient",intensity)
        wrap("self_repulsion_gradient",lambda args,kwargs,r:state["part"].update(repulsion=r.copy()))
        wrap("average_signed_gradients",lambda args,kwargs,r:state["part"].update(averaged=r.copy()))
        wrap("spring_gradient",lambda args,kwargs,r:state["part"].update({kwargs["direction"]:r.copy()}))
        wrap("quadratic_curvature",lambda args,kwargs,r:state["part"].update(curvature=r.copy()))
        def collision(args,kwargs,result):
            state["part"]["trials"].append((args[2].copy(),result[0].copy()))
        wrap("asynchronous_first_step",collision)
        def callback(step,outer,coordinates,diagnostics):
            part=state["part"];arrays={k:v for k,v in part.items() if k!="trials"}
            arrays["accepted"]=coordinates
            for i,(proposal,candidate) in enumerate(part["trials"]):
                arrays[f"proposal_{i}"]=proposal;arrays[f"candidate_{i}"]=candidate
            path=out/f"{backend}.step{step:02d}.npz";np.savez(path,**arrays)
            row["trace"].append(dict(step=step,pass_index=outer,diagnostics=diagnostics,
                coordinate_sha256=hashlib.sha256(coordinates.tobytes()).hexdigest(),snapshot_sha256=sha(path)))
            save()
        gpu=backend=="torch_snapshot"
        if gpu:torch.cuda.synchronize(a.device)
        started=time.perf_counter()
        try:
            result=stage.place_white_preaparc(subject_dir=subject,hemi=h,
                output=subject/f"surf/{h}.white.preaparc",output_volume=subject/"mri/mrisps.wpa.mgz",
                max_steps=400,regularization_backend="cpu",sampling_backend="cpu",candidate_backend=backend,
                candidate_grid_cells_per_axis=3 if gpu else 2,retained_mht_backend="compiled",
                cleanup_marking_backend="source_torch" if gpu else "source_numba",
                cleanup_candidate_grid_cells_per_axis=3 if gpu else 2,device=a.device if gpu else "cpu",
                trace_callback=callback)
            if gpu:torch.cuda.synchronize(a.device)
        finally:
            for name,original in originals.items():setattr(stage,name,original)
        row.update(status="complete",seconds_including_diagnostic_IO=time.perf_counter()-started,
            result=result,final=mesh(subject/f"surf/{h}.white.preaparc",native_subject/f"surf/{h}.white.preaparc"))
        comparisons=[]
        for tr in row["trace"]:
            step=tr["step"];base=Path(f"{prefix}.step{step:02d}")
            with np.load(out/f"{backend}.step{step:02d}.npz") as py:
                if not Path(f"{base}.clear").exists():
                    comparisons.append(dict(step=step,status="native_capture_unavailable"));continue
                clear=native_state(Path(f"{base}.clear"),len(native_xyz))
                after=native_state(Path(f"{base}.after_collision"),len(native_xyz))
                item=dict(step=step,entry_coordinates=compare(clear["floats"][:,:3],py["current"]),
                    accepted_coordinates=compare(after["floats"][:,:3],py["accepted"]),
                    entry_normals=compare(clear["floats"][:,3:6],py["normals"]),
                    rip_flags=compare(clear["flags"][:,0],py["ripped"]))
                targets=np.fromfile(f"{base}.intensity_input",dtype="<f4").reshape(-1,2)
                item["targets"]=compare(targets[:,0],py["targets"]);item["sigmas"]=compare(targets[:,1],py["sigmas"])
                cumulative=np.float32(py["averaged"]+py["repulsion"])
                expected={"intensity":py["intensity"],"pre_normal_spring":cumulative,
                          "normal_spring":np.float32(cumulative+py["normal"])}
                expected["curvature"]=np.float32(expected["normal_spring"]+np.float32(py["curvature"][:,None]*py["normals"]))
                expected["tangential_spring"]=np.float32(expected["curvature"]+py["tangent"])
                for name,val in expected.items():
                    path=Path(f"{base}.{name}")
                    if path.exists():item[name]=compare(native_state(path,len(native_xyz))["floats"][:,6:9],val)
                comparisons.append(item)
        row["per_step_comparison"]=comparisons
        row["first_different_accepted_coordinate_step"]=next((v["step"] for v in comparisons if v.get("accepted_coordinates",{}).get("different_elements")),None)
        save()
    report["candidate_source_sha256"]={name:sha(m.__file__) for name,m in list(sys.modules.items())
        if name.startswith("fnit.recon_all.place_") and getattr(m,"__file__",None)}
    report["input_sha256_after"]={str(v):sha(a.subject/v) for v in inputs}
    report["status"]="complete" if report["input_sha256_after"]==report["input_sha256"] else "input_changed"
    save()


if __name__=="__main__":main()
