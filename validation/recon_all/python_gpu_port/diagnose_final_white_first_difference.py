"""最终white完整同输入诊断：原生透明probe先过几何门，再比较首步状态。

此脚本保存私有诊断影像/逐步文件，计时含额外写出，不用于性能验收。
候选读取自产七输入；原生参考只用于诊断，不传给优化器。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject",type=Path,required=True)
    p.add_argument("--candidate-directory",type=Path,required=True)
    p.add_argument("--native-probe",type=Path,required=True)
    p.add_argument("--native-reference",type=Path,required=True)
    p.add_argument("--assets-directory",type=Path,required=True)
    p.add_argument("--output-directory",type=Path,required=True)
    p.add_argument("--hemisphere",choices=("lh","rh"),default="lh")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--code-base-commit",required=True)
    a=p.parse_args()
    if a.output_directory.exists(): raise FileExistsError(a.output_directory)
    if not (a.candidate_directory/"place_final_white_python.py").is_file():
        raise FileNotFoundError("candidate directory must directly contain final white module")
    h=a.hemisphere;out=a.output_directory;out.mkdir(parents=True);os.chmod(out,0o700)
    inputs=[Path(f"surf/{h}.white.preaparc"),Path(f"surf/autodet.gw.stats.{h}.dat"),
        *[Path(f"label/{h}.{n}") for n in ("cortex.label","aparc.annot")],
        *[Path(f"mri/{n}.mgz") for n in ("brain.finalsurfs","wm","aseg.presurf")]]
    report={"scope":"same_input_complete_final_white_probe_and_candidate_first_state",
        "input_sha256":{str(v):sha(a.subject/v) for v in inputs},
        "script_sha256":sha(__file__),"probe_sha256":sha(a.native_probe),
        "code_base_commit":a.code_base_commit,"reference_surface_sha256":sha(a.native_reference),
        "threads":a.threads,"device":a.device,"status":"started",
        "timing_scope":"diagnostic with extra checkpoints; not a performance gate"}
    def save():
        tmp=out/"report.tmp";tmp.write_text(json.dumps(report,indent=2)+"\n");tmp.replace(out/"report.json")
    save();native=out/"native_subject";prefix=out/"native_gradient"
    for rel in inputs:
        dest=native/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.subject/rel,dest)
    env=dict(os.environ,PLACE_GRAD_PREFIX=str(prefix),PLACE_VALUES_PREFIX=str(out/"native_targets"))
    started=time.perf_counter()
    with (out/"native.private.log").open("w") as stream:
        process=subprocess.run([sys.executable,"-m","fnit.recon_all.final_white_conda",str(native),h,
            "--binary",str(a.native_probe),"--assets-dir",str(a.assets_directory),"--threads",str(a.threads)],
            env=env,stdout=stream,stderr=stream)
    report["native_probe_seconds_with_extra_IO"]=time.perf_counter()-started
    report["native_returncode"]=process.returncode
    if process.returncode:
        report["status"]="failed_native";save();raise RuntimeError("native probe failed; inspect private log")
    xyz,faces=fs.read_geometry(str(native/f"surf/{h}.white"));ref,rf=fs.read_geometry(str(a.native_reference))
    report["probe_admission"]={"coordinates":compare(xyz,ref),"ordered_faces_equal":bool(np.array_equal(faces,rf))}
    if not np.array_equal(xyz,ref) or not np.array_equal(faces,rf):
        report["status"]="failed_probe_admission";save();raise RuntimeError("probe altered final geometry")
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(a.candidate_directory))
    from fnit.recon_all import place_white_preaparc_python as stage
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    report["TF32_matmul"]=True;report["TF32_cudnn"]=True;report["half_precision"]=False
    trace=[]
    def callback(step,outer,coordinates,diagnostics):
        trace.append({"step":step,"pass":outer,"coordinate_sha256":hashlib.sha256(coordinates.tobytes()).hexdigest(),
                      "diagnostics":diagnostics})
        report["trace"]=trace;save()
    torch.cuda.synchronize(torch.device(a.device));started=time.perf_counter()
    result=stage._place_white_preaparc(subject_dir=a.subject,hemi=h,output=out/f"{h}.white.python",
        steps=400,complete=True,final_white=True,diagnostics=out/"python_complete.npz",
        regularization_backend="cpu",sampling_backend="cpu",candidate_backend="torch_snapshot",
        candidate_grid_cells_per_axis=3,retained_mht_backend="compiled",
        cleanup_marking_backend="source_torch",cleanup_candidate_grid_cells_per_axis=3,
        device=a.device,trace_callback=callback)
    torch.cuda.synchronize(torch.device(a.device))
    report["candidate_seconds_with_extra_IO"]=time.perf_counter()-started
    report["candidate_result"]=result
    report["source_sha256"]={name:sha(m.__file__) for name,m in list(sys.modules.items())
        if name.startswith("fnit.recon_all.place_") and getattr(m,"__file__",None)}
    count=len(xyz)
    with np.load(out/"python_complete.npz") as py:
        clear=native_state(Path(f"{prefix}.step01.clear"),count)
        targets=np.fromfile(f"{prefix}.step01.intensity_input",dtype="<f4").reshape(-1,2)
        first={"initial_coordinates":compare(clear["floats"][:,:3],py["initial"]),
            "rip_flags":compare(clear["flags"][:,0],py["ripped"]),
            "targets":compare(targets[:,0],py["target_values"])}
        for native_term,key in (("intensity","intensity"),("pre_normal_spring","pre_normal_spring"),
                                ("normal_spring","normal_spring"),("curvature","curvature"),
                                ("tangential_spring","tangential_spring"),("after_collision","after_collision")):
            state=native_state(Path(f"{prefix}.step01.{native_term}"),count)
            values=state["floats"][:,:3] if native_term=="after_collision" else state["floats"][:,6:9]
            first[native_term]=compare(values,py[key])
        report["first_step_comparison"]=first
    y,yf=fs.read_geometry(str(out/f"{h}.white.python"))
    report["final_comparison"]={"coordinates":compare(y,xyz),"ordered_faces_equal":bool(np.array_equal(yf,faces))}
    report["input_sha256_after"]={str(v):sha(a.subject/v) for v in inputs}
    report["status"]="complete" if report["input_sha256"]==report["input_sha256_after"] else "input_changed"
    save();print(json.dumps({"status":report["status"],"first_step":report["first_step_comparison"]}))


if __name__=="__main__":main()
