"""完整透明目标探针录取后，以相同逐轮输入复查 CBV 与有序平滑。

仅用于隔离诊断；候选自产 MRI 预处理后逐体素核验，参考状态不得用于生产。
完整 probe 包含读写但增加快照开销，因此耗时不能当作性能验收。
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

import nibabel as nib
import numpy as np

from diagnose_pial_first_difference import compare


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(path, count):
    dtype=np.dtype([("floats", "<f4", 13), ("flags", "<i4", 4)])
    result=np.fromfile(path, dtype=dtype)
    if len(result)!=count:
        raise ValueError("native target checkpoint size differs")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject",type=Path,required=True)
    p.add_argument("--reference-surface",type=Path,required=True)
    p.add_argument("--native-probe",type=Path,required=True)
    p.add_argument("--assets-directory",type=Path,required=True)
    p.add_argument("--output-directory",type=Path,required=True)
    p.add_argument("--hemisphere",choices=("lh","rh"),default="lh")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--code-base-commit",required=True)
    a=p.parse_args();out=a.output_directory;h=a.hemisphere
    if out.exists():raise FileExistsError(out)
    out.mkdir(mode=0o700,parents=True)
    subject=out/"native_probe";inputs=[f"surf/{h}.{n}" for n in ("orig","orig.premesh")]
    inputs += [f"mri/{n}.mgz" for n in ("brain.finalsurfs","wm","aseg.presurf")]
    for rel in inputs:
        dest=subject/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.subject/rel,dest)
    report=dict(scope="same_input_all_target_pass_diagnostic_not_production",
        code_base_commit=a.code_base_commit, script_sha256=sha(__file__),
        input_sha256={v:sha(a.subject/v) for v in inputs},
        native_probe_sha256=sha(a.native_probe),status="started",passes=[])
    def save():
        temp=out/"report.pending";temp.write_text(json.dumps(report,indent=2)+"\n");temp.replace(out/"report.json")
    save();prefix=out/"native_target"
    env=dict(os.environ,PLACE_TARGET_PASS_PREFIX=str(prefix))
    started=time.perf_counter()
    with (out/"native.private.log").open("w") as stream:
        process=subprocess.run([sys.executable,"-m","fnit.recon_all.white_preaparc_conda",str(subject),h,
            "--binary",str(a.native_probe),"--assets-dir",str(a.assets_directory),"--threads",str(a.threads)],
            stdout=stream,stderr=stream,env=env)
    x,f=nib.freesurfer.read_geometry(str(subject/f"surf/{h}.white.preaparc"))
    y,g=nib.freesurfer.read_geometry(str(a.reference_surface))
    report["probe_admission"]=dict(returncode=process.returncode,
        seconds_including_extra_IO=time.perf_counter()-started,ordered_faces_equal=bool(np.array_equal(f,g)),
        coordinates=compare(x,y))
    save()
    if process.returncode or not np.array_equal(f,g) or not np.array_equal(x,y):
        report["status"]="failed_probe_admission";save();raise RuntimeError("probe output differs")
    from fnit.recon_all.place_surface_border import compute_border_values_first_pass
    from fnit.recon_all.place_surface_smoothing import average_marked_values
    from fnit.recon_all.place_surface_volume import prepare_placement_volume
    from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
    brain=nib.load(str(a.subject/"mri/brain.finalsurfs.mgz"))
    wm=nib.load(str(a.subject/"mri/wm.mgz"));seg=nib.load(str(a.subject/"mri/aseg.presurf.mgz"))
    stats_path=subject/f"surf/autodet.gw.stats.{h}.dat"
    stats=dict(line.split()[:2] for line in stats_path.read_text().splitlines() if len(line.split())>=2)
    volume,_=prepare_placement_volume(np.asarray(brain.dataobj),np.asarray(wm.dataobj),
        surface="white",mid_gray=float(stats["MID_GRAY"]))
    native_volume=nib.load(str(subject/"mri/mrisps.wpa.mgz"))
    report["own_prepared_volume_vs_native"]=compare(volume,np.asarray(native_volume.dataobj))
    _,_,metadata=nib.freesurfer.read_geometry(str(a.subject/f"surf/{h}.orig"),read_metadata=True)
    affine=surface_ras_to_voxel(brain.header,metadata)
    thresholds=np.array([float(stats[f"white_{n}"]) for n in
        ("inside_hi","border_hi","border_low","outside_low","outside_hi")])
    original=state(f"{prefix}.pass00.entry",len(x))["floats"][:,:3].copy()
    for i in range(4):
        entry=state(f"{prefix}.pass{i:02d}.entry",len(x))
        before=state(f"{prefix}.pass{i:02d}.before_average",len(x))
        after=state(f"{prefix}.pass{i:02d}.after_average",len(x))
        ripped=before["flags"][:,0].astype(bool);active=~ripped
        started=time.perf_counter()
        cbv=compute_border_values_first_pass(volume,np.asarray(seg.dataobj),
            before["floats"][:,:3],before["floats"][:,3:6],original,ripped,
            entry["floats"][:,6],affine,thresholds,hemisphere=h,surface="white",sigma=2.0/(1<<i))
        native_smoothed_replay=average_marked_values(before["floats"][:,6],before["flags"][:,1],
            ripped,f,5,device="cpu")
        own_smoothed=average_marked_values(cbv[0],cbv[4],ripped,f,5,device="cpu")
        names=("values","distance","mean","target_coordinates","marked","sigma")
        native=(before["floats"][:,6],before["floats"][:,8],before["floats"][:,9],
            before["floats"][:,10:13],before["flags"][:,1],before["floats"][:,7])
        row=dict(pass_index=i,active_vertices=int(active.sum()),
            replay_seconds=time.perf_counter()-started,
            raw={name:compare(n[active],p[active]) for name,n,p in zip(names,native,cbv)},
            averaging_same_native_input=compare(after["floats"][active,6],native_smoothed_replay[active]),
            averaging_self_border_input=compare(after["floats"][active,6],own_smoothed[active]))
        different=np.flatnonzero(active & (native[0]!=cbv[0]))
        row["raw_different_value_vertices"]=different.tolist()
        row["different_value_examples"]=[dict(vertex=int(v),native=float(native[0][v]),python=float(cbv[0][v]),
            native_distance=float(native[1][v]),python_distance=float(cbv[1][v])) for v in different[:20]]
        np.savez(out/f"pass{i:02d}.replay.npz",native=before,entry=entry,after=after,
            python_values=cbv[0],python_distance=cbv[1],python_mean=cbv[2],python_targets=cbv[3],
            python_marked=cbv[4],python_sigma=cbv[5],original=original,affine=affine,thresholds=thresholds)
        report["passes"].append(row);save()
    report["status"]="complete";save()


if __name__=="__main__":main()
