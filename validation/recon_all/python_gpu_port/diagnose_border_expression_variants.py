"""以已录取的固定逐轮状态隔离两处 CBV 表达式精度，禁止进入生产。

候选代码来自现有 FNIT 搜索函数的私有诊断副本；只改变所列表达式，
保存实际副本 SHA。需要完整目标探针几何录取以及相同 MRI 的报告。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

from diagnose_pial_first_difference import compare
from diagnose_preaparc_target_passes import state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints-directory",type=Path,required=True)
    p.add_argument("--subject",type=Path,required=True)
    p.add_argument("--border-source",type=Path,required=True)
    p.add_argument("--output-directory",type=Path,required=True)
    p.add_argument("--hemisphere",choices=("lh","rh"),default="lh")
    a=p.parse_args();out=a.output_directory;h=a.hemisphere
    admission=json.loads((a.checkpoints_directory/"report.json").read_text())
    if admission["status"]!="complete" or admission["probe_admission"]["coordinates"]["different_elements"]:
        raise RuntimeError("full probe admission not complete")
    out.mkdir(mode=0o700,parents=True,exist_ok=False)
    from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
    from fnit.recon_all.place_surface_volume import prepare_placement_volume
    brain=nib.load(str(a.subject/"mri/brain.finalsurfs.mgz"))
    seg=np.asarray(nib.load(str(a.subject/"mri/aseg.presurf.mgz")).dataobj)
    wm=np.asarray(nib.load(str(a.subject/"mri/wm.mgz")).dataobj)
    stats_file=a.checkpoints_directory/f"native_probe/surf/autodet.gw.stats.{h}.dat"
    stats=dict(line.split()[:2] for line in stats_file.read_text().splitlines() if len(line.split())>=2)
    volume,_=prepare_placement_volume(np.asarray(brain.dataobj),wm,surface="white",mid_gray=float(stats["MID_GRAY"]))
    vertices,_,meta=nib.freesurfer.read_geometry(str(a.subject/f"surf/{h}.orig"),read_metadata=True)
    affine=surface_ras_to_voxel(brain.header,meta)
    thresholds=np.array([float(stats[f"white_{n}"]) for n in
        ("inside_hi","border_hi","border_low","outside_low","outside_hi")])
    count=len(vertices)
    original=state(a.checkpoints_directory/"native_target.pass00.entry",count)["floats"][:,:3].copy()
    old=a.border_source.read_text()
    projection="orig_dist = abs((px - float(original[v, 0])) * nx + (py - float(original[v, 1])) * ny + (pz - float(original[v, 2])) * nz)"
    new_projection="orig_dist = abs(np.float64(px - original[v, 0]) * np.float64(nx) + np.float64(py - original[v, 1]) * np.float64(ny) + np.float64(pz - original[v, 2]) * np.float64(nz))"
    outside="px + nx * (d + 1.0), py + ny * (d + 1.0), pz + nz * (d + 1.0)"
    new_outside="np.float32(px + np.float32(nx * np.float32(d + 1.0))), np.float32(py + np.float32(ny * np.float32(d + 1.0))), np.float32(pz + np.float32(nz * np.float32(d + 1.0)))"
    if old.count(projection)!=1 or old.count(outside)!=2:
        raise RuntimeError("FNIT search expressions changed")
    report=dict(scope="same_input_diagnostic_expression_variants_not_production",
        baseline_source_sha256=hashlib.sha256(old.encode()).hexdigest(),
        probe_admission_sha256=hashlib.sha256((a.checkpoints_directory/"report.json").read_bytes()).hexdigest(),
        variants={})
    for name,do_projection,do_outside in (("baseline",False,False),("projection_double",True,False),
        ("outside_float",False,True),("combined",True,True)):
        source=old
        if do_projection:source=source.replace(projection,new_projection)
        if do_outside:source=source.replace(outside,new_outside)
        file=out/f"border_{name}.py";file.write_text(source)
        module_name=f"border_expression_{name}"
        spec=importlib.util.spec_from_file_location(module_name,file)
        module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module;spec.loader.exec_module(module)
        rows=[]
        for i in range(4):
            entry=state(a.checkpoints_directory/f"native_target.pass{i:02d}.entry",count)
            before=state(a.checkpoints_directory/f"native_target.pass{i:02d}.before_average",count)
            active=before["flags"][:,0]==0
            started=time.perf_counter()
            result=module.compute_border_values_first_pass(volume,seg,before["floats"][:,:3],
                before["floats"][:,3:6],original,~active,entry["floats"][:,6],affine,thresholds,
                hemisphere=h,surface="white",sigma=2.0/(1<<i))
            native=(before["floats"][:,6],before["floats"][:,8],before["floats"][:,9],
                before["floats"][:,10:13],before["flags"][:,1],before["floats"][:,7])
            diff=np.flatnonzero(active & (native[0]!=result[0]))
            row=dict(pass_index=i,seconds_including_first_JIT=time.perf_counter()-started,
                different_target_vertices=diff.tolist(),
                fields={k:compare(n[active],v[active]) for k,n,v in zip(
                    ("values","distance","mean","target_coordinates","marked","sigma"),native,result)})
            rows.append(row)
        report["variants"][name]=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),passes=rows)
        (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
