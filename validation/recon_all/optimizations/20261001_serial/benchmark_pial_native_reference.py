"""隔离诊断：同一自产七项前置+aparc，当前Conda与官方pial各完整运行一次。"""
import argparse,hashlib,json,os,platform,shutil,time
from pathlib import Path
import nibabel.freesurfer.io as fs
import numpy as np
from fnit.recon_all.native_free import _run_native_pial
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
p.add_argument("--conda-binary",type=Path,required=True);p.add_argument("--official-binary",type=Path,required=True)
p.add_argument("--assets",type=Path,required=True);p.add_argument("--official-home",type=Path,required=True)
p.add_argument("--python-surface",type=Path,required=True);p.add_argument("--commit",required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
paths=["surf/lh.white","surf/autodet.gw.stats.lh.dat","label/lh.cortex.label","label/lh.cortex+hipamyg.label","label/lh.aparc.annot",*["mri/"+x+".mgz" for x in ("brain.finalsurfs","wm","aseg.presurf")]]
report={"commit":a.commit,"host":platform.node(),"scope":"isolated_same_input_full_pial; no reference inputs in production",
 "threads":4,"input_sha256":{x:sha(a.source/x) for x in paths},"benchmark_sha256":sha(__file__),"runs":{}}
surfaces={}
for name,binary,home in (("conda",a.conda_binary,a.assets),("official",a.official_binary,a.official_home)):
    subject=a.output/name
    for path in paths:
        dest=subject/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.source/path,dest)
    (subject/"scripts").mkdir()
    tick=time.perf_counter();stage=_run_native_pial(binary,subject,"lh",home,4)
    report["runs"][name]={"stage":stage,"seconds_including_stage_io":time.perf_counter()-tick,"binary_sha256":sha(binary),"output_sha256":sha(stage["output"])}
    surfaces[name]=Path(stage["output"])
surfaces.update(python=a.python_surface,frozen_conda=a.source/"surf/lh.pial.T1")
ref,rf=fs.read_geometry(str(surfaces["official"]));comparisons={}
for name,path in surfaces.items():
    xyz,faces=fs.read_geometry(str(path));same=xyz.shape==ref.shape and np.array_equal(rf,faces)
    delta=np.linalg.norm(xyz-ref,axis=1) if same else None
    comparisons[name]={"ordered_correspondence":same,"different_coordinate_components":int(np.count_nonzero(xyz!=ref)) if same else None,
        "mean_mm":float(delta.mean()) if same else None,"max_mm":float(delta.max()) if same else None,"p99_mm":float(np.quantile(delta,.99)) if same else None,
        "surface_sha256":sha(path)}
report["comparison_to_official"]=comparisons
(a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
