"""完整四轮pial同输入回归/CPU剖析，参考表面只在计算后诊断。"""
import argparse,cProfile,hashlib,inspect,json,os,platform,pstats,resource,time
from pathlib import Path
import nibabel.freesurfer.io as fs
import numpy as np
import numba,torch
from fnit.recon_all.place_pial_python import place_pial_t1

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--subject",type=Path,required=True);p.add_argument("--hemi",choices=["lh","rh"],required=True)
p.add_argument("--output",type=Path,required=True);p.add_argument("--commit",required=True)
p.add_argument("--profile",action="store_true");p.add_argument("--baseline-surface",type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);numba.set_num_threads(4)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
files=[a.subject/("surf/"+a.hemi+".white"),a.subject/("surf/autodet.gw.stats."+a.hemi+".dat"),
       a.subject/("label/"+a.hemi+".cortex.label"),a.subject/("label/"+a.hemi+".cortex+hipamyg.label"),
       *[a.subject/("mri/"+x+".mgz") for x in ("brain.finalsurfs","wm","aseg.presurf")]]
inputs={str(x):sha(x) for x in files}
prof=cProfile.Profile() if a.profile else None
if prof:prof.enable()
tick=time.perf_counter();result=place_pial_t1(subject=a.subject,hemisphere=a.hemi,output=a.output/"pial.T1",max_steps=200)
seconds=time.perf_counter()-tick
if prof:
    prof.disable();prof.dump_stats(str(a.output/"profile.pstats"))
    with (a.output/"profile.txt").open("w") as stream:pstats.Stats(prof,stream=stream).sort_stats("cumulative").print_stats(80)
candidate,cf=fs.read_geometry(str(a.output/"pial.T1"))
comparisons={}
for name,path in (("frozen_conda",a.subject/("surf/"+a.hemi+".pial.T1")),("optimization",a.baseline_surface)):
    if path is None:continue
    ref,rf=fs.read_geometry(str(path));same=ref.shape==candidate.shape and np.array_equal(rf,cf)
    delta=np.linalg.norm(ref-candidate,axis=1) if same else None
    comparisons[name]={"reference_sha256":sha(path),"ordered_correspondence":same,
        "different_coordinate_components":int(np.count_nonzero(ref!=candidate)) if same else None,
        "mean_mm":float(delta.mean()) if same else None,"max_mm":float(delta.max()) if same else None,"p99_mm":float(np.quantile(delta,.99)) if same else None}
report={"commit":a.commit,"host":platform.node(),"threads":{"torch":torch.get_num_threads(),"numba":numba.get_num_threads()},
    "scope":"frozen_self_generated_full_four_pass_pial; not_whole_case",
    "profile":a.profile,"timing_scope":"includes JIT, input IO and output write; cProfile overhead included when enabled",
    "seconds":seconds,"max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"stage":result,
    "input_sha256":inputs,"output_sha256":sha(a.output/"pial.T1"),"comparisons":comparisons,
    "source_sha256":{str(x):sha(x) for x in Path(inspect.getfile(place_pial_t1)).parent.glob("place*.py")},"benchmark_sha256":sha(__file__)}
(a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)
if a.baseline_surface and (not comparisons["optimization"]["ordered_correspondence"] or comparisons["optimization"]["different_coordinate_components"]):
    raise RuntimeError("changed ordered pial geometry; do not enable optimization")
