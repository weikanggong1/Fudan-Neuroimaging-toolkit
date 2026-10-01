"""冻结自产inflated/smoothwm的完整standard sphere配对，包含JIT和I/O。"""
import argparse,cProfile,hashlib,inspect,json,os,platform,pstats,time
from pathlib import Path
import nibabel.freesurfer.io as fs
import numpy as np
import numba,torch
from fnit.recon_all.sphere_standard_run import run_standard_sphere
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--subject",type=Path,required=True);p.add_argument("--hemi",choices=["lh","rh"],required=True)
p.add_argument("--output",type=Path,required=True);p.add_argument("--commit",required=True)
p.add_argument("--profile",action="store_true");p.add_argument("--averaging-device",default="cpu")
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);numba.set_num_threads(4)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
surf=a.subject/"surf";kwargs={"finish_device":"cpu"}
if "averaging_device" in inspect.signature(run_standard_sphere).parameters:
 kwargs["averaging_device"]=a.averaging_device
prof=cProfile.Profile() if a.profile else None
if prof:prof.enable()
tick=time.perf_counter()
result=run_standard_sphere(inflated=surf/(a.hemi+".inflated"),smoothwm=surf/(a.hemi+".smoothwm"),output=a.output/"sphere",**kwargs)
seconds=time.perf_counter()-tick
if prof:
 prof.disable();prof.dump_stats(str(a.output/"profile.pstats"))
 with (a.output/"profile.txt").open("w") as s:pstats.Stats(prof,stream=s).sort_stats("cumulative").print_stats(60)
ref,rf=fs.read_geometry(str(surf/(a.hemi+".sphere")));cand,cf=fs.read_geometry(str(a.output/"sphere"))
same=ref.shape==cand.shape and np.array_equal(rf,cf)
delta=np.abs(ref-cand) if same else None
report={"commit":a.commit,"host":platform.node(),"torch":torch.__version__,"numba":numba.__version__,
 "threads":{"torch":torch.get_num_threads(),"numba":numba.get_num_threads()},
 "scope":"frozen_self_generated_same_input_full_standard_sphere; not_raw_T1_whole_case",
 "profile":a.profile,"seconds":seconds,"requested_averaging_device":a.averaging_device,
 "input_sha256":{k:sha(surf/(a.hemi+"."+k)) for k in ("inflated","smoothwm","sphere")},
 "output_sha256":sha(a.output/"sphere"),"stage":result,
 "comparison":{"correspondence":same,"different_coordinates":int(np.count_nonzero(delta)) if same else None,
 "max_mm":float(delta.max()) if same else None,"p99_mm":float(np.quantile(delta,.99)) if same else None},
 "source_sha256":{str(x):sha(x) for x in Path(inspect.getfile(run_standard_sphere)).parent.glob("sphere_standard*.py")},
 "benchmark_sha256":sha(__file__),"gpu_uuid":os.environ.get("CUDA_VISIBLE_DEVICES")}
(a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(seconds,report["comparison"],flush=True)
