"""自产冻结输入的两轮归一化实测，包含读写和传输。"""
import argparse,hashlib,json,os,platform,time
from pathlib import Path
import nibabel as nib
import numpy as np
import numba,torch
from fnit.recon_all.normalization.pipeline import normalize_t1
from fnit.recon_all.normalization.aseg_pipeline import normalize_t1_aseg
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--subject",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--commit",required=True)
p.add_argument("--device",default="cuda:0")
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);numba.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
cuda=torch.device(a.device).type=="cuda"
stats_valid=cuda and os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") is None
rows=[]
for name,func,kwargs in [
 ("T1",normalize_t1,{"input_file":a.subject/"mri/nu.mgz",
                    "xfm_file":a.subject/"mri/transforms/talairach.xfm"}),
 ("brain",normalize_t1_aseg,{"norm_file":a.subject/"mri/norm.mgz",
       "aseg_file":a.subject/"mri/aseg.presurf.mgz","brainmask_file":a.subject/"mri/brainmask.mgz"})]:
 inputs={k:sha(v) for k,v in kwargs.items()}
 if cuda:
  torch.cuda.synchronize(a.device)
  if stats_valid:torch.cuda.reset_peak_memory_stats(a.device)
 tick=time.perf_counter()
 result=func(**kwargs,output_file=a.output/(name+".mgz"),device=a.device,three_d_iterations=2)
 if cuda:torch.cuda.synchronize(a.device)
 seconds=time.perf_counter()-tick
 reference=nib.load(a.subject/"mri"/(name+".mgz"));candidate=nib.load(a.output/(name+".mgz"))
 diff=np.abs(np.asarray(reference.dataobj,dtype=np.float64)-np.asarray(candidate.dataobj,dtype=np.float64))
 rows.append({"name":name,"seconds":seconds,"function_report":result,"input_sha256":inputs,
   "output_sha256":sha(a.output/(name+".mgz")),
   "versus_saved_c248":{"different_voxels":int(np.count_nonzero(diff)),
       "max":float(diff.max()),"p99":float(np.quantile(diff,.99)),
       "geometry_equal":bool(np.array_equal(reference.affine,candidate.affine)),
       "dtype_equal":reference.get_data_dtype()==candidate.get_data_dtype()},
   "allocated":torch.cuda.max_memory_allocated(a.device) if stats_valid else None,
   "reserved":torch.cuda.max_memory_reserved(a.device) if stats_valid else None})
 (a.output/"report.json").write_text(json.dumps({"commit":a.commit,"host":platform.node(),
    "device":a.device,"allocator_stats_valid":stats_valid,"allocator_cache_environment":os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),"torch":torch.__version__,"numba":numba.__version__,
    "threads":{"torch":torch.get_num_threads(),"numba":numba.get_num_threads()},
    "rows":rows,"script_sha256":sha(__file__),"whole_case":False},indent=2)+"\n")
 print(name,seconds,flush=True)
