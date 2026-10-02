"""合并 main 后复用 SynthStrip/SynthSeg 的真实冻结输入回归。
source 提供自产 orig.mgz；输出为空目录。只读 baseline 的分割用于事后比较。
计时含构造、推理、传输与写出，目标 GPU 同步；不称整例加速。
device 默认 cuda:0，threads 默认4；weights/assets 必须来自已校验清单。
输出 run.json、原空间去颅骨图/标签/体积CSV和实际前向设置；无官方输入参与计算。
"""
import argparse,hashlib,json,os,platform,time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from fnit.synthstrip import SynthStrip
from fnit.synthseg_parc import SynthSeg

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def compare(a,b,labels=False):
    x,y=nib.load(a),nib.load(b);u,v=np.asarray(x.dataobj),np.asarray(y.dataobj)
    if u.shape!=v.shape:raise ValueError("different shape")
    d=np.abs(u.astype(float)-v.astype(float))
    row={"different_voxels":int(np.count_nonzero(u!=v)),"maximum":float(d.max()),
         "p99":float(np.percentile(d,99)),"geometry_equal":bool(np.array_equal(x.affine,y.affine)),
         "dtype_equal":bool(x.get_data_dtype()==y.get_data_dtype()),"before_sha256":sha(a),"after_sha256":sha(b)}
    if labels:
        row["dice"]={str(int(k)):float(2*np.count_nonzero((u==k)&(v==k))/
            (np.count_nonzero(u==k)+np.count_nonzero(v==k))) for k in np.union1d(u,v) if k!=0}
    return row

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ["source","output","weights","assets"]:p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--threads",type=int,default=4)
    p.add_argument("--commit",required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    image=a.source/"mri/orig.mgz";torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    cuda=torch.device(a.device).type=="cuda"
    if cuda:torch.cuda.synchronize(a.device);torch.cuda.reset_peak_memory_stats(a.device)
    tick=time.perf_counter(); forwards=[]
    strip=SynthStrip(weights=a.weights,device=a.device,threads=a.threads,configure_precision=False)
    # flags的默认enabled=False会关闭cuDNN；显式匹配生产SynthStrip策略。
    strip_backend={"enabled":True,"benchmark":False,"deterministic":True,"allow_tf32":False}
    with torch.backends.cudnn.flags(**strip_backend):
        actual_strip_backend={key:bool(getattr(torch.backends.cudnn,key)) for key in strip_backend}
        if actual_strip_backend!=strip_backend:raise ValueError("actual cuDNN policy differs")
        stripped=strip(image,precision_report=forwards)
    stripped.image.save(str(a.output/"synthstrip.mgz"))
    stripped.mask.save(str(a.output/"synthstrip-mask.nii.gz"))
    stripped.distance.save(str(a.output/"synthstrip-distance.nii.gz"))
    del stripped,strip
    if cuda:torch.cuda.synchronize(a.device)
    strip_seconds=time.perf_counter()-tick
    tick=time.perf_counter()
    model=SynthSeg(weights=a.weights,device=a.device,threads=a.threads,cudnn_tf32=False)
    result=model(image,keep_geometry=True,color_lut=a.assets/"FreeSurferColorLUT.txt")
    result.segmentation.save(str(a.output/"synthseg.rca.mgz"))
    result.write_volumes_csv(image,a.output/"synthseg.vol.csv")
    precision=result.precision
    del result,model
    if cuda:torch.cuda.synchronize(a.device)
    seg_seconds=time.perf_counter()-tick
    report={"code_commit":a.commit,"scope":"same_input_integration_stage; not_whole",
        "host":platform.node(),"threads":a.threads,"gpu_uuid":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "input_sha256":sha(image),"benchmark_sha256":sha(__file__),
        "precision":{"SynthStrip":forwards,"SynthSeg":precision},
        "SynthStrip_backend_policy":strip_backend,
        "SynthStrip_actual_backend":actual_strip_backend,
        "seconds":{"SynthStrip_including_extra_mask_distance_io":strip_seconds,"SynthSeg":seg_seconds},
        "allocator_cache_disabled_env":os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")=="1",
        "torch_memory_stats_status":"unavailable_allocator_disabled" if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")=="1" else "available",
        "allocated_bytes":torch.cuda.max_memory_allocated(a.device) if cuda and os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")!="1" else None,
        "reserved_bytes":torch.cuda.max_memory_reserved(a.device) if cuda and os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")!="1" else None,
        "comparisons":{"SynthStrip":compare(a.source/"mri/synthstrip.mgz",a.output/"synthstrip.mgz"),
           "SynthSeg":compare(a.source/"mri/synthseg.rca.mgz",a.output/"synthseg.rca.mgz",labels=True)},
        "overall_metric_equivalence":"not_assessed"}
    (a.output/"run.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"seconds":report["seconds"],"comparisons":report["comparisons"]}),flush=True)
if __name__=="__main__":main()
