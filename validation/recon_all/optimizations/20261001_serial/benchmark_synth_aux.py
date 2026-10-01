"""冻结自产输入的Synth辅助链计时；每次在空目录运行，影像/资源均记录SHA。"""
import argparse, hashlib, inspect, json, os, platform, shutil, time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from fnit.recon_all.mni_aux_chain import run_mni_aux_chain
from fnit.recon_all.sclimbic import mri_entowm_seg

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "weights", "assets", "report"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",required=True)
    p.add_argument("--commit",required=True)
    p.add_argument("--threads",type=int,default=4)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    inputs={}
    for name in ("orig.mgz","nu.mgz","synthseg.rca.mgz","transforms/talairach.xfm.lta"):
        src=a.source/"mri"/name; dst=a.output/"mri"/name
        inputs[name]=sha(src); dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
    (a.output/"stats").mkdir()
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    cuda=torch.device(a.device).type=="cuda"
    if cuda:torch.cuda.synchronize(a.device);torch.cuda.reset_peak_memory_stats(a.device)
    forwards=[]
    kwargs={"device":a.device,"stats_path":a.output/"stats/entowm.stats",
            "talairach_lta":a.output/"mri/transforms/talairach.xfm.lta"}
    if "precision_report" in inspect.signature(mri_entowm_seg).parameters:
        kwargs["precision_report"]=forwards
    start=time.perf_counter()
    mri_entowm_seg(a.output/"mri/nu.mgz",a.output/"mri/entowm.mgz",a.weights,**kwargs)
    if cuda:torch.cuda.synchronize(a.device)
    ento=time.perf_counter()-start
    start=time.perf_counter()
    result=run_mni_aux_chain(a.output,a.weights,a.assets,device=a.device,threads=a.threads)
    if cuda:torch.cuda.synchronize(a.device)
    aux=time.perf_counter()-start
    def info(path):
        im=nib.load(path)
        return {"sha256":sha(path),"shape":list(im.shape),"dtype":str(im.get_data_dtype()),
                "affine":im.affine.tolist()}
    outputs={n:info(a.output/"mri"/n) for n in ("entowm.mgz","mca-dura.mgz","vsinus.mgz")}
    weights={n:sha(a.weights/n) for n in (
        "synthmorph.affine.2.h5","entowm.fsm31.t1.nstd00-30.nstd21-108.h5",
        "entowm.ctab","mca-dura.both-lh.nstd21.fhs.h5","vsinus.fhs.1.h5") if (a.weights/n).is_file()}
    assets={str(x.relative_to(a.assets)):sha(x) for x in a.assets.rglob("*")
            if x.is_file() and ("mni152.1.0mm" in x.name or "prior" in x.name)}
    report={"code_commit":a.commit,"host":platform.node(),"threads":a.threads,
            "torch":torch.__version__,"device":a.device,"input_sha256":inputs,
            "weight_sha256":weights,"asset_sha256":assets,"outputs":outputs,
            "seconds":{"entowm":ento,"mni_aux":aux,"combined":ento+aux},
            "precision":{"entowm":forwards,"mni_aux":result.get("runtime")},
            "allocated_bytes":torch.cuda.max_memory_allocated(a.device) if cuda else None,
            "reserved_bytes":torch.cuda.max_memory_reserved(a.device) if cuda else None,
            "gpu_uuid":os.environ.get("CUDA_VISIBLE_DEVICES"),"benchmark_sha256":sha(__file__),
            "source_sha256":{str(x):sha(x) for x in Path(inspect.getfile(run_mni_aux_chain)).parent.glob("*.py")},
            "scope":"same_input_stage_includes_loading_transfer_and_output_io; checkpoint_copy_excluded",
            "overall_metric_equivalence":"not_assessed"}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["seconds"]),flush=True)
if __name__=="__main__":main()
