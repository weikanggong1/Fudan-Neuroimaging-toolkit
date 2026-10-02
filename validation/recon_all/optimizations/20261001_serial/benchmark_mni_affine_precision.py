"""冻结自产 MNI 输入的连续非线性链验证，仅用于 benchmark。
--source 提供 orig 与 aff.lta/invol.crop；--output 必须不存在。
--baseline/--prior 指已完成的 FNIT 检查点，仅在计算结束后只读比较。
--weights/--assets 为固定资源；--native-bin 为 Conda 源码构建目录。
--device/--threads 固定目标设备和线程；--commit 绑定实际源码；--report 新 JSON。
输入/输出 scanner RAS 变换、位移 mm；返回三个文件的几何、dtype、差异统计。
计时包括加载、计算、传输和写出，不含检查点复制和事后比较；GPU 同步后计时。
不改变验收阈值，不把本测试称为原始 T1 整例。
"""
import argparse, hashlib, inspect, json, os, platform, shutil, time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from fnit.recon_all.mni_nonlinear_chain import run_mni_nonlinear_chain

T="transforms/synthmorph.1.0mm.1.0mm/"
FILES=[T+"warp.to.mni152.1.0mm.1.0mm.nii.gz",
       T+"warp.to.mni152.1.0mm.1.0mm.inv.nii.gz", T+"test.nii.gz"]

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def compare(before,after):
    a,b=nib.load(before),nib.load(after)
    x,y=np.asarray(a.dataobj),np.asarray(b.dataobj)
    if x.shape!=y.shape:raise ValueError("different grids")
    if not np.all(np.isfinite(y)):raise ValueError("nonfinite output")
    delta=np.abs(x.astype(np.float64)-y.astype(np.float64))
    return {"before_sha256":sha(before),"after_sha256":sha(after),
        "geometry_equal":bool(np.array_equal(a.affine,b.affine)),
        "dtype_equal":bool(a.get_data_dtype()==b.get_data_dtype()),
        "shape":list(x.shape),"different_elements":int(np.count_nonzero(x!=y)),
        "max_abs_error":float(delta.max()),"p99_abs_error":float(np.percentile(delta,99)),
        "mean_abs_error":float(delta.mean()),"units":"mm" if "warp." in str(after) else "intensity"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("source","output","baseline","prior","weights","assets","native-bin","report"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--threads",type=int,default=4)
    p.add_argument("--commit",required=True);a=p.parse_args()
    if a.output.exists() or a.report.exists():raise FileExistsError(a.output)
    inputs={}
    for n in ["orig.mgz",T+"aff.lta",T+"invol.crop.nii.gz"]:
        src=a.source/"mri"/n;dst=a.output/"mri"/n
        inputs[n]=sha(src);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    cuda=torch.device(a.device).type=="cuda"
    if cuda:torch.cuda.synchronize(a.device);torch.cuda.reset_peak_memory_stats(a.device)
    start=time.perf_counter()
    result=run_mni_nonlinear_chain(subject_dir=a.output,weights_dir=a.weights,assets_dir=a.assets,
        warp_convert=a.native_bin/"mri_warp_convert",ca_register=a.native_bin/"mri_ca_register",
        mri_convert=a.native_bin/"mri_convert",device=a.device,threads=a.threads)
    if cuda:torch.cuda.synchronize(a.device)
    seconds=time.perf_counter()-start
    for f in result["precision"]["actual_forwards"]:
        assert f["device"]==a.device and f["input_dtype"]=="torch.float32"
        assert not f["matmul_tf32"] and not f["cudnn_tf32"] and not f["autocast"]["enabled"]
    report={"code_commit":a.commit,"scope":"frozen_self_generated_input_continuous_MNI_chain",
        "host":platform.node(),"threads":a.threads,"device":a.device,"gpu_uuid":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "input_sha256":inputs,"seconds":seconds,"result":result,
        "allocated_bytes":torch.cuda.max_memory_allocated(a.device) if cuda else None,
        "reserved_bytes":torch.cuda.max_memory_reserved(a.device) if cuda else None,
        "benchmark_sha256":sha(__file__),"source_sha256":sha(inspect.getfile(run_mni_nonlinear_chain)),
        "program_sha256":{n:sha(a.native_bin/n) for n in ["mri_warp_convert","mri_ca_register","mri_convert"]},
        "weight_sha256":sha(a.weights/"synthmorph.deform.3.h5"),
        "comparisons":{k:{n:compare(path/"mri"/n,a.output/"mri"/n) for n in FILES}
                       for k,path in [("baseline_cpu_affine",a.baseline),("prior_gpu_matmul_tf32",a.prior)]},
        "overall_metric_equivalence":"not_assessed"}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"seconds":seconds,"comparisons":report["comparisons"]}),flush=True)
if __name__=="__main__":main()
