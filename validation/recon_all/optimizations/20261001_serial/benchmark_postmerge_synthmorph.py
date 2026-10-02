"""合并 main 后验证公用 SynthMorph 采样器对 recon-all 的影响。
输入 source 是已完成 FNIT 整例目录；只读 orig、synthstrip 和自产注册输出。
output 必须不存在；weights/assets/native-bin 使用声明资源和 Conda 原生组件。
设备 cuda:0、threads=4；模型 FP32 例外，与生产注册设置一致，不使用半精度。
输出新目录中的 Talairach/MNI 注册及 report.json；矩阵逐元素、warp逐体素比较。
LTA 世界矩阵平移为mm、线性系数无量纲；warp为RAS位移mm，检查图为强度。
这是冻结真实输入阶段回归，不是从原始T1开始的整例。计时同步CUDA，包含I/O。
缺少资源、已有输出、推理或比较失败抛异常。原命令见 MNI_AUX_CHAIN 与 Talairach 文档。
"""
import argparse, hashlib, json, os, pathlib, platform, shutil, time
import nibabel as nib
import numpy as np
import torch
from fnit.recon_all.talairach_synthmorph import register_talairach
from fnit.recon_all.mni_aux_chain import register_mni152_affine
from fnit.recon_all.mni_nonlinear_chain import run_mni_nonlinear_chain

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def compare(a,b,matrix=False):
    if matrix:
        def read_matrix_geometry(path):
            lines=path.read_text().splitlines(); start=lines.index("1 4 4")+1
            values=np.asarray([[float(v) for v in row.split()] for row in lines[start:start+4]],np.float64)
            # 同类型LTA直接比较矩阵；忽略filename文本，保留两端几何的数值语义。
            info={}
            for section in ("src volume info","dst volume info"):
                at=lines.index(section)+1
                info[section]={k.strip():v.strip() for row in lines[at:at+8] if "=" in row
                               for k,v in [row.split("=",1)] if k.strip()!="filename"}
            kind=int(lines[0].split("=")[1].split("#")[0])
            return values,info,kind
        x,gx,kx=read_matrix_geometry(a);y,gy,ky=read_matrix_geometry(b)
        geometry=gx==gy and kx==ky;dtype=x.dtype==y.dtype
    else:
        aa,bb=nib.load(a),nib.load(b)
        x,y=np.asarray(aa.dataobj),np.asarray(bb.dataobj)
        geometry=bool(np.array_equal(aa.affine,bb.affine));dtype=aa.get_data_dtype()==bb.get_data_dtype()
    if x.shape!=y.shape: raise ValueError("shape mismatch")
    d=np.abs(x.astype(np.float64)-y.astype(np.float64))
    return {"different_elements":int(np.count_nonzero(x!=y)), "max_abs_error":float(d.max()),
            "p99_abs_error":float(np.percentile(d,99)),"geometry_equal":geometry,"dtype_equal":bool(dtype),
            "before_sha256":sha(a),"after_sha256":sha(b)}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("source","output","weights","assets","native-bin"):
        p.add_argument("--"+n,type=pathlib.Path,required=True)
    p.add_argument("--commit",required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--threads",type=int,default=4);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True);(a.output/"mri").mkdir()
    shutil.copyfile(a.source/"mri/orig.mgz",a.output/"mri/orig.mgz")
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=True;torch.backends.cudnn.deterministic=True
    torch.cuda.synchronize(a.device);started=time.perf_counter()
    forwards=[]
    register_talairach(moving=a.source/"mri/synthstrip.mgz",
        template=a.assets/"average/mni305.cor.stripped.mgz",weights=a.weights,
        output_xfm=a.output/"talairach.xfm",output_lta=a.output/"talairach.aff.lta",
        device=a.device,threads=a.threads,precision_report=forwards)
    torch.cuda.synchronize(a.device);tal=time.perf_counter()-started
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter()
    # 生产mni_aux的外层flags只传allow_tf32；PyTorch2.5.1实际默认关闭cuDNN。
    # 此回归匹配已测生产语义，不能把cuDNN-on控制与之混比。
    affine_backend={"enabled":False,"benchmark":False,"deterministic":False,"allow_tf32":False}
    with torch.backends.cudnn.flags(**affine_backend):
        assert all(bool(getattr(torch.backends.cudnn,k))==v for k,v in affine_backend.items())
        register_mni152_affine(subject_dir=a.output,weights_dir=a.weights,assets_dir=a.assets,
            device=a.device,threads=a.threads,precision_report=forwards)
    torch.cuda.synchronize(a.device);aff=time.perf_counter()-started
    started=time.perf_counter()
    result=run_mni_nonlinear_chain(subject_dir=a.output,weights_dir=a.weights,assets_dir=a.assets,
        warp_convert=a.native_bin/"mri_warp_convert",ca_register=a.native_bin/"mri_ca_register",
        mri_convert=a.native_bin/"mri_convert",device=a.device,threads=a.threads)
    torch.cuda.synchronize(a.device);nonlin=time.perf_counter()-started
    t="mri/transforms/synthmorph.1.0mm.1.0mm/"
    comps={"talairach_affine":compare(a.source/"mri/transforms/synthmorph.mni305/aff.lta",a.output/"talairach.aff.lta",True)}
    for n in ("aff.lta","reg.targ_to_invol.lta","warp.to.mni152.1.0mm.1.0mm.nii.gz",
              "warp.to.mni152.1.0mm.1.0mm.inv.nii.gz","test.nii.gz"):
        comps[n]=compare(a.source/(t+n),a.output/(t+n),n.endswith(".lta"))
    r={"code_commit":a.commit,"scope":"post_merge_frozen_FNIT_input_real_stage_regression",
       "host":platform.node(),"device":a.device,"threads":a.threads,"gpu_uuid":os.environ.get("CUDA_VISIBLE_DEVICES"),
       "helper_sha256":sha(__file__),"input_sha256":{n:sha(a.source/("mri/"+n)) for n in ("orig.mgz","synthstrip.mgz")},
       "seconds":{"talairach":tal,"mni_affine":aff,"mni_nonlinear":nonlin},
       "actual_forwards":forwards,"mni_affine_backend":affine_backend,"nonlinear":result,"comparisons":comps,
       "passed":all(v["different_elements"]==0 and v["geometry_equal"] and v["dtype_equal"] for v in comps.values()),
       "equivalence_scope":"these affected same-input stages only; not a new whole execution"}
    (a.output/"report.json").write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps({"passed":r["passed"],"seconds":r["seconds"],"comparisons":comps}),flush=True)
    if not r["passed"]:raise AssertionError("post-merge regression")
if __name__=="__main__":main()
