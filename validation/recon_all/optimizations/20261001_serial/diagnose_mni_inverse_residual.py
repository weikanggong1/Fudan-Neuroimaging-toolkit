"""对两例MNI逆场进行只读局部误差诊断。
config提供完整比较cases和code_commit；candidate-mode默认stage，
stage-prefix为当前冻结链输出前缀；whole模式读取cases中完成的自产candidate。
output必须不存在，返回全场及基线brainmask内的向量最大/P99/均值和异常数。
0.1仅作定位分箱，不是验收阈值；不改变流程或掩盖全场最大值。
逆场shape为原始conform网格×1×3；记录NIfTI意图/单位/体素尺寸，不从shape臆测位移。
"""
import argparse,hashlib,json
from pathlib import Path
import nibabel as nib
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--stage-prefix",type=Path)
    p.add_argument("--candidate-mode",choices=("stage","whole"),default="stage")
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists() and a.output.stat().st_size:raise FileExistsError(a.output)
    if a.candidate_mode=="stage" and a.stage_prefix is None:
        p.error("stage mode requires --stage-prefix")
    c=json.loads(a.config.read_text());result={"code_commit":c["code_commit"],"scope":"readonly inverse residual diagnostic; not acceptance gate",
     "script_sha256":sha(__file__),"config_sha256":sha(a.config),"diagnostic_only_cutoff":0.1,"cases":{}}
    name="transforms/synthmorph.1.0mm.1.0mm/warp.to.mni152.1.0mm.1.0mm.inv.nii.gz"
    for case in c["cases"]:
        before=Path(case["baseline"])/"mri"
        candidate=Path(case["candidate"]) if a.candidate_mode=="whole" else Path(str(a.stage_prefix)+case["id"][-2:]+"_retry1")
        if a.candidate_mode=="whole" and json.loads((candidate/"fnit-native-free-run.json").read_text())["status"]!="complete":
            raise ValueError("whole reconstruction incomplete: "+str(candidate))
        after=candidate/"mri"
        first,second=before/name,after/name
        x,y=nib.load(first),nib.load(second)
        u,v=np.asarray(x.dataobj),np.asarray(y.dataobj)
        if u.shape!=v.shape or not np.array_equal(x.affine,y.affine):raise ValueError("inverse grids differ")
        d=np.linalg.norm((u.astype(np.float64)-v.astype(np.float64)).squeeze(axis=3),axis=-1)
        maskfile=before/"brainmask.mgz";mask=np.asarray(nib.load(maskfile).dataobj)>0
        idx=np.unravel_index(np.argmax(d),d.shape)
        def stats(vals):return {"maximum":float(vals.max()),"p99":float(np.percentile(vals,99)),"mean":float(vals.mean()),"count_over_0_1":int(np.count_nonzero(vals>.1))}
        result["cases"][case["id"]]={"input_sha256":{str(z):sha(z) for z in [first,second,maskfile]},
          "field_header":{"intent":x.header.get_intent(),"xyzt_units":x.header.get_xyzt_units(),"zooms":[float(z) for z in x.header.get_zooms()]},
          "whole_vector_error":stats(d),"maximum_index":list(map(int,idx)),"maximum_inside_brainmask":bool(mask[idx]),
          "brainmask_vector_error":stats(d[mask]),"baseline_brainmask_voxels":int(mask.sum())}
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["cases"]),flush=True)
if __name__=="__main__":main()
