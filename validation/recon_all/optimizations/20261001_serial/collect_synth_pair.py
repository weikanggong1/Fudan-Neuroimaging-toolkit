"""汇总Synth三路配对和非线性链；不会修改生产结果。"""
import argparse,hashlib,json
from pathlib import Path
import nibabel as nib
import numpy as np
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def image_pair(a,b,labels=False):
 x,y=nib.load(a),nib.load(b)
 u,v=np.asarray(x.dataobj),np.asarray(y.dataobj)
 out={"shape_equal":u.shape==v.shape,"geometry_equal":bool(np.array_equal(x.affine,y.affine)),
      "dtype_equal":x.get_data_dtype()==y.get_data_dtype()}
 if u.shape!=v.shape:return out
 d=np.abs(u.astype(np.float64)-v.astype(np.float64))
 out.update(different_elements=int(np.count_nonzero(d)),max=float(d.max()),p99=float(np.quantile(d,.99)),
            candidate_sha256=sha(b),baseline_sha256=sha(a))
 if labels:
  out["label_dice"]={int(i):float(2*np.count_nonzero((u==i)&(v==i))/(np.count_nonzero(u==i)+np.count_nonzero(v==i))) for i in np.union1d(u,v) if i!=0}
 return out
def lta(p):
 lines=p.read_text().splitlines();i=lines.index("1 4 4")
 return np.array([[float(x) for x in r.split()] for r in lines[i+1:i+5]])
p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True)
p.add_argument("--weights",type=Path,required=True)
a=p.parse_args();out={"scope":"frozen_self_generated_same_input_stage","subjects":{},
 "strict_reproduction":"see_each_comparison","overall_metric_equivalence":"not_assessed",
 "stage_criterion_before_execution":"same_gpu_old_new_labels_and_geometry_unchanged; nonlinear_positive_transform_unchanged",
 "whole_speedup":None}
for sub in ("01","02"):
 entry={}
 for v in ("oldcpu","oldgpu","newgpu"):
  f=a.root/f"stage1r2_sub{sub}_{v}.json";entry[v]=json.loads(f.read_text())
  entry[v]["monitor"]=json.loads((a.root/f"stage1r2_sub{sub}_{v}_monitor/monitor.json").read_text())
 entry["comparisons"]={}
 for v in ("oldcpu","oldgpu"):
  r=a.root/f"stage1r2_sub{sub}_{v}";c=a.root/f"stage1r2_sub{sub}_newgpu"
  cmp={n:image_pair(r/"mri"/n,c/"mri"/n,True) for n in ("entowm.mgz","mca-dura.mgz","vsinus.mgz")}
  transform="mri/transforms/synthmorph.1.0mm.1.0mm/reg.targ_to_invol.lta"
  cmp["affine_max_abs_matrix_delta"]=float(np.abs(lta(r/transform)-lta(c/transform)).max())
  entry["comparisons"][v]=cmp
 entry["nonlinear"]={}
 for v in ("oldgpu","newgpu"):
  entry["nonlinear"][v]=json.loads((a.root/f"stage1r2_mni_sub{sub}_{v}.json").read_text())
  entry["nonlinear"][v]["monitor"]=json.loads((a.root/f"stage1r2_mni_sub{sub}_{v}_monitor/monitor.json").read_text())
 path="mri/transforms/synthmorph.1.0mm.1.0mm/"
 entry["nonlinear"]["comparisons"]={n:image_pair(a.root/f"stage1r2_mni_sub{sub}_oldgpu"/path/n,
    a.root/f"stage1r2_mni_sub{sub}_newgpu"/path/n) for n in (
       "warp.to.mni152.1.0mm.1.0mm.nii.gz","warp.to.mni152.1.0mm.1.0mm.inv.nii.gz","test.nii.gz")}
 out["subjects"][sub]=entry
out["deform_weight_sha256"]=sha(a.weights/"synthmorph.deform.3.h5")
out["source_archive_sha256"]={f.name:sha(f) for f in a.root.glob("*.tar")}
out["script_sha256"]=sha(__file__)
(a.root/"stage1_summary.json").write_text(json.dumps(out,indent=2)+"\n")
print("stage1 summary saved")
