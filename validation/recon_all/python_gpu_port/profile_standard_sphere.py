"""对冻结的真实 standard sphere 输入做 CPU cProfile；含剖析开销，不能用作提速基线。
--subject 是自产目录，读取同面序 inflated/smoothwm 和保存的 sphere；
--hemi 默认 lh，--threads 默认4，--code-commit 必须提供，--output 必须不存在。
表面为 surface RAS/mm；输出新 sphere、stage.json、report.json、profile.txt。
仅改观测，不修改生产；严格比较只在顶点和面对应时进行。
原阶段对应 mris_sphere -seed 1234 inflated sphere，本包装器无官方独立 CLI。
"""
import argparse,cProfile,hashlib,json,os,platform,pstats,time
from pathlib import Path
import nibabel.freesurfer.io as fsio
import numba,numpy as np,torch
from fnit.recon_all.sphere_standard_run import run_standard_sphere
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--subject",type=Path,required=True);p.add_argument("--hemi",choices=("lh","rh"),default="lh")
p.add_argument("--threads",type=int,default=4);p.add_argument("--code-commit",required=True);p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if a.threads<1:raise ValueError("threads must be positive")
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(a.threads);numba.set_num_threads(a.threads)
inputs={k:a.subject/"surf"/(a.hemi+"."+k) for k in ["inflated","smoothwm","sphere"]}
hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in inputs.items()}
cp=cProfile.Profile();tick=time.perf_counter();cp.enable()
stage=run_standard_sphere(inflated=inputs["inflated"],smoothwm=inputs["smoothwm"],output=a.output/"sphere",finish_device="cpu")
cp.disable();seconds=time.perf_counter()-tick;cp.dump_stats(str(a.output/"profile.pstats"))
with (a.output/"profile.txt").open("w") as stream:pstats.Stats(cp,stream=stream).sort_stats("cumulative").print_stats(80)
ref,rf=fsio.read_geometry(str(inputs["sphere"]));cand,cf=fsio.read_geometry(str(a.output/"sphere"))
cmp={"same_vertex_shape":ref.shape==cand.shape,"ordered_faces_equal":bool(np.array_equal(rf,cf))}
if cmp["same_vertex_shape"] and cmp["ordered_faces_equal"]:
 delta=np.abs(ref-cand);cmp.update(different_coordinates=int(np.count_nonzero(delta)),max_mm=float(delta.max(initial=0)),p99_mm=float(np.quantile(delta,.99)))
rows=[{"file":k[0],"line":k[1],"function":k[2],"calls":v[1],"self_seconds":v[2],"cumulative_seconds":v[3]} for k,v in pstats.Stats(cp).stats.items()]
import sys
sources={n:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for n,m in sys.modules.items() if n.startswith("fnit.recon_all.sphere") and getattr(m,"__file__",None)}
report={"scope":"same frozen FNIT inputs CPU profile; profile overhead included; saved final file comparator is diagnostic",
"code_commit":a.code_commit,"host":platform.node(),"thread_budget":{"torch":torch.get_num_threads(),"numba":numba.get_num_threads()},
"input_paths":{k:str(v) for k,v in inputs.items()},"input_sha256":hashes,"source_sha256":sources,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
"profile_wall_seconds":seconds,"versus_saved_surface":cmp,"functions":sorted(rows,key=lambda x:x["cumulative_seconds"],reverse=True)}
(a.output/"stage.json").write_text(json.dumps(stage,indent=2)+"\n");(a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"seconds":seconds,"comparison":cmp}))

