"""按例等待原始 T1 命令完成，只读候选/基线/官方输出进行性能与精度比较。
--config 为基准路径与候选绑定 JSON。输出 comparative_launch/comparison/quality/figures。
不会将参考文件写回生产输出；没有批准整体等效门槛，保持 not_assessed。
包含138项诊断、同名ROI、标签Dice、网格/扩展质量和脑图；失败保留日志。
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True);a=p.parse_args()
c=json.loads(a.config.read_text());h=Path(c["hot"]);root=Path(c["comparison_root"])
root.mkdir(parents=True,exist_ok=True)
progress_path=root/"progress.json"
if progress_path.exists():
 previous=json.loads(progress_path.read_text())
 if previous.get("config")!=c or previous.get("completed"):
  raise ValueError("existing comparison must be a waiting run of the same config")
scripts=h/"comparison_scripts_c248520";env=dict(os.environ)
env.update(PYTHONPATH=str(h/"code_final_candidate/src"),CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",NUMBA_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4",NUMBA_CACHE_DIR=str(root/"numba_cache"))
manifest={"config":c,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
"comparator_sha256":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in scripts.glob("*.py")},"completed":[],"status":"waiting"}
(root/"progress.json").write_text(json.dumps(manifest,indent=2)+"\n")
try:
 for case in c["cases"]:
  completion_path=Path(case["launch"])/"completion.json"
  manifest["status"]="waiting_"+case["id"]
  (root/"progress.json").write_text(json.dumps(manifest,indent=2)+"\n")
  while not completion_path.exists():time.sleep(15)
  manifest["status"]="comparing_"+case["id"]
  (root/"progress.json").write_text(json.dumps(manifest,indent=2)+"\n")
  s=case["id"];completion=json.loads((Path(case["launch"])/"completion.json").read_text())
  if completion["exit_code"]: raise RuntimeError(s+" original T1 execution failed")
  subject=Path(case["candidate"]);folder=root/s;folder.mkdir()
  commands=[
   [c["python"],str(scripts/"compare_performance_pair.py"),"--baseline",case["baseline"],"--candidate",str(subject),"--official",case["official"],"--label-table",c["label_table"],"--output-dir",str(folder/"paired"),"--code-commit",c["code_commit"]],
   [c["python"],str(scripts/"plot_recon_all_comparison.py"),"--reference",case["official"],"--candidate",str(subject),"--region-report",str(folder/"paired/region_vs_official.json"),"--dice-report",str(folder/"paired/dice_vs_official.json"),"--output-dir",str(folder/"figures"),"--code-commit",c["code_commit"]],
   [c["python"],str(scripts/"benchmark_surface_quality_extended.py"),"--subject",str(subject),"--output",str(folder/"quality"),"--code-version",c["code_commit"],"--threads","4","--cross-timeout-seconds","180","--max-bbox-pairs","20000000"]]
  with (folder/"command.log").open("w") as log:
   for command in commands:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  manifest["completed"].append(s);(root/"progress.json").write_text(json.dumps(manifest,indent=2)+"\n")
 manifest["status"]="complete"
except BaseException as exc:
 manifest["status"]="failed";manifest["error_type"]=type(exc).__name__;manifest["error"]=str(exc)
 raise
finally:
 (root/"progress.json").write_text(json.dumps(manifest,indent=2)+"\n")

