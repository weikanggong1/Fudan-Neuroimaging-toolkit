"""从原始 T1 与空目录运行热点优化整例；记录代码、环境及包含解释器启动的墙钟。
--config 指 JSON：python/code_root/code_commit/source_archive_sha256/input/output/
weights/assets/device/threads/invocation/monitor_script/api_script/gpu_uuid/fs_license。
输出 launch.json、command.log、completion.json；GPU 另有父子采样目录。
不读取官方或旧候选输出。异常返回非零，完整运行后由独立比较器验收。
属于 FNIT benchmark 启动器，没有官方独立等价命令。
"""
import argparse,hashlib,json,os,platform,socket,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True)
a=p.parse_args(); c=json.loads(a.config.read_text()); root=Path(c["diagnostic_root"])
root.mkdir(parents=True,exist_ok=False)
out=Path(c["output"])
if out.exists() and any(out.iterdir()): raise ValueError("subject output must be empty")
if not Path(c["input"]).is_file(): raise FileNotFoundError(c["input"])
license_path=Path(c["fs_license"])
if not license_path.is_file() or not os.access(license_path,os.R_OK):
 raise FileNotFoundError("declared upstream-source license is not readable")
threads=int(c["threads"])
env=dict(os.environ)
env.update({k:str(threads) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS","NUMEXPR_NUM_THREADS"]})
env.update(PYTHONPATH=str(Path(c["code_root"])/"src"),NUMBA_CACHE_DIR=str(root/"numba_cache"))
env.update(FS_LICENSE=str(license_path),FREESURFER_HOME=c["assets"])
env["CUDA_VISIBLE_DEVICES"]=c.get("gpu_uuid","") if c["device"].startswith("cuda") else ""
if c["device"].startswith("cuda"):env["PYTORCH_NO_CUDA_MEMORY_CACHING"]="1"
base=[c["python"]]
if c["invocation"]=="initialized_cuda_api": base += [c["api_script"]]
else: base += ["-m","fnit.recon_all.native_free"]
base += [c["input"],c["output"],"--weights-dir",c["weights"],"--assets-dir",c["assets"],"--device",c["device"],"--threads",str(threads),"--profile-stages"]
if c.get("native_bin_dir"):base += ["--native-bin-dir",c["native_bin_dir"]]
if c["device"].startswith("cuda"):base += ["--cuda-allocator-cache","disabled"]
command=base
if c["device"].startswith("cuda"):command=[c["python"],c["monitor_script"],"--gpu-uuid",c["gpu_uuid"],"--output",str(root/"gpu_monitor"),"--interval","1","--",*base]
launch={**c,"command":command,"host":socket.gethostname(),"platform":platform.platform(),"cpu":platform.processor(),
"loadavg_at_launch":os.getloadavg(),"started_utc":datetime.now(timezone.utc).isoformat(),
"thread_environment":{k:env[k] for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS","NUMEXPR_NUM_THREADS"]},
"CUDA_VISIBLE_DEVICES":env["CUDA_VISIBLE_DEVICES"],"PYTORCH_NO_CUDA_MEMORY_CACHING":env.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
"launcher_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"config_sha256":hashlib.sha256(a.config.read_bytes()).hexdigest(),
"timing_scope":"outer wall includes process imports, validation, models, H2D/D2H, all stages and output IO; CPU has no GPU memory observation",
"precision_policy":"float32; TF32 default; existing SynthSeg/SynthStrip FP32 exceptions retained"}
(root/"launch.json").write_text(json.dumps(launch,indent=2)+"\n")
start=time.perf_counter()
with (root/"command.log").open("w") as log:
 process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 (root/"pid.json").write_text(json.dumps({"launcher_pid":os.getpid(),"command_pid":process.pid})+"\n")
 code=process.wait()
completion={"exit_code":code,"command_seconds":time.perf_counter()-start,"finished_utc":datetime.now(timezone.utc).isoformat(),
"execution_status":"complete" if code==0 else "failed","numerical_acceptance":"not_assessed","whole_metric_equivalence":"not_assessed; no confirmed gates",
"code_commit":c["code_commit"],"source_archive_sha256":c["source_archive_sha256"]}
(root/"completion.json").write_text(json.dumps(completion,indent=2)+"\n")
raise SystemExit(code)
