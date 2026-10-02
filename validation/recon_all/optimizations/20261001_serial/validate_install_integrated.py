"""既有主页 Conda 环境中的当前源码 wheel 构建、隔离 target 安装及 CLI/API 导入。
不重建 Conda 环境，不构成无预装软件整例隔离证明；仅使用 Conda 编译器构建 FNIT 扩展。
输出 JSON、pip/CLI 日志和 wheel SHA；无 GPU 计算，无资源下载。
"""
import hashlib,json,os,pathlib,subprocess,sys,time,platform
R=pathlib.Path("/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001");C=pathlib.Path("/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/candidate_integrated_ff372d7")
D=R/"installation_ff372d7";D.mkdir()
P=pathlib.Path(sys.executable).parent
env=dict(os.environ,CUDA_VISIBLE_DEVICES="",CC=str(P/"x86_64-conda-linux-gnu-cc"),CXX=str(P/"x86_64-conda-linux-gnu-c++"))
for n in ["CC","CXX"]:
 if not pathlib.Path(env[n]).is_file():raise FileNotFoundError(env[n])
report={"code_commit":"ff372d73f106e850b999fbc95ae0b324cd315cbf","source_archive_sha256":"071e48c97a1f8d88bd920e4753ed676033e47bed5a3fa0bc22e06c24338fd126","scope":"existing homepage Conda; wheel build and target install; not clean environment/whole runtime isolation","host":platform.node(),"python":sys.version,"compiler_paths":{n:env[n] for n in ["CC","CXX"]},"steps":{}}
def run(name,args,extra=None):
 e=dict(env)
 if extra:e.update(extra)
 start=time.perf_counter()
 with (D/(name+".log")).open("w") as f:r=subprocess.run(args,env=e,cwd=C,stdout=f,stderr=subprocess.STDOUT)
 report["steps"][name]={"seconds":time.perf_counter()-start,"exit_code":r.returncode,"command":args,"log_sha256":hashlib.sha256((D/(name+".log")).read_bytes()).hexdigest()}
 (D/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 if r.returncode:raise RuntimeError(name+" failed")
run("wheel",[sys.executable,"-m","pip","wheel","--no-deps","--no-build-isolation",str(C),"-w",str(D/"wheels")])
wheel=next((D/"wheels").glob("*.whl"));report["wheel_sha256"]=hashlib.sha256(wheel.read_bytes()).hexdigest()
run("install",[sys.executable,"-m","pip","install","--no-deps","--target",str(D/"installed"),str(wheel)])
extra={"PYTHONPATH":str(D/"installed")}
run("recon_cli",[sys.executable,"-m","fnit.recon_all.native_free","--help"],extra)
run("api_import",[sys.executable,"-c","import fnit,fnit.recon_all.native_free,fnit.recon_all.normalization._normalization_cuda,fnit.synthstrip,fnit.synthseg_parc; print(fnit.__file__)"],extra)
report["status"]="passed";report["isolation_status"]="not_verified"
report["script_sha256"]=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
(D/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report["steps"]),flush=True)
