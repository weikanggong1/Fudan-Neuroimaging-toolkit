"""重新实测输入、权重、资产和Conda源码编译程序SHA；只读。
--config含expected_manifest/roots/input_paths/code_commit/archive_sha256；--output新JSON。
记录主机、CPU/GPU、依赖版本；不能作为物理隔离运行证明。
"""
import argparse, datetime, hashlib, json, platform, subprocess
from pathlib import Path
import torch, numpy, scipy, nibabel, numba
def info(path):
    p=Path(path); h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return {"size_bytes":p.stat().st_size,"sha256":h.hexdigest()}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text())
    if a.output.exists():raise FileExistsError(a.output)
    expected=json.loads(Path(c["expected_manifest"]).read_text())
    result={"code_commit":c["code_commit"],"source_archive_sha256":c["source_archive_sha256"],
      "host":platform.node(),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "torch":torch.__version__,"torch_cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
      "numpy":numpy.__version__,"scipy":scipy.__version__,"nibabel":nibabel.__version__,"numba":numba.__version__,
      "expected_manifest_sha256":info(c["expected_manifest"])["sha256"],
      "script_sha256":info(__file__)["sha256"],"mismatches":[],"isolation":"not_verified"}
    for group in ["weights","assets","binaries","inputs"]:
        result[group]={}
        for name,value in expected[group].items():
            path=c["input_paths"][name] if group=="inputs" else str(Path(c["roots"][group])/name)
            observed=info(path);observed["resolved_path"]=str(Path(path).resolve())
            result[group][name]=observed
            if any(observed[k]!=value[k] for k in ["size_bytes","sha256"]):
                result["mismatches"].append({"group":group,"name":name,"expected":value,"actual":observed})
    result["gpu_snapshot"]=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu","--format=csv,noheader"],text=True).splitlines()
    result["cpu"]=next(line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n")
    if result["mismatches"]:raise ValueError(result["mismatches"])
    print(json.dumps({"resources_match":True,"code_commit":c["code_commit"],"host":result["host"]}),flush=True)
if __name__=="__main__":main()
