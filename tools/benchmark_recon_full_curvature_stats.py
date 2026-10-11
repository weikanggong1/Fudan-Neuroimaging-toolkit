"""公开真实自产smoothwm：完整自有curv.stats/八图对独立Conda双参考。

CPU/GPU/GPU/CPU完整API含验证、JIT、传输和读写；不将本阶段替换当作整例。
原有八图探索门保持max≤1e-5/P99≤1e-6，严格文本、数值差异单独保留。
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

import nibabel.freesurfer.io as fsio
import numpy as np
import torch
import fnit.recon_all


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def map_comparison(first, second):
    a, b = fsio.read_morph_data(str(first)), fsio.read_morph_data(str(second))
    if a.shape != b.shape:
        raise ValueError("map vertex correspondence differs")
    error = np.abs(a.astype(np.float64)-b.astype(np.float64))
    maximum, p99 = float(error.max(initial=0)), float(np.percentile(error, 99))
    return {"different_elements": int(np.count_nonzero(a != b)), "max_abs": maximum,
            "p99_abs": p99, "mean_abs": float(error.mean()),
            "within_existing_eight_map_exploration_gate": maximum <= 1e-5 and p99 <= 1e-6}


def stats_comparison(first, second):
    a, b = Path(first).read_text(), Path(second).read_text()
    pattern = r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    av, bv = np.asarray([float(v) for v in re.findall(pattern,a)]), np.asarray([float(v) for v in re.findall(pattern,b)])
    if av.shape != bv.shape:
        raise ValueError("full statistic fields differ in count")
    error = np.abs(av-bv)
    return {"same_file_bytes": a==b,"numeric_fields":len(av),"different_numeric_fields":int(np.count_nonzero(av!=bv)),
            "max_numeric_abs":float(error.max(initial=0)),
            "different_lines":[{"line":i+1,"reference":x,"candidate":y}
              for i,(x,y) in enumerate(zip(a.splitlines(),b.splitlines())) if x!=y]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject",type=Path,nargs="+",required=True)
    p.add_argument("--candidate-directory",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--reference-program",type=Path,required=True)
    p.add_argument("--assets",type=Path,required=True)
    p.add_argument("--code-version",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--angle-backend",choices=("source_libm","torch"),default="source_libm")
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    fnit.recon_all.__path__.insert(0,str(a.candidate_directory.resolve()))
    module=importlib.import_module("fnit.recon_all.curvature_stats_python")
    import fnit.recon_all.discrete_curvature_torch as discrete
    import fnit.recon_all.curvature_stats_torch as derivatives
    import fnit.recon_all.inflate_python as inflate
    import fnit.recon_all.inflate_topology as topology
    import fnit.recon_all.place_surface_normals as normals
    import fnit.recon_all.mris_register_objective as atan_source
    source_modules=(module,discrete,derivatives,inflate,topology,normals,atan_source)
    from fnit.recon_all.curvature_stats_python import write_curvature_stats, MAP_ORDER
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    cuda_setup_start=time.perf_counter()
    torch.cuda.set_device(a.device)
    torch.empty(1,device=a.device)
    torch.cuda.synchronize(a.device)
    cuda_setup_seconds=time.perf_counter()-cuda_setup_start
    environment=dict(os.environ,SUBJECTS_DIR=str(a.output.resolve()),FREESURFER_HOME=str(a.assets.resolve()),
                     OMP_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads))
    report={"scope":"complete_same_input_discrete_curv_stats_and_eight_maps_not_raw_T1",
        "code_base_commit":a.code_version,"source_sha256":{m.__name__:sha(m.__file__) for m in source_modules},
        "common_dependency_package":str(Path(fnit.recon_all.__file__).resolve()),
        "script_sha256":sha(__file__),"reference_program_sha256":sha(a.reference_program),
        "hostname":platform.node(),"cpu_affinity":sorted(os.sched_getaffinity(0)),"threads":a.threads,
        "cpu_model":next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),None),
        "system_libc":platform.libc_ver(),
        "torch":torch.__version__,"numpy":np.__version__,"device":a.device,"angle_backend":a.angle_backend,
        "cuda_setup_seconds":cuda_setup_seconds,"api_wall_scope":"includes validation, reads, JIT, transfer, maps/statistics and writes; imports/global CUDA setup separate",
        "tf32_matmul":torch.backends.cuda.matmul.allow_tf32,"tf32_cudnn":torch.backends.cudnn.allow_tf32,
        "half_precision":False,"cuda_allocator_environment":{k:os.environ.get(k) for k in
           ("PYTORCH_NO_CUDA_MEMORY_CACHING","PYTORCH_CUDA_ALLOC_CONF","PYTORCH_ALLOC_CONF")},
        "threshold_before_test":{"max_abs":1e-5,"p99_abs":1e-6},"cases":[],"status":"running",
        "whole_recon_all":"not_run","overall_metric_equivalence":"not_assessed"}
    save=lambda:(a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    save()
    for index,subject in enumerate(a.subject):
      for h in ("lh","rh"):
        files=[subject/"surf"/f"{h}.{name}" for name in ("smoothwm","curv","sulc")]
        xyz,faces=fsio.read_geometry(str(files[0]))
        row={"case_index":index,"hemisphere":h,"input_sha256":{f.name:sha(f) for f in files},
             "input_mesh":{"vertices":len(xyz),"faces":len(faces),"ordered_face_sha256":hashlib.sha256(faces.tobytes()).hexdigest(),
                           "coordinates_sha256":hashlib.sha256(xyz.astype(np.float32).tobytes()).hexdigest(),"coordinate_space":"surface RAS/mm"},
             "native_seconds":[],"native_commands":[],"trials":[]}
        report["cases"].append(row);save()
        native=[]
        for repeat in range(2):
            folder=a.output/f"reference_{index}_{h}_{repeat}"
            for name in ("surf","stats","scripts"):(folder/name).mkdir(parents=True)
            for f in files:shutil.copyfile(f,folder/"surf"/f.name)
            command=[str(a.reference_program.resolve()),"-m","--writeCurvatureFiles","-G","-o",str(folder/"stats"/f"{h}.curv.stats"),"-F","smoothwm",folder.name,h,"curv","sulc"]
            row["native_commands"].append(command)
            started=time.perf_counter()
            with (folder/"native.log").open("w") as stream:
                subprocess.run(command,env=environment,cwd=folder/"scripts",stdout=stream,stderr=subprocess.STDOUT,check=True)
            row["native_seconds"].append(time.perf_counter()-started);native.append(folder);save()
        row["native_repeat_maps"]={name:map_comparison(native[0]/"surf"/f"{h}.smoothwm.{name}.crv",native[1]/"surf"/f"{h}.smoothwm.{name}.crv") for name in MAP_ORDER}
        row["native_repeat_stats"]=stats_comparison(native[0]/"stats"/f"{h}.curv.stats",native[1]/"stats"/f"{h}.curv.stats")
        row["native_output_sha256"]=[{path.name:sha(path) for path in [folder/"stats"/f"{h}.curv.stats",*[folder/"surf"/f"{h}.smoothwm.{name}.crv" for name in MAP_ORDER]]} for folder in native]
        for iteration,device in enumerate(("cpu",a.device,a.device,"cpu")):
            output=a.output/f"candidate_{index}_{h}_{iteration}"
            if device.startswith("cuda"):
                torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            result=write_curvature_stats(subject_dir=subject,hemisphere=h,stats_file=output/f"{h}.curv.stats",
                output_prefix=output/f"{h}.smoothwm",device=device,angle_backend=a.angle_backend)
            trial={"iteration":iteration,"device":device,"result":result,
                   "output_sha256":{name:sha(path) for name,path in result["outputs"].items()},
                   "maps":{name:map_comparison(native[0]/"surf"/f"{h}.smoothwm.{name}.crv",result["outputs"][name]) for name in MAP_ORDER},
                   "stats":stats_comparison(native[0]/"stats"/f"{h}.curv.stats",result["outputs"]["stats"])}
            if device.startswith("cuda"):
                trial["peak_allocated_bytes"]=torch.cuda.max_memory_allocated(device)
                trial["peak_reserved_bytes"]=torch.cuda.max_memory_reserved(device)
            row["trials"].append(trial);save()
        row["input_sha256_after"]={f.name:sha(f) for f in files};save()
    report["source_sha256_after"]={m.__name__:sha(m.__file__) for m in source_modules}
    report["status"]="complete";save()


if __name__=="__main__":main()
