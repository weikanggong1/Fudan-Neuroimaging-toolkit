"""最终执行模块SHA的完整WM文件API回归；只读带SHA参考检查点。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","reference-report","reference-directory","module-directory","output-directory"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",default="cuda:1")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--code-version",required=True)
    args=parser.parse_args()
    frozen=json.loads(args.reference_report.read_text())
    reference=args.reference_directory/"output.mgz"
    if frozen["status"]!="complete" or sha(args.source)!=frozen["source_sha256"]:
        raise ValueError("frozen source or completion differs")
    if sha(reference)!=frozen["comparisons"]["output.mgz"]["reference_sha256"]:
        raise ValueError("frozen reference output SHA differs")
    args.output_directory.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import nibabel as nib
    import torch,numba
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(args.module_directory.resolve()))
    from fnit.recon_all import mri_segment as wm
    torch.set_num_threads(args.threads);numba.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    target=torch.device(args.device)
    if target.type=="cuda":torch.cuda.synchronize(target)
    output=args.output_directory/"output.mgz"
    started=time.perf_counter()
    api=wm.segment_white_matter_mgz(source_path=args.source,output_path=output,device=args.device,
        histogram_backend="torch",histogram_batch_size=2048,planar_backend="cached",planar_batch_size=256)
    if target.type=="cuda":torch.cuda.synchronize(target)
    seconds=time.perf_counter()-started
    a,b=[nib.load(str(p)) for p in (output,reference)]
    aa,bb=[np.asarray(i.dataobj) for i in (a,b)]
    absdiff=np.abs(aa.astype(np.int16)-bb.astype(np.int16))
    report={"status":"complete","scope":"same-input final-source complete WM file API, frozen native reference",
        "code_version":args.code_version,"module_sha256":sha(wm.__file__),"script_sha256":sha(__file__),
        "source_sha256":sha(args.source),"reference_report_sha256":sha(args.reference_report),
        "reference_sha256":sha(reference),"output_sha256":sha(output),"cpu_affinity":sorted(os.sched_getaffinity(0)),
        "device":args.device,"threads":args.threads,"api":api,"file_api_wall_seconds":seconds,
        "different_voxels":int(np.count_nonzero(absdiff)),"maximum_absolute_error":int(absdiff.max()),
        "p99_absolute_error":float(np.percentile(absdiff,99)),"affine_exact":bool(np.array_equal(a.affine,b.affine)),
        "dtype_exact":a.get_data_dtype()==b.get_data_dtype(),"memory_process_ownership":None}
    (args.output_directory/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    if report["different_voxels"] or not report["affine_exact"] or not report["dtype_exact"]:
        raise ValueError("final-source WM regression failed")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
