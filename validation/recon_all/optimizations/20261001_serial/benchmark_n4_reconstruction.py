"""两例自产orig：旧程序/new1/new4；固定拟合1线程，比较完整float输出及MGZ。"""
import argparse,hashlib,json,os,platform,subprocess,time
from pathlib import Path
import nibabel as nib
import numpy as np
from fnit.recon_all.n4_itk import _to_uchar
from fnit.recon_all.mgh_compat import save_same_dtype_mgh

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--orig",type=Path,required=True)
p.add_argument("--old-binary",type=Path,required=True)
p.add_argument("--new-binary",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--commit",required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
reports={};baseline=None
for name,binary,threads in (("old",a.old_binary,1),("new1",a.new_binary,1),("new4",a.new_binary,4)):
    tick=time.perf_counter()
    source=nib.load(str(a.orig));raw=a.output/(name+".input.f32");out=a.output/(name+".output.f32")
    np.asarray(source.dataobj,dtype=np.float32).ravel(order="F").tofile(raw)
    command=[str(binary),str(raw),str(out),*map(str,source.shape),*map(str,source.header.get_zooms()[:3])]
    profile=a.output/(name+".profile.json")
    if name!="old":command.extend([str(threads),str(profile)])
    subprocess.run(command,check=True)
    result=np.fromfile(out,dtype=np.float32)
    if result.size!=np.prod(source.shape):raise RuntimeError("output shape differs")
    uchar=_to_uchar(result.reshape(source.shape,order="F"))
    mgz=a.output/(name+".mgz");save_same_dtype_mgh(a.orig,mgz,uchar)
    seconds=time.perf_counter()-tick
    if baseline is None:baseline=result.copy()
    delta=np.abs(result.astype(np.float64)-baseline.astype(np.float64))
    reports[name]={"command":command,"seconds_including_io":seconds,"binary_sha256":sha(binary),
        "float_sha256":sha(out),"mgz_sha256":sha(mgz),"profile":json.loads(profile.read_text()) if profile.exists() else None,
        "comparison_to_old":{"different_float32_values":int(np.count_nonzero(result!=baseline)),
            "max_float_error":float(delta.max()),"p99_float_error":float(np.quantile(delta,.99)),
            "different_uint8_voxels":int(np.count_nonzero(uchar.ravel(order="F")!=_to_uchar(baseline))),
            "geometry_equal":np.array_equal(nib.load(str(mgz)).affine,source.affine)}}
    if reports[name]["comparison_to_old"]["different_uint8_voxels"]:
        raise RuntimeError("N4 output regression; do not enable threaded reconstruction")
    raw.unlink()
report={"commit":a.commit,"host":platform.node(),"scope":"frozen_self_generated_orig_full_N4; not_raw_T1_whole_case",
    "input_sha256":sha(a.orig),"benchmark_sha256":sha(__file__),"threads":{k:os.getenv(k) for k in ("OMP_NUM_THREADS","ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")},"results":reports}
(a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report),flush=True)
