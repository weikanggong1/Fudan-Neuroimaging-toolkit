"""冻结完整N4数组比较后，实际文件API、几何和量化再回归；仅诊断。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("fnit-directory","module-directory","router-directory","output-directory"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--case",choices=("sub06","sub07"),required=True)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--code-version",required=True)
    args=parser.parse_args()
    import numpy as np
    import nibabel as nib
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(args.module_directory.resolve()))
    fnit.recon_all.__path__.insert(0,str(args.router_directory.resolve()))
    from fnit.recon_all import n4_simpleitk,n4_itk,input_n4_chain
    baseline=args.fnit_directory/"runs/recon_n4_python_adapter_20261010_v1"/args.case
    previous=json.loads((baseline/"report.json").read_text())
    if previous["status"]!="complete" or sha(n4_simpleitk.__file__)!=previous["module_sha256"]:
        raise ValueError("previous array result/source not complete or changed")
    source=args.fnit_directory/"runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1"/(args.case+"-candidate")/"subject/mri/orig.mgz"
    if sha(source)!=previous["source_sha256"]:raise ValueError("input hash changed")
    expected_file=baseline/"python_float.npy"
    native_file=baseline/"native-0.f32"
    if sha(expected_file)!=previous["python_float_sha256"] or sha(native_file)!=previous["native_float_sha256"]:
        raise ValueError("diagnostic reference checkpoint changed")
    args.output_directory.mkdir(parents=True,exist_ok=False)
    output=args.output_directory/"nu0.mgz"
    profile=args.output_directory/"profile.json"
    # 参考仅在调用后读取；完整文件生产API不接收参考输入。
    detail=input_n4_chain.run_n4_stage(input_path=source,output_path=output,
        n4_backend="simpleitk",n4_execution="in-process",threads=args.threads,
        device="cpu",report_path=profile)
    actual,original=[nib.load(str(p)) for p in (output,source)]
    data=np.asarray(actual.dataobj)
    expected=n4_itk._to_uchar(np.load(expected_file))
    native=n4_itk._to_uchar(np.fromfile(native_file,dtype=np.float32).reshape(original.shape,order="F"))
    def comparison(reference):
        difference=np.abs(data.astype(np.int16)-reference.astype(np.int16))
        return {"different_voxels":int(np.count_nonzero(difference)),"maximum_absolute_error":int(difference.max()),
                "p99_absolute_error":float(np.percentile(difference,99)),"mean_absolute_error":float(difference.mean())}
    report={"status":"complete","scope":"full Python file N4 stage; not raw-T1 continuous reconstruction",
        "code_version":args.code_version,"script_sha256":sha(__file__),
        "module_sha256":sha(n4_simpleitk.__file__),"wrapper_sha256":sha(n4_itk.__file__),
        "input_chain_sha256":sha(input_n4_chain.__file__),"source_sha256":sha(source),
        "baseline_report_sha256":sha(baseline/"report.json"),"output_sha256":sha(output),
        "comparisons":{"previous_python_quantized":comparison(expected),"native_quantized":comparison(native)},
        "geometry_exact":bool(np.array_equal(actual.affine,original.affine)),
        "shape_exact":actual.shape==original.shape,"output_dtype":str(actual.get_data_dtype()),
        "detail":detail,"threads":args.threads}
    (args.output_directory/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    if report["comparisons"]["previous_python_quantized"]["different_voxels"] or not report["geometry_exact"] or not report["shape_exact"]:
        raise RuntimeError("complete Python file API changed frozen result/geometry")
    print(json.dumps(report["comparisons"]),flush=True)


if __name__=="__main__":main()
