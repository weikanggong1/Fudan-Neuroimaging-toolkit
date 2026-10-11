"""清理旧未调用兼容入口后，最终CPU源码对冻结同输入native完整回归。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("fnit-directory","module-directory","output-directory"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--code-version",required=True)
    args=parser.parse_args()
    import numpy as np
    import nibabel as nib
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(args.module_directory.resolve()))
    from fnit.recon_all import edit_wm_aseg_core_python as core
    from fnit.recon_all import wm_aseg_io as io
    args.output_directory.mkdir(parents=True,exist_ok=False)
    for case in ("sub07","sub06"):
        baseline=args.fnit_directory/"runs/recon_wm_aseg_independent_20261010_v1"/case
        producer=json.loads((baseline/"report.json").read_text())
        mri=args.fnit_directory/"runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1"/(case+"-candidate")/"subject/mri"
        inputs={name:mri/(name+".mgz") for name in ("wm.seg","brain","aseg.presurf","entowm")}
        for name,path in inputs.items():
            if sha(path)!=producer["inputs_sha256"][name]:raise ValueError("producer input hash differs")
        reference=baseline/"0-native.mgz"
        if sha(reference)!=producer["records"][0]["output_sha256"]:raise ValueError("native checkpoint hash differs")
        folder=args.output_directory/case;folder.mkdir()
        output=folder/"output.mgz"
        detail=core.write_wm_asegedit_profile(wm_file=inputs["wm.seg"],brain_file=inputs["brain"],
            aseg_file=inputs["aseg.presurf"],entowm_file=inputs["entowm"],output_file=output,fill_seg_wm=True)
        a,b=[nib.load(str(p)) for p in (output,reference)]
        aa,bb=[np.asarray(i.dataobj) for i in (a,b)]
        absolute=np.abs(aa.astype(np.int16)-bb.astype(np.int16))
        report={"status":"complete","scope":"final CPU module full file API against SHA-frozen native checkpoint; no timing claim",
            "code_version":args.code_version,"script_sha256":sha(__file__),"core_sha256":sha(core.__file__),
            "io_sha256":sha(io.__file__),"inputs_sha256":producer["inputs_sha256"],"reference_sha256":sha(reference),
            "output_sha256":sha(output),"detail":detail,"threads":1,"different_voxels":int(np.count_nonzero(absolute)),
            "maximum_absolute_error":int(absolute.max()),"p99_absolute_error":float(np.percentile(absolute,99)),
            "affine_exact":bool(np.array_equal(a.affine,b.affine)),"dtype_exact":a.get_data_dtype()==b.get_data_dtype()}
        (folder/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        if report["different_voxels"] or not report["affine_exact"] or not report["dtype_exact"]:
            raise ValueError("final CPU regression failed")
        print(case,report["different_voxels"],flush=True)


if __name__=="__main__":main()
