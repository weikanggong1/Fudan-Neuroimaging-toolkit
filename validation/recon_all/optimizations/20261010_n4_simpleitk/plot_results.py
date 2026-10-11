"""仅从已声明公开输入和N4诊断输出生成脑图，不读取私人影像。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-directory",type=Path,required=True)
    parser.add_argument("--output-directory",type=Path,required=True)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    args.output_directory.mkdir(parents=True,exist_ok=False)
    records=[]
    for case in ("sub07","sub06"):
        directory=args.fnit_directory/"runs/recon_n4_python_adapter_20261010_v1"/case
        report=json.loads((directory/"report.json").read_text())
        source=args.fnit_directory/"runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1"/(case+"-candidate")/"subject/mri/orig.mgz"
        if report["status"]!="complete" or sha(source)!=report["source_sha256"]:
            raise ValueError("input producer/source differs")
        if sha(directory/"python_float.npy")!=report["python_float_sha256"] or sha(directory/"native-0.f32")!=report["native_float_sha256"]:
            raise ValueError("N4 output hash differs")
        original=np.asarray(nib.load(str(source)).dataobj,dtype=np.float32)
        candidate=np.load(directory/"python_float.npy")
        reference=np.fromfile(directory/"native-0.f32",dtype=np.float32).reshape(original.shape,order="F")
        quantize=lambda image:np.floor(np.clip(image,0,255)+.5).astype(np.uint8)
        differences=quantize(candidate)!=quantize(reference)
        absolute=np.abs(candidate-reference)
        figure,axes=plt.subplots(2,3,figsize=(12,8))
        cuts=[]
        for row,axis in enumerate((1,2)):
            counts=differences.sum(axis=tuple(n for n in range(3) if n!=axis))
            index=int(counts.argmax());cuts.append({"voxel_axis":axis,"voxel_slice":index,"quantized_different_voxels":int(counts[index])})
            take=lambda image:np.rot90(np.take(image,index,axis=axis))
            axes[row,0].imshow(take(reference),cmap="gray",vmin=0,vmax=180)
            axes[row,0].set_title("Native N4 / voxel axis "+str(axis)+"="+str(index))
            image=axes[row,1].imshow(take(absolute),cmap="magma",vmin=0,vmax=report["comparisons"]["float_python_vs_native"]["maximum_absolute_error"])
            figure.colorbar(image,ax=axes[row,1],fraction=.046)
            axes[row,1].set_title("Absolute FP32 error")
            axes[row,2].imshow(take(original),cmap="gray",vmin=0,vmax=180)
            mask=take(differences)
            axes[row,2].imshow(np.ma.masked_where(~mask,mask),cmap="autumn",vmin=0,vmax=1,alpha=.9)
            y,x=np.nonzero(mask);axes[row,2].scatter(x,y,s=25,facecolors="none",edgecolors="red",linewidths=.7)
            axes[row,2].set_title("T1 + uint8 rounding differences")
        for axis in axes.ravel():axis.axis("off")
        figure.suptitle(case+" / complete same-input SimpleITK N4 / unchanged XYZ conformed grid")
        figure.tight_layout()
        output=args.output_directory/(case+"_n4.png");figure.savefig(output,dpi=140);plt.close(figure)
        records.append({"case":case,"source_sha256":sha(source),"array_report_sha256":sha(directory/"report.json"),
            "native_float_sha256":sha(directory/"native-0.f32"),"python_float_sha256":sha(directory/"python_float.npy"),
            "plot_sha256":sha(output),"cuts":cuts,"coordinates":"conformed XYZ voxel indices; same mm affine, no resampling"})
    (args.output_directory/"plot_provenance.json").write_text(json.dumps(records,indent=2)+"\n")


if __name__=="__main__":main()
