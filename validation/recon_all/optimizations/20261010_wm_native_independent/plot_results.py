"""使用已声明公开T1及实际前后WM结果生成叠加图；不修改算法输入。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-directory",type=Path,required=True)
    parser.add_argument("--output-directory",type=Path,required=True)
    args=parser.parse_args()
    import nibabel as nib
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.output_directory.mkdir(parents=True,exist_ok=True)
    receipts={}
    for case in ("sub-07","sub-06"):
        paths={"t1":args.fnit_directory/"runs/recon_torch_wm_fill_20261009/public_wm_fill_inputs_v1/inputs"/case/"antsdn.brain.mgz"}
        for name,version,backend in (("old","v1","candidate"),("new","v2","candidate"),("native","v2","native")):
            paths[name]=args.fnit_directory/"runs"/("recon_wm_native_independent_20261010_"+version)/case/backend/"output.mgz"
        arrays={name:np.asarray(nib.load(str(path)).dataobj) for name,path in paths.items()}
        delta=arrays["old"]!=arrays["native"]
        z=int(np.argmax(np.count_nonzero(delta,axis=(0,1))))
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
        for ax,title in zip(axes,("T1 + reference WM","Previous WM difference","Repaired WM difference")):
            ax.imshow(arrays["t1"][:,:,z].T,cmap="gray",origin="lower",vmin=0,vmax=130);ax.set_title(title);ax.axis("off")
        axes[0].imshow(np.ma.masked_where(~(arrays["native"][:,:,z]>=5).T,(arrays["native"][:,:,z]>=5).T),cmap="Greens",origin="lower",alpha=.3,vmin=0,vmax=1)
        for ax,name in zip(axes[1:],("old","new")):
            changed=(arrays[name][:,:,z]!=arrays["native"][:,:,z]).T
            ax.imshow(np.ma.masked_where(~changed,changed),cmap="Reds",origin="lower",alpha=.9,vmin=0,vmax=1)
        fig.suptitle(case+" same-input WM; slice z="+str(z)+"; whole-volume differences "+str(int(delta.sum()))+" -> "+str(int(np.count_nonzero(arrays["new"]!=arrays["native"]))))
        output=args.output_directory/(case+"_wm_repair.png");fig.savefig(output,dpi=160);plt.close(fig)
        receipts[output.name]={"source_sha256":{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()},
            "png_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"slice_z":z}
    (args.output_directory/"plot_provenance.json").write_text(json.dumps(receipts,indent=2)+"\n")


if __name__=="__main__":main()
