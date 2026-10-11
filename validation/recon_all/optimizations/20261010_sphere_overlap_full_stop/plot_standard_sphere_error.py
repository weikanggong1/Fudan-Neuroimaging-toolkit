"""真实同网格球面误差投到inflated显示，完整网格计算/抽样绘图分开。"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inflated", type=Path, required=True)
    parser.add_argument("--old-sphere", type=Path, required=True)
    parser.add_argument("--fixed-sphere", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mesh, faces = fsio.read_geometry(args.inflated)
    values = []
    for path in [args.old_sphere,args.fixed_sphere,args.reference]:
        coordinates, candidate_faces = fsio.read_geometry(path)
        if coordinates.shape != mesh.shape or not np.array_equal(faces,candidate_faces):
            raise ValueError("ordered vertex/face correspondence required for error display")
        values.append(coordinates)
    old,fixed,reference=values
    errors=[np.linalg.norm(old-reference,axis=1),np.linalg.norm(fixed-reference,axis=1)]
    sample=np.arange(0,len(mesh),8)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,error,title in zip(axes[:2],errors,["Before: archived FNIT v4","Fixed source expressions"]):
        colored=ax.scatter(mesh[sample,1],mesh[sample,2],c=error[sample],s=3,vmin=0,vmax=10,cmap="viridis")
        ax.set_title(title);ax.set_xlabel("inflated surface RAS Y (mm)");ax.set_ylabel("Z (mm)");ax.set_aspect("equal")
        fig.colorbar(colored,ax=ax,label="same-index sphere distance (mm)")
    triangle=reference[faces]
    cross=np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0])
    radial=np.einsum("ij,ij->i",cross,triangle.sum(axis=1))
    negative=radial<0
    centers=mesh[faces[negative]].mean(axis=1)
    axes[2].scatter(mesh[sample,1],mesh[sample,2],c="lightgray",s=2)
    axes[2].scatter(centers[:,1],centers[:,2],c="crimson",s=3)
    axes[2].set_title(f"Reference and fixed: {negative.sum()} folds")
    axes[2].set_xlabel("inflated surface RAS Y (mm)");axes[2].set_ylabel("Z (mm)");axes[2].set_aspect("equal")
    fig.savefig(args.output,dpi=160)
    report={"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256":{key:hashlib.sha256(getattr(args,key).read_bytes()).hexdigest()
                             for key in ["inflated","old_sphere","fixed_sphere","reference"]},
            "plot_sample_stride":8,"metrics_use_complete_mesh":True,"vertices":len(mesh),"faces":len(faces),
            "reference_negative_faces_double":int(negative.sum()),
            "old_mean_p99_max_mm":[float(errors[0].mean()),float(np.percentile(errors[0],99)),float(errors[0].max())],
            "fixed_mean_p99_max_mm":[float(errors[1].mean()),float(np.percentile(errors[1],99)),float(errors[1].max())],
            "output_sha256":hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
