"""Real GM stage slices; intensity errors use the same color range in both arms."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("slices","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    arrays=np.load(args.slices)
    mask=arrays["mask"]
    slice_index=int(arrays["slice_index"])
    figure,axes=plt.subplots(2,3,figsize=(10,6.6),layout="constrained")
    for row,(name,label) in enumerate((
        ("warped","Warped GM"),
        ("nonlinear_jacobian","Nonlinear Jacobian"),
    )):
        target=arrays[name+"_official"]
        errors=[arrays[name+"_"+arm+"_error"] for arm in ("baseline","candidate")]
        error_range=float(arrays[name+"_error_range"])
        axes[row,0].imshow(target.T,origin="lower",cmap="gray")
        axes[row,0].set_title(label+" official")
        for column,(error,title) in enumerate(zip(errors,("Baseline absolute error","CPU orientation absolute error")),start=1):
            shown=np.ma.masked_where(~mask.T,error.T)
            image=axes[row,column].imshow(shown,origin="lower",cmap="magma",vmin=0,vmax=error_range)
            axes[row,column].set_title(title)
        figure.colorbar(image,ax=axes[row,1:],label="Absolute error")
    for axis in axes.ravel(): axis.set_axis_off()
    figure.suptitle(f"Fixed real official GM / FLIRT; axial index {slice_index}\nShared P99 color range per row; metrics use all voxels")
    figure.savefig(args.output,dpi=170)


if __name__=="__main__":
    main()
