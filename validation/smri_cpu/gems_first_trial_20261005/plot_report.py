"""Render scalar diagnostics already in this directory; no MRI/mesh is read."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    directory=Path(__file__).resolve().parent
    structures=["hippo-amygdala-right","thalamus","hippo-amygdala-left"]
    python=[];raw=[];normalized=[]
    for structure in structures:
        python.append(json.loads((directory/(structure+"_python_summary.public.json")).read_text()))
        native_name=structure+"_native_summary.public.json" if structure.endswith("right") else structure+"_native_v2_summary.public.json"
        raw.append(json.loads((directory/native_name).read_text()))
        new=directory/(structure+"_native_v5_summary.public.json")
        normalized.append(json.loads(new.read_text()) if new.exists() else None)
    figure,axes=plt.subplots(1,2,figsize=(10.5,4.0),layout="constrained")
    x=np.arange(3);labels=["Right HA","Thalamus","Left HA"]
    for offset,key,color,label in [(-.24,"armijo","#42759d","FNIT Armijo"),(0,"reference","#77a55d","Source-defined first step")]:
        axes[0].bar(x+offset,[p["searches"][key]["selected_max_node_displacement"] for p in python],width=.24,color=color,label=label)
    axes[0].bar(x+.24,[r["native_step"]["returned_maximal_deformation"] for r in raw],width=.24,color="#d3a45c",label="Native raw-parity diagnostic")
    axes[0].set(xticks=x,xticklabels=labels,ylabel="Maximum accepted movement (voxel)",ylim=(0,1.25))
    axes[0].legend(fontsize=7,loc="upper right")
    axes[0].set_title("Captured first update")
    axes[1].semilogy(x,[r["same_points"]["armijo_selected"]["complete_projected_gradient_vs_python"]["relative_l2"] for r in raw],"o-",color="#b45c5c",label="Raw set_positions oracle (v1/v2)")
    if all(normalized):
        axes[1].semilogy(x,[r["same_points"]["armijo_selected"]["complete_projected_gradient_vs_python"]["relative_l2"] for r in normalized],"s-",color="#42759d",label="Native recipe parity oracle (v5)")
    axes[1].axhline(1e-5,linestyle="--",color="0.5",linewidth=.8,label="Predeclared same-point gradient gate")
    axes[1].set(xticks=x,xticklabels=labels,ylabel="Complete projected gradient relative L2")
    axes[1].set_title("Identical FP32 accepted points")
    axes[1].legend(fontsize=7,loc="upper left")
    figure.savefig(directory/"first_trial_scalars.png",dpi=180)
    plt.close(figure)


if __name__=="__main__":main()
