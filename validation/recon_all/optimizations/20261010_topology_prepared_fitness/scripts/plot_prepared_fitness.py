"""公开T1的准备后候选评分图；不生成生产输入或修改候选。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from benchmark_recon_topology_mri_likelihood import load_statistics_capture


def run(*, statistics_capture, candidate_npz, report_json, output_png, output_receipt):
    started = time.perf_counter()
    if output_png.exists() or output_receipt.exists():
        raise FileExistsError("new diagnostic figure and receipt required")
    fields = load_statistics_capture(capture_path=statistics_capture)
    with np.load(candidate_npz,allow_pickle=False) as archive:
        distance = archive["distance_mm"].copy()
        cores = archive["core_fields"].copy()
    report = json.loads(report_json.read_text())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xyz = fields["vertex_fields"][:, :3]
    ids = fields["selected_vertices"]
    fig,axes = plt.subplots(1,3,figsize=(15,4.4))
    axes[0].scatter(xyz[::8,1],xyz[::8,2],color="#c6cbd0",s=.3,rasterized=True)
    axes[0].scatter(xyz[ids,1],xyz[ids,2],color="#b43a37",s=12)
    axes[0].set_aspect("equal");axes[0].set_xlabel("Surface RAS Y (mm)");axes[0].set_ylabel("Z (mm)")
    axes[0].set_title("Public sub-07 LH: 99-vertex candidate")
    middle = distance[:,:,distance.shape[2]//2].copy()
    middle[middle == np.float32(1000)] = np.nan
    image = axes[1].imshow(middle.T,origin="lower",cmap="coolwarm",vmin=-2,vmax=2)
    axes[1].set_title("FNIT geometry-derived distance (mm)");axes[1].set_xlabel("Local grid X");axes[1].set_ylabel("Y")
    fig.colorbar(image,ax=axes[1],shrink=.75)
    labels = ["face","vertex","normal","curvature","unMRI"]
    axes[2].bar(labels,cores,color=["#487bb2"]*5)
    axes[2].set_ylabel("Source TP log likelihood (float32)")
    axes[2].set_title("Five recomputed terms: all 0-bit error")
    axes[2].tick_params(axis="x",rotation=35)
    fig.suptitle("Prepared-stage diagnostic; upstream SFF/MRI preparation and GA are not assessed",fontsize=11)
    fig.tight_layout();output_png.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output_png,dpi=160);plt.close(fig)
    sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt=dict(scope="public derived diagnostic figure only, not final mesh QC",
        input_sha256=dict(statistics_capture=sha(statistics_capture),candidate_npz=sha(candidate_npz),report_json=sha(report_json)),
        script_sha256=sha(__file__),output_sha256=sha(output_png),seconds_including_load_plot_write=time.perf_counter()-started,
        selected_vertices=len(ids),distance_shape=list(distance.shape),
        strict_stage_output_bytes=report["strict_stage_output_bytes"],
        overall_internal_byte_reproduction=report["overall_internal_byte_reproduction"])
    output_receipt.parent.mkdir(parents=True,exist_ok=True)
    output_receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("statistics-capture","candidate-npz","report-json","output-png","output-receipt"):
        parser.add_argument("--"+name,type=Path,required=True)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))


if __name__ == "__main__":
    main()
