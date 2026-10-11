"""真实完整组脑图与CLI墙钟；仅显示变换，不改变生产体素、标签或空间。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(*, fnit_root: Path, summary: Path, output: Path) -> dict:
    """输入授权公开T1运行根/本轮摘要，输出两PNG及来源SHA、RAS切片毫米坐标。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    results = json.loads(summary.read_bytes())
    if results["status"] != "complete_same_input_groups":
        raise ValueError("plots require complete actual groups")
    provenance = {"scope": "derived visualization of authorized public ds000114 T1 and own complete group outputs only",
        "plot_script_sha256": digest(Path(__file__)), "summary_sha256": digest(summary),
        "display_orientation": "nibabel.as_closest_canonical RAS display only; output geometry not changed", "cases": {}}
    figure, axes = plt.subplots(2, 3, figsize=(11, 7.5), constrained_layout=True)
    for row, case in enumerate(("sub07", "sub06")):
        original = fnit_root / f"runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1/{case}-candidate/subject/mri/orig.mgz"
        run = fnit_root / f"runs/input_stage_parallel_20261010_v2/ABBA/{case}/1-serial"
        paths = {"orig": original, "nu0": run / "nu0.mgz", "segmentation": run / "synthseg.mgz"}
        images = {name: nib.as_closest_canonical(nib.load(str(path))) for name, path in paths.items()}
        if any(image.shape != images["orig"].shape or not np.array_equal(image.affine, images["orig"].affine) for image in images.values()):
            raise ValueError("not same production output grid")
        z = images["orig"].shape[2]//2
        provenance["cases"][case] = {"inputs_sha256": {name: digest(path) for name,path in paths.items()},
            "RAS_slice_index": z, "scanner_RAS_z_mm": float((images["orig"].affine @ [0,0,z,1])[2]),
            "parallel_outputs": "numerically identical to serial, backed by full ABBA voxel/dtype/affine/tkr/Dice/CSV comparisons"}
        gray = np.rot90(np.asanyarray(images["nu0"].dataobj)[:,:,z])
        for col, name in enumerate(("orig", "nu0", "segmentation")):
            axis = axes[row,col]
            data = np.rot90(np.asanyarray(images[name].dataobj)[:,:,z])
            axis.imshow(gray if name == "segmentation" else data, cmap="gray", vmin=0,vmax=255)
            if name == "segmentation":
                ids = np.unique(data[data != 0])
                color = np.zeros((*data.shape,4),dtype=np.float32)
                palette = plt.get_cmap("tab20")
                for index,label in enumerate(ids):
                    color[data == label] = (*palette(index % 20)[:3],.6)
                axis.imshow(color)
            axis.set_title(f"{case}: { {'orig':'Conformed T1','nu0':'Complete N4','segmentation':'Full GPU SynthSeg'}[name] }")
            axis.axis("off")
    figure.suptitle("Same-input complete groups: serial / parallel output differences = 0\nDisplay colours only; original label values and spatial geometry preserved")
    picture=output/"brain_outputs.png"
    figure.savefig(picture,dpi=170)
    plt.close(figure)
    figure,axis=plt.subplots(figsize=(7.5,4.5),constrained_layout=True)
    for index,case in enumerate(("sub07","sub06")):
        row=results["cases"][case]
        for offset,mode,color in ((-.17,"serial","#61778a"),(.17,"parallel","#1c9b8e")):
            x=index+offset
            actual=[trial["complete_CLI_seconds"] for trial in row["trials"] if trial["execution"]==mode]
            value=row[mode+"_complete_CLI_median_seconds"]
            axis.bar(x,value,width=.30,color=color,label=mode if index==0 else None)
            axis.scatter([x-.04,x+.04],actual,color="black",s=16,zorder=3)
            axis.text(x,value+2,f"{value:.2f}",ha="center",fontsize=10)
    axis.set(xticks=[0,1],xticklabels=["sub-07","sub-06"],ylabel="Complete group CLI wall clock (s)",
             title="N4 + SynthSeg, same machine / GPU / total CPU4\nABBA observations; not whole recon-all speedup")
    axis.legend()
    axis.spines[["top","right"]].set_visible(False)
    picture=output/"complete_group_times.png"
    figure.savefig(picture,dpi=170)
    plt.close(figure)
    provenance["image_sha256"]={path.name:digest(path) for path in output.glob("*.png")}
    (output/"plot_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    return provenance


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("fnit-root","summary","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    result=plot(**vars(parser.parse_args()))
    print(json.dumps(result["image_sha256"]))
