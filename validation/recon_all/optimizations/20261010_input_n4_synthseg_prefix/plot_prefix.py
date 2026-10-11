"""两例公开原始T1、conform、WM/filled与真实差异图，只导出PNG及来源JSON。

显式传runs/已完成比较目录及两份原始T1。所有画图输入须与比较SHA
完全匹配，串行/并行空间相同；原始与conform各按自身canonical RAS网格
展示，不把不同网格当作同体素。输出目录必须不存在，不保存影像数组。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def plot(*, runs_root: Path, comparisons: Path, raw_sub06: Path,
         raw_sub07: Path, output: Path) -> dict:
    """返回JSON来源记录；PNG显示RAS轴切面及全部体素误差计数。

    rows为原始T1、conform、串行/并行WM和filled以及差异叠加。WM>=5
    保留WM_MIN_VAL语义，filled255/127表示左/右；切面不是整例验收。
    serial-minus-parallel符号不反转：差异为parallel-serial。使用
    对应orig同一RAS切面，误差全零时选择非零脑区包围盒中心。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap
    import nibabel as nib
    import numpy as np
    path = Path(__file__).with_name("compare_prefix.py")
    spec = importlib.util.spec_from_file_location("input_prefix_compare", path)
    comparator = importlib.util.module_from_spec(spec); spec.loader.exec_module(comparator)
    summary = json.loads((comparisons/"summary.json").read_bytes())
    if summary["status"] != "complete_prefix_comparison" or summary["full_workflow_complete"]:
        raise ValueError("two complete prefix comparisons required")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    provenance = {"scope": "public ds000114 real raw T1-to-filled serial/parallel prefix; not full recon-all acceptance",
        "plot_script_sha256": comparator.sha(Path(__file__)),
        "comparator_sha256": comparator.sha(path), "comparison_summary_sha256": comparator.sha(comparisons/"summary.json"),
        "coordinate_display": "closest canonical RAS; raw and conformed rows use own grid; display axes X/Y/Z",
        "slice_rule": "conformed RAS plane with maximum changed voxels; if all zero, nonzero-orig bounding-box center; raw uses own nonzero bounding-box center",
        "cases": []}
    for case, raw in (("sub07", raw_sub07), ("sub06", raw_sub06)):
        comparison_path = comparisons/(case+".json")
        row = json.loads(comparison_path.read_bytes())
        if comparator.sha(raw) != row["original_T1_sha256"]:
            raise ValueError("raw T1 is not comparison input")
        dirs = [runs_root/name/"subject/mri" for name in row["producer_directories"]]
        names = ("orig.mgz", "T1.mgz", "wm.mgz", "filled.mgz")
        images, data, bindings = {}, {}, {}
        for name in names:
            paths = [directory/name for directory in dirs]
            hashes = [comparator.sha(path) for path in paths]
            detail = row["volumes"][name]
            if hashes != detail["files_sha256"] or not detail["geometry"]["voxelwise_comparable"]:
                raise ValueError("plot SHA/grid differs from comparator: "+name)
            images[name] = [nib.as_closest_canonical(nib.load(str(path))) for path in paths]
            data[name] = [np.asarray(image.dataobj) for image in images[name]]
            delta = data[name][1].astype(np.float64)-data[name][0].astype(np.float64)
            if int(np.count_nonzero(delta)) != detail["different_voxels"]:
                raise ValueError("plot values differ from comparator")
            bindings[name] = {"files_sha256": hashes, "full_volume_different_voxels": int(np.count_nonzero(delta)),
                              "full_volume_max_abs": float(np.abs(delta).max())}
        orig = data["orig.mgz"][0]
        for name in names:
            if any(image.shape != images["orig.mgz"][0].shape or not np.array_equal(image.affine,images["orig.mgz"][0].affine)
                   for image in images[name]):
                raise ValueError("conformed overlay grid mismatch: "+name)
        raw_image = nib.as_closest_canonical(nib.load(str(raw)))
        raw_values = np.asarray(raw_image.dataobj)
        if raw_values.ndim != 3 or not np.isfinite(raw_values).all():
            raise ValueError("finite 3D public T1 required")
        changed = np.logical_or.reduce([data[name][0] != data[name][1] for name in ("T1.mgz","wm.mgz","filled.mgz")])
        def indices(values, mask=None):
            mask = values != 0 if mask is None else mask
            occupied = np.argwhere(mask)
            center = ((occupied.min(axis=0)+occupied.max(axis=0))//2 if len(occupied) else np.asarray(values.shape)//2)
            return [int(np.argmax(mask.sum(axis=tuple(j for j in range(3) if j != axis))))
                    if mask is changed and mask.any() else int(center[axis]) for axis in range(3)]
        slices, raw_slices = indices(orig, changed if changed.any() else None), indices(raw_values)
        fig, axes = plt.subplots(7,3,figsize=(12,22),constrained_layout=True)
        raw_limit = float(np.quantile(raw_values[raw_values>0],.99)) if (raw_values>0).any() else 1
        intensity_delta = data["T1.mgz"][1].astype(np.float64)-data["T1.mgz"][0].astype(np.float64)
        wm_changed = data["wm.mgz"][0] != data["wm.mgz"][1]
        filled_changed = data["filled.mgz"][0] != data["filled.mgz"][1]
        for axis in range(3):
            take = lambda a: np.take(a,slices[axis],axis=axis).T
            ax = axes[0,axis]; ax.imshow(np.take(raw_values,raw_slices[axis],axis=axis).T,cmap="gray",origin="lower",vmin=0,vmax=raw_limit)
            ax.set_title(f"raw T1: RAS {'XYZ'[axis]} slice {raw_slices[axis]} (own grid)")
            axes[1,axis].imshow(take(orig),cmap="gray",origin="lower",vmin=0,vmax=160)
            axes[1,axis].set_title(f"conform: RAS {'XYZ'[axis]} slice {slices[axis]}")
            for trial,label in enumerate(("serial","parallel")):
                ax = axes[2+trial,axis]; ax.imshow(take(data["T1.mgz"][trial]),cmap="gray",origin="lower",vmin=0,vmax=160)
                ax.imshow(np.ma.masked_equal(take(data["wm.mgz"][trial]>=5),False),cmap=ListedColormap(["cyan"]),origin="lower",alpha=.6,interpolation="nearest")
                ax.set_title(label+" T1 + WM >=5 (cyan)")
                ax = axes[4+trial,axis]; ax.imshow(take(data["T1.mgz"][trial]),cmap="gray",origin="lower",vmin=0,vmax=160)
                for value,color in ((255,"orange"),(127,"deepskyblue")):
                    ax.imshow(np.ma.masked_equal(take(data["filled.mgz"][trial]==value),False),cmap=ListedColormap([color]),origin="lower",alpha=.55,interpolation="nearest")
                ax.set_title(label+" filled: LH255 orange / RH127 blue")
            ax = axes[6,axis]; ax.imshow(take(orig),cmap="gray",origin="lower",vmin=0,vmax=160)
            delta = take(intensity_delta); limit=max(1,float(np.abs(intensity_delta).max()))
            ax.imshow(np.ma.masked_equal(delta,0),cmap="coolwarm",origin="lower",vmin=-limit,vmax=limit,interpolation="nearest")
            for mask,color in ((wm_changed,"cyan"),(filled_changed,"yellow")):
                ax.imshow(np.ma.masked_equal(take(mask),False),cmap=ListedColormap([color]),origin="lower",alpha=.7,interpolation="nearest")
            ax.set_title(f"parallel - serial: T1 {np.count_nonzero(delta)}, WM {np.count_nonzero(take(wm_changed))}, filled {np.count_nonzero(take(filled_changed))} pixels")
        for ax in axes.flat: ax.axis("off")
        fig.suptitle(case+": current complete raw T1-to-filled prefix\nAll-volume differences T1/WM/filled: "+
            "/".join(str(bindings[name]["full_volume_different_voxels"]) for name in ("T1.mgz","wm.mgz","filled.mgz"))+"; overall equivalence not assessed",fontsize=13)
        axes[6,1].legend(handles=[Patch(color="cyan",label="WM value mismatch"),Patch(color="yellow",label="filled label mismatch")],loc="lower center",fontsize=8)
        png=output/(case+"_raw_conform_wm_filled.png");fig.savefig(png,dpi=130);plt.close(fig)
        provenance["cases"].append({"case":case,"comparison_sha256":comparator.sha(comparison_path),
            "raw_T1_sha256":comparator.sha(raw),"raw_display_shape_RAS":list(raw_image.shape),
            "conformed_display_shape_RAS":list(orig.shape),"raw_slices_RAS_XYZ":raw_slices,
            "conformed_slices_RAS_XYZ":slices,"volumes":bindings,"png_sha256":comparator.sha(png)})
    (output/"plot_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    return provenance


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("runs-root","comparisons","raw-sub06","raw-sub07","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    print(json.dumps(plot(**vars(parser.parse_args()))))
