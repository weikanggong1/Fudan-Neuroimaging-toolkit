"""按真实角色画两例N4/WM/filled差异；不导出影像数组或原始路径。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(*, specification: Path, comparisons: Path, output: Path) -> dict:
    """读取完成的比较与显式原输入，写两张PNG及不可变哈希/切面来源。

    specification为compare_n4_prefix使用的同一JSON；comparisons含成功
    summary和两例JSON；output必须新目录。原始T1单独在自身canonical
    RAS网格展示；其他图必须和比较文件同SHA并同conform/RAS网格。
    切面选取WM/filled差异最多的位置，零差异时取orig脑区中心；差值为
    Torch-SimpleITK存储值，filled255/127为左/右。失败抛异常，不更改源图。
    没有独立官方命令，属于真实诊断可视化，不代替全体素数值/质量验收。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import nibabel as nib
    import numpy as np
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(specification.read_text())
    summary_path = comparisons / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary["status"] != "complete_n4_prefix_comparison" or summary["full_workflow_complete"]
            or sha(specification) != summary["specification_sha256"]):
        raise ValueError("complete same-specification N4 comparison required")
    output.mkdir(parents=True)
    report = {"scope": "real raw T1-to-filled N4 backend comparison; not whole reconstruction",
        "plot_script_sha256": sha(Path(__file__)), "comparison_summary_sha256": sha(summary_path),
        "coordinate_display": "canonical RAS; raw row uses own grid; all conformed rows use same grid",
        "delta": "Torch candidate minus SimpleITK reference; stored intensity unit",
        "cases": []}
    for case_spec in spec["cases"]:
        case = case_spec["case"]
        comparison_path = comparisons / (case + ".json")
        comparison = json.loads(comparison_path.read_text())
        raw = Path(case_spec["raw_t1"])
        if sha(raw) != comparison["original_T1_sha256"]:
            raise ValueError("raw T1 differs from comparison")
        subjects = [Path(case_spec[role]["trial_dir"]) / "subject" for role in ("reference", "candidate")]
        volumes, geometries = {}, {}
        for name in ("orig.mgz", "tmp/nu0.mgz", "T1.mgz", "wm.mgz", "filled.mgz"):
            files = [subject / "mri" / name for subject in subjects]
            detail = comparison["volumes"][name]
            if [sha(file) for file in files] != detail["files_sha256"] or not detail["geometry"]["voxelwise_comparable"]:
                raise ValueError("plot file hash/grid differs from comparator: " + name)
            images = [nib.as_closest_canonical(nib.load(str(file))) for file in files]
            arrays = [np.asarray(image.dataobj) for image in images]
            if int(np.count_nonzero(arrays[0] != arrays[1])) != detail["different_voxels"]:
                raise ValueError("plot values differ from completed comparison")
            volumes[name], geometries[name] = arrays, images
        orig = volumes["orig.mgz"][0]
        for images in geometries.values():
            if any(image.shape != orig.shape or not np.array_equal(image.affine, geometries["orig.mgz"][0].affine) for image in images):
                raise ValueError("conformed overlays do not share a grid")
        raw_image = nib.as_closest_canonical(nib.load(str(raw)))
        raw_values = np.asarray(raw_image.dataobj)
        changed = np.logical_or(volumes["wm.mgz"][0] != volumes["wm.mgz"][1],
                                volumes["filled.mgz"][0] != volumes["filled.mgz"][1])

        def center(values):
            occupied = np.argwhere(values != 0)
            return ((occupied.min(0) + occupied.max(0)) // 2).tolist() if len(occupied) else (np.asarray(values.shape)//2).tolist()

        slices = [int(np.argmax(changed.sum(axis=tuple(other for other in range(3) if other != axis))))
                  for axis in range(3)] if changed.any() else center(orig)
        raw_slices = center(raw_values)
        delta_n4 = volumes["tmp/nu0.mgz"][1].astype(np.float32) - volumes["tmp/nu0.mgz"][0].astype(np.float32)
        delta_t1 = volumes["T1.mgz"][1].astype(np.float32) - volumes["T1.mgz"][0].astype(np.float32)
        wm_changed = volumes["wm.mgz"][0] != volumes["wm.mgz"][1]
        filled_changed = volumes["filled.mgz"][0] != volumes["filled.mgz"][1]
        figure, axes = plt.subplots(8, 3, figsize=(12, 24), constrained_layout=True)
        raw_limit = float(np.quantile(raw_values[raw_values > 0], .99)) if (raw_values > 0).any() else 1
        for axis in range(3):
            def take(values): return np.take(values, slices[axis], axis=axis).T
            axes[0, axis].imshow(np.take(raw_values, raw_slices[axis], axis=axis).T, cmap="gray", origin="lower", vmin=0, vmax=raw_limit)
            axes[0, axis].set_title(f"raw T1: own RAS {'XYZ'[axis]} slice {raw_slices[axis]}")
            axes[1, axis].imshow(take(orig), cmap="gray", origin="lower", vmin=0, vmax=160)
            axes[1, axis].set_title(f"same orig: RAS {'XYZ'[axis]} slice {slices[axis]}")
            axes[2, axis].imshow(take(orig), cmap="gray", origin="lower", vmin=0, vmax=160)
            n4_limit = max(1, float(np.abs(delta_n4).max()))
            axes[2, axis].imshow(np.ma.masked_equal(take(delta_n4), 0), cmap="coolwarm", origin="lower", vmin=-n4_limit, vmax=n4_limit)
            axes[2, axis].set_title(f"nu0 Torch - SimpleITK: {np.count_nonzero(take(delta_n4))} pixels")
            for index, label in enumerate(("SimpleITK reference", "Torch candidate")):
                axis_wm = axes[3 + index, axis]
                axis_wm.imshow(take(volumes["T1.mgz"][index]), cmap="gray", origin="lower", vmin=0, vmax=160)
                axis_wm.imshow(np.ma.masked_equal(take(volumes["wm.mgz"][index] >= 5), False), cmap=ListedColormap(["cyan"]), origin="lower", alpha=.55)
                axis_wm.set_title(label + ": T1 + WM >=5")
                axis_fill = axes[5 + index, axis]
                axis_fill.imshow(take(volumes["T1.mgz"][index]), cmap="gray", origin="lower", vmin=0, vmax=160)
                for value, color in ((255, "orange"), (127, "deepskyblue")):
                    axis_fill.imshow(np.ma.masked_equal(take(volumes["filled.mgz"][index] == value), False), cmap=ListedColormap([color]), origin="lower", alpha=.55)
                axis_fill.set_title(label + ": filled LH255/RH127")
            ax = axes[7, axis]
            ax.imshow(take(orig), cmap="gray", origin="lower", vmin=0, vmax=160)
            limit = max(1, float(np.abs(delta_t1).max()))
            ax.imshow(np.ma.masked_equal(take(delta_t1), 0), cmap="coolwarm", origin="lower", vmin=-limit, vmax=limit)
            for mask, color in ((wm_changed, "cyan"), (filled_changed, "yellow")):
                ax.imshow(np.ma.masked_equal(take(mask), False), cmap=ListedColormap([color]), origin="lower", alpha=.7)
            ax.set_title(f"T1 / WM / filled differences: {np.count_nonzero(take(delta_t1))} / {np.count_nonzero(take(wm_changed))} / {np.count_nonzero(take(filled_changed))}")
        for ax in axes.flat: ax.axis("off")
        figure.suptitle(case + ": full-volume differences nu0/T1/WM/filled = " + "/".join(
            str(comparison["volumes"][name]["different_voxels"]) for name in ("tmp/nu0.mgz", "T1.mgz", "wm.mgz", "filled.mgz"))
            + "\nCross-version/time observations; overall equivalence not assessed", fontsize=13)
        png = output / (case + "_n4_wm_filled.png")
        figure.savefig(png, dpi=120); plt.close(figure)
        report["cases"].append({"case": case, "png_sha256": sha(png), "comparison_sha256": sha(comparison_path),
            "raw_T1_sha256": sha(raw), "raw_slices_RAS_XYZ": raw_slices, "conformed_slices_RAS_XYZ": slices,
            "files_sha256": {name: comparison["volumes"][name]["files_sha256"] for name in volumes}})
    (output / "plot_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("specification", "comparisons", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(plot(**vars(parser.parse_args()))))
