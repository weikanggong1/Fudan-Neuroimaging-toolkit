"""Summarize matched real-DWI ProbtrackX outputs without exporting subject images."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from probtrackx_validation import compare, _time

REGIONS = ("genu_cc", "cst_right", "cst_left", "slf_right", "slf_left")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--private-figure", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    report = {"reference": "FSL 6.0.7.22 probtrackx2 / probtrackx2_gpu",
              "posterior": "same original FSL BEDPOSTX full-brain three-fibre posterior",
              "settings": {"nsamples_per_seed_voxel": 200, "nsteps": 400,
                           "steplength_mm": 0.5, "curvature_threshold": 0.2,
                           "fibthresh": 0.01, "rseed": 20260927,
                           "fnit_batch_size": 256, "seed_voxels_each": 7},
              "regions": {}}
    for region in REGIONS:
        row = {"jhu_label": {"genu_cc": 3, "cst_right": 7, "cst_left": 8,
                             "slf_right": 41, "slf_left": 42}[region]}
        for device in ("cpu", "gpu"):
            fsl = root / f"fsl_{device}_{region}"
            fnit = root / f"fnit_{device}_{region}"
            for directory in (fsl, fnit):
                if not (directory / "fdt_paths.nii.gz").is_file() or not (directory / "waytotal").is_file():
                    raise FileNotFoundError(f"incomplete tracking output: {directory}")
            metrics = compare(fsl, fnit)
            metrics["wall_seconds_including_load_and_write"] = {
                "fsl": _time(root / f"fsl_{device}_{region}.time"),
                "fnit": _time(root / f"fnit_{device}_{region}.time")}
            row[device] = metrics
        report["regions"][region] = row
    repeat = compare(root / "fsl_cpu_genu_cc", root / "fsl_cpu_genu_cc_repeat")
    report["fsl_cpu_repeat_different_seed"] = {
        "seeds": [20260927, 20260928],
        "first_waytotal": repeat["fsl_waytotal"],
        "repeat_waytotal": repeat["fnit_waytotal"],
        "pearson_on_union_nonzero": repeat["pearson_on_union_nonzero"],
        "support_dice": repeat["support_dice"],
        "top_tenth_dice": repeat["top_tenth_dice"],
    }
    report["network_cpu"] = compare(root / "fsl_cpu_network", root / "fnit_cpu_network")
    fsl_matrix = np.atleast_2d(np.loadtxt(root / "fsl_cpu_network" / "fdt_network_matrix", dtype=np.int64))
    fnit_matrix = np.atleast_2d(np.loadtxt(root / "fnit_cpu_network" / "fdt_network_matrix", dtype=np.int64))
    if fsl_matrix.shape != (len(REGIONS), len(REGIONS)) or fnit_matrix.shape != fsl_matrix.shape:
        raise ValueError("invalid 5x5 ROI network matrix")
    report["network_cpu"]["fsl_matrix"] = fsl_matrix.tolist()
    report["network_cpu"]["fnit_matrix"] = fnit_matrix.tolist()
    report["network_cpu"]["wall_seconds_including_load_and_write"] = {
        "fsl": _time(root / "fsl_cpu_network.time"),
        "fnit": _time(root / "fnit_cpu_network.time")}
    high_fsl = root / "fsl_cpu_network_2000_20260927"
    high_fnit = root / "fnit_cpu_network_2000"
    high_repeat = root / "fsl_cpu_network_2000_20260928"
    report["network_cpu_2000"] = compare(high_fsl, high_fnit)
    report["network_cpu_2000"]["nsamples_per_seed_voxel"] = 2000
    for name, directory in (("fsl_matrix", high_fsl), ("fnit_matrix", high_fnit)):
        matrix = np.atleast_2d(np.loadtxt(directory / "fdt_network_matrix", dtype=np.int64))
        if matrix.shape != (len(REGIONS), len(REGIONS)):
            raise ValueError(f"invalid 5x5 ROI network matrix: {directory}")
        report["network_cpu_2000"][name] = matrix.tolist()
    report["network_cpu_2000"]["wall_seconds_including_load_and_write"] = {
        "fsl": _time(root / "fsl_cpu_network_2000_20260927.time"),
        "fnit": _time(root / "fnit_cpu_network_2000.time")}
    repeat = compare(high_fsl, high_repeat)
    report["network_cpu_2000_fsl_repeat"] = {
        "seeds": [20260927, 20260928],
        "pearson_on_union_nonzero": repeat["pearson_on_union_nonzero"],
        "support_dice": repeat["support_dice"],
        "top_tenth_dice": repeat["top_tenth_dice"],
        "first_matrix": report["network_cpu_2000"]["fsl_matrix"],
        "repeat_matrix": np.atleast_2d(np.loadtxt(
            high_repeat / "fdt_network_matrix", dtype=np.int64)).tolist(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(REGIONS), 3, figsize=(10, 2.4 * len(REGIONS)),
                                layout="constrained")
    for i, region in enumerate(REGIONS):
        fsl = np.asarray(nib.load(str(root / f"fsl_cpu_{region}" / "fdt_paths.nii.gz")).dataobj)
        fnit = np.asarray(nib.load(str(root / f"fnit_cpu_{region}" / "fdt_paths.nii.gz")).dataobj)
        seed = np.asarray(nib.load(str(root / f"seed_{region}.nii.gz")).dataobj) > 0
        z = int(np.median(np.argwhere(seed)[:, 2]))
        maximum = np.log1p(max(float(fsl[:, :, z].max()), float(fnit[:, :, z].max())))
        for axis, volume, title in zip(axes[i], (fsl, fnit, fnit - fsl),
                                       ("FSL CPU", "FNIT CPU", "FNIT - FSL")):
            image = volume[:, :, z]
            if title == "FNIT - FSL":
                limit = max(1, float(np.abs(image).max()))
                axis.imshow(np.rot90(image), cmap="coolwarm", vmin=-limit, vmax=limit)
            else:
                axis.imshow(np.rot90(np.log1p(image)), cmap="magma", vmin=0, vmax=maximum)
            axis.set_title(f"{region}: {title}")
            axis.axis("off")
    args.private_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.private_figure, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
