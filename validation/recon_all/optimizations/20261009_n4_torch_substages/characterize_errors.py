"""真实 N4 差异的强度、脑内/脑外、标签和连通簇诊断；不修补候选。"""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def quantize(a):
    return np.floor(np.clip(a, 0, 255) + .5).astype(np.uint8)


def group_stats(mask, error, source):
    count = int(mask.sum())
    if count == 0:
        return {"voxels": 0}
    return {"voxels": count, "max_float_error": float(error[mask].max()),
            "p99_float_error": float(np.quantile(error[mask], .99)),
            "input_intensity_min": float(source[mask].min()),
            "input_intensity_median": float(np.median(source[mask])),
            "input_intensity_max": float(source[mask].max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-raw", type=Path, required=True)
    parser.add_argument("--source-volume", type=Path, required=True)
    parser.add_argument("--candidate-raw", type=Path, required=True)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--brainmask", type=Path)
    parser.add_argument("--aseg", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path)
    args = parser.parse_args()
    image = nib.load(str(args.source_volume))
    source = np.asarray(image.dataobj, dtype=np.float32)
    raw = np.fromfile(args.input_raw, np.float32).reshape(source.shape, order="F")
    if not np.array_equal(source, raw):
        raise ValueError("source MRI data does not exactly match frozen raw input; tissue attribution refused")
    candidate = np.fromfile(args.candidate_raw, np.float32).reshape(source.shape, order="F")
    reference = np.fromfile(args.reference_raw, np.float32).reshape(source.shape, order="F")
    error = np.abs(candidate.astype(np.float64)-reference.astype(np.float64))
    signed = quantize(candidate).astype(np.int16)-quantize(reference).astype(np.int16)
    different = signed != 0
    coordinates = np.argwhere(different)
    clusters, count = ndimage.label(different)
    sizes = np.bincount(clusters.ravel())[1:]
    report = {"kind": "diagnostic_only_no_production_inputs_or_corrections", "shape": list(source.shape),
              "affine": image.affine.tolist(), "uint8_difference": group_stats(different, error, source),
              "signed_uint8_difference_counts": {str(v): int((signed==v).sum()) for v in np.unique(signed)},
              "connectivity": "6 face neighbours", "cluster_count": int(count),
              "largest_cluster_voxels": int(sizes.max()) if count else 0,
              "largest_20_clusters_voxels": sorted(sizes.tolist(), reverse=True)[:20],
              "bbox_voxel_xyz": [coordinates.min(0).tolist(), coordinates.max(0).tolist()] if coordinates.size else None,
              "intensity_bins": [], "source_sha256": {"script": sha(__file__), "raw": sha(args.input_raw),
                  "source_volume": sha(args.source_volume), "candidate": sha(args.candidate_raw), "reference": sha(args.reference_raw)}}
    foreground = reference > 0
    signed_float = candidate.astype(np.float64)-reference.astype(np.float64)
    report["foreground_voxels"] = int(foreground.sum())
    if foreground.any():
        values = signed_float[foreground]
        report["signed_float_foreground"] = {"mean": float(values.mean()), "median": float(np.median(values)),
            "p01": float(np.quantile(values, .01)), "p99": float(np.quantile(values, .99)),
            "positive_voxels": int((values > 0).sum()), "negative_voxels": int((values < 0).sum())}
        relative = error[foreground]/reference[foreground]
        report["relative_error_foreground"] = {"max": float(relative.max()), "p99": float(np.quantile(relative, .99))}
    bins = (0, 20, 60, 100, 140, 180, 220, 256)
    for low, high in zip(bins[:-1], bins[1:]):
        report["intensity_bins"].append({"min_inclusive": low, "max_exclusive": high,
            **group_stats(different & (source>=low) & (source<high), error, source)})
    masks = {}
    for name, path in (("brainmask", args.brainmask), ("aseg", args.aseg)):
        if path is None:
            continue
        labelled = nib.load(str(path))
        if labelled.shape != image.shape or not np.allclose(labelled.affine, image.affine, rtol=0, atol=1e-5):
            raise ValueError(f"{name} must share the exact conformed MRI grid")
        masks[name] = np.asarray(labelled.dataobj)
        report["source_sha256"][name] = sha(path)
    if "brainmask" in masks:
        report["brain_inside"] = group_stats(different & (masks["brainmask"] != 0), error, source)
        report["brain_outside"] = group_stats(different & (masks["brainmask"] == 0), error, source)
    if "aseg" in masks:
        report["labels"] = {str(int(label)): group_stats(different & (masks["aseg"] == label), error, source)
                            for label in np.unique(masks["aseg"][different])}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    if args.output_plot is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 非特例修补：仅选择最大浮点差异位置作可视化。
        centre = np.unravel_index(np.argmax(error), error.shape)
        figure, axes = plt.subplots(3, 3, figsize=(10, 9))
        for axis in range(3):
            plane = [slice(None)]*3; plane[axis] = centre[axis]; plane = tuple(plane)
            for col, values in enumerate((source[plane], error[plane], signed[plane])):
                kwargs = ({"cmap": "gray", "vmin": 0, "vmax": 255} if col == 0 else
                          {"cmap": "magma", "vmin": 0, "vmax": float(error.max())} if col == 1 else
                          {"cmap": "coolwarm", "vmin": -1, "vmax": 1})
                plotted = axes[axis,col].imshow(values.T, origin="lower", **kwargs)
                axes[axis,col].set_title(f"{'xyz'[axis]}={centre[axis]}: " + ("input T1" if col==0 else "float error" if col==1 else "uint8 delta"))
                axes[axis,col].axis("off")
                if col: figure.colorbar(plotted, ax=axes[axis,col], shrink=.7)
        figure.tight_layout(); args.output_plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output_plot, dpi=150); plt.close(figure)


if __name__ == "__main__":
    main()
