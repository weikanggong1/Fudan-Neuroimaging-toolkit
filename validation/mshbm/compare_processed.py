"""比较真实处理时序的 MS-HBM 标签及固定参照划分下的 17 网络连接。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import nibabel as nib

from fnit.mshbm import load_assets, network_timeseries, project_volume
from fnit.mshbm.cli import read_cortex


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def correlation(first, second):
    first = first.astype(np.float64)
    second = second.astype(np.float64)
    first -= first.mean(axis=0)
    second -= second.mean(axis=0)
    denominator = np.sqrt((first * first).sum(0) * (second * second).sum(0))
    result = np.full(denominator.shape, np.nan)
    np.divide((first * second).sum(0), denominator, out=result, where=denominator > 0)
    return result


def compare(candidate, reference, candidate_labels, reference_labels, mask):
    if candidate.shape != reference.shape:
        raise ValueError("time and cortical vertex counts differ")
    first = candidate_labels[mask]
    second = reference_labels[mask]
    dice = [2 * np.count_nonzero((first == k) & (second == k)) /
            max(1, np.count_nonzero(first == k) + np.count_nonzero(second == k))
            for k in range(1, 18)]
    # Freeze the reference partition to avoid confusing signal differences with label changes.
    candidate_networks = network_timeseries(candidate, reference_labels, mask)
    reference_networks = network_timeseries(reference, reference_labels, mask)
    candidate_fc = np.corrcoef(candidate_networks.T)
    reference_fc = np.corrcoef(reference_networks.T)
    edges = np.triu_indices(17, k=1)
    vertex_r = correlation(candidate, reference)
    network_r = correlation(candidate_networks, reference_networks)
    return {
        "frames": len(candidate), "cortical_vertices": int(mask.sum()),
        "label_agreement": float(np.mean(first == second)),
        "network_dice": dice, "network_dice_mean": float(np.mean(dice)),
        "network_dice_min": float(np.min(dice)),
        "candidate_network_vertices": [int(np.sum(first == k)) for k in range(1, 18)],
        "reference_network_vertices": [int(np.sum(second == k)) for k in range(1, 18)],
        "vertex_temporal_r_valid": int(np.isfinite(vertex_r).sum()),
        "vertex_temporal_r_mean": float(np.nanmean(vertex_r)),
        "vertex_temporal_r_median": float(np.nanmedian(vertex_r)),
        "fixed_reference_network_temporal_r": network_r.tolist(),
        "fixed_reference_network_temporal_r_mean": float(np.nanmean(network_r)),
        "fixed_reference_fc_upper_triangle_r": float(np.corrcoef(
            candidate_fc[edges], reference_fc[edges])[0, 1]),
        "fixed_reference_fc_mae": float(np.mean(np.abs(
            candidate_fc[edges] - reference_fc[edges]))),
        "fixed_reference_fc_max_absolute_difference": float(np.max(np.abs(
            candidate_fc[edges] - reference_fc[edges]))),
    }, (candidate_fc, reference_fc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["surface", "volume"], required=True)
    for name in ("candidate", "reference", "candidate-labels", "reference-labels", "report"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--left-surface")
    parser.add_argument("--right-surface")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    assets = load_assets()
    if args.kind == "surface":
        candidate = read_cortex(args.candidate, assets["cortex_mask"])
        reference = read_cortex(args.reference, assets["cortex_mask"])
    else:
        if not args.left_surface or not args.right_surface:
            parser.error("volume requires anatomical left/right surfaces")
        candidate = project_volume(args.candidate, args.left_surface, args.right_surface,
                                   assets=assets, device=args.device)
        reference = project_volume(args.reference, args.left_surface, args.right_surface,
                                   assets=assets, device=args.device)
    report, matrices = compare(candidate, reference, np.load(args.candidate_labels),
                               np.load(args.reference_labels), assets["cortex_mask"])
    if args.kind == "volume":
        volume_paths = [Path(path).parent / "labels_mni.nii.gz" for path in
                        (args.candidate_labels, args.reference_labels)]
        volume_labels = [np.asarray(nib.load(path).dataobj) for path in volume_paths]
        first, second = volume_labels
        support = (first > 0) | (second > 0)
        volume_dice = [2 * np.count_nonzero((first == k) & (second == k)) /
                       max(1, np.count_nonzero(first == k) + np.count_nonzero(second == k))
                       for k in range(1, 18)]
        report["mni_volume_labels"] = {
            "candidate_nonzero_voxels": int(np.count_nonzero(first)),
            "reference_nonzero_voxels": int(np.count_nonzero(second)),
            "union_nonzero_voxels": int(support.sum()),
            "agreement_in_union": float(np.mean(first[support] == second[support])),
            "network_dice": volume_dice,
            "network_dice_mean": float(np.mean(volume_dice)),
            "network_dice_min": float(np.min(volume_dice)),
            "sha256": [sha256(path) for path in volume_paths],
        }
    report["input_sha256"] = {name: sha256(getattr(args, name)) for name in (
        "candidate", "reference", "candidate_labels", "reference_labels")}
    report["model"] = {"prior": "CBIG HCP_40 fsLR32k 17", "w": 200, "c": 50,
                       "single_run_split": "two contiguous 245-frame pseudo-sessions"}
    Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # Numerical FC matrices remain beside private inputs for figure generation.
    np.savez(Path(args.report).with_suffix(".private.npz"),
             candidate_fc=matrices[0], reference_fc=matrices[1])
    print(json.dumps({key: report[key] for key in (
        "label_agreement", "network_dice_mean", "network_dice_min",
        "vertex_temporal_r_mean", "fixed_reference_fc_upper_triangle_r")}))


if __name__ == "__main__":
    main()
