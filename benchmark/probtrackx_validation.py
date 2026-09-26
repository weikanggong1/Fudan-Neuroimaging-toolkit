"""Compare matched FSL/FNIT volume tractography outputs without publishing DWI."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def _load_paths(directory):
    image = nib.load(str(Path(directory) / "fdt_paths.nii.gz"))
    return image, np.asarray(image.dataobj, dtype=np.float64)


def _counts(directory):
    return np.atleast_1d(np.loadtxt(Path(directory) / "waytotal", dtype=np.int64))


def _time(path):
    for line in Path(path).read_text().splitlines():
        if line.startswith("wall_s="):
            return float(line.split("=", 1)[1])
    raise ValueError(f"no wall_s in {path}")


def _dice(a, b):
    denominator = int(a.sum() + b.sum())
    return float(2 * np.logical_and(a, b).sum() / denominator) if denominator else 1.0


def compare(reference_dir, candidate_dir):
    ref_image, ref = _load_paths(reference_dir)
    new_image, new = _load_paths(candidate_dir)
    if ref.shape != new.shape or not np.allclose(ref_image.affine, new_image.affine, atol=1e-4):
        raise ValueError("reference and candidate density geometry differ")
    support = (ref > 0) | (new > 0)
    r = float(np.corrcoef(ref[support], new[support])[0, 1]) if support.sum() > 1 else None
    k = max(1, int((ref > 0).sum() * 0.1))
    ref_top = np.zeros(ref.size, dtype=bool)
    new_top = np.zeros(new.size, dtype=bool)
    ref_top[np.argpartition(ref.ravel(), -k)[-k:]] = True
    new_top[np.argpartition(new.ravel(), -k)[-k:]] = True
    return {
        "shape": list(ref.shape),
        "affine_max_abs_difference": float(np.abs(ref_image.affine - new_image.affine).max()),
        "fsl_waytotal": _counts(reference_dir).tolist(),
        "fnit_waytotal": _counts(candidate_dir).tolist(),
        "fsl_nonzero_voxels": int((ref > 0).sum()),
        "fnit_nonzero_voxels": int((new > 0).sum()),
        "fsl_density_sum": float(ref.sum()),
        "fnit_density_sum": float(new.sum()),
        "fsl_density_max": float(ref.max()),
        "fnit_density_max": float(new.max()),
        "pearson_on_union_nonzero": r,
        "support_dice": _dice(ref > 0, new > 0),
        "top_tenth_dice": _dice(ref_top, new_top),
        "mean_abs_error_on_union": float(np.abs(ref[support] - new[support]).mean())
        if support.any() else 0.0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-label", default="FSL 6.0.7.22 probtrackx2")
    parser.add_argument("--fsl-seed", required=True)
    parser.add_argument("--fnit-seed", required=True)
    parser.add_argument("--fsl-network", required=True)
    parser.add_argument("--fnit-network", required=True)
    parser.add_argument("--fsl-seed-time", required=True)
    parser.add_argument("--fnit-seed-time", required=True)
    parser.add_argument("--fsl-network-time", required=True)
    parser.add_argument("--fnit-network-time", required=True)
    parser.add_argument("--slice-z", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-figure", required=True)
    args = parser.parse_args(argv)
    seed = compare(args.fsl_seed, args.fnit_seed)
    network = compare(args.fsl_network, args.fnit_network)
    fsl_matrix = np.atleast_2d(np.loadtxt(Path(args.fsl_network) / "fdt_network_matrix",
                                        dtype=np.int64))
    fnit_matrix = np.atleast_2d(np.loadtxt(Path(args.fnit_network) / "fdt_network_matrix",
                                         dtype=np.int64))
    if fsl_matrix.shape != fnit_matrix.shape:
        raise ValueError("network matrix shapes differ")
    network["fsl_matrix"] = fsl_matrix.tolist()
    network["fnit_matrix"] = fnit_matrix.tolist()
    network["matrix_abs_error_sum"] = int(np.abs(fsl_matrix - fnit_matrix).sum())
    report = {
        "reference": args.reference_label,
        "seed_to_voxel": seed,
        "network": network,
        "wall_seconds_including_load_and_write": {
            "fsl_seed": _time(args.fsl_seed_time),
            "fnit_seed": _time(args.fnit_seed_time),
            "fsl_network": _time(args.fsl_network_time),
            "fnit_network": _time(args.fnit_network_time),
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, fsl = _load_paths(args.fsl_seed)
    _, fnit = _load_paths(args.fnit_seed)
    if not 0 <= args.slice_z < fsl.shape[2]:
        raise ValueError("slice-z is outside image")
    maximum = np.log1p(max(float(fsl[:, :, args.slice_z].max()),
                           float(fnit[:, :, args.slice_z].max())))
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.8), layout="constrained")
    for axis, data, title in zip(axes[:2], (fsl, fnit), ("FSL", "FNIT")):
        axis.imshow(np.rot90(np.log1p(data[:, :, args.slice_z])), cmap="magma",
                    vmin=0, vmax=maximum)
        axis.set_title(title)
        axis.set_axis_off()
    difference = fnit[:, :, args.slice_z] - fsl[:, :, args.slice_z]
    limit = max(1, float(np.abs(difference).max()))
    axes[2].imshow(np.rot90(difference), cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[2].set_title("FNIT − FSL")
    axes[2].set_axis_off()
    figure.suptitle(f"Seed-to-voxel density, diffusion slice z={args.slice_z}")
    figure.savefig(args.output_figure, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
