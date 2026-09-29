"""按真实 GMWMI 种子的皮层/皮层下组织比较两套独立流线长度。"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.stats import ks_2samp


def _sha(path: Path) -> str:
    """返回单个输入文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lengths(path: Path) -> np.ndarray:
    """读取 TCK，返回按文件顺序排列的逐流线弦长，单位 mm。"""
    streamlines = nib.streamlines.load(str(path)).streamlines
    return np.asarray([np.linalg.norm(np.diff(track, axis=0), axis=1).sum()
                       for track in streamlines])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official-tracks", "official-seeds", "fnit-tracks", "fnit-seeds",
                 "five-tissue", "output", "figure"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--affine-mode", choices=("header-spacing", "exact"),
                        default="header-spacing")
    args = parser.parse_args()
    official_seeds = np.loadtxt(args.official_seeds, delimiter=",", comments="#",
                                usecols=(2, 3, 4))
    seeds = {"official": official_seeds, "fnit": np.load(args.fnit_seeds)}
    lengths = {"official": _lengths(args.official_tracks),
               "fnit": _lengths(args.fnit_tracks)}
    image = nib.load(str(args.five_tissue))
    affine = image.affine.copy()
    if args.affine_mode == "header-spacing":
        affine[:3, :3] *= np.asarray(image.header.get_zooms()[:3]) / np.linalg.norm(
            affine[:3, :3], axis=0)
    tissue = np.asarray(image.dataobj, dtype=np.float32)
    classes = {}
    for name, points in seeds.items():
        if len(points) != len(lengths[name]):
            raise ValueError(f"{name}: accepted-seed count differs from TCK track count")
        voxel = nib.affines.apply_affine(np.linalg.inv(affine), points).T
        cgm = map_coordinates(tissue[..., 0], voxel, order=1, mode="constant", cval=0)
        sgm = map_coordinates(tissue[..., 1], voxel, order=1, mode="constant", cval=0)
        classes[name] = sgm > cgm
    report = {"input_sha256": {name: _sha(getattr(args, name.replace("-", "_")))
                               for name in ("official-tracks", "official-seeds", "fnit-tracks",
                                            "fnit-seeds", "five-tissue")},
              "affine_mode": args.affine_mode, "classes": {}, "comparison": {}}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for axis, (label, title, is_sgm) in zip(
        axes, (("皮层", "Cortical", False), ("皮层下", "Subcortical GM", True))
    ):
        pair = {}
        for name, color in (("official", "#2675a8"), ("fnit", "#d76527")):
            subset = lengths[name][classes[name] == is_sgm]
            pair[name] = subset
            report["classes"].setdefault(name, {})[label] = {
                "count": len(subset),
                "length_quantiles_mm": np.quantile(subset, [0, .1, .25, .5, .75, .9, 1]).tolist(),
            }
            axis.hist(subset, bins=np.arange(0, 255, 5), density=True,
                      histtype="step", color=color, label=name)
        report["comparison"][label] = {
            "length_ks": float(ks_2samp(pair["official"], pair["fnit"]).statistic),
        }
        axis.set(title=title, xlabel="Streamline length (mm)", ylabel="Density")
        axis.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(json.dumps(report["comparison"], ensure_ascii=False))


if __name__ == "__main__":
    main()
