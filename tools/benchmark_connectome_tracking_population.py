"""真实 TCK 的流线长度、端点和轨迹密度分布对照。"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.stats import ks_2samp


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path, affine, shape):
    tractogram = nib.streamlines.load(str(path), lazy_load=False)
    tracks = list(tractogram.streamlines)
    lengths = np.asarray([np.linalg.norm(np.diff(track, axis=0), axis=1).sum()
                          for track in tracks])
    endpoints = np.concatenate([np.stack((track[0], track[-1])) for track in tracks])
    points = np.concatenate(tracks)
    voxel = nib.affines.apply_affine(np.linalg.inv(affine), points)
    index = np.rint(voxel).astype(np.int32)
    inside = ((index >= 0) & (index < np.asarray(shape))).all(-1)
    tdi = np.zeros(shape, dtype=np.int32)
    np.add.at(tdi, tuple(index[inside].T), 1)
    return lengths, endpoints, tdi


def _correlation(left, right):
    support = (left + right) > 0
    return float(np.corrcoef(left[support], right[support])[0, 1])


def _coarse_tdi(tdi):
    if any(size % 4 for size in tdi.shape):
        raise ValueError("FOD grid must divide into four-voxel spatial blocks")
    x, y, z = (size // 4 for size in tdi.shape)
    return tdi.reshape(x, 4, y, 4, z, 4).sum(axis=(1, 3, 5)).ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, nargs=3, required=True, help="三份官方真实 TCK")
    parser.add_argument("--fnit", type=Path, required=True, help="一份 FNIT 真实 TCK")
    parser.add_argument("--fnit-repeat", type=Path, nargs=2,
                        help="可选：再接两份 FNIT TCK，输出三次内部与九次跨软件比较")
    parser.add_argument("--grid", type=Path, required=True, help="同输入 FOD NIfTI 网格")
    parser.add_argument("--output", type=Path, required=True, help="指标 JSON")
    parser.add_argument("--figure", type=Path, required=True, help="长度和 TDI 对照 PNG")
    args = parser.parse_args()
    image = nib.load(str(args.grid))
    shape = image.shape[:3]
    fnit_names = ["fnit"] if args.fnit_repeat is None else ["fnit_0", "fnit_1", "fnit_2"]
    names = ["official_0", "official_1", "official_2", *fnit_names]
    paths = [*args.official, args.fnit, *(args.fnit_repeat or [])]
    data = {name: _read(path, image.affine, shape)
            for name, path in zip(names, paths)}
    all_endpoints = np.concatenate([item[1] for item in data.values()])
    bins = [np.arange(all_endpoints[:, axis].min() - 8,
                      all_endpoints[:, axis].max() + 16, 8.) for axis in range(3)]
    histogram = {name: np.histogramdd(item[1], bins=bins)[0].ravel()
                 for name, item in data.items()}
    pairs = {}
    official_names = names[:3]
    comparisons = [(official_names[i], official_names[j])
                   for i, j in ((0, 1), (0, 2), (1, 2))]
    comparisons += [(left, right) for left in official_names for right in fnit_names]
    if len(fnit_names) == 3:
        comparisons += [(fnit_names[i], fnit_names[j])
                        for i, j in ((0, 1), (0, 2), (1, 2))]
    for left, right in comparisons:
        first, second = data[left], data[right]
        pairs[f"{left}_vs_{right}"] = {
            "length_ks": float(ks_2samp(first[0], second[0]).statistic),
            "endpoint_8mm_histogram_pearson": _correlation(histogram[left], histogram[right]),
            "tdi_native_voxel_pearson": _correlation(first[2].ravel(), second[2].ravel()),
            "tdi_8mm_block_pearson": _correlation(_coarse_tdi(first[2]), _coarse_tdi(second[2])),
        }
    report = {
        "dataset": "OpenNeuro ds004666 real fixed WM FOD and independent official/FNIT tracks",
        "input_sha256": {name: _hash(path) for name, path in zip(names, paths)},
        "grid_sha256": _hash(args.grid),
        "track_counts": {name: len(item[0]) for name, item in data.items()},
        "length_quantiles_mm": {name: np.quantile(item[0], [0, .1, .25, .5, .75, .9, 1]).tolist()
                                for name, item in data.items()},
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = ["#2675a8", "#73a9ce", "#accce1", "#d76527", "#eb984e", "#f5c087"]
    for name, color in zip(names, colors):
        axes[0, 0].hist(data[name][0], bins=np.arange(0, 255, 5), density=True,
                        histtype="step", linewidth=1.5, label=name, color=color)
    axes[0, 0].set(xlabel="Length (mm)", ylabel="Density", title="Accepted streamline lengths")
    axes[0, 0].legend(frameon=False)
    z = shape[2] // 2
    official_tdi = data["official_0"][2][:, :, z - 2:z + 3].sum(-1).T
    fnit_tdi = data[fnit_names[0]][2][:, :, z - 2:z + 3].sum(-1).T
    vmax = np.quantile(np.concatenate((official_tdi.ravel(), fnit_tdi.ravel())), .995)
    for axis, matrix, title in ((axes[0, 1], official_tdi, "MRtrix TDI"),
                                (axes[1, 0], fnit_tdi, "FNIT TDI")):
        axis.imshow(matrix, origin="lower", vmin=0, vmax=vmax, cmap="magma")
        axis.set(title=title, xlabel="FOD voxel x", ylabel="FOD voxel y")
    difference = fnit_tdi - official_tdi
    limit = np.quantile(np.abs(difference), .995)
    axes[1, 1].imshow(difference, origin="lower", vmin=-limit, vmax=limit,
                      cmap="coolwarm")
    axes[1, 1].set(title="FNIT minus MRtrix TDI", xlabel="FOD voxel x", ylabel="FOD voxel y")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
