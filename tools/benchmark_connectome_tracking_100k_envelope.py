"""三次官方 100k 四矩阵内部波动与一次 FNIT 100k 对照。"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from benchmark_connectome_rng_envelope import compact_metrics, load_matrices
from compare_connectome_matrices import NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, nargs=3, required=True)
    parser.add_argument("--fnit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    official = [load_matrices(path) for path in args.official]
    fnit = load_matrices(args.fnit)
    within = [compact_metrics(official[i][0], official[j][0])
              for i, j in itertools.combinations(range(3), 2)]
    cross = [compact_metrics(item[0], fnit[0]) for item in official]
    fields = ("relative_l1_full_upper", "nonzero_mean_scaled_mae", "support_dice")
    ranges = {
        name: {field: {
            "official_min_max": [float(min(item[name][field] for item in within)),
                                 float(max(item[name][field] for item in within))],
            "fnit_vs_official": [float(item[name][field]) for item in cross],
        } for field in fields} for name in NAMES
    }
    report = {
        "dataset": "OpenNeuro ds004666 corrected paired T1/DWI, 100000 seed attempts",
        "design": "three MRtrix seeds 0..2 and one FNIT seed 0; same FOD, 5TT, GMWMI, FA and atlas",
        "official_input_sha256": [item[1] for item in official],
        "fnit_input_sha256": fnit[1],
        "ranges": ranges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for row, name in enumerate(("count", "mean_fa")):
            reference = official[0][0][name]
            candidate = fnit[0][name]
            difference = candidate - reference
            if name == "count":
                images = (np.log1p(reference), np.log1p(candidate),
                          np.sign(difference) * np.log1p(np.abs(difference)))
            else:
                images = (reference, candidate, difference)
            maximum = max(float(images[0].max()), float(images[1].max()), 1e-12)
            limit = max(float(np.abs(images[2]).max()), 1e-12)
            for column, (image, title) in enumerate(zip(
                images, ("MRtrix seed 0", "FNIT seed 0", "FNIT - MRtrix")
            )):
                axes[row, column].imshow(
                    image, origin="lower", interpolation="nearest",
                    cmap="coolwarm" if column == 2 else "viridis",
                    vmin=-limit if column == 2 else 0,
                    vmax=limit if column == 2 else maximum,
                )
                axes[row, column].set(title=f"{name}: {title}", xlabel="node", ylabel="node")
        fig.suptitle("ds004666, 100,000 attempts; 20 region structural connectome")
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=180)
        plt.close(fig)
    print(json.dumps(ranges))


if __name__ == "__main__":
    main()
