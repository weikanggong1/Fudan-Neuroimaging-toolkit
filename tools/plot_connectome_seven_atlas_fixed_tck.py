"""绘制同一真实 TCK 的七套官方与 FNIT count 矩阵。"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.matrix_dir / "report.json").read_text())
    profiles = list(report["profiles"])
    fig, axes = plt.subplots(len(profiles), 2, figsize=(10, 19), constrained_layout=True)
    for row, profile in enumerate(profiles):
        with np.load(args.matrix_dir / f"{profile}.npz") as data:
            official = np.log1p(data["official_count"])
            fnit = np.log1p(data["fnit_count"])
        maximum = max(float(official.max()), float(fnit.max()))
        for col, (matrix, title) in enumerate(((official, "MRtrix"), (fnit, "FNIT"))):
            axes[row, col].imshow(matrix, cmap="viridis", origin="lower",
                                  interpolation="nearest", vmin=0, vmax=maximum)
            axes[row, col].set(title=f"{profile} · {title}", xlabel="node", ylabel="node")
    fig.suptitle("ds004666 · same 27,616 MRtrix streamlines · log(1 + count)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
