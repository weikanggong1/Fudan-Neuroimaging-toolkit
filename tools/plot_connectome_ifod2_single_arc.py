"""绘制真实 FOD 单弧概率与 MRtrix 官方实现的逐弧误差。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="单弧 benchmark 逐弧 CSV")
    parser.add_argument("--output", type=Path, required=True, help="输出 PNG 文件")
    args = parser.parse_args()
    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    arc = np.arange(len(data))
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    axes[0].plot(arc, data["official"], ".", label="MRtrix", markersize=4)
    axes[0].plot(arc, data["mrtrix_lookup"], "x", label="PyTorch lookup", markersize=3)
    axes[0].set(xlabel="Arc index", ylabel="Path probability")
    axes[0].legend(frameon=False)
    for name, label in (("direct_sh", "Direct SH"), ("mrtrix_lookup", "PyTorch lookup")):
        error = np.abs(data[name] - data["official"])
        axes[1].semilogy(arc, np.maximum(error, 1e-9), ".", label=label, markersize=4)
    axes[1].set(xlabel="Arc index", ylabel="Absolute probability error")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
