"""从已校验阶段摘要绘制实际墙钟图，不读取或重算MRI。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_results(*, summary_file: Path, output_file: Path) -> None:
    """输入完整summary.json，输出独立PNG；两参数必填，不覆盖已有图。

    图展示秒数、两次原始观测及中位数；输入非complete_stages时拒绝。
    不改变指标，不据共享负载时间宣布整例加速，无原软件独立命令。
    """
    if output_file.exists():
        raise FileExistsError(output_file)
    value = json.loads(summary_file.read_text())
    if value["status"] != "complete_stages" or value["complete_stage_count"] != 8:
        raise ValueError("eight completed stage calls required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for axis, subject in zip(axes, ("sub06", "sub07")):
        times = value["subjects"][subject]["times"]
        keys = ("cpu64", "torch1024")
        axis.bar([0, 1], [times[key]["median_seconds"] for key in keys],
                 width=.58, color=("#617586", "#247c9c"), alpha=.75)
        for i, key in enumerate(keys):
            for j, seconds in enumerate(times[key]["wall_seconds"]):
                axis.scatter(i + (j - .5) * .12, seconds, color="black", s=18, zorder=3)
            axis.text(i, max(times[key]["wall_seconds"]) + 5,
                      f'{times[key]["median_seconds"]:.2f}s', ha="center", fontsize=9)
        axis.set_xticks([0, 1], ["CPU inverse / chunk 64", "Torch inverse / chunk 1024"], fontsize=8)
        axis.set_title(subject + " (all 4 LTA files identical)", fontsize=10)
        axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
        axis.set_ylim(0, 140)
    axes[0].set_ylabel("Complete isolated GCA API wall seconds")
    figure.suptitle("Same-input self-owned GCA: shared-load A/B/A/B observations", fontsize=11)
    figure.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=180); plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    plot_results(**vars(parser.parse_args()))
