"""从真实 GMWMI 种子和官方初始方向生成固定 ACT 检查路径。"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, required=True, help="真实种子位置 N×3")
    parser.add_argument("--directions", type=Path, required=True, help="官方初始方向 N×5")
    parser.add_argument("--output", type=Path, required=True, help="S/P/R 文本路径")
    parser.add_argument("--count", type=int, default=300, help="成功初始方向的种子数")
    args = parser.parse_args()
    seeds = np.loadtxt(args.seeds, dtype=np.float32)
    directions = np.loadtxt(args.directions, dtype=np.float32)
    if seeds.shape != (10000, 3) or directions.shape != (10000, 5):
        raise ValueError("expected 10000 real positions and official directions")
    selected = np.flatnonzero(directions[:, 0] == 1)[:args.count]
    if len(selected) != args.count:
        raise ValueError("fewer than requested successful initial directions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for index in selected:
            position = seeds[index]
            direction = directions[index, 1:4]
            stream.write("S " + " ".join(f"{value:.9g}" for value in position) + "\n")
            for sign in (1, -1):
                if sign == -1:
                    stream.write("R\n")
                for step in range(1, 21):
                    point = position + sign * .5 * step * direction
                    stream.write("P " + " ".join(f"{value:.9g}" for value in point) + "\n")


if __name__ == "__main__":
    main()
