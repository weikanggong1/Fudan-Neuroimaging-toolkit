"""同一真实输入下，一份 FNIT 四矩阵对三次官方四矩阵的随机范围。"""

import argparse
import itertools
import json
from pathlib import Path

from benchmark_connectome_rng_envelope import NAMES, compact_metrics, load_matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", nargs=3, type=Path, required=True,
                        help="三次官方独立追踪的四矩阵目录，顺序为 RNG 0/1/2")
    parser.add_argument("--fnit", type=Path, required=True,
                        help="一份 FNIT 独立追踪的四矩阵目录")
    parser.add_argument("--output", type=Path, required=True,
                        help="输入哈希、官方互比和 FNIT 对官方逐对指标 JSON")
    args = parser.parse_args()
    official = [load_matrices(path) for path in args.official]
    fnit = load_matrices(args.fnit)
    within = [{"official_seeds": [i, j],
               "metrics": compact_metrics(official[i][0], official[j][0])}
              for i, j in itertools.combinations(range(3), 2)]
    cross = [{"official_seed": i,
              "metrics": compact_metrics(official[i][0], fnit[0])}
             for i in range(3)]
    fields = ("relative_l1_full_upper", "nonzero_mean_scaled_mae", "support_dice")
    ranges = {}
    for name in NAMES:
        ranges[name] = {}
        for field in fields:
            reference = [pair["metrics"][name][field] for pair in within]
            candidate = [pair["metrics"][name][field] for pair in cross]
            ranges[name][field] = {"official": [min(reference), max(reference)],
                                   "fnit": [min(candidate), max(candidate)],
                                   "fnit_pairs_inside_official_range": sum(
                                       min(reference) <= value <= max(reference)
                                       for value in candidate)}
    report = {
        "input_sha256": {"official": [item[1] for item in official],
                         "fnit": fnit[1]},
        "matrix_shape": list(fnit[0]["count"].shape),
        "within_official": within,
        "fnit_vs_official": cross,
        "ranges": ranges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: values["relative_l1_full_upper"]
                      for name, values in ranges.items()}))


if __name__ == "__main__":
    main()
