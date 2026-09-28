"""同一 84 节点 atlas 的多次独立整链连接矩阵稳定性。"""

import argparse
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np


NAMES = ("count", "sift2_fbc", "mean_length", "mean_fa")
UPPER = np.triu_indices(84, 1)


def load(directory: Path, prefix: str) -> dict[str, np.ndarray]:
    matrices = {name: np.loadtxt(directory / f"{prefix}{name}.csv", delimiter=",")
                for name in NAMES}
    if any(matrix.shape != (84, 84) or not np.isfinite(matrix).all()
           for matrix in matrices.values()):
        raise ValueError("all four matrices must be finite 84x84 arrays")
    return matrices


def compare(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict:
    count_a, count_b = candidate["count"][UPPER], reference["count"][UPPER]
    support_a, support_b = count_a > 0, count_b > 0
    shared = support_a & support_b
    result = {
        "count_support_dice": float(2 * shared.sum() / max(support_a.sum() + support_b.sum(), 1)),
        "common_edges": int(shared.sum()),
        "count_relative_l1": float(np.abs(count_a - count_b).sum() /
                                   max(np.abs(count_b).sum(), 1e-12)),
    }
    for name in NAMES:
        a, b = candidate[name][UPPER], reference[name][UPPER]
        evaluated = shared if name in ("mean_length", "mean_fa") else np.ones_like(shared)
        result[f"{name}_pearson_r"] = (float(np.corrcoef(a[evaluated], b[evaluated])[0, 1])
                                         if evaluated.sum() > 1 else None)
        result[f"{name}_common_edge_nmae"] = (float(np.abs(a[shared] - b[shared]).mean() /
                                                    max(np.abs(b[shared]).mean(), 1e-12))
                                               if shared.any() else None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-dir", type=Path, action="append", required=True)
    parser.add_argument("--reference-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    if len(args.reference_dir) < 2:
        raise ValueError("at least two independent MRtrix repeats are required")
    fnit = [load(path, "connectome_") for path in args.fnit_dir]
    reference = [load(path, "") for path in args.reference_dir]
    report = {
        "dataset": args.dataset,
        "atlas": "same FNIT-built 84-node DWI grid",
        "fnit_repeats": len(fnit),
        "reference_repeats": len(reference),
        "reference_pairs": {f"reference_{a}__reference_{b}": compare(reference[a], reference[b])
                            for a, b in combinations(range(len(reference)), 2)},
        "fnit_pairs": {f"fnit_{a}__fnit_{b}": compare(fnit[a], fnit[b])
                       for a, b in combinations(range(len(fnit)), 2)},
        "cross_pairs": {f"fnit_{a}__reference_{b}": compare(fnit[a], reference[b])
                        for a, b in product(range(len(fnit)), range(len(reference)))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"fnit_repeats": len(fnit), "reference_repeats": len(reference)}))


if __name__ == "__main__":
    main()
