"""比较官方与 FNIT 的 GEMS 后验软体积表。"""

import argparse
import json
from pathlib import Path


def read_volumes(path: Path) -> dict[str, float]:
    values = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            values[fields[0]] = float(fields[1])
    return values


def compare_volumes(reference: Path, candidate: Path) -> dict:
    ref, got = read_volumes(reference), read_volumes(candidate)
    rows = []
    for label in sorted(ref.keys() | got.keys()):
        a, b = ref.get(label), got.get(label)
        difference = abs(a - b) / a if a is not None and b is not None and a > 0 else None
        rows.append({"label": label, "reference_mm3": a, "fnit_mm3": b,
                     "relative_difference": difference,
                     "accepted": difference is not None and difference <= .05})
    return {"reference": str(reference), "candidate": str(candidate),
            "labels": len(rows), "accepted": sum(row["accepted"] for row in rows), "regions": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = compare_volumes(reference=args.reference, candidate=args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{report['accepted']}/{report['labels']} 条软体积达标")


if __name__ == "__main__":
    main()
