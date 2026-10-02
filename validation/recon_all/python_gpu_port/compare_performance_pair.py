"""比较真实 recon-all 优化前后及官方参考；严格失败仍继续报告脑区指标。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from compare_complete_subject import compare as compare_files
from compare_surface_chain import compare as compare_surfaces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "official", "label-table", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).parent
    summary = {"candidate_code_commit": args.code_commit,
               "overall_metric_equivalence": "not_assessed; no confirmed whole-case gates",
               "comparisons": {},
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for label, reference in (("baseline", args.baseline), ("official", args.official)):
        strict = compare_files(reference.resolve(), args.candidate.resolve())
        strict_path = args.output_dir / f"strict_vs_{label}.json"
        strict_path.write_text(json.dumps(strict, indent=2) + "\n")
        print(f"{label}: strict {strict['passed']}/{strict['checked']}", flush=True)
        region_path = args.output_dir / f"region_vs_{label}.json"
        dice_path = args.output_dir / f"dice_vs_{label}.json"
        common = ["--reference", str(reference), "--candidate", str(args.candidate),
                  "--code-commit", args.code_commit]
        subprocess.run([sys.executable, str(scripts / "compare_region_stats.py"),
                        *common, "--output", str(region_path)], check=True)
        subprocess.run([sys.executable, str(scripts / "compare_parcellation_dice.py"),
                        *common, "--label-table", str(args.label_table),
                        "--output", str(dice_path)], check=True)
        surface = compare_surfaces(reference, args.candidate)
        surface["code_commit"] = args.code_commit
        surface["comparator_sha256"] = hashlib.sha256(
            (scripts / "compare_surface_chain.py").read_bytes()).hexdigest()
        surface_path = args.output_dir / f"surface_vs_{label}.json"
        surface_path.write_text(json.dumps(surface, indent=2) + "\n")
        summary["comparisons"][label] = {
            "strict_all_pass": strict["all_pass"], "strict_passed": strict["passed"],
            "strict_checked": strict["checked"],
            "reports": [path.name for path in (strict_path, region_path, dice_path, surface_path)]}
    summary["execution_status"] = "complete"
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
