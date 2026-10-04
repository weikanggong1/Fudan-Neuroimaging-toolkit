"""Reuse the repository's fixed-grid subregion audit after timing has ended."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


FILES = {
    "brainstem": ("brainstemSsLabels", ["brainstemSsLabels.volumes.txt"]),
    "thalamus": ("ThalamicNuclei", ["ThalamicNuclei.volumes.txt"]),
    "hippo-amygdala-left": ("lh.hippoAmygLabels", ["lh.hippoSfVolumes.txt", "lh.amygNucVolumes.txt"]),
    "hippo-amygdala-right": ("rh.hippoAmygLabels", ["rh.hippoSfVolumes.txt", "rh.amygNucVolumes.txt"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--like", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=("same_stage_inputs_cpu8",
                 "fnit_raw_end_to_end_vs_official_saved_stage_outputs"),
        default="same_stage_inputs_cpu8",
        help="Record the comparison scope; raw-input scoring reuses saved official stages.")
    parser.add_argument("--audit-helper", type=Path,
                        default=Path(__file__).resolve().parents[2] / "subregions/reproducibility_20261002/analyze_repeatability.py")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("fixed_grid_audit", args.audit_helper)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    report = json.loads((args.candidate_dir / "report.json").read_text())
    groups = []
    for family in report["structures"]:
        metadata = {int(label): row for label, row in report["labels"].items() if row["source"] == family}
        if not metadata:
            raise ValueError("Missing declared family labels: " + family)
        stem, volumes = FILES[family]
        for space in ["native", "hr"]:
            official = args.official_dir / (stem + (".FSvoxelSpace.mgz" if space == "native" else ".mgz"))
            candidate = (args.candidate_dir / "subregions_native.nii.gz" if space == "native"
                         else args.candidate_dir / "highres" / (family + ".nii.gz"))
            definition = {"id": family + "_" + space, "family": family, "space": space,
                          "label_ids": sorted(metadata), "label_names": {str(k): v["name"] for k, v in metadata.items()},
                          "grid": {"like": str(args.like)} if space == "native" else {"union_like": str(official)},
                          "official": [{"id": "official_cpu8", "labels": str(official),
                                        "label_offset": 10000 if family.endswith("-right") else 0,
                                        "soft_volumes_files": [str(args.official_dir / name) for name in volumes],
                                        "provenance": {"scope": args.scope}}],
                          "fnit": [{"id": "candidate_cpu8", "labels": str(candidate),
                                    "soft_volumes_file": str(args.candidate_dir / "volumes.tsv"),
                                    "provenance": {"scope": args.scope}}]}
            result = audit.audit_group(definition, Path("/"))
            pair = next(row for row in result["pairs"] if row["kind"] == "cross_method")
            for region in pair["regions"]:
                reference_count = region["first_voxels"]
                relative = (abs(region["first_voxels"] - region["second_voxels"]) / reference_count
                            if reference_count else None)
                region["hard_volume_relative_to_official"] = relative
                region["preexisting_gate_passed"] = (region["dice"] >= .95 and relative <= .05
                                                       if region["dice"] is not None and relative is not None else False)
            groups.append(result)
    output = {"status": "scored", "both_empty_dice": None,
              "comparison_scope": args.scope,
              "threshold": {"per_label_dice_minimum": .95, "hard_volume_relative_to_official_maximum": .05},
              "audit_helper_sha256": hashlib.sha256(args.audit_helper.read_bytes()).hexdigest(),
              "candidate_report_sha256": hashlib.sha256((args.candidate_dir / "report.json").read_bytes()).hexdigest(),
              "groups": groups}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for group in groups:
        pair = next(row for row in group["pairs"] if row["kind"] == "cross_method")
        print(group["id"], sum(row["preexisting_gate_passed"] for row in pair["regions"]),
              "/", len(pair["regions"]), flush=True)


if __name__ == "__main__":
    main()
