"""Recompute scalar stage totals from the saved profile; no MRI/Torch imports."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]


def main():
    profile_path = HERE / "PROFILE.public.json"
    data = json.loads(profile_path.read_text())
    assert data["status"] == "complete_gates_passed"
    stages = defaultdict(lambda: {"calls": 0, "inclusive_seconds": 0.0, "exclusive_seconds": 0.0})
    layers = defaultdict(lambda: {"calls": 0, "inclusive_seconds": 0.0})
    for row in data["events"]:
        for key in ("inclusive_seconds", "exclusive_seconds"):
            stages[row["name"]][key] += row[key]
        stages[row["name"]]["calls"] += 1
        if row["name"] == "layer_conv":
            value = layers[row["metadata"]["layer"]]
            value["calls"] += 1
            value["inclusive_seconds"] += row["inclusive_seconds"]
    api = stages["API"]["inclusive_seconds"]
    for row in stages.values():
        row["fraction_of_observed_API_using_inclusive_clock"] = row["inclusive_seconds"] / api
    functionals = [row for row in data["events"] if row["name"] == "functional_conv"]
    groups = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    for row in functionals:
        key = str(row["metadata"]["groups"])
        groups[key]["calls"] += 1
        groups[key]["seconds"] += row["inclusive_seconds"]
    padding = defaultdict(float)
    for row in data["events"]:
        if row["name"] == "pad":
            padding[row["parent"]] += row["inclusive_seconds"]
    reference_path = REPOSITORY / "validation/smri_cpu/seg_memory_20261005/CPU_FULL.public.json"
    reference = json.loads(reference_path.read_text())
    official = reference["pairs"]["seg33"]["official_vs_candidate"]
    summary = {
        "schema": "fnit_ordinary33_cpu_observed_timing_summary/v1",
        "scope": "diagnostic exclusive/nested clocks; no candidate change or formal speed benchmark",
        "profile_report_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "source_plan_sha256": hashlib.sha256((HERE / "PLAN.json").read_bytes()).hexdigest(),
        "observed_API_seconds": data["observed_api_seconds"],
        "observed_outer_seconds": data["queue"]["observed_outer_seconds"],
        "stage_totals": dict(stages),
        "layer_convolution_totals": dict(sorted(layers.items(), key=lambda item: -item[1]["inclusive_seconds"])),
        "functional_convolution_by_group": dict(groups),
        "padding_by_parent": dict(padding),
        "functional_convolution_cpu_mkldnn_disabled_all": all(
            row["metadata"]["device"] == "cpu" and row["metadata"]["mkldnn_enabled"] is False
            for row in functionals),
        "complete_saved_output_parity": data["complete_saved_output_comparison"]["passed"],
        "maximum_RSS_bytes": data["maximum_rss_kib"] * 1024,
        "official_accuracy_reused_due_saved_file_SHA_exact": {
            "reference_report": str(reference_path.relative_to(REPOSITORY)),
            "different_voxels": official["different_voxels"],
            "note": "Profile matches existing FNIT file, not a new native fit; pre-existing native residual remains.",
        },
        "metadata_not_a_mathematical_change": True,
        "current_cpu_official_speed_goal_met": False,
        "accounting_check": {
            "exclusive_event_sum_seconds": sum(row["exclusive_seconds"] for row in data["events"]),
            "API_inclusive_seconds": api,
            "difference_seconds": sum(row["exclusive_seconds"] for row in data["events"]) - api,
        },
        "timing_caveat": "Only one observed run. Nested clocks overlap; sum exclusive clocks for accounting. Whole observed timing is not a new formal benchmark.",
    }
    assert abs(summary["accounting_check"]["difference_seconds"]) < 1e-6
    assert stages["CNN"]["calls"] == stages["gaussian33"]["calls"] == 2
    assert stages["layer_conv"]["calls"] == 38 and len(functionals) == 142
    (HERE / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"profile_sha256": summary["profile_report_sha256"],
                      "CNN_seconds": stages["CNN"]["inclusive_seconds"],
                      "blur33_seconds": stages["gaussian33"]["inclusive_seconds"],
                      "module_conv_seconds": stages["layer_conv"]["inclusive_seconds"],
                      "functional_conv_groups": dict(groups), "accounting": summary["accounting_check"]}))


if __name__ == "__main__":
    main()
