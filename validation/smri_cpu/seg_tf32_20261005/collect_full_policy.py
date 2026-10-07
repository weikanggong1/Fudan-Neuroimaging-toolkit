"""Compare saved complete raw-T1 policy maps/CSV/headers; never run inference."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "binding", "comparator", "official", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    import nibabel as nib
    import numpy as np
    assert sha(args.comparator) == "61eaa50d008e1958f48c5f718d6ad6805fccf7142a91de83c266a91733b7bbf8"
    spec = importlib.util.spec_from_file_location("saved_compare", args.comparator)
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    queue = json.loads((args.run / "queue.private.json").read_text())
    assert queue["status"] == "complete" and all(row["returncode"] == 0 for row in queue["jobs"])
    binding = json.loads(args.binding.read_text())
    report = {"schema": "fnit_parc_tf32_complete_saved_pairs/v1", "queue": queue,
        "binding_sha256": sha(args.binding), "collector_sha256": sha(__file__), "comparator_sha256": sha(args.comparator),
        "baseline_repository_commit": binding["baseline_repository_commit"],
        "candidate_repository_commit": binding["candidate_repository_commit"],
        "scope": "Original raw CC0 T1 complete API; default old/new lossless gate, new False/None policy effects separately scored; no reference injected.",
        "official_reference_provenance": "Reuse prior same-input source-bound nodecw7 eight-core official maps/CSV; native inference was not rerun in this stage.",
        "jobs": {}, "default_pairs": {}, "inheritance_pairs": {}, "new_policy_effects": {}}
    for row in queue["jobs"]:
        worker = json.loads((args.run / row["name"] / "full.private.json").read_text())
        assert worker["status"] == "complete" and worker["source_files"] == binding["source_files"][row["arm"]]
        assert worker["worker_sha256"] == queue["worker_sha256"] and worker["binding_sha256"] == queue["binding_sha256"]
        report["jobs"][row["name"]] = worker

    def geometry(first, second):
        return (first.shape == second.shape and first.get_data_dtype() == second.get_data_dtype()
            and np.array_equal(first.affine, second.affine) and first.header.binaryblock == second.header.binaryblock
            and [(entry.get_code(), entry._raw) for entry in first.header.extensions] ==
                [(entry.get_code(), entry._raw) for entry in second.header.extensions])

    def pair(first_name, second_name):
        first, second = args.run / first_name, args.run / second_name
        result = {"first": first_name, "second": second_name, "saved_maps": {},
                  "csv": compare.compare_csv(first / "volumes.csv", second / "volumes.csv")}
        for name in ("segmentation", "cortical_parcellation", "combined"):
            old, new = first / (name + ".nii.gz"), second / (name + ".nii.gz")
            row = compare.compare_segmentation(old, new)
            row["header_and_world_geometry_exact"] = geometry(nib.load(old), nib.load(new))
            row["compressed_file_exact"] = sha(old) == sha(new)
            result["saved_maps"][name] = row
        result["csv"]["numeric_exact"] = (result["csv"]["comparison_status"] == "compared"
            and result["csv"]["max_absolute_difference_mm3"] == 0)
        result["csv"]["saved_file_exact"] = sha(first / "volumes.csv") == sha(second / "volumes.csv")
        result["lossless_passed"] = (all(row["comparison_status"] == "compared" and row["different_voxels"] == 0
            and row["header_and_world_geometry_exact"] for row in result["saved_maps"].values()) and result["csv"]["numeric_exact"])
        return result

    def official(mode, name):
        reference_map, reference_csv = args.official / (mode + ".nii.gz"), args.official / (mode + ".csv")
        output = args.run / name
        return {"map_sha256": sha(reference_map), "csv_sha256": sha(reference_csv),
                "combined": compare.compare_segmentation(reference_map, output / "combined.nii.gz"),
                "csv": compare.compare_csv(reference_csv, output / "volumes.csv")}

    for mode in ("parc", "parc-fast"):
        old, new = mode + "_baseline_true", mode + "_candidate_true"
        result = pair(old, new)
        result["official_vs_baseline"] = official(mode, old)
        result["official_vs_candidate"] = official(mode, new)
        a, b = report["jobs"][old], report["jobs"][new]
        result["candidate_constructor_and_inference_restore"] = a["before_construction"] == b["before_construction"] == b["after_construction"] == b["after_inference"]
        result["baseline_constructor_side_effect"] = a["before_construction"] != a["after_inference"]
        if queue["device"] != "cpu":
            result["allocated_peak_exact"] = a["gpu"]["max_allocated_bytes"] == b["gpu"]["max_allocated_bytes"]
            result["reserved_peak_exact"] = a["gpu"]["max_reserved_bytes"] == b["gpu"]["max_reserved_bytes"]
            result["memory_budget_passed"] = all(row["gpu"][key] <= 20_000_000_000 for row in (a, b)
                for key in ("max_allocated_bytes", "max_reserved_bytes"))
        result["passed"] = result["lossless_passed"] and result["candidate_constructor_and_inference_restore"]
        if queue["device"] != "cpu":
            result["passed"] = result["passed"] and result["allocated_peak_exact"] and result["reserved_peak_exact"] and result["memory_budget_passed"]
        report["default_pairs"][mode] = result
        if queue["device"] != "cpu":
            declared, inherited = mode + "_candidate_false", mode + "_candidate_none"
            result = pair(declared, inherited)
            result["scope_restore"] = all(report["jobs"][name]["caller_precision_restored"] for name in (declared, inherited))
            result["memory_budget_passed"] = all(report["jobs"][name]["gpu"][key] <= 20_000_000_000 for name in (declared, inherited)
                for key in ("max_allocated_bytes", "max_reserved_bytes"))
            result["passed"] = result["lossless_passed"] and result["scope_restore"] and result["memory_budget_passed"]
            report["inheritance_pairs"][mode] = result
            report["new_policy_effects"][mode] = {"default_true_vs_declared_false": pair(new, declared),
                "official_vs_declared_false": official(mode, declared), "official_vs_inherit_false": official(mode, inherited)}
    report["default_complete_passed"] = len(report["default_pairs"]) == 2 and all(row["passed"] for row in report["default_pairs"].values())
    report["new_policy_inheritance_passed"] = (all(row["passed"] for row in report["inheritance_pairs"].values())
        if queue["device"] != "cpu" else "not_applicable_CPU_flags_unchanged")
    report["passed"] = report["default_complete_passed"] and report["new_policy_inheritance_passed"] is not False
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"device": queue["device"], "arms": len(report["jobs"]), "passed": report["passed"],
        "default_complete_passed": report["default_complete_passed"], "new_policy_inheritance_passed": report["new_policy_inheritance_passed"]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
