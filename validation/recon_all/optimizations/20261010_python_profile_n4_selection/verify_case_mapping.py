"""按实际case字段、输出路径和输入SHA核对N4表；仅写新派生收据。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main(*, repository: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("preserve existing reports; use a new summary path")
    root = repository.resolve()
    originals: dict[str, bytes] = {}

    def read(relative):
        raw = (root / relative).read_bytes()
        originals[relative] = raw
        return json.loads(raw), {"path": relative, "sha256": hashlib.sha256(raw).hexdigest()}

    cached, cached_ref = read("validation/recon_all/optimizations/20261009_n4_torch_substages/reports/input_chain_a100_20261009_v2/cached_worker/summary.json")
    rows = []
    for case in ("sub06", "sub07"):
        array, array_ref = read(f"validation/recon_all/optimizations/20261010_n4_simpleitk/reports/{case}_array.json")
        file, file_ref = read(f"validation/recon_all/optimizations/20261010_n4_simpleitk/reports/{case}_file.json")
        prefix, prefix_ref = read(f"validation/recon_all/optimizations/20261010_n4_continuous_prefix/reports/{case}.json")
        assert f"/{case}/" in file["detail"]["output"]
        assert f"/{case}-candidate/" in prefix["candidate"]
        assert array["source_sha256"] == file["source_sha256"] == file["detail"]["input_sha256"]
        sitk_difference = file["comparisons"]["native_quantized"]["different_voxels"]
        assert sitk_difference == array["comparisons"]["uint8_python_vs_native"]["different_voxels"]
        label = "ds000114_sub-" + case[-2:]
        selected = [row for row in cached["rows"] if row["case"] == label]
        assert len(selected) == 2 and {row["mode"] for row in selected} == {"cold_cli", "initialized_api"}
        torch_difference = selected[0]["to_native"]["whole"]["different_voxels"]
        assert all(row["to_native"]["whole"]["different_voxels"] == torch_difference for row in selected)
        assert all(row["to_native"]["whole"]["negative_differences"] == 0 for row in selected)
        assert all(row["to_native"]["whole"]["positive_differences"] == torch_difference for row in selected)
        assert prefix["files"]["mri/tmp/nu0.mgz"]["different_voxels"] == torch_difference
        filled = prefix["files"]["mri/filled.mgz"]
        rows.append({"case": case,
            "simpleitk_same_input_stage": {"array_report": array_ref, "file_report": file_ref,
                "orig_file_sha256": file["source_sha256"], "code_version": file["code_version"],
                "different_uint8_voxels_to_native": sitk_difference,
                "maximum_absolute_error": file["comparisons"]["native_quantized"]["maximum_absolute_error"],
                "full_file_API_seconds": file["detail"]["total_seconds"]},
            "torch_cached_same_input_stage": {"case_field": label, "report": cached_ref,
                "orig_file_sha256": selected[0]["input_sha256"],
                "code_version": selected[0]["worker"]["code_version"],
                "different_uint8_voxels_to_native": torch_difference,
                "negative_differences": 0, "positive_differences": torch_difference,
                "cold_CLI_seconds": next(x["wall_seconds_including_exec_load_transfer_io"] for x in selected if x["mode"] == "cold_cli")},
            "torch_continuous_prefix": {"report": prefix_ref,
                "raw_T1_sha256": prefix["input_sha256"], "code_version": prefix["code_version"],
                "nu0_different_voxels": prefix["files"]["mri/tmp/nu0.mgz"]["different_voxels"],
                "filled_different_voxels": filled["different_voxels"],
                "filled_minimum_Dice": filled["labels"]["minimum_dice"]}})
    for path, raw in originals.items():
        assert (root / path).read_bytes() == raw, "original changed during read-only collection: " + path
    report = {"schema_version": 1,
        "scope": "case mapping correction derived from existing original N4 reports; no new MRI execution",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "correction": {"document": "docs/recon_all/PYTHON_PROFILE_N4_SELECTION.md",
            "previous_simpleitk_sub06_sub07": [28, 99], "correct_simpleitk_sub06_sub07": [99, 28],
            "torch_nu0_sub06_sub07_unchanged": [4016, 3259], "filled_sub06_sub07_unchanged": [4671, 2291]},
        "rows": rows, "original_reports_unchanged": True,
        "input_boundary": "SimpleITK and cached-Torch rows use their own frozen upstream/version; do not claim identical inputs across batches",
        "overall_metric_equivalence": "not_assessed"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"correct_simpleitk_sub06_sub07": [row["simpleitk_same_input_stage"]["different_uint8_voxels_to_native"] for row in rows],
                      "summary_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    main(repository=options.repository, output=options.output)
