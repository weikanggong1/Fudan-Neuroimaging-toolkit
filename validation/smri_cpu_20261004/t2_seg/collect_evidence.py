"""Export measured task2 results without credentials or private run arguments."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def collect(run_root, workspace, tables_dir=None):
    report = {"schema": "fnit_smri_cpu_seg_evidence/v1", "records": [],
              "comparisons": {}, "preprocessing": {}, "source_files": {}}
    groups = ["corrected_numa1_records_v1", "corrected_case02_records_v1",
              "selective_numa1_v1_records", "gpu_corrected_records_v1",
              "gpu_selective_v1_records", "gpu_header_v1_records", "wmh_numa1_v2_records",
              "original_backend_numa1_v1_records", "gpu_original_backend_v1_records"]
    for group in groups:
        for path in sorted((run_root / group).glob("*/record.json")):
            record = json.loads(path.read_text())
            row = {key: record.get(key) for key in (
                "status", "hostname", "cpu_affinity", "max_cpu_threads",
                "started_utc", "finished_utc", "wall_seconds", "returncode",
                "maximum_sampled_tree_rss_bytes", "maximum_sampled_tree_threads",
                "load_before", "load_after")}
            row.update(group=group, job_id=path.parent.name,
                       job_sha256=record.get("job_sha256"))
            time_file = path.parent / "time.txt"
            if time_file.exists():
                for line in time_file.read_text().splitlines():
                    if "Maximum resident set size (kbytes):" in line:
                        row["time_maximum_rss_kbytes"] = int(line.rsplit(":", 1)[1])
            guard = list(run_root.glob("*/*" + path.parent.name + ".guard.json"))
            if guard:
                metrics = json.loads(guard[0].read_text())
                row["gpu"] = {key: metrics[key] for key in (
                    "cli_api_seconds", "api_seconds_excluding_preprocessing",
                    "max_allocated_bytes", "max_reserved_bytes", "memory_budget_bytes")
                    if key in metrics}
            report["records"].append(row)
    for path in sorted((run_root / "comparisons").glob("*.json")):
        if path.name.startswith(("corrected_numa1_", "case02_", "selective_",
                                 "gpu_corrected_", "gpu_selective_", "ctab_corrected_",
                                 "keepgeom_", "wmh_", "original_backend_", "gpu_original_backend_")):
            comparison = json.loads(path.read_text())
            comparison.pop("reference_library", None)
            detailed = ("official1_candidate3" in path.name
                        or "baseline2_candidate3" in path.name
                        or path.name.startswith(("case02_", "keepgeom_"))
                        or (path.name.startswith("wmh_") and "official1_candidate2" in path.name)
                        or (path.name.startswith("original_backend_")
                            and "baseline2_candidate3" in path.name)
                        or (path.name.startswith("selective_")
                            and any(name in path.name for name in (
                                "official_candidate", "baseline1_candidate2"))))
            segmentation = comparison.get("segmentation", {})
            volumes = comparison.get("soft_volumes", {})
            # Detailed per-case/mode error tables are retained once. Exact
            # GPU controls and repetitions need their own hashes, geometry,
            # status and time, rather than repeating every unchanged label.
            if not detailed:
                segmentation.pop("per_label", None)
                if segmentation.get("different_voxels") == 0:
                    segmentation["all_labels_array_equal"] = True
                for key in ("columns", "reference_columns", "candidate_columns"):
                    volumes.pop(key, None)
                if volumes.get("max_absolute_difference_mm3") == 0:
                    volumes["all_numeric_columns_equal"] = True
            elif volumes.get("column_names_and_order_equal"):
                volumes.pop("candidate_columns", None)
            if detailed:
                segmentation["per_label"] = [{key: row[key] for key in (
                    "label", "reference_voxels", "dice", "reference_hard_volume_mm3",
                    "hard_volume_signed_difference_mm3", "hard_volume_relative_difference")}
                    for row in segmentation.get("per_label", [])]
                # Names already accompany each volume. Repeating the full
                # header twice adds no new evidence after its equality check.
                volumes.pop("reference_columns", None)
            report["comparisons"][path.name] = comparison
    for index in (1, 2, 3):
        item = {}
        for prefix in ("official", "fnit"):
            name = (f"official_case{index:02d}.json" if prefix == "official"
                    else f"fnit_case{index:02d}_corrected.json")
            path = run_root / "preprocess" / name
            if path.exists():
                item[prefix] = json.loads(path.read_text())
        report["preprocessing"][f"case{index:02d}"] = item
    for freeze in ("baseline_corrected", "candidate_corrected", "candidate_selective",
                   "baseline_final_frontend", "candidate_header", "candidate_original_backend"):
        source = workspace / freeze / "src/fnit"
        if not source.exists():
            continue
        paths = list((source / "synthseg_parc").glob("*.py"))
        paths += [source / "_nib.py", source / "cli.py"]
        report["source_files"][freeze] = {
            str(path.relative_to(source.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}
    report["precision_acceptance"] = {
        "case01_cpu_soft_volumes": "not_assessed: no prospective numerical gate",
        "official_bitwise_equivalence": "not achieved on case01",
        "case02": {}}
    acceptance = workspace / "acceptance_case02.json"
    if acceptance.exists():
        report["precision_acceptance"]["acceptance_case02_sha256"] = hashlib.sha256(
            acceptance.read_bytes()).hexdigest()
    for feature in ("synthseg", "parc_fast", "parc"):
        comparison = report["comparisons"].get(f"case02_{feature}_baseline_candidate.json")
        if comparison is None:
            report["precision_acceptance"]["case02"][feature] = {"status": "pending"}
            continue
        segmentation, volumes = comparison["segmentation"], comparison["soft_volumes"]
        hard_equal = segmentation.get("different_voxels") == 0
        soft_close = all(abs(row["signed_difference_mm3"]) <=
                         0.01 + 1e-5 * abs(row["reference_mm3"]) for row in volumes["columns"])
        geometry_equal = (segmentation["reference_geometry"]["shape"] ==
                          segmentation["candidate_geometry"]["shape"] and
                          segmentation["affine_max_abs_mm"] == 0)
        names_equal = volumes.get("column_names_and_order_equal", False)
        report["precision_acceptance"]["case02"][feature] = {
            "status": "passed" if hard_equal and soft_close and geometry_equal and names_equal else "failed",
            "hard_labels_exact": hard_equal, "csv_rtol_1e_5_atol_0_01": soft_close,
            "shape_affine_exact": geometry_equal, "csv_columns_equal": names_equal,
            "evaluated_freeze": "candidate_corrected_all_slab"}
    report["precision_acceptance"]["original_backend_formal"] = {}
    acceptance = workspace / "acceptance_original_backend.json"
    if acceptance.exists():
        report["precision_acceptance"]["original_backend_acceptance_sha256"] = hashlib.sha256(
            acceptance.read_bytes()).hexdigest()
    for case in ("case01", "case02"):
        name = f"original_backend_{case}_baseline2_candidate3.json"
        comparison = report["comparisons"].get(name)
        if comparison is None:
            report["precision_acceptance"]["original_backend_formal"][case] = {"status": "pending"}
            continue
        segmentation, volumes = comparison["segmentation"], comparison["soft_volumes"]
        hard_equal = segmentation.get("different_voxels") == 0
        soft_close = all(abs(row["signed_difference_mm3"]) <=
                         0.01 + 1e-5 * abs(row["reference_mm3"]) for row in volumes["columns"])
        first, second = segmentation["reference_geometry"], segmentation["candidate_geometry"]
        geometry_equal = (segmentation["affine_max_abs_mm"] == 0 and
                          all(first[key] == second[key] for key in (
                              "shape", "dtype", "zooms", "qform_code", "sform_code")))
        names_equal = volumes.get("column_names_and_order_equal", False)
        report["precision_acceptance"]["original_backend_formal"][case] = {
            "status": "passed" if hard_equal and soft_close and geometry_equal and names_equal else "failed",
            "hard_labels_exact": hard_equal, "csv_rtol_1e_5_atol_0_01": soft_close,
            "geometry_exact": geometry_equal, "csv_columns_equal": names_equal,
            "evaluated_freeze": "candidate_original_backend"}
    if tables_dir is not None:
        tables_dir.mkdir(parents=True, exist_ok=True)
        label_rows, volume_rows = [], []
        for name, comparison in report["comparisons"].items():
            segmentation = comparison.get("segmentation", {})
            volumes = comparison.get("soft_volumes", {})
            label_rows.extend({"comparison": name, **row}
                              for row in segmentation.pop("per_label", []))
            volume_rows.extend({"comparison": name, **row}
                               for row in volumes.pop("columns", []))
        tables = {}
        for filename, rows, fields in [
            ("per_label_errors.public.csv", label_rows, ["comparison", "label", "reference_voxels",
             "dice", "reference_hard_volume_mm3", "hard_volume_signed_difference_mm3",
             "hard_volume_relative_difference"]),
            ("soft_volume_errors.public.csv", volume_rows, ["comparison", "name", "reference_mm3",
             "candidate_mm3", "signed_difference_mm3", "relative_difference"])]:
            path = tables_dir / filename
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            tables[filename] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        report["detailed_error_tables"] = tables
    # Record each distinct saved geometry once. Each paired arm retains a
    # content identifier, including genuinely different pixdim/extensions.
    geometries = {}
    for comparison in report["comparisons"].values():
        for field in ("segmentation", "lesion_probability"):
            values = comparison.get(field, {})
            for arm in ("reference", "candidate"):
                geometry = values.pop(arm + "_geometry", None)
                if geometry is None:
                    continue
                encoded = json.dumps(geometry, sort_keys=True).encode()
                identifier = hashlib.sha256(encoded).hexdigest()[:16]
                geometries[identifier] = geometry
                values[arm + "_geometry_id"] = identifier
    report["saved_geometries"] = geometries
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.run_root, args.workspace, args.tables_dir), indent=2, ensure_ascii=False))
