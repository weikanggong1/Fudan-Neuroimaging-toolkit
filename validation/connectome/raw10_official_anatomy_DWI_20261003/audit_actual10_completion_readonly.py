"""Read saved official anatomy/reference provenance; never start MRI programs.

The ten-case check revalidates immutable reports and same-case bindings. Only
CON10 receives an additional image-byte/grid audit here; the completed source
map records Task04's full saved-file audit, and CON11 has its separate audit.
Run on the existing CPU Python environment with CUDA hidden, writing stdout to
a new publication JSON. The caller must verify the executed source SHA first.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import pathlib
import socket
import time

import nibabel as nib
import numpy as np


MAP_SHA = "a12765a9dd1257ec1ba6281c61ebdebd6f654c12f173dfc597c025527d1446aa"
ANATOMY_SHA = "8daf5e5b13aa579268738d69e156beb02aaf88e6423aa3d8c8c3f330c8cd2f5a"
REFERENCE_SHA = "0c6191ec2fc950352548d2244a8ee1c03ab22cdd42a2f347bd36ffcb83dd487e"
REFERENCE_HELPER_SHA = "d10befcb9400e7989e224c6054bc34e9d2d1b5d2a41c9f2b23a0a205ac7eb754"
MATRIX_SHA = "9581300f2db3e96aca6914493e8f6dab7b2bff9efdbfd15ada47d744bac9802c"
CASES = ["sub-CON01", "sub-CON03", *[f"sub-CON{x:02d}" for x in range(4, 12)]]


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--executed-source-sha256", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.root
    verified = {}

    def bound(row):
        path = pathlib.Path(row["path"])
        if str(path) not in verified:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(chunk)
            verified[str(path)] = {"path": str(path), "size_bytes": path.stat().st_size,
                                   "sha256": digest.hexdigest()}
        actual = verified[str(path)]
        require(actual["sha256"] == row["sha256"], f"SHA changed: {path}")
        if row.get("size_bytes") is not None:
            require(actual["size_bytes"] == row["size_bytes"], f"size changed: {path}")
        return actual

    def read(row):
        return json.loads(pathlib.Path(bound(row)["path"]).read_text())

    mapping_record = bound({"path": str(root / "task_04/explicit_case_map_v2_completed_view/case_origin_binding.json"),
                            "sha256": MAP_SHA})
    mapping = read(mapping_record)
    require(mapping["state"] == "completed" and mapping["execution_completed"] is True
            and sorted(mapping["cases"]) == sorted(CASES), "actual completed ten-case map required")
    raw_record = bound(mapping["source_identity"]["raw_manifest"])
    raw = read(raw_record)
    helper_record = bound({"path": str(root / "official_anatomy_reference_cb06c0ca/tools/reference/benchmark_connectome_anatomy_official.py"),
                           "sha256": ANATOMY_SHA})
    spec = importlib.util.spec_from_file_location("frozen_official_anatomy_readonly", helper_record["path"])
    anatomy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anatomy)
    result = {"schema_version": 1, "scope": "actual completed ten-case staged official anatomy/reference provenance",
              "state": "completed", "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "host": socket.gethostname(), "gpu_used": False, "MRI_commands_started": False,
              "executed_audit_source_sha256": args.executed_source_sha256,
              "source_map": mapping_record, "source_map_utc": mapping["utc"], "raw_manifest": raw_record,
              "dataset": raw.get("dataset"), "snapshot": raw.get("snapshot"),
              "frozen_anatomy_worker": helper_record,
              "anatomy_DWI_cases_completed": 10, "reference_cases_completed": 10,
              "reference_seeds_per_case": [0, 1, 2, 3, 4], "seed_attempts_per_repeat": 100000,
              "scientific_FNIT_vs_official_parity": "not_assessed_by_this_provenance_audit",
              "continuous_cold_chain": False, "cases": {}}
    for case in CASES:
        entry = mapping["cases"][case]
        require(entry["state"] == "completed" and entry["actual_command_count"] == 198, "reference incomplete: " + case)
        c_record = bound(entry["producer_contracts"]["anatomy_contract"])
        c = read(c_record)
        d_record = bound(entry["producer_contracts"]["official_dwi_contract"])
        d = read(d_record)
        m_record = bound(entry["reference_manifest"])
        m = read(m_record)
        require(c["state"] == d["state"] == m["state"] == "completed"
                and c["case_id"] == d["case_id"] == m["case_id"] == case, "same-case completed inputs required")
        require(c["official_dwi_contract"]["sha256"] == d_record["sha256"]
                and c["official_dwi_contract"]["path"] == d_record["path"], "DWI consumer route changed")
        for name, record in (("anatomy_contract", c_record), ("official_dwi_contract", d_record)):
            require(m["source"][name]["path"] == record["path"]
                    and m["source"][name]["sha256"] == record["sha256"], "reference producer differs")
        report_record = bound(c["official_anatomy_report"])
        report = read(report_record)
        prepared_record = bound(c["prepared_report"])
        prepared = read(prepared_record)
        require(report["case_id"] == prepared["case_id"] == case
                and report["execution_completed"] and prepared["execution_completed"]
                and report["state"] == "official_anatomy_and_dwi_atlas_completed"
                and prepared["state"] == "official_structural_reference_completed", "anatomy execution incomplete")
        require(report["prepared_origin"] == c["prepared_report"]
                and report["official_dwi_origin"]["contract"] == c["official_dwi_contract"], "anatomy origin mismatch")
        require(len(report["outputs"]) == 20 and len(prepared["outputs"]) == 27
                and all(x["returncode"] == 0 for x in report["commands"]), "anatomy outputs/commands incomplete")
        require(report["script_sha256"] == ANATOMY_SHA
                and m["script_sha256"] == REFERENCE_SHA
                and m["reference_command_helper_sha256"] == REFERENCE_HELPER_SHA
                and m["matrix_helper_sha256"] == MATRIX_SHA, "frozen scientific worker changed")
        require(m["execution_completed"] and len(m["completed_commands"]) == 198
                and all(x["returncode"] == 0 for x in m["completed_commands"])
                and m["seeds"] == [0, 1, 2, 3, 4]
                and m["parameters"]["n_seed_attempts"] == 100000, "five-repeat reference incomplete")
        require(sorted(c["atlases"]) == sorted(["fs-aparc", *anatomy.PROFILES]), "eight atlas profiles required")
        fresh = c["fresh_fs_origin"]
        fs = read(fresh["report"])
        original = read(fresh["original_report"]) if fresh.get("original_report") else fs
        require(fs["status"] == "completed" and fs["case_id"] == case
                and original["exit_code"] == 0 and fresh["recon_all_rerun"] is False, "fresh FS exit proof absent")
        require(c["raw_t1w"] == fresh["raw_t1w"] == prepared["preflight"]["anatomy"]["raw_t1w"]
                and m["source"]["raw_t1w"] == c["raw_t1w"], "fresh T1 binding changed")
        original_input = pathlib.Path(original["command"][original["command"].index("-i") + 1]).resolve()
        require(original_input == pathlib.Path(c["raw_t1w"]["path"]).resolve(), "recon-all input differs")
        row = {"state": "completed", "origin_group": entry["controller_binding"],
               "anatomy_consumer": c_record, "official_dwi_consumer": d_record,
               "prepared_report": prepared_record, "official_anatomy_report": report_record,
               "reference_manifest": m_record, "raw_t1w": c["raw_t1w"],
               "fresh_fs_report": fresh["report"], "fresh_fs_original_report": fresh.get("original_report"),
               "prepare_outputs": 27, "anatomy_complete_outputs": 20,
               "complete_mode": report["mode"], "new_complete_commands": len(report["commands"]),
               "new_complete_entry_seconds": report["total_wall_seconds"],
               "new_complete_command_seconds": sum(x["seconds_inclusive"] for x in report["commands"]),
               "official_FLIRT_seconds": next((x["seconds_inclusive"] for x in report["commands"] if x["stage"] == "flirt"), None),
               "reference_commands_exit_zero": 198, "reference_worker_seconds": m["total_wall_seconds"],
               "accepted_tracks": [m["outputs"][str(seed)]["accepted_tracks"] for seed in m["seeds"]],
               "atlas_nodes": {k:v["n_nodes"] for k,v in c["atlases"].items()},
               "prior_source_map_files_verified": entry["file_verification"]["actual_unique_files_verified"]}
        result["cases"][case] = row
        if case == "sub-CON10":
            # Existing frozen validators read files only; no preflight or native execution.
            anatomy.verify_anatomy(report["config"])
            _, paths = anatomy.verify_dwi_contract(c["official_dwi_contract"], case)
            for record in [*prepared["outputs"].values(), *report["outputs"].values(),
                           c["raw_t1w"], *fresh["files"].values(),
                           *[d["files"][k] for k in ("corrected_dwi", "mean_b0", "mean_b0_brain", "brain_mask")]]:
                bound(record)
            grid = nib.load(str(paths["corrected_dwi"]))
            atlases = {}
            for name, atlas in c["atlases"].items():
                image = nib.load(atlas["atlas_dwi"]["path"])
                data = np.asanyarray(image.dataobj)
                nodes = anatomy.read_nodes(atlas["nodes"]["path"])
                require([x["index"] for x in nodes] == list(range(1, atlas["n_nodes"] + 1)), "non-contiguous nodes")
                require(image.shape == grid.shape[:3] and np.allclose(image.affine, grid.affine, atol=1e-5, rtol=0), "DWI grid changed")
                require(np.isfinite(data).all() and (data >= 0).all() and (data <= atlas["n_nodes"]).all()
                        and np.equal(data, np.round(data)).all(), "invalid atlas labels")
                atlases[name] = {"n_nodes": len(nodes), "shape": list(image.shape), "dtype": str(data.dtype),
                                 "present_positive_nodes": int(np.count_nonzero(np.unique(data))),
                                 "maximum_affine_error": float(np.abs(image.affine - grid.affine).max()),
                                 "atlas_dwi": bound(atlas["atlas_dwi"]), "nodes": bound(atlas["nodes"])}
            row["additional_readonly_image_audit"] = {"state": "passed", "fresh_FS_files_verified": len(fresh["files"]),
                "prepared_output_SHA_checks": 27, "complete_output_SHA_checks": 20,
                "same_case_official_DWI_four_files_verified": True, "DWI_shape": list(grid.shape),
                "eight_atlases_full_grid_integer_labels_and_contiguous_nodes": atlases,
                "reference_saved_matrix_audit": "not repeated here; prior immutable source-map file audit retained"}
    old_path = root / "official_anatomy_raw10_CPU_budget_v4/cohort_reference.json"
    old = json.loads(old_path.read_text())
    require(all(old["cases"][k]["state"] == "completed" for k in CASES[:-1])
            and old["cases"]["sub-CON11"]["state"] == "waiting_official_dwi", "old namespace unexpectedly changed")
    result["old_anatomy_controller_observed"] = {"path": str(old_path), "state": old["state"],
        "execution_completed": old["execution_completed"], "completed_cases": CASES[:-1],
        "old_CON11_state": old["cases"]["sub-CON11"]["state"],
        "policy": "old mutable status observed only; new CON11 published by explicit new origin, no aliases or old-state edits"}
    result["verification_scope"] = "Rechecked ten immutable reports/contracts and same-case frozen-source/exit proofs; additionally rehashed CON10 prepare/complete/fresh-FS/four-DWI files and read all eight atlas grids/labels. Original map full-file audits and independent CON11 audit retain their own times. No solver, native MRI executable or GPU was run."
    result["verified_unique_files"] = len(verified)
    result["verified_unique_bytes"] = sum(v["size_bytes"] for v in verified.values())
    result["readonly_audit_seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
