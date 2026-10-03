#!/usr/bin/env python3
"""Adapt a completed frozen-v1 oracle report after rechecking its evidence.

No MRI processing or new pipeline timing is performed. The original continuous
wall boundary stays intact; the schema adaptation/check cost is recorded alone.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    target = args.case / "report.public.json"
    while True:
        if target.is_file():
            report = json.loads(target.read_text())
            if report.get("status") not in ("starting", "running"):
                break
        if not args.wait:
            raise RuntimeError("Reference is not yet terminal")
        time.sleep(20)
    if report.get("status") not in ("completed", "complete"):
        raise RuntimeError("Original reference failed; report retained without complete status")
    started = time.perf_counter()
    manifest = json.loads((args.raw / "public_manifest.json").read_text())
    matches = [row for row in manifest["subjects"] if row["subject"] == report["subject"]]
    if len(matches) != 1:
        raise RuntimeError("Original paired public inputs cannot be bound")
    inputs = matches[0]
    initial = report["input_sha256"]
    before = {"t1w": initial.get("t1w", initial.get("T1w")),
              "bold": initial.get("bold", initial.get("BOLD"))}
    after = {"t1w": sha256(args.raw / inputs["T1w"]["relative_path"]),
             "bold": sha256(args.raw / inputs["BOLD"]["relative_path"])}
    source_after = sha256(args.source)
    report["input_sha256"] = before
    report["input_after_sha256"] = after
    report["input_unchanged_during_run"] = before == after
    report["source_after_sha256"] = source_after
    report["source_unchanged_during_run"] = report["launcher_sha256"] == source_after
    report["source_kind"] = "Frozen official reference harness; does not wrap FNIT source"
    report["frames"] = report["input_frames"]
    report["repetition_time"] = report["input_TR_seconds"]
    report["command_exit_code"] = report["container_exit_code"]
    versions = report["versions"]
    report["software_versions"] = {"fmriprep": versions["fmriprep"],
        "freesurfer": re.search(r"-(\d+\.\d+\.\d+)-", versions["FreeSurfer"]).group(1),
        "smriprep": versions["smriprep"], "nipype": versions["nipype"],
        "nibabel": versions["nibabel"], "python": versions["python"]}
    report["output_checks"] = {}
    files = {}
    for item in report["QC"]["outputs"]:
        key = {"MNI152NLin6Asym_res2_preproc": "preproc_mni",
               "T1w_preproc": "preproc_t1w", "fsLR91k_CIFTI": "dtseries"}.get(item["kind"])
        if key is not None:
            path = args.case / item["relative_path"]
            if sha256(path) != item["sha256"]:
                raise RuntimeError("Saved reference output changed after primary QC")
            report["output_checks"][key] = item
            files[key] = str(path)
    if not all(key in files for key in ("preproc_mni", "dtseries")):
        raise RuntimeError("Complete saved volume/surface outputs missing")
    files["recon_all"] = str(args.case / "derivatives/sourcedata/freesurfer" / ("sub-" + report["subject"]))
    files["metadata"] = str(target)
    (args.case / "files.private.json").write_text(json.dumps(files, indent=2) + "\n")
    report["wall_seconds"] = report["continuous_wall_through_saved_QC_seconds"]
    report["schema_adapter_seconds"] = time.perf_counter() - started
    report["schema_adapter_timing_note"] = "Original continuous MRI+QC wall is unchanged; this later input/source/output recheck and schema adaptation is listed separately and not summed into whole wall"
    report["status"] = "complete" if (report["source_unchanged_during_run"]
        and report["input_unchanged_during_run"] and report["QC"]["passed"]
        and report["command_exit_code"] == 0 and report["node_runtime_extraction_exit_code"] == 0) else "failed"
    temporary = target.with_name(target.name + ".schema.partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    print(json.dumps({"subject": report["subject"], "status": report["status"],
        "source_unchanged_during_run": report["source_unchanged_during_run"],
        "input_unchanged_during_run": report["input_unchanged_during_run"]}), flush=True)


if __name__ == "__main__":
    main()
