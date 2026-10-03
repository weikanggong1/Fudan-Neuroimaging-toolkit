#!/usr/bin/env python3
"""修正独立参考检查的BIDS文件名匹配，保留原失败报告和实际生产时钟。

不运行MRI、不修改冻结harness。完整原产物SHA绑定与额外保存后检查独立计时。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
import traceback

import nibabel as nib
import validate_reference_saved as validator

sha = validator.sha256
save = validator.save
CASES = ["CON01"] + [f"CON{x:02d}" for x in range(3, 12)]
GLOB_ERROR = "Expected two fsnative BOLD GIFTI outputs, observed 0"


def resolve_harness_source(case: Path, source_root: Path, expected_sha256: str) -> tuple[Path, str]:
    snapshot = case / "launcher.snapshot.py"
    if snapshot.is_file():
        if sha(snapshot) != expected_sha256:
            raise ValueError("Current fresh attempt launcher snapshot differs from its initial source SHA")
        return snapshot, "exact current attempt launcher.snapshot.py SHA"
    sources = [p for p in sorted(source_root.glob("run_reference_cohort*.py")) if sha(p) == expected_sha256]
    if not sources:
        raise ValueError("Exact legacy frozen harness SHA cannot be resolved")
    return sources[0], "legacy exact source SHA; duplicate same-SHA aliases are equivalent"


def recover(original_path: Path, raw: Path, source_root: Path) -> dict:
    if (original_path.parent / "report.corrected.public.json").exists() or (original_path.parent / "files.corrected.private.json").exists():
        raise FileExistsError("Corrected evidence already exists; refuse to overwrite original hashes, bindings or timing")
    started = time.perf_counter()
    recovery_started_utc = datetime.now(timezone.utc).isoformat()
    own_before = sha(Path(__file__))
    checker_before = sha(Path(validator.__file__))
    original_sha = sha(original_path)
    original = json.loads(original_path.read_text())
    case = original_path.parent
    if original.get("container_exit_code") != 0 or original.get("node_runtime_extraction_exit_code") != 0:
        raise ValueError("MRI process or runtime extraction failed; no QC-only recovery is allowed")
    errors = original.get("QC", {}).get("errors", [])
    if original.get("status") == "failed" and not errors:
        raise ValueError("Unknown failed original report without the explicitly identified QC cause")
    for key in ("source_unchanged_during_run", "input_unchanged_during_run"):
        if key in original and original[key] is not True:
            raise ValueError("Original source/input guard failed; QC-only recovery is forbidden")
    if original.get("status") not in ("complete", "completed", "failed") or any(x != GLOB_ERROR for x in errors):
        raise ValueError("Original failure includes causes beyond the known GIFTI entity-order check")
    report = deepcopy(original)
    subject = report["subject"]
    manifest_path = raw / "public_manifest.json"
    manifest_sha = sha(manifest_path)
    rows = [x for x in json.loads(manifest_path.read_text())["subjects"] if x["subject"] == subject]
    if len(rows) != 1:
        raise ValueError("Public paired source identity is ambiguous")
    inputs = rows[0]
    before = {"t1w": report["input_sha256"].get("t1w", report["input_sha256"].get("T1w")),
              "bold": report["input_sha256"].get("bold", report["input_sha256"].get("BOLD"))}
    raw_paths = {key: raw / inputs[role]["relative_path"] for key, role in (("t1w", "T1w"), ("bold", "BOLD"))}
    after = {key: sha(path) for key, path in raw_paths.items()}
    if before != after or any(after[key] != inputs[role]["sha256"] for key, role in (("t1w", "T1w"), ("bold", "BOLD"))):
        raise ValueError("Raw inputs no longer match original or pinned public acquisition")
    source, binding_method = resolve_harness_source(case, source_root, report["launcher_sha256"])
    report.update(input_sha256=before, input_after_sha256=after,
                  input_unchanged_during_run=True, source_after_sha256=sha(source),
                  source_unchanged_during_run=True, source_binding_method=binding_method,
                  source_kind="Exact original frozen official-reference harness, independently rehashed after run; not FNIT runtime",
                  frames=report["input_frames"], repetition_time=report["input_TR_seconds"],
                  command_exit_code=report["container_exit_code"])
    versions = report["versions"]
    match = re.search(r"-(\d+\.\d+\.\d+)-", versions["FreeSurfer"])
    if match is None:
        raise ValueError("Actual FreeSurfer version cannot be parsed")
    report["software_versions"] = {"fmriprep": versions["fmriprep"], "freesurfer": match.group(1),
                                   **{key: versions[key] for key in ("smriprep", "nipype", "nibabel", "python")}}
    checked = {path: digest for path, digest in ((p, after[k]) for k, p in raw_paths.items())}
    outputs = []
    for item in report["QC"]["outputs"]:
        path = (case / item["relative_path"]).resolve()
        if not path.is_relative_to(case.resolve()) or sha(path) != item["sha256"]:
            raise ValueError("Original saved output binding changed or escapes this fresh attempt")
        checked[path] = item["sha256"]
        if item["kind"] != "fsnative_BOLD":
            outputs.append(item)
    native = sorted((case / "derivatives" / ("sub-" + subject)).rglob("*hemi-*_space-fsnative_bold.func.gii"))
    if len(native) != 2 or {re.search(r"hemi-([LR])_", path.name).group(1) for path in native} != {"L", "R"}:
        raise ValueError("Correct entity-order search did not find exactly two current hemispheres")
    for path in native:
        image = nib.load(str(path))
        digest = sha(path)
        checked[path] = digest
        outputs.append({"kind": "fsnative_BOLD", "relative_path": str(path.relative_to(case)),
                        "bytes": path.stat().st_size, "sha256": digest, "frames": len(image.darrays)})
    report["QC"].update(passed=True, errors=[], outputs=outputs)
    if not report["QC"].get("final_HTML_saved") or report["QC"].get("saved_QC_SVG_count", 0) <= 0:
        raise ValueError("Original full QC report/brain figures were not saved")
    subject_dir = case / "derivatives/sourcedata/freesurfer" / ("sub-" + subject + "_ses-preop")
    if not (subject_dir / "mri/orig.mgz").is_file() or not (subject_dir / "scripts/recon-all.done").is_file():
        raise ValueError("Exact new session FreeSurfer reconstruction is not complete")
    for relative in ("mri/orig.mgz", "scripts/recon-all.done"):
        checked[subject_dir / relative] = sha(subject_dir / relative)
    report["status"] = "complete"
    extra = validator.check(original_path, report_override=report)
    if extra["status"] != "passed":
        raise ValueError("Additional strict saved-output checks did not pass")
    report["output_checks"] = {}
    files = {}
    for item in outputs:
        key = {"MNI152NLin6Asym_res2_preproc": "preproc_mni", "T1w_preproc": "preproc_t1w", "fsLR91k_CIFTI": "dtseries"}.get(item["kind"])
        if key:
            report["output_checks"][key] = item
            files[key] = str(case / item["relative_path"])
    if set(files) != {"preproc_mni", "preproc_t1w", "dtseries"}:
        raise ValueError("Original primary full-run output set is incomplete")
    guards = all(sha(p) == digest for p, digest in checked.items())
    if not guards or sha(source) != report["launcher_sha256"] or sha(original_path) != original_sha or sha(manifest_path) != manifest_sha:
        raise ValueError("Inputs, source, original report or final saved files changed during recovery")
    if sha(Path(__file__)) != own_before or sha(Path(validator.__file__)) != checker_before:
        raise ValueError("Independent recovery/checker source changed")
    target = case / "report.corrected.public.json"
    files.update(recon_all=str(subject_dir), metadata=str(target), original_report=str(original_path))
    save(case / "files.corrected.private.json", files)
    finished = datetime.now(timezone.utc)
    previous_end = datetime.fromisoformat(report["end_utc"])
    report.update(original_harness_status=original["status"], original_harness_QC_errors=errors,
                  original_report_sha256=original_sha, corrective_harness_sha256=own_before,
                  corrective_validator_sha256=checker_before,
                  recovered_saved_QC=extra, corrective_saved_file_guards_equal=guards,
                  recovery_started_utc=recovery_started_utc,
                  recovery_finished_utc=finished.isoformat(),
                  recovery_gap_since_original_end_seconds=(finished - previous_end).total_seconds(),
                  recovered_QC_seconds=time.perf_counter() - started)
    report["corrective_QC_seconds"] = report["recovered_QC_seconds"]
    original_boundary = "Original whole ended at a failed filename-only QC check" if errors else "Original whole completed its initial saved-output QC"
    report["recovery_timing_note"] = "Container-process and original whole clock remain unchanged. " + original_boundary + "; later strict saved QC has its own elapsed time and intervening gap, and is not added or claimed contiguous. Container exited 0; original report remains immutable."
    report["continuous_wall_through_saved_QC_boundary_status"] = "original filename-check failed" if errors else "original QC passed"
    report["wall_seconds"] = report["continuous_wall_through_saved_QC_seconds"]
    save(target, report)
    return {"case_id": subject, "attempt": case.name, "status": "complete", "report_sha256": sha(target),
            "files_private_sha256": sha(case / "files.corrected.private.json"),
            "container_process_wall_seconds": report["container_process_wall_seconds"],
            "original_continuous_wall_through_saved_QC_seconds": report["continuous_wall_through_saved_QC_seconds"],
            "recovered_QC_seconds": report["recovered_QC_seconds"], "original_harness_status": original["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, help="optional legacy source archive root; case launcher.snapshot.py is always preferred")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    state = {"status": "waiting", "cases": {}, "production_timing_includes_recovery": False,
             "recovery_sha256": sha(Path(__file__)), "validator_sha256": sha(Path(validator.__file__))}
    pending = list(args.subjects)
    while pending:
        for subject in list(pending):
            original = sorted((args.reference_root / "cases" / ("sub-" + subject)).glob("attempt-*/report.public.json"))
            ready = [p for p in original if json.loads(p.read_text()).get("container_exit_code") == 0
                     and json.loads(p.read_text()).get("status") not in ("running", "starting")]
            if not ready:
                continue
            try:
                if len(ready) != 1:
                    raise ValueError("Multiple MRI-successful fresh attempts need explicit selection")
                row = recover(ready[0], args.raw_root, args.source_root or args.reference_root)
            except Exception as error:
                (args.output_root / (subject + ".private.txt")).write_text(traceback.format_exc())
                row = {"case_id": subject, "status": "failed", "error_type": type(error).__name__, "detail": "Independent recovery failed; original MRI/QC evidence and private diagnostic retained."}
            state["cases"][subject] = row
            save(args.output_root / (subject + ".public.json"), row)
            print(json.dumps(row), flush=True)
            pending.remove(subject)
        state.update(pending_cases=list(pending), status="waiting" if pending else "complete" if all(x["status"] == "complete" for x in state["cases"].values()) else "failed")
        save(args.output_root / "cohort.public.json", state)
        if pending and args.watch:
            time.sleep(args.poll_seconds)
        else:
            break


if __name__ == "__main__":
    main()
