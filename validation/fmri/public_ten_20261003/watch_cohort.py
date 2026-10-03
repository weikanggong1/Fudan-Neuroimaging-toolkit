#!/usr/bin/env python3
"""只读等待正式十例的新 complete 配对，具名绑定 SHA 后写 fresh 收集/脑图批次。

输入 --manifest PRIVATE.json；输出 --output-root NEW_DIRECTORY。
不启动 MRI、不修改 raw/正式报告/源码，也不按 glob 选择旧 attempt。
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

# Analysis helpers are frozen as source bytes. Never create deployment pyc files,
# nor let a previously cached helper replace the source whose SHA was verified.
sys.dont_write_bytecode = True

CASES = ("CON01", "CON03", "CON04", "CON05", "CON06", "CON07", "CON08", "CON09", "CON10", "CON11")
TOOLS = ("collect_cohort.py", "compare_subject.py", "render_cohort.py", "prepare_display_geometry.py")


def verify_analysis_tools(analysis, expected):
    if set(expected) != set(TOOLS):
        raise ValueError("watch must freeze all four verified analysis tools")
    if any(sha256(analysis / name) != digest for name, digest in expected.items()):
        raise RuntimeError("frozen analysis source guard failed before helper import")


def import_verified_comparison(analysis, expected):
    verify_analysis_tools(analysis, expected)
    path = analysis / "compare_subject.py"
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected[path.name]:
        raise RuntimeError("comparison source changed between verification and import")
    module = types.ModuleType("compare_subject")
    module.__file__ = str(path)
    sys.path.insert(0, str(analysis))
    sys.modules[module.__name__] = module
    exec(compile(payload, str(path), "exec"), module.__dict__)
    verify_analysis_tools(analysis, expected)
    return module


def batch_passed(report, exit_code, ready_cases):
    figures = report.get("figures", {})
    return (exit_code == 0 and report.get("status") in ("partial", "complete")
            and tuple(report.get("completed_comparisons", [])) == tuple(ready_cases)
            and report.get("analysis_tools_unchanged") is True
            and isinstance(figures, dict) and figures.get("status") in ("partial", "complete")
            and figures.get("exit_code") == 0
            and isinstance(figures.get("provenance_sha256"), str))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_bytes(path):
    payload = Path(path).read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("watch inputs must be named JSON objects")
    return value, hashlib.sha256(payload).hexdigest()


def read_bound(entry):
    path = Path(entry["path"]).resolve()
    value, digest = read_json_bytes(path)
    if digest != entry["sha256"]:
        raise ValueError("bound watcher configuration/geometry changed")
    return value, digest


def write_json_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o600)


def utc():
    return datetime.now(timezone.utc).isoformat()


def metadata_ready(case, report, raw, *, reference, source_revision, recovery_validator):
    """Cheap metadata gates; the frozen collector rechecks actual output SHA/arrays."""
    if report.get("status") != "complete":
        return False
    expected = {"t1w": raw["T1w"]["sha256"], "bold": raw["BOLD"]["sha256"]}
    tr = report.get("repetition_time")
    if (report.get("subject") != case or report.get("frames") != 180
            or report.get("input_sha256") != expected
            or not isinstance(tr, (int, float)) or isinstance(tr, bool)
            or not math.isclose(tr, raw["repetition_time_seconds"], rel_tol=1e-6, abs_tol=1e-7)
            or report.get("source_unchanged_during_run") is not True):
        raise ValueError("completed run metadata contradicts the bound raw cohort")
    if reference:
        if "recovered_QC_seconds" not in report or recovery_validator(report) is None:
            raise ValueError("automatic collection waits for strict corrected reference")
        versions = report.get("software_versions", {})
        if versions.get("fmriprep") != "25.2.4" or versions.get("freesurfer") != "7.3.2":
            raise ValueError("reference software differs from the fixed complete workflow")
    elif (report.get("source_revision") != source_revision
          or report.get("backend") != "fnit" or report.get("volume_executed") is not True
          or any(report.get(key) is not True for key in
                 ("raw_inputs_unchanged", "configuration_unchanged", "driver_unchanged"))):
        raise ValueError("candidate must be the guarded raw automatic-volume/FNIT formal run")
    return True


def bind_available(config, raw_cases, recovery_validator):
    """Use only each explicitly configured fresh attempt; future corrected files auto-bind."""
    bound = copy.deepcopy(config)
    statuses, ready, watched = {}, {}, {}
    for case in CASES:
        statuses[case] = {"candidate": "pending", "reference": "pending", "pair": "pending"}
        specs = {}
        for side in ("candidate", "reference"):
            original = config[side + "_cases"][case]
            spec = copy.deepcopy(original)
            if side == "reference":
                attempt = Path(original["report"]).resolve().parent
                if not attempt.name.startswith("attempt-") or attempt.parent.name != "sub-" + case:
                    raise ValueError("reference binding must identify its exact fresh case attempt")
                spec["report"] = str(attempt / "report.corrected.public.json")
                spec["files"] = str(attempt / "files.corrected.private.json")
            bound[side + "_cases"][case] = spec
            report_path, files_path = Path(spec["report"]), Path(spec["files"])
            if not report_path.is_file():
                continue
            try:
                report, report_digest = read_json_bytes(report_path)
                statuses[case][side] = report.get("status", "unknown")
                if not metadata_ready(case, report, raw_cases[case], reference=side == "reference",
                                      source_revision=config["source_revision"], recovery_validator=recovery_validator):
                    continue
                if original.get("report_sha256") and original["report_sha256"] != report_digest:
                    raise ValueError("already-bound immutable report changed")
                files, files_digest = read_json_bytes(files_path)
                if original.get("files_sha256") and original["files_sha256"] != files_digest:
                    raise ValueError("already-bound immutable output map changed")
                if not all(isinstance(files.get(key), str) for key in ("preproc_mni", "dtseries")):
                    raise ValueError("complete output map lacks precise final output paths")
                if side == "reference":
                    original_path = report_path.parent / "report.public.json"
                    actual_original, original_digest = read_json_bytes(original_path)
                    if (original_digest != report["original_report_sha256"]
                            or actual_original.get("status") != report["original_harness_status"]
                            or actual_original.get("subject") != case):
                        raise ValueError("corrected reference differs from its actual preserved original report")
                    watched[str(original_path)] = original_digest
                else:
                    queue_path = Path(spec["queue_report"])
                    queue, _ = read_json_bytes(queue_path)
                    row = queue.get("cases", {}).get(case, {})
                    seconds = row.get("process_wall_seconds")
                    if (queue.get("source_revision") != config["source_revision"]
                            or row.get("status") != "complete" or row.get("exit_code") != 0
                            or not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                            or not math.isfinite(seconds) or seconds < 0):
                        statuses[case][side] = "waiting_complete_queue_clock"
                        continue
                spec.update(report_sha256=report_digest, files_sha256=files_digest)
                bound[side + "_cases"][case] = spec
                watched[str(report_path)] = report_digest
                watched[str(files_path)] = files_digest
                specs[side] = {"report_sha256": report_digest, "files_sha256": files_digest}
            except Exception as error:
                statuses[case][side] = "invalid"
                statuses[case][side + "_failure_type"] = type(error).__name__
        if len(specs) == 2:
            statuses[case]["pair"] = "ready"
            ready[case] = specs
    signature = hashlib.sha256(json.dumps(ready, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return bound, statuses, ready, signature, watched


def watch(manifest_path, output_root, *, poll_seconds=30.0, maximum_polls=None,
          maximum_attempts_per_signature=3, retry_seconds=60.0):
    if (not math.isfinite(poll_seconds) or poll_seconds < 5
            or not math.isfinite(retry_seconds) or retry_seconds < 5
            or isinstance(maximum_attempts_per_signature, bool)
            or not isinstance(maximum_attempts_per_signature, int) or maximum_attempts_per_signature < 1
            or (maximum_polls is not None and
                (isinstance(maximum_polls, bool) or not isinstance(maximum_polls, int) or maximum_polls < 1))):
        raise ValueError("poll/retry intervals must be finite >=5 seconds; poll/attempt counts must be positive integers")
    manifest_path, output_root = Path(manifest_path).resolve(), Path(output_root).resolve()
    manifest, manifest_digest = read_json_bytes(manifest_path)
    config, config_digest = read_bound(manifest["cohort_config"])
    analysis = Path(manifest["analysis_root"]).resolve()
    verify_analysis_tools(analysis, manifest["analysis_tools_sha256"])
    if config["cohort_id"] != "formal-v4" or config["source_revision"] != "1128bc52c7a0233266e5b8a8d7dc0b382994e676":
        raise ValueError("this watcher is bound to the final formal-v4 scientific source")
    if set(config["candidate_cases"]) != set(CASES) or set(config["reference_cases"]) != set(CASES):
        raise ValueError("watch requires every predeclared case once on both sides")
    for protected in (analysis, Path(config["source_root"]).resolve(), Path(config["raw_root"]).resolve(),
                      Path(config["candidate_root"]).resolve(),
                      *(Path(row["report"]).resolve().parent for row in config["reference_cases"].values())):
        if output_root.is_relative_to(protected) or protected.is_relative_to(output_root):
            raise ValueError("watch output must remain outside all protected input/source roots")
    compare_subject = import_verified_comparison(analysis, manifest["analysis_tools_sha256"])
    data, _ = read_bound(config["data_manifest"])
    if tuple(data["selected_subjects"]) != CASES or data["license"] != "CC0":
        raise ValueError("watch raw cohort differs from approved fixed CC0 cases")
    raw_cases = {row["subject"]: row for row in data["subjects"]}
    geometry, geometry_digest = read_bound(manifest["render_geometry"])
    if geometry.get("cohort_id") != config["cohort_id"]:
        raise ValueError("display geometry must belong to the same formal cohort")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    script_digest = sha256(__file__)
    completed_signatures, attempts, batches, previous_status, poll = set(), {}, [], None, 0
    needs_attention = set()
    started = time.perf_counter()
    with (output_root / "events.public.jsonl").open("x", buffering=1) as events:
        while maximum_polls is None or poll < maximum_polls:
            poll += 1
            if (sha256(manifest_path) != manifest_digest
                    or sha256(manifest["cohort_config"]["path"]) != config_digest
                    or sha256(manifest["render_geometry"]["path"]) != geometry_digest
                    or sha256(__file__) != script_digest
                    or any(sha256(analysis / name) != digest for name, digest in manifest["analysis_tools_sha256"].items())):
                raise RuntimeError("watch/config/geometry/frozen analysis source guard failed")
            bound, statuses, ready, signature, watched = bind_available(config, raw_cases, compare_subject.reference_recovery_provenance)
            if statuses != previous_status:
                events.write(json.dumps({"host_clock_utc": utc(), "poll": poll, "cases": statuses,
                                        "ready_cases": list(ready)}, allow_nan=False) + "\n")
                previous_status = statuses
            previous_attempt = attempts.get(signature, {"count": 0, "retry_after": 0.0})
            if (ready and signature not in completed_signatures and signature not in needs_attention
                    and time.monotonic() >= previous_attempt["retry_after"]):
                attempt_count = previous_attempt["count"] + 1
                batch = output_root / f"batch_{len(batches)+1:03d}"
                batch.mkdir(mode=0o700)
                bound["render"] = geometry
                config_path = batch / "cohort_config.private.json"
                write_json_new(config_path, bound)
                child_environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
                exit_code, failure_type = None, None
                with (batch / "collector.private.log").open("x") as log:
                    try:
                        result = subprocess.run([sys.executable, str(analysis / "collect_cohort.py"), "--config", str(config_path),
                                                 "--output-root", str(batch / "collection")], stdout=log,
                                                stderr=subprocess.STDOUT, env=child_environment)
                        exit_code = result.returncode
                    except Exception as error:
                        failure_type = type(error).__name__
                        log.write(f"watcher collector launch failed: {failure_type}\n")
                report_path = batch / "collection/cohort.public.json"
                report, report_digest = {}, None
                try:
                    report, report_digest = read_json_bytes(report_path)
                except Exception as error:
                    failure_type = type(error).__name__
                guards = all(sha256(path) == digest for path, digest in watched.items())
                passed = guards and batch_passed(report, exit_code, ready)
                row = {"batch": batch.name, "report_sha256": report_digest, "collector_exit_code": exit_code,
                       "status": report.get("status"), "completed_comparisons": report.get("completed_comparisons", []),
                       "figures": report.get("figures"), "bound_complete_reports_unchanged": guards,
                       "private_configuration_sha256": sha256(config_path), "ready_signature_sha256": signature,
                       "source_revision": config["source_revision"], "cohort_id": config["cohort_id"],
                       "attempt_for_signature": attempt_count, "collection_passed": passed,
                       "failure_type": failure_type, "ready_cases": list(ready)}
                write_json_new(batch / "dispatch.public.json", row)
                batches.append(row)
                events.write(json.dumps({"host_clock_utc": utc(), "new_batch": row}, allow_nan=False) + "\n")
                if not guards:
                    raise RuntimeError("complete source bindings changed during collection")
                if passed:
                    completed_signatures.add(signature)
                else:
                    attempts[signature] = {"count": attempt_count, "retry_after": time.monotonic() + retry_seconds}
                    if attempt_count >= maximum_attempts_per_signature:
                        needs_attention.add(signature)
                        attention = {"status": "needs_attention", "ready_signature_sha256": signature,
                                     "ready_cases": list(ready), "attempts": attempt_count,
                                     "failure_type": failure_type, "last_batch": batch.name,
                                     "scope": "analysis collection failed; MRI inputs and reports unchanged; new case bindings continue to be observed"}
                        write_json_new(batch / "needs_attention.public.json", attention)
                        events.write(json.dumps({"host_clock_utc": utc(), "analysis_needs_attention": attention}, allow_nan=False) + "\n")
                if (passed and report.get("status") == "complete"
                        and tuple(report.get("completed_comparisons", [])) == CASES
                        and report.get("analysis_tools_unchanged") is True
                        and report.get("figures", {}).get("status") == "complete"):
                    final = {"status": "complete", "cohort_id": config["cohort_id"], "source_revision": config["source_revision"],
                             "completed_comparisons": list(CASES), "batches": batches, "watch_script_sha256": script_digest,
                             "private_manifest_sha256": manifest_digest, "analysis_tools_sha256": manifest["analysis_tools_sha256"],
                             "elapsed_watch_seconds_excluded_from_MRI": time.perf_counter()-started}
                    write_json_new(output_root / "watch_complete.public.json", final)
                    return final
            if maximum_polls is None or poll < maximum_polls:
                time.sleep(poll_seconds)
    return {"status": "needs_attention" if needs_attention else "waiting", "polls": poll,
            "batches": batches, "needs_attention_signatures": sorted(needs_attention)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", default=30.0, type=float)
    parser.add_argument("--maximum-polls", type=int, help="验收探针限制；默认持续等待完整十例")
    parser.add_argument("--maximum-attempts-per-signature", type=int, default=3,
                        help="同一组已绑定结果的分析重试总次数；用 fresh 目录保留每次失败")
    parser.add_argument("--retry-seconds", type=float, default=60.0,
                        help="失败分析批次的最短重试间隔；不重新执行 MRI")
    args = parser.parse_args()
    if (args.poll_seconds < 5 or not math.isfinite(args.poll_seconds)
            or (args.maximum_polls is not None and args.maximum_polls < 1)
            or args.maximum_attempts_per_signature < 1 or args.retry_seconds < 5 or not math.isfinite(args.retry_seconds)):
        parser.error("poll/retry intervals must be finite >=5 seconds; poll/attempt counts must be positive")
    try:
        report = watch(args.manifest, args.output_root, poll_seconds=args.poll_seconds,
                       maximum_polls=args.maximum_polls, maximum_attempts_per_signature=args.maximum_attempts_per_signature,
                       retry_seconds=args.retry_seconds)
    except BaseException as error:
        # The launcher owns its private log. Never write into a rejected existing
        # output path: it may be a protected MRI/source directory.
        print(json.dumps({"status": "failed", "failure_type": type(error).__name__,
                          "watch_script_sha256": sha256(__file__)}), file=sys.stderr, flush=True)
        raise
    print(json.dumps({"status": report["status"], "completed_comparisons": report.get("completed_comparisons", [])}), flush=True)
    return 2 if report["status"] == "needs_attention" else 0


if __name__ == "__main__":
    raise SystemExit(main())
