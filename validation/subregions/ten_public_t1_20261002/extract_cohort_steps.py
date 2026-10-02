"""Read actual step timers after the ten-subject cohort analysis is final.

CPU/metadata only: no fitting, image loading, or official-software invocation.
Missing timers remain NA. Subsets/residuals are labelled and never added to
their parent timers. No intermediate-step segmentation Dice is inferred.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


STRUCTURES = ("brainstem", "thalamus", "hippo-amygdala-left", "hippo-amygdala-right")
OFFICIAL_COMPONENTS = ("reconall", "official_brainstem", "official_thalamus", "official_hippo_amygdala")
TERMINAL = {"completed", "completed_with_failures", "failed"}
EXPLICIT = (("preprocessing", r"Preprocessing took (\d+) seconds"),
            ("atlas_alignment", r"Initial atlas alignment took (\d+) seconds"),
            ("synthetic_prepare_and_fit", r"Initial mesh fitting took (\d+) seconds"),
            ("intensity_prepare_and_fit", r"Mesh fitting took (\d+) seconds"))
FSTIME = re.compile(r"^(?:@#@|#@)FSTIME\s+(?P<timestamp>\S+)\s+(?P<command>.+?)\s+N\s+\d+\s+e\s+(?P<seconds>[0-9.eE+\-]+)(?:\s|$)")
HELPER = Path(__file__).resolve().parents[1] / "reproducibility_20261002/extract_final_step_timing_v2.py"


def load_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(value, base):
    value = {"path": str(value)} if not isinstance(value, dict) else value
    path = Path(value["path"])
    path = path if path.is_absolute() else base / path
    actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if any(actual[key] != value[key] for key in ("bytes", "sha256") if key in value):
        raise ValueError("Timing artifact changed after final analysis: " + str(path))
    return actual


def finite_seconds(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or value < 0):
        raise ValueError("Actual timer is nonfinite, negative, or nonnumeric")
    return value


def fnit_extractor():
    # Reuse the exact previous pure extraction function. Its retired CLI imports
    # refer to archived helpers; load this one function's AST, not that CLI or
    # any old paths/main calls. The function needs only math and its arguments.
    tree = ast.parse(HELPER.read_text(), filename=str(HELPER))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "extract_fnit"]
    if len(nodes) != 1:
        raise ValueError("Expected one audited pure FNIT timer extractor")
    namespace = {"math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HELPER), "exec"), namespace)
    return namespace["extract_fnit"]


def row(case, method, mode, structure, step, seconds, scope, field, source, *, status="completed", kind="recorded_float", output=None):
    seconds = finite_seconds(seconds)
    return dict(case_id=case["id"], development_seen=case["development_seen"], method=method, mode=mode,
                structure=structure, step=step, seconds=seconds,
                measurement_status=status if seconds is not None or status != "completed" else "missing_timer",
                measurement_kind=kind if seconds is not None else "missing", scope=scope, field=field,
                source_reference=source, output=output, intermediate_cross_dice=None)


def fnit_rows(case, mode, run, api, source, extract):
    result = []
    for value in extract(api, {**run, "mode": mode}):
        kind = "derived_residual" if value["step"] == "recipe_unallocated_overhead" else "recorded_float"
        scope = value["scope"]
        if value["step"] == "process_wall":
            scope = run["process_wall_scope"]
        elif value["step"] == "source_input_preflight":
            scope = "Current parent per-case frozen source/input identity checking before Popen; initial queue asset hashing is recorded separately."
        result.append(row(case, "fnit", mode, value["structure"], value["step"], value["seconds"],
                          scope, value["field"], source, kind=kind, output=run["output"]))
    shared = api["initialization"]["shared_preprocessing"]
    if "seconds" in shared.get("intensity_preprocessing", {}):
        result.append(row(case, "fnit", mode, "shared", "raw_intensity_preprocessing_subset",
                          shared["intensity_preprocessing"]["seconds"],
                          "Recorded raw intensity preprocessing subtotal inside shared preprocessing; includes grid conversion, TorchFAST and WM scaling. Do not add to parent total.",
                          "initialization/shared_preprocessing/intensity_preprocessing/seconds", source))
    bs = api["initialization"]["brainstem"]
    result.append(row(case, "fnit", mode, "brainstem", "synthetic_core_fit_subset", bs.get("segmentation_fit", {}).get("seconds"),
                      "Recorded custom BrainstemSS synthetic core fit timer inside the independent recipe synthetic/preparation timer; do not add to enclosing timer.",
                      "initialization/brainstem/segmentation_fit/seconds", source))
    # BS records its independent enclosing recipe timers, but not these generic
    # multiscale subdivisions. Expose missingness instead of inventing values.
    for phase, path in (("synthetic", "segmentation_fit/mesh_solver"), ("intensity", "intensity_mesh_solver")):
        for key, step in (("preparation_seconds", "multiscale_prepare_subset"), ("gems_fit_seconds", "multiscale_gems_fit_subset"),
                          ("post_fit_seconds", "multiscale_postfit_subset"), ("total_seconds", "multiscale_total_subset")):
            name = phase + "_" + step
            if not any(v["structure"] == "brainstem" and v["step"] == name for v in result):
                result.append(row(case, "fnit", mode, "brainstem", name, None,
                                  "No independently recorded generic multiscale timer for the custom BrainstemSS solver; missing, not zero.",
                                  "initialization/brainstem/" + path + "/" + key, source))
    result.append(row(case, "fnit", mode, "all", "input_wait", run.get("input_wait_seconds"),
                      ("Parent wait for fresh successful recon-all stage inputs; outside FNIT subprocess/API timers." if mode == "stage" else
                       "Parent raw input-readiness branch timer before identity preflight; outside FNIT subprocess/API timers."),
                      "run_audit/input_wait_seconds", source))
    return result


def official_rows(case, report, source, base):
    result, fstime = [], []
    for component_name in OFFICIAL_COMPONENTS:
        component = report.get("components", {}).get(component_name)
        structure = {"reconall": "reconall", "official_brainstem": "brainstem", "official_thalamus": "thalamus",
                     "official_hippo_amygdala": "hippo-amygdala"}[component_name]
        completed = bool(component and component.get("state") == "completed" and component.get("exit_code") == 0)
        result.append(row(case, "official", "raw", structure, "command_process_wall",
                          component.get("process_wall_seconds") if component else None,
                          component.get("process_wall_scope", "No saved component timing scope.") if component else "Official component unavailable after final outcome.",
                          "components/" + component_name + "/process_wall_seconds", source,
                          status="completed" if completed else "failed_or_unavailable_component"))
        if not completed:
            continue
        log_record = artifact(component["log"], base)
        log_lines = Path(log_record["path"]).read_text(errors="replace").splitlines()
        if component_name == "reconall":
            internal = component.get("reconall_internal_log")
            if internal:
                log_record = artifact(internal, base)
                log_lines = Path(log_record["path"]).read_text(errors="replace").splitlines()
            for saved in component.get("reconall_explicit_resource_timer_lines", []):
                index = int(saved["line"])
                if not 1 <= index <= len(log_lines) or log_lines[index - 1] != saved["text"]:
                    raise ValueError("Saved recon-all resource line differs from actual log")
                match = FSTIME.fullmatch(saved["text"]) or FSTIME.match(saved["text"])
                fstime.append(dict(case_id=case["id"], development_seen=case["development_seen"], source_reference=source,
                                   log_path=log_record["path"], log_sha256=log_record["sha256"], line=index, text=saved["text"],
                                   command=match["command"] if match else None, recorded_timestamp=match["timestamp"] if match else None,
                                   seconds=finite_seconds(float(match["seconds"])) if match else None,
                                   timing_status="explicit_elapsed_e_field" if match else "context_line_without_parsed_elapsed",
                                   scope="Actual recon-all resource log line; e is recorded elapsed seconds. Context timestamps are retained only; no timestamp subtraction or additive recon-all step total."))
            continue
        timers = component.get("explicit_stage_timers", [])
        expected = 8 if component_name == "official_hippo_amygdala" else 4
        if len(timers) != expected:
            raise ValueError("Completed official component lacks exact explicit timer sequence")
        for index, timer in enumerate(timers):
            step, pattern = EXPLICIT[index % 4]
            match = re.fullmatch(pattern, timer["text"])
            if not match or timer["step"] != step or int(match[1]) != timer["seconds"]:
                raise ValueError("Official integer timer text/value/order differs")
            line = int(timer["line"])
            if not 1 <= line <= len(log_lines) or log_lines[line - 1].strip() != timer["text"]:
                raise ValueError("Official integer timer differs from actual log line")
            target = ("hippo-amygdala-left" if index < 4 else "hippo-amygdala-right") if expected == 8 else structure
            if timer["structure"] != target:
                raise ValueError("Official bilateral timer hemisphere differs")
            result.append(row(case, "official", "raw", target, step, timer["seconds"], timer["scope"],
                              f"{log_record['path']}:{line}", source, kind="recorded_integer"))
        for target in (STRUCTURES[2:] if expected == 8 else [structure]):
            result.append(row(case, "official", "raw", target, "postprocess", None,
                              "Official extract/postprocess/cleanup has no independent saved timer; missing, not zero.", None, source))
    completed = report.get("state") == "completed"
    for step, key in (("complete_pipeline_wall", "complete_pipeline_wall_seconds"),
                      ("summed_component_process_wall", "summed_component_process_wall_seconds")):
        result.append(row(case, "official", "raw", "all", step, report.get(key) if completed else None,
                          report.get("complete_wall_scope") if step == "complete_pipeline_wall" else
                          "Saved sum of recon-all and three component process walls; excludes between-component verification. Not the sum of overlapping internal/subset timers.",
                          key, source))
    return result, fstime


def summarize(rows, cases):
    groups = sorted({(r["method"], r["mode"], r["structure"], r["step"]) for r in rows})
    result = {}
    for name, selected in (("cohort_all", cases), ("cohort_new_subjects", [c for c in cases if not c["development_seen"]])):
        ids = {c["id"] for c in selected}
        summaries = []
        for key in groups:
            current = [r for r in rows if r["case_id"] in ids and (r["method"], r["mode"], r["structure"], r["step"]) == key]
            if len({r["case_id"] for r in current}) != len(current):
                raise ValueError("Duplicate subject/step; cannot treat invocations as independent subjects")
            values = [r["seconds"] for r in current if r["measurement_status"] == "completed" and r["seconds"] is not None]
            summaries.append(dict(zip(("method", "mode", "structure", "step"), key)) |
                             dict(planned_subjects=len(ids), defined_subjects=len(values), missing_or_failed_subjects=len(ids)-len(values),
                                  mean=statistics.fmean(values) if values else None, median=statistics.median(values) if values else None,
                                  min=min(values) if values else None, max=max(values) if values else None,
                                  std_sample=statistics.stdev(values) if len(values)>1 else None,
                                  scope=sorted({r["scope"] for r in current if r["scope"]}),
                                  measurement_kinds=sorted({r["measurement_kind"] for r in current if r["seconds"] is not None})))
        result[name] = dict(planned_subjects=len(ids), case_ids=sorted(ids), rows=summaries)
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_tsv(path, rows):
    columns = list(dict.fromkeys(k for row_value in rows for k in row_value))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=columns)
        writer.writeheader()
        for value in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v for k,v in value.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--analysis", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    analysis_path = args.analysis or root / "analysis/cohort_analysis.json"
    manifest_path = args.manifest or root / "cohort_manifest.json"
    if not analysis_path.is_file():
        raise ValueError("Final ten-subject cohort analysis is unavailable; no final step artifacts written")
    analysis, manifest = load_json(analysis_path), load_json(manifest_path)
    if analysis.get("final_outcome_ready") is not True:
        raise ValueError("Ten-subject outcome is not final; no final step artifacts written")
    cases = manifest["cases"]
    ids = {c["id"] for c in cases}
    if (len(cases) != 10 or len(ids) != 10 or analysis.get("planned_subjects") != 10
            or {c["case_id"] for c in analysis["cases"]} != ids or len(analysis["cases"]) != 10
            or sha256(manifest_path) != analysis["manifest_sha256"]):
        raise ValueError("Final analysis and predetermined ten-case manifest differ")
    base = manifest_path.parent
    source_manifest = artifact(analysis["source_audit"]["source_manifest"], base)
    frozen = load_json(source_manifest["path"])
    runtime = {x["path"].removeprefix("src/fnit/"):x["sha256"] for x in frozen["files"]
               if x["path"].startswith("src/fnit/") and x["path"].endswith(".py")}
    runs, queue_artifacts = [], []
    for entry in analysis["queue_sources"]:
        value = artifact(entry, base)
        queue = load_json(value["path"])
        if queue.get("state", queue.get("status")) not in TERMINAL:
            raise ValueError("A final analysis queue is still running")
        runs.extend(queue.get("runs", [])); queue_artifacts.append(value)
    analyzed = {c["case_id"]:c for c in analysis["cases"]}
    extract = fnit_extractor()
    rows, fstime, sources, outcomes = [], [], {}, []
    for case in cases:
        outcome = dict(case_id=case["id"], development_seen=case["development_seen"], analysis_status=analyzed[case["id"]]["status"], components={})
        for mode in ("raw", "stage"):
            matched = [r for r in runs if r["case_id"] == case["id"] and r["component"] == "fnit_"+mode
                       and r.get("state") == "completed" and r.get("status", "success") == "success" and r.get("exit_code") == 0]
            approved = analyzed[case["id"]].get("timings", {}).get("fnit", {}).get(mode)
            source = "fnit:"+case["id"]+":"+mode
            if len(matched)>1:
                raise ValueError("More than one successful full FNIT outcome for a subject/mode")
            if not matched or approved is None:
                outcome["components"]["fnit_"+mode] = "failed_or_unavailable_after_final_analysis"
                rows.append(row(case, "fnit", mode, "all", "process_wall", None,
                                "No successful timing validated by the final cohort analysis; planned subject retained.", None, None,
                                status="failed_or_unavailable_after_final_analysis"))
                continue
            run = matched[0]
            api_record = artifact(run["api_report"], base)
            report_record = artifact(run["report"], base)
            api, report = load_json(api_record["path"]), load_json(report_record["path"])
            if set(api["structures"]) != set(STRUCTURES) or report["source_sha256"] != runtime:
                raise ValueError("Step source does not match the all-structure frozen runtime")
            for actual, expected in ((api["timings"]["compute_seconds"], approved["api_compute_seconds"]),
                                     (report["api_total_seconds"], approved["api_total_seconds"]),
                                     (report["output_save_seconds"], approved["output_save_seconds"]),
                                     (run["process_wall_seconds"], approved["process_wall_seconds"])):
                if not math.isclose(finite_seconds(actual), finite_seconds(expected), rel_tol=1e-9, abs_tol=1e-6):
                    raise ValueError("FNIT recorded timers differ from final analysis")
            sources[source] = dict(api_report=api_record, report=report_record, source_manifest=source_manifest,
                                   context_identity=artifact(run["context_identity"], base), actual_command=run["command"],
                                   process_wall_scope=run["process_wall_scope"])
            rows.extend(fnit_rows(case, mode, run, api, source, extract))
            outcome["components"]["fnit_"+mode] = "completed"
        source = "official:"+case["id"]
        report_path = Path(case["official"]["status_file"])
        if not report_path.is_file():
            outcome["components"]["official"] = "missing_final_report"
            rows.append(row(case, "official", "raw", "all", "complete_pipeline_wall", None,
                            "No fresh official report after final outcome; planned subject retained.", None, None,
                            status="failed_or_unavailable_after_final_analysis"))
        else:
            declared = analyzed[case["id"]].get("official_report", str(report_path))
            report_record = artifact(declared, base)
            report = load_json(report_record["path"])
            if report["case_id"] != case["id"] or report["input"]["sha256"] != case["raw_t1"]["sha256"]:
                raise ValueError("Official step report belongs to another case/input")
            software = report["software"]
            sources[source] = dict(official_report=report_record, software=software,
                                   software_metadata_sha256=hashlib.sha256(json.dumps(software,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
                                   official_validation_driver=report["driver"],
                                   component_record_artifacts={k:artifact(v["record"],base) for k,v in report.get("components",{}).items() if v.get("record")})
            current, provenance = official_rows(case, report, source, base)
            rows.extend(current); fstime.extend(provenance)
            outcome["components"]["official"] = report["state"]
        outcomes.append(outcome)
    summary = summarize(rows, cases)
    output = args.output_dir or analysis_path.parent / "steps"
    output.mkdir(parents=True, exist_ok=True)
    result = dict(schema_version=1, final_outcome_ready=True, planned_subjects=10, script=artifact(Path(__file__),base),
                  reused_fnit_timer_extractor=artifact(HELPER,base), analysis=artifact(analysis_path,base),
                  manifest=artifact(manifest_path,base), queue_artifacts=queue_artifacts, sources=sources, cases=outcomes,
                  timing_rules="Actual recorded timers only; subset timers are not additional totals; missing values are NA; derived recipe residuals remain unallocated; observer overhead remains included.",
                  intermediate_accuracy=dict(status="not_quantified_without_comparable_saved_stage_labels",
                    explanation="Final labels are quantified in cohort_analysis. Objective values and initialization mask Dice are not equivalent stage segmentation Dice."),
                  rows=rows, reconall_explicit_resource_lines=fstime, summary=summary)
    write_json(output/"cohort_steps.json",result)
    write_tsv(output/"cohort_steps.tsv",rows)
    write_tsv(output/"cohort_reconall_fstime.tsv",fstime)
    write_json(output/"cohort_steps_summary.json",summary)
    write_tsv(output/"cohort_steps_summary.tsv",[dict(cohort=name,**r) for name,d in summary.items() for r in d["rows"]])
    print(json.dumps(dict(state="completed",planned_subjects=10,rows=len(rows),reconall_provenance_lines=len(fstime),
                         defined_reconall_elapsed_rows=sum(r["seconds"] is not None for r in fstime),output=str(output))))


if __name__ == "__main__":
    main()
