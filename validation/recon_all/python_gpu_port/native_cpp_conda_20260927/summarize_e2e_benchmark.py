"""Summarize a completed Conda C++ recon-all run against the archived same-T1 reference.

The official command times come from the archived FreeSurfer recon-all FSTIME log.
They can overlap with wrapper commands; do not sum them to obtain total time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SURFACES = {"orig", "smoothwm", "inflated", "white", "white.preaparc",
            "pial", "pial.T1", "sphere", "sphere.reg"}
KEY_FILES = (
    "mri/orig.mgz", "mri/nu.mgz", "mri/T1.mgz", "mri/synthseg.rca.mgz",
    "mri/aseg.mgz", "mri/aparc+aseg.mgz", "mri/wmparc.mgz",
    *(f"surf/{hemi}.{name}" for hemi in ("lh", "rh")
      for name in ("orig", "white", "pial", "sphere", "sphere.reg",
                   "thickness", "area", "volume", "curv", "sulc")),
    *(f"label/{hemi}.aparc.annot" for hemi in ("lh", "rh")),
    "stats/aseg.stats", "stats/wmparc.stats",
    *(f"stats/{hemi}.aparc.stats" for hemi in ("lh", "rh")),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group(name: str) -> str:
    if name.startswith("mri/"):
        return "mri"
    if name.startswith("surf/"):
        return "surface" if name.split("/", 1)[1].split(".", 1)[1] in SURFACES else "vertex map"
    return name.split("/", 1)[0]


def detail(row: dict) -> str:
    if row.get("missing_candidate") or row.get("missing_reference"):
        return "missing candidate" if row.get("missing_candidate") else "missing reference"
    for label, key in (("voxels", "voxels"), ("xyz", "coordinates_mm"),
                       ("vertex IDs", "vertex_ids")):
        part = row.get(key)
        if isinstance(part, dict):
            return f"{label}: {part.get('exact', '?')}/{part.get('elements', '?')} exact; max |Δ| {part.get('max_abs', '?')}"
    if "elements" in row:
        return f"vertices: {row['exact']}/{row['elements']} exact; max |Δ| {row['max_abs']}"
    if "outlier_rows" in row:
        return f"outlier rows: {row['outlier_rows']}; max numeric |Δ| {row['max_numeric_abs']}"
    return "see strict JSON"


def summarize(run_path: Path, strict_path: Path, baseline_path: Path) -> tuple[dict, str]:
    run = json.loads(run_path.read_text())
    strict = json.loads(strict_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    if run.get("status") != "complete":
        raise ValueError(f"candidate status is {run.get('status')!r}, expected 'complete'")
    if strict.get("checked") != 138 or len(strict.get("files", {})) != 138:
        raise ValueError("strict comparator report must contain the fixed 138 outputs")
    if Path(strict["candidate"]).resolve() != Path(run["subject_dir"]).resolve():
        raise ValueError("strict comparator candidate differs from timed subject")
    if Path(strict["reference"]).resolve() != Path(baseline["official_log"]).parent.parent.resolve():
        raise ValueError("strict comparator reference differs from official benchmark subject")
    input_path = Path(run["input"])
    if sha256(input_path) != baseline["input_sha256"]:
        raise ValueError("candidate T1 SHA-256 differs from archived official benchmark T1")
    official = baseline["official_executable_elapsed_seconds"]
    stages = run["stages"]
    candidate_seconds = float(run["total_seconds"])
    official_seconds = float(baseline["official_total_seconds"])
    counts: dict[str, dict[str, int]] = {}
    for name, row in strict["files"].items():
        state = counts.setdefault(group(name), {"checked": 0, "passed": 0, "missing_candidate": 0})
        state["checked"] += 1
        state["passed"] += bool(row["pass"])
        state["missing_candidate"] += bool(row.get("missing_candidate"))
    summary = {
        "input_sha256": baseline["input_sha256"],
        "candidate_run_sha256": sha256(run_path),
        "strict_report_sha256": sha256(strict_path),
        "official_log_sha256": baseline["official_log_sha256"],
        "official_benchmark_sha256": sha256(baseline_path),
        "candidate_status": run["status"],
        "candidate_total_seconds": candidate_seconds,
        "official_archived_total_seconds": official_seconds,
        "candidate_over_official_observed_ratio": candidate_seconds / official_seconds,
        "strict_checked": strict["checked"], "strict_passed": strict["passed"],
        "strict_all_pass": strict["all_pass"], "strict_groups": counts,
        "timing_note": "Archived official and candidate runs were not paired under controlled host load. Differences in output or scope prevent an equivalent-speed claim.",
    }
    lines = [
        "# Conda C++ recon-all: same-T1 end-to-end observation",
        "",
        f"Candidate profile: `{run['profile']}`; device `{run['device']}`; {run['threads']} CPU threads.",
        f"Input SHA-256: `{summary['input_sha256']}`. Candidate status: **complete**.",
        f"Strict outputs: **{strict['passed']}/{strict['checked']}** passed; all passed: **{strict['all_pass']}**.",
        "",
        "## Observed total wall time",
        "",
        "| Conda candidate | Archived FreeSurfer 8.2 | Candidate / official |",
        "|---:|---:|---:|",
        f"| {candidate_seconds:.1f} s | {official_seconds:.1f} s | {candidate_seconds / official_seconds:.3f} |",
        "",
        "These are one-run observations on a shared host at different times. "
        "The official result is an archived same-T1 run, not a paired cold-start trial. "
        + ("All 138 fixed outputs passed, but timing needs paired repeats before a stable speed claim."
           if strict["all_pass"] else
           "The fixed outputs do not all match, so this ratio is not an equivalent-reconstruction speedup."),
        "",
        "## Candidate stages (inclusive wall time)",
        "",
        "| Stage | Seconds |",
        "|---|---:|",
    ]
    lines.extend(f"| `{row['name']}` | {float(row['seconds']):.2f} |" for row in stages)
    staged_seconds = sum(float(row["seconds"]) for row in stages)
    lines += [f"| Other overhead | {candidate_seconds - staged_seconds:.2f} |", "",
              "`surface_lh/rh` include their nested topology, sphere, and metric steps; do not add the next table to this total.",
              "", "## Candidate hemisphere substeps", "",
              "| Hemisphere | Step | Seconds |", "|---|---|---:|"]
    for hemi, data in run.get("surfaces", {}).items():
        for key in ("topology_python_seconds", "topology_native_seconds"):
            if data.get(key) is not None:
                lines.append(f"| {hemi} | `{key}` | {float(data[key]):.2f} |")
        for outer, label in (("native_sphere_seconds", "sphere"),
                             ("native_metric_seconds", "metric")):
            for name, seconds in (data.get(outer) or {}).items():
                lines.append(f"| {hemi} | `{label}.{name}` | {float(seconds):.2f} |")
    lines += ["", "## Archived official command timing", "",
              "Each row is the sum of that command's `@#@FSTIME` elapsed entries. "
              "Wrapper commands can contain other commands, so these rows must not be summed as an end-to-end total.",
              "", "| Command | Calls | Seconds |", "|---|---:|---:|"]
    lines.extend(f"| `{name}` | {row['calls']} | {float(row['seconds']):.2f} |"
                 for name, row in official.items())
    lines += ["", "## Strict 138-output comparison", "",
              "| Output class | Passed | Checked | Missing candidate |",
              "|---|---:|---:|---:|"]
    lines.extend(f"| {name} | {row['passed']} | {row['checked']} | {row['missing_candidate']} |"
                 for name, row in counts.items())
    lines += ["", "Selected outputs (details and every region/vertex remain in the strict and diagnostic JSON):",
              "", "| Output | Pass | Error summary |", "|---|---|---|"]
    for name in KEY_FILES:
        row = strict["files"][name]
        lines.append(f"| `{name}` | {row['pass']} | {detail(row)} |")
    lines += ["", "## Provenance", "",
              f"- Candidate run SHA-256: `{summary['candidate_run_sha256']}`.",
              f"- Strict report SHA-256: `{summary['strict_report_sha256']}`.",
              f"- Archived official log SHA-256: `{summary['official_log_sha256']}`.",
              f"- Archived benchmark JSON SHA-256: `{summary['official_benchmark_sha256']}`.",
              ""]
    return summary, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--strict-report", type=Path, required=True)
    parser.add_argument("--official-benchmark", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    result, markdown = summarize(args.candidate_run, args.strict_report,
                                 args.official_benchmark)
    args.markdown_out.write_text(markdown)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['strict_passed']}/{result['strict_checked']} outputs; "
          f"candidate {result['candidate_total_seconds']:.1f} s; "
          f"archived official {result['official_archived_total_seconds']:.1f} s")


if __name__ == "__main__":
    main()
