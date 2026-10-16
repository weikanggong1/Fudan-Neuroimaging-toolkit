"""Record exact inputs and build provenance for a paired pial stage diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

INPUT_OPTIONS = (
    "--adgws-in", "--seg", "--wm", "--invol", "--i", "--rip-label",
    "--pin-medial-wall", "--aparc", "--repulse-surf", "--white-surf",
)


def digest(path: Path) -> dict:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def command(log: str) -> list[str]:
    line = next(
        line.strip() for line in log.splitlines()
        if line.startswith("/") and "mris_place_surface" in line
    )
    return shlex.split(line)


def inputs(argv: list[str]) -> dict:
    return {
        flag: digest(Path(argv[argv.index(flag) + 1]))
        for flag in INPUT_OPTIONS
    }


def normalized(argv: list[str]) -> list[str]:
    result = argv.copy()
    result[0] = "<binary>"
    for flag in ("--i", "--o", "--target"):
        result[result.index(flag) + 1] = f"<{flag}>"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--installed-log", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--source-main", type=Path, required=True)
    parser.add_argument("--source-gradient", type=Path, required=True)
    parser.add_argument("--historical-log", type=Path)
    parser.add_argument("--historical-lh-output", type=Path)
    parser.add_argument("--pristine-source-binary", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source_log = args.source_log.read_text()
    installed_log = args.installed_log.read_text()
    source_argv = command(source_log)
    installed_argv = command(installed_log)
    source_inputs = inputs(source_argv)
    installed_inputs = inputs(installed_argv)
    source_files = (
        args.source_root / "mris_make_surfaces/mris_place_surface.cpp",
        args.source_root / "utils/mrisurf_mri.cpp",
        args.source_root / "utils/mrisurf_timeStep.cpp",
        args.source_main,
        args.source_gradient,
        args.build_root / "mris_make_surfaces/CMakeFiles/mris_place_surface.dir/link.txt",
        args.build_root / "utils/CMakeFiles/utils.dir/flags.make",
        args.build_root / "mris_make_surfaces/CMakeFiles/mris_place_surface.dir/flags.make",
    )
    compiler = "/home1/benchmark-user/anaconda3/bin/x86_64-conda-linux-gnu-g++"
    report = {
        "source_git_commit": subprocess.check_output(
            ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "compiler": compiler,
        "compiler_version": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "source_compile_flags": {
            name: [line for line in (args.build_root / relative).read_text().splitlines()
                   if line.startswith("CXX_FLAGS =") or line.startswith("CXX_DEFINES =")]
            for name, relative in (
                ("main", "mris_make_surfaces/CMakeFiles/mris_place_surface.dir/flags.make"),
                ("utils", "utils/CMakeFiles/utils.dir/flags.make"),
            )
        },
        "thread_configuration": {
            "argv_threads": installed_argv[installed_argv.index("--threads") + 1],
            "openmp_compiler_flag": "-fopenmp",
            "installed_binary_compile_flags": "not embedded in the stripped installed binary; exact original build flags unavailable",
        },
        "installed_binary": digest(Path(installed_argv[0])),
        "source_probe_binary": digest(Path(source_argv[0])),
        "source_and_build_files": [digest(path) for path in source_files],
        "source_command_argv": source_argv,
        "installed_command_argv": installed_argv,
        "normalized_argv_equal": normalized(source_argv) == normalized(installed_argv),
        "source_input_files": source_inputs,
        "installed_input_files": installed_inputs,
        "all_scientific_input_hashes_equal": all(
            source_inputs[flag]["sha256"] == installed_inputs[flag]["sha256"]
            for flag in INPUT_OPTIONS
        ),
        "source_initial_objective": re.findall(
            r"PY_OBJ_REF initial rms=(\S+) sse=(\S+) orig_area=(\S+) total_area=(\S+)",
            source_log,
        ),
        "installed_initial_objective_line": next(
            (line for line in installed_log.splitlines() if re.match(r"^000: dt:", line)),
            "",
        ),
        "non_scientific_runtime_environment": {
            "FREESURFER_HOME": "/opt/reference/Freesurfer/8.2.0-1",
            "SUBJECTS_DIR": "/opt/reference/Freesurfer/8.2.0-1/subjects",
            "FS_LICENSE": "provided to both processes; credential content omitted",
        },
    }
    if args.pristine_source_binary:
        report["pristine_source_binary"] = digest(args.pristine_source_binary)
    if args.historical_log:
        historical = args.historical_log.read_bytes().decode(errors="replace").split("\n")
        sections = {}
        for index, line in enumerate(historical, 1):
            match = re.search(r"#@# T1PialSurf (lh|rh)\b", line)
            if match:
                hemisphere = match.group(1)
                sections[hemisphere] = {"stage_line_number": index, "stage_line": line}
                for next_index in range(index, len(historical)):
                    objective = historical[next_index]
                    if re.match(r"^000: dt:", objective):
                        sections[hemisphere]["initial_line_number"] = next_index + 1
                        sections[hemisphere]["initial_objective_line"] = objective
                        break
        report["historical_original_recon_all"] = {
            "log_path": str(args.historical_log),
            "hemisphere_sections": sections,
            "left_initial_matches_current_installed": (
                sections["lh"]["initial_objective_line"]
                == report["installed_initial_objective_line"]
            ),
            "previous_28453128_association": "28453128 is the right pial initial SSE, not the left",
        }
    if args.historical_lh_output:
        report["historical_original_left_pial_output"] = digest(args.historical_lh_output)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "normalized_argv_equal": report["normalized_argv_equal"],
        "all_scientific_input_hashes_equal": report["all_scientific_input_hashes_equal"],
        "source_initial_objective": report["source_initial_objective"],
        "installed_initial_objective_line": report["installed_initial_objective_line"],
    }, indent=2))


if __name__ == "__main__":
    main()
