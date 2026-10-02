"""One fresh official FS8.2 CPU pipeline from the unchanged public T1.

Validation only. Does not import FNIT or change the official algorithms.
The outer manager owns subject scheduling. Every component writes its own
record; no old recon-all subject or previous timing is reused. Images remain
inside the server case directory. License contents are never inspected/copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time


TIMER_PATTERNS = (
    ("preprocessing", re.compile(r"^Preprocessing took (\d+) seconds$")),
    ("atlas_alignment", re.compile(r"^Initial atlas alignment took (\d+) seconds$")),
    ("synthetic_prepare_and_fit", re.compile(r"^Initial mesh fitting took (\d+) seconds$")),
    ("intensity_prepare_and_fit", re.compile(r"^Mesh fitting took (\d+) seconds$")),
)
STRUCTURES = ("brainstem", "thalamus", "hippo-amygdala")


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def resources():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            memory[key + "_kib"] = int(value.split()[0])
    return {"unix_time": time.time(), "host": os.uname().nodename,
            "logical_cpus": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "loadavg_1_5_15": list(os.getloadavg()), **memory}


def environment(fs, license_path, subjects):
    env = dict(os.environ, FREESURFER_HOME=str(fs), FS_LICENSE=str(license_path),
               SUBJECTS_DIR=str(subjects), FS_V8_XOPTS="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               NUMEXPR_NUM_THREADS="4", ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="4",
               TF_NUM_INTRAOP_THREADS="4", TF_NUM_INTEROP_THREADS="1")
    env.pop("PYTHONPATH", None)
    # Transfer the official setup's complete exported environment in memory.
    # This child stdout is captured internally; never print it, write it to a
    # report/log, or replace inherited setup values with a partial whitelist.
    code = "import json,os;print(json.dumps(dict(os.environ)))"
    command = ["/bin/bash", "-c",
               'source "$1/SetUpFreeSurfer.sh" >/dev/null; "$2" -c "$3"',
               "fnit-official-setup", str(fs), sys.executable, code]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    env = json.loads(result.stdout)
    env["PATH"] = str(fs / "bin") + os.pathsep + env["PATH"]
    # A setup script must not override the declared CPU-only four-thread run.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        env[key] = "4"
    env.update(CUDA_VISIBLE_DEVICES="", FS_V8_XOPTS="1")
    # recon-all's V8 expert-options path uses FREESURFER, whereas its -version
    # fast path only checks FREESURFER_HOME. Preserve the setup's own alias.
    if env.get("FREESURFER") != str(fs):
        raise ValueError("Official setup did not export the selected FREESURFER root")
    return env


def preflight(fs, license_path, env):
    if not license_path.is_file() or not os.access(license_path, os.R_OK):
        raise ValueError("Existing external FreeSurfer license is unavailable")
    executables = [fs / "bin" / name for name in (
        "recon-all", "segment_subregions", "fspython", "mri_robust_register")]
    for path in executables:
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"Required installed executable unavailable: {path}")
    result = subprocess.run([str(fs / "bin/recon-all"), "-version"], env=env,
                            capture_output=True, text=True, check=True, timeout=60)
    version = result.stdout.strip()
    if "8.2.0" not in version:
        raise ValueError(f"The declared official benchmark requires FS8.2: {version}")
    package = fs / "python/lib/python3.8/site-packages/samseg"
    source = [identity(p) for p in sorted((package / "subregions").glob("*.py"))]
    source.append(identity(package / "cli/segment_subregions.py"))
    native = [identity(p) for p in sorted((package / "gems").glob("*.so"))]
    if not source or not native:
        raise ValueError("Installed official SAMSEG/GEMS files are incomplete")
    atlas = []
    for name in ("BrainstemSS", "ThalamicNuclei", "HippoSF"):
        for filename in ("AtlasMesh.gz", "AtlasDump.mgz", "compressionLookupTable.txt"):
            atlas.append(identity(fs / "average" / name / "atlas" / filename))
    return {"build_stamp": (fs / "build-stamp.txt").read_text().strip(),
            "reconall_version_output": version, "executables": [identity(p) for p in executables],
            "official_python_source": source, "native_libraries": native, "atlas_files": atlas,
            "license": {"path": str(license_path), "exists": True, "readable": True,
                        "bytes": license_path.stat().st_size, "content_read_or_copied": False},
            "cpu_only": True, "default_v8_mode": 1}


def explicit_timers(log, structure):
    result = []
    for number, text in enumerate(log.read_text(errors="replace").splitlines(), 1):
        for step, pattern in TIMER_PATTERNS:
            match = pattern.fullmatch(text.strip())
            if match:
                result.append({"step": step, "seconds": int(match[1]),
                               "line": number, "text": text.strip()})
    expected = 8 if structure == "hippo-amygdala" else 4
    if len(result) != expected or [r["step"] for r in result] != [p[0] for p in TIMER_PATTERNS] * (expected // 4):
        raise ValueError("Completed official component lacks the expected ordered explicit timers")
    for index, row in enumerate(result):
        row["structure"] = ("hippo-amygdala-left" if index < 4 else "hippo-amygdala-right") if structure == "hippo-amygdala" else structure
        row["scope"] = "Official integer process.py timer; phase includes its preparation. Not an additional independent command duration."
    return result


def run_component(component, command, official, env, state):
    record_path = official / (component + "_record.json")
    log_path = official / (component + ".log")
    record = {"component": component, "case_id": state["case_id"], "state": "running",
              "command": command, "input_sha256": state["input"]["sha256"],
              "threads": 4, "cpu_only": True, "resources_before": resources()}
    save(record_path, record)
    with log_path.open("x") as log:
        started = time.monotonic()
        started_unix = time.time()
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        ended = {}
        def wait_exit():
            process.wait()
            ended.update(monotonic=time.monotonic(), unix=time.time())
        watcher = threading.Thread(target=wait_exit, daemon=True)
        watcher.start()
        record.update(pid=process.pid, started_unix=started_unix)
        save(record_path, record)
        watcher.join()
    record.update(exit_code=process.returncode, finished_unix=ended["unix"],
                  process_wall_seconds=ended["monotonic"] - started,
                  process_wall_scope="Immediately before Popen to independent wait() observer; excludes input/software preflight.",
                  resources_after=resources(), log=identity(log_path),
                  state="process_completed" if process.returncode == 0 else "failed")
    save(record_path, record)
    state["components"][component] = {"record": str(record_path), **record}
    save(official / "official_report.json", state)
    if process.returncode:
        raise RuntimeError(f"Official {component} exited {process.returncode}; preserve its log and fresh subject")
    return record, record_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--case-root", required=True, type=Path)
    parser.add_argument("--input-t1", required=True, type=Path)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--freesurfer-home", type=Path, default=Path("/public/software/apps/Freesurfer/8.2.0-1"))
    parser.add_argument("--fs-license", type=Path, default=Path("/public/software/apps/Freesurfer/8.0.0-1/license.txt"))
    parser.add_argument("--threads", type=int, choices=(4,), default=4)
    parser.add_argument("--cpu-affinity", help="Optional comma-separated allowed logical CPU IDs; inherited by all official descendants")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"sub-\d+", args.case_id):
        raise ValueError("case-id must be the fixed dataset subject identifier sub-XX")
    if args.cpu_affinity:
        chosen = {int(v) for v in args.cpu_affinity.split(",")}
        if len(chosen) < 4 or not chosen <= os.sched_getaffinity(0):
            raise ValueError("CPU affinity needs at least four distinct currently allowed logical CPUs")
        os.sched_setaffinity(0, chosen)
    input_t1, case_root, fs = args.input_t1.resolve(), args.case_root.resolve(), args.freesurfer_home.resolve()
    official = case_root / "official"
    subjects = official / "subjects"
    subject = subjects / args.case_id
    started_preflight = time.monotonic()
    original = identity(input_t1)
    if original["sha256"] != args.expected_input_sha256:
        raise ValueError("Public snapshot T1 content differs from the independently selected input SHA")
    if subject.exists() or (official / "official_report.json").exists():
        raise ValueError("Official subject/report already exists; this complete run must start fresh")
    env = environment(fs, args.fs_license.resolve(), subjects)
    software = preflight(fs, args.fs_license.resolve(), env)
    recon = [str(fs / "bin/recon-all"), "-i", str(input_t1), "-s", args.case_id,
             "-sd", str(subjects), "-all", "-openmp", "4"]
    commands = {"reconall": recon}
    for structure in STRUCTURES:
        component = "official_" + structure.replace("-", "_")
        commands[component] = [str(fs / "bin/segment_subregions"), structure,
                               "--cross", args.case_id, "--sd", str(subjects), "--threads", "4",
                               "--out-dir", str(official / "subregions" / structure),
                               "--temp-dir", str(official / ("temp_" + structure))]
    state = {"case_id": args.case_id, "state": "preflight_completed", "input": original,
             "input_description": "Unmodified downloaded public snapshot T1; the public release already includes defacing.",
             "subject_dir": str(subject), "software": software, "commands": commands,
             "driver": identity(Path(__file__)), "resources_preflight": resources(),
             "preflight_seconds": time.monotonic() - started_preflight,
             "environment": {k: env.get(k) for k in ("FREESURFER_HOME", "SUBJECTS_DIR", "FS_LICENSE", "FS_V8_XOPTS",
                 "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
             "reference_from_previous_subject_or_timing": False, "components": {}}
    if args.preflight_only:
        print(json.dumps(state, indent=2))
        return
    official.mkdir(parents=True, exist_ok=True)
    subjects.mkdir(exist_ok=False)
    state.update(state="running", pipeline_started_unix=time.time())
    save(official / "official_report.json", state)
    pipeline_started = time.monotonic()
    try:
        record, record_path = run_component("reconall", recon, official, env, state)
        required = [subject / "mri" / name for name in ("norm.mgz", "aseg.mgz", "wmparc.mgz")]
        done = subject / "scripts/recon-all.done"
        if not done.is_file() or not all(p.is_file() for p in required):
            raise ValueError("recon-all returned success without its completion marker and required full coarse outputs")
        record.update(state="completed", outputs={p.name: identity(p) for p in required}, completion_marker=identity(done))
        internal_log = subject / "scripts/recon-all.log"
        if internal_log.is_file():
            record["reconall_internal_log"] = identity(internal_log)
            record["reconall_explicit_resource_timer_lines"] = [
                {"line": number, "text": text} for number, text in enumerate(internal_log.read_text(errors="replace").splitlines(), 1)
                if text.startswith("#@#") or "#@FSTIME" in text or "@#@FSTIME" in text or "runtime" in text.lower()]
        save(record_path, record)
        state["components"]["reconall"] = {"record": str(record_path), **record}
        # The independent FNIT stage worker may start as soon as these fresh
        # norm/aseg/wmparc outputs have passed their completion/hash gate.
        save(official / "official_report.json", state)
        for structure in STRUCTURES:
            component = "official_" + structure.replace("-", "_")
            record, record_path = run_component(component, commands[component], official, env, state)
            directory = official / "subregions" / structure
            native = ([directory / "brainstemSsLabels.FSvoxelSpace.mgz"] if structure == "brainstem" else
                      [directory / "ThalamicNuclei.FSvoxelSpace.mgz"] if structure == "thalamus" else
                      [directory / (side + ".hippoAmygLabels.FSvoxelSpace.mgz") for side in ("lh", "rh")])
            if not all(p.is_file() for p in native):
                raise ValueError(f"Official {structure} returned success without its native label output")
            record.update(state="completed", native_outputs=[identity(p) for p in native],
                          output_files=[identity(p) for p in sorted(directory.glob("*")) if p.is_file()],
                          explicit_stage_timers=explicit_timers(Path(record["log"]["path"]), structure),
                          untimed_postprocess_seconds=None)
            save(record_path, record)
            state["components"][component] = {"record": str(record_path), **record}
            save(official / "official_report.json", state)
        state.update(state="completed", finished_unix=time.time(),
                     complete_pipeline_wall_seconds=time.monotonic() - pipeline_started,
                     summed_component_process_wall_seconds=sum(r["process_wall_seconds"] for r in state["components"].values()),
                     complete_wall_scope="One current raw public T1: fresh recon-all -all -openmp4 then three sequential subregion commands; includes between-component verification/hashing, excludes initial input/software preflight.")
        if identity(input_t1) != original:
            raise ValueError("Original public T1 changed during the full official run")
        save(official / "official_report.json", state)
        print(json.dumps({"case_id": args.case_id, "state": state["state"], "report": str(official / "official_report.json"),
                          "complete_pipeline_wall_seconds": state["complete_pipeline_wall_seconds"]}))
    except BaseException as error:
        state.update(state="failed", finished_unix=time.time(), failure=repr(error),
                     incomplete_elapsed_seconds=time.monotonic() - pipeline_started)
        save(official / "official_report.json", state)
        raise


if __name__ == "__main__":
    main()
