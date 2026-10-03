#!/usr/bin/env python3
"""Run fresh complete official fMRIPrep 25.2.4 references for public ds001226.

Each subject has independent work, derivatives and runtime-home directories.
Only verified immutable templates and the fixed container image are cached.
The whole wall clock begins with container-process creation and includes all
image saves and post-run output QC. Concurrent node durations are never summed
to stand in for that whole wall clock. This is an oracle harness, not FNIT runtime.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

SIF_SHA256 = "8e32238619053c1f9d1739b26f4afd72df809d914f5a5771707bf5da4b1d0f39"
SUBJECTS = ["CON01"] + [f"CON{number:02d}" for number in range(3, 12)]
PROTOCOL = {
    "reference": "Official complete fMRIPrep 25.2.4 with its bundled FreeSurfer",
    "input": "Matched original T1w and complete original resting-state BOLD",
    "discarded_volumes": 0, "slice_timing_correction": False,
    "fieldmaps_supplied": False, "syn_sdc_requested": False,
    "surface_registration": "Default complete MSMSulc, no precomputed subject surfaces",
    "outputs": ["MNI152NLin6Asym:res-2", "T1w", "fsnative", "fsLR 91k CIFTI"],
    "denoising": "None; preprocessed BOLD is the principal comparison",
    "nprocs": 8, "omp_nthreads": 4, "memory_limit_mb": 49152,
    "whole_timing": "Continuous container process creation through completed output validation and saved QC; includes fresh anatomical reconstruction, all volume/surface work, output saves and QC",
    "excluded_from_whole_timing": ["SIF staging/verification", "input checksum preflight", "template cache snapshot copying", "reference version probe"],
    "algorithm_boundary": "Reference ANTs normalization and FNIT chosen FNIRT normalization are different algorithms; complete pipeline comparisons do not assert algorithmic equivalence",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def hardware():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable"):
            memory[key + "_GiB"] = int(value.split()[0]) / 1024**2
    model = next((line.split(":", 1)[1].strip()
                  for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), "unknown")
    return {"CPU_model": model, "logical_cpus": os.cpu_count(),
            "load_average": list(os.getloadavg()), **memory}


def runtime_extractor():
    # This executes only in the fixed oracle container against its own results.
    return '''import hashlib,json,shlex
from pathlib import Path
from nipype.utils.filemanip import loadpkl
root=Path('/case/work')
rows=[]
errors=[]
for path in sorted(root.rglob('result_*.pklz')):
 try:
  result=loadpkl(str(path))
  runtimes=result.runtime if isinstance(result.runtime,list) else [result.runtime]
  for index,runtime in enumerate(runtimes):
   command=str(getattr(runtime,'cmdline','') or '')
   row={'relative_result_path':str(path.relative_to(root)),'runtime_index':index,
        'interface':str(getattr(result,'interface','')),
        'duration_seconds':float(getattr(runtime,'duration',0) or 0),
        'start_time':str(getattr(runtime,'startTime','') or ''),
        'end_time':str(getattr(runtime,'endTime','') or '')}
   if command:
    row['command_sha256']=hashlib.sha256(command.encode()).hexdigest()
    row['command_executable']=Path(shlex.split(command)[0]).name
   for name in ('mem_peak_gb','cpu_percent'):
    value=getattr(runtime,name,None)
    if isinstance(value,(int,float)):row[name]=float(value)
   rows.append(row)
 except Exception as error:
  errors.append({'relative_result_path':str(path.relative_to(root)),
                 'error_type':type(error).__name__})
target=Path('/case/node_runtime.public.json')
target.write_text(json.dumps({'nodes':rows,'extraction_errors':errors,
 'timing_boundary':'Durations come from original trusted locally generated Nipype node runtime records; parallel or nested nodes are not summed into whole wall time'},indent=2)+'\\n')
print(json.dumps({'node_runtime_records':len(rows),'extraction_errors':len(errors)}))
'''


def validate_outputs(case, subject, repetition_time, expected_frames):
    import nibabel as nib
    import numpy as np
    derivatives = case / "derivatives"
    subject_root = derivatives / f"sub-{subject}"
    outputs = []
    errors = []
    patterns = {
        "MNI152NLin6Asym_res2_preproc": "*space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
        "T1w_preproc": "*space-T1w_desc-preproc_bold.nii.gz",
        "fsLR91k_CIFTI": "*den-91k_bold.dtseries.nii",
    }
    for kind, pattern in patterns.items():
        paths = sorted(subject_root.rglob(pattern))
        if len(paths) != 1:
            errors.append(f"{kind}: expected one output, observed {len(paths)}")
            continue
        path = paths[0]
        image = nib.load(str(path))
        record = {"kind": kind, "relative_path": str(path.relative_to(case)),
                  "bytes": path.stat().st_size, "sha256": sha256(path),
                  "shape": list(image.shape)}
        if kind == "fsLR91k_CIFTI":
            axis = image.header.get_axis(0)
            frames, tr = int(image.shape[0]), float(axis.step)
            record["grayordinates"] = int(image.shape[1])
            if image.shape != (expected_frames, 91282):
                errors.append("CIFTI shape differs from complete original frames × 91282")
            if type(axis).__name__ != "SeriesAxis":
                errors.append("CIFTI first axis is not a time series")
            record["finite_fraction"] = float(np.isfinite(np.asanyarray(image.dataobj)).mean())
        else:
            frames, tr = int(image.shape[3]), float(image.header.get_zooms()[3])
            record["zooms"] = [float(v) for v in image.header.get_zooms()]
            # Read complete saved NIfTI once in bounded slabs; no cropped data QC.
            finite_count = 0
            total_count = 0
            for start in range(0, expected_frames, 12):
                block = np.asanyarray(image.dataobj[..., start:start + 12])
                finite_count += int(np.isfinite(block).sum())
                total_count += int(block.size)
            record["finite_fraction"] = finite_count / total_count
            if kind.startswith("MNI") and not np.allclose(record["zooms"][:3], 2, atol=1e-6):
                errors.append("MNI output does not use 2 mm voxels")
        record.update({"frames": frames, "repetition_time_seconds": tr})
        if frames != expected_frames or abs(tr - repetition_time) > 1e-5:
            errors.append(kind + ": complete frame count or original TR not preserved")
        if record["finite_fraction"] != 1.0:
            errors.append(kind + ": non-finite saved values")
        outputs.append(record)
    spheres = sorted(subject_root.rglob("*desc-msmsulc_sphere.surf.gii"))
    if len(spheres) != 2:
        errors.append(f"Expected two newly generated MSMSulc spheres, observed {len(spheres)}")
    for path in spheres:
        image = nib.load(str(path))
        outputs.append({"kind": "new_MSMSulc_sphere", "relative_path": str(path.relative_to(case)),
            "bytes": path.stat().st_size, "sha256": sha256(path),
            "point_count": int(image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data.shape[0])})
    native = sorted(subject_root.rglob("*space-fsnative_hemi-*_bold.func.gii"))
    if len(native) != 2:
        errors.append(f"Expected two fsnative BOLD GIFTI outputs, observed {len(native)}")
    for path in native:
        image = nib.load(str(path))
        frames = len(image.darrays)
        if frames != expected_frames:
            errors.append("fsnative BOLD GIFTI does not preserve full frames")
        outputs.append({"kind": "fsnative_BOLD", "relative_path": str(path.relative_to(case)),
            "bytes": path.stat().st_size, "sha256": sha256(path), "frames": frames})
    html = derivatives / f"sub-{subject}.html"
    svg = list(subject_root.rglob("*.svg"))
    if not html.is_file():
        errors.append("fMRIPrep final HTML report is missing")
    return {"passed": not errors, "errors": errors, "outputs": outputs,
            "final_HTML_saved": html.is_file(), "saved_QC_SVG_count": len(svg)}


def worker(args, subject):
    root, raw = args.root, args.raw
    source_sha256_before = sha256(__file__)
    manifest = json.loads((raw / "public_manifest.json").read_text())
    matches = [row for row in manifest["subjects"] if row["subject"] == subject]
    if len(matches) != 1:
        raise ValueError("Requested complete subject is not yet acquired")
    inputs = matches[0]
    for key in ("T1w", "BOLD", "T1w_json", "BOLD_json"):
        path = raw / inputs[key]["relative_path"]
        if path.stat().st_size != inputs[key]["bytes"] or sha256(path) != inputs[key]["sha256"]:
            raise ValueError("Input differs from acquired public manifest: " + key)
    if inputs["complete_original_frames"] != 180:
        raise ValueError("Pilot and cohort must consume the complete 180 frames")
    if sha256(args.image) != SIF_SHA256:
        raise ValueError("Fixed node-local SIF hash differs")
    if not args.license.is_file() or not os.access(args.license, os.R_OK):
        raise ValueError("Valid existing reference license path is unreadable")
    case = root / "cases" / f"sub-{subject}" / f"attempt-{args.attempt:02d}"
    case.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, case / "launcher.snapshot.py")
    (case / "runtime_home").mkdir()
    shutil.copytree(args.template_cache, case / "templateflow", symlinks=False)
    bindings = [f"{raw}:/data:ro", f"{case}:/case",
                f"{case / 'templateflow'}:/opt/templateflow",
                f"{args.license}:/opt/fs-license/license.txt:ro"]
    container = [str(args.singularity), "exec", "--cleanenv", "--containall",
                 "--bind", ",".join(bindings), "--home", f"{case / 'runtime_home'}:/home/fmriprep", "--pwd", "/case",
                 "--env", "TEMPLATEFLOW_HOME=/opt/templateflow,FS_LICENSE=/opt/fs-license/license.txt,TZ=UTC,OMP_NUM_THREADS=4,MKL_NUM_THREADS=4,OPENBLAS_NUM_THREADS=4,ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4",
                 str(args.image)]
    command = container + ["fmriprep", "/data", "/case/derivatives", "participant",
        "--participant-label", subject, "--ignore", "slicetiming",
        "--cifti-output", "91k", "--output-spaces", "MNI152NLin6Asym:res-2", "T1w", "fsnative",
        "--fs-license", "/opt/fs-license/license.txt", "--fs-subjects-dir", "/case/derivatives/sourcedata/freesurfer", "--nprocs", "8",
        "--omp-nthreads", "4", "--mem-mb", "49152", "--work-dir", "/case/work",
        "--resource-monitor", "--stop-on-first-crash", "--notrack"]
    canonical_command = command[len(container):]
    version = subprocess.run(container + ["python", "-c",
        "import json,sys,fmriprep,smriprep,nipype,nibabel;from nipype.interfaces.freesurfer import Info;print(json.dumps({'fmriprep':fmriprep.__version__,'smriprep':smriprep.__version__,'nipype':nipype.__version__,'nibabel':nibabel.__version__,'python':sys.version.split()[0],'FreeSurfer':Info.version()}))"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    (case / "version_probe.private.log").write_text(version.stdout + version.stderr)
    if version.returncode:
        raise RuntimeError("Reference version probe failed: " + version.stderr[-1000:])
    versions = json.loads(version.stdout.strip().splitlines()[-1])
    if versions["fmriprep"] != "25.2.4":
        raise ValueError("Unexpected installed official fMRIPrep version")
    freesurfer_version = re.search(r"-(\d+\.\d+\.\d+)-", versions["FreeSurfer"]).group(1)
    if freesurfer_version != "7.3.2":
        raise ValueError("Fixed 25.2.4 oracle image has unexpected bundled FreeSurfer")
    write_json(case / "command.private.json", {"actual_host_command": command})
    report = {"schema_version": 1, "subject": subject, "status": "starting",
        "protocol": PROTOCOL, "versions": versions, "SIF_sha256": SIF_SHA256,
        "software_versions": {"fmriprep": versions["fmriprep"], "freesurfer": freesurfer_version,
                              "smriprep": versions["smriprep"], "nipype": versions["nipype"],
                              "nibabel": versions["nibabel"], "python": versions["python"]},
        "launcher_sha256": source_sha256_before,
        "source_kind": "Official reference harness; this does not wrap FNIT source",
        "canonical_container_command": canonical_command,
        "canonical_command_sha256": hashlib.sha256(json.dumps(canonical_command,separators=(",", ":")).encode()).hexdigest(),
        "actual_host_command_sha256": hashlib.sha256(json.dumps(command,separators=(",", ":")).encode()).hexdigest(),
        "input_sha256": {"t1w": inputs["T1w"]["sha256"], "bold": inputs["BOLD"]["sha256"]},
        "input_sidecar_sha256": {key: inputs[key]["sha256"] for key in ("T1w_json", "BOLD_json")},
        "input_frames": 180, "input_TR_seconds": inputs["repetition_time_seconds"],
        "frames": 180, "repetition_time": inputs["repetition_time_seconds"],
        "hardware_before": hardware(), "fresh_anatomical_cache": True,
        "subject_work_existed_before_process": (case / "work").exists(),
        "subject_derivatives_existed_before_process": (case / "derivatives").exists(),
    }
    write_json(case / "report.public.json", report)
    report["start_utc"] = utc()
    started = time.perf_counter()
    with (case / "fmriprep.private.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        report["status"] = "running"
        write_json(case / "process.private.json", {"worker_pid": os.getpid(),
            "container_process_pid": process.pid, "node": os.uname().nodename,
            "start_utc": report["start_utc"]})
        write_json(case / "report.public.json", report)
        print(json.dumps({"subject": subject, "status": "running", "worker_pid": os.getpid(),
                          "container_process_pid": process.pid}), flush=True)
        returncode = process.wait()
    report["container_process_wall_seconds"] = time.perf_counter() - started
    report["container_exit_code"] = returncode
    report["command_exit_code"] = returncode
    report["process_exit_utc"] = utc()
    extractor = case / "extract_runtime.py"
    extractor.write_text(runtime_extractor())
    with (case / "runtime_extraction.private.log").open("w") as log:
        extraction = subprocess.run(container + ["python", "/case/extract_runtime.py"],
            stdout=log, stderr=subprocess.STDOUT)
    report["node_runtime_extraction_exit_code"] = extraction.returncode
    if returncode == 0:
        try:
            report["QC"] = validate_outputs(case, subject, inputs["repetition_time_seconds"], 180)
        except Exception as error:
            report["QC"] = {"passed": False, "errors": [type(error).__name__ + ": " + str(error)]}
    else:
        report["QC"] = {"passed": False, "errors": ["Official complete container process failed; original log and work are retained"]}
    report["hardware_after"] = hardware()
    report["source_after_sha256"] = sha256(__file__)
    report["source_unchanged_during_run"] = report["source_after_sha256"] == source_sha256_before
    report["input_after_sha256"] = {"t1w": sha256(raw / inputs["T1w"]["relative_path"]),
                                      "bold": sha256(raw / inputs["BOLD"]["relative_path"])}
    report["input_unchanged_during_run"] = report["input_after_sha256"] == report["input_sha256"]
    report["output_checks"] = {}
    files = {}
    for item in report["QC"].get("outputs", []):
        key = {"MNI152NLin6Asym_res2_preproc": "preproc_mni",
               "T1w_preproc": "preproc_t1w", "fsLR91k_CIFTI": "dtseries"}.get(item["kind"])
        if key is not None:
            report["output_checks"][key] = item
            files[key] = str(case / item["relative_path"])
    files["recon_all"] = str(case / "derivatives" / "sourcedata" / "freesurfer" / f"sub-{subject}")
    files["metadata"] = str(case / "report.public.json")
    write_json(case / "files.private.json", files)
    report["end_utc"] = utc()
    report["status"] = "complete" if returncode == 0 and report["QC"]["passed"] and extraction.returncode == 0 and report["source_unchanged_during_run"] and report["input_unchanged_during_run"] else "failed"
    write_json(case / "report.public.json", report)
    report["continuous_wall_through_saved_QC_seconds"] = time.perf_counter() - started
    report["wall_seconds"] = report["continuous_wall_through_saved_QC_seconds"]
    report["timing_record_note"] = "The continuous clock stops after the primary final QC report is fsynced; the final scalar timing record is then appended. Parallel node runtime sums are not used as whole wall."
    write_json(case / "report.public.json", report)
    print(json.dumps({"subject": subject, "status": report["status"],
                     "wall_seconds": report["continuous_wall_through_saved_QC_seconds"]}), flush=True)
    return 0 if report["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--template-cache", required=True, type=Path)
    parser.add_argument("--license", required=True, type=Path)
    parser.add_argument("--singularity", type=Path, default=Path("/public/software/apps/singularity/4.2.2/bin/singularity"))
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=SUBJECTS)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--node", default="nodecw10")
    parser.add_argument("--control-socket", type=Path, default=Path("/tmp/common30-nodecw10-20260909.sock"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if len(set(args.subjects)) != len(args.subjects):
        raise ValueError("Duplicate subjects are forbidden")
    args.root.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if len(args.subjects) != 1:
            raise ValueError("One fresh subject per worker")
        raise SystemExit(worker(args, args.subjects[0]))
    write_json(args.root / ("launch_" + "_".join(args.subjects) + ".public.json"),
        {"subjects": args.subjects, "protocol": PROTOCOL, "launcher_sha256": sha256(__file__),
         "concurrency": max(1, min(4, args.concurrency)), "launch_utc": utc()})
    def launch(subject):
        command = ["ssh", "-S", str(args.control_socket), "-o", "BatchMode=yes", args.node,
            "python3", "-u", str(Path(__file__).resolve()), "--worker", "--subjects", subject,
            "--root", str(args.root), "--raw", str(args.raw), "--image", str(args.image),
            "--template-cache", str(args.template_cache), "--license", str(args.license),
            "--singularity", str(args.singularity), "--attempt", str(args.attempt)]
        return subject, subprocess.run(command, stdin=subprocess.DEVNULL).returncode
    with ThreadPoolExecutor(max_workers=max(1, min(4, args.concurrency))) as pool:
        futures = [pool.submit(launch, subject) for subject in args.subjects]
        failed = []
        for future in as_completed(futures):
            subject, code = future.result()
            if code:
                failed.append(subject)
    write_json(args.root / ("dispatch_" + "_".join(args.subjects) + ".public.json"),
        {"subjects": args.subjects, "failed_subjects": failed, "end_utc": utc()})
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

