"""Run fresh raw-T1/raw-DWI cohort benchmarks on separate CPU and GPU hosts.

This stdlib-only driver measures official recon-all followed by the real FNIT
raw-BIDS wall runner. It does not implement, approximate, or compare MRI maths.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import uuid

SCHEMA_VERSION = 1
GPU_LOCK = "/tmp/fnit-recon-five-20261002-gongwk.gpu.lock"
MATRICES = ("count", "sift2_fbc", "mean_length", "mean_fa")
ANATOMY = ("mri/brain.mgz", "mri/aparc+aseg.mgz", "mri/ribbon.mgz",
           "surf/lh.white", "surf/rh.white", "surf/lh.pial", "surf/rh.pial")
ATLAS_NAMES = ("fs-aparc", "fs-aparc-a2009s", "aparc+tian-s1", "aparc.a2009s+tian-s1",
               "glasser+tian-s1", "glasser+tian-s4", "schaefer200+tian-s1",
               "schaefer500+tian-s4", "schaefer1000+tian-s4")
RESOURCE_OPTIONS = ("--atlas-templates-dir", "--fsaverage-dir", "--mni-template",
                    "--synthmorph-weights", "--tian-fnirt-coeff")
LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA = re.compile(r"^[0-9a-fA-F]{64}$")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def absolute_path(value, field):
    if not isinstance(value, str) or not Path(value).is_absolute() or "\0" in value:
        raise ValueError(f"{field} must be an absolute path")
    return value


def validate_manifest(manifest, selected=None, *, pilot=False):
    """Validate provenance declarations; remote workers verify actual bytes."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    for key in ("dataset", "snapshot", "license", "source_url", "download_completed_utc"):
        if not manifest.get(key):
            raise ValueError(f"manifest lacks {key}")
    if manifest.get("downloaded_new") is not True:
        raise ValueError("manifest must explicitly declare downloaded_new=true")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest cases must be a nonempty list")
    identifiers, subjects = set(), set()
    for case in cases:
        identifier = case.get("case_id", "")
        subject = case.get("subject", "").removeprefix("sub-")
        if not LABEL.fullmatch(identifier) or not re.fullmatch(r"[A-Za-z0-9]+", subject):
            raise ValueError("case_id/subject must be safe distinct BIDS identifiers")
        if identifier in identifiers or subject in subjects:
            raise ValueError("one case per distinct subject is required")
        identifiers.add(identifier)
        subjects.add(subject)
        root = Path(absolute_path(case.get("bids_root"), "bids_root"))
        t1 = Path(absolute_path(case.get("t1w"), "t1w"))
        files = case.get("input_files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"{identifier}: input_files is required")
        kinds, paths = {}, set()
        for record in files:
            path = Path(absolute_path(record.get("path"), "input_files.path"))
            if not SHA.fullmatch(record.get("sha256", "")):
                raise ValueError(f"{identifier}: missing/invalid SHA-256 for {path}")
            if path in paths or not path.is_relative_to(root) or "derivatives" in path.relative_to(root).parts:
                raise ValueError(f"{identifier}: duplicate/nonraw/outside-BIDS input {path}")
            paths.add(path)
            kinds.setdefault(record.get("kind"), []).append(path)
        for kind in ("raw_t1w", "raw_dwi", "bval", "bvec", "dwi_json", "dataset_description"):
            if kind not in kinds:
                raise ValueError(f"{identifier}: input_files lacks {kind}")
        if len(kinds["raw_t1w"]) != 1 or kinds["raw_t1w"][0] != t1:
            raise ValueError(f"{identifier}: t1w must be the single declared raw T1w")
        if len(kinds["raw_dwi"]) != 1:
            raise ValueError(f"{identifier}: select exactly one primary raw DWI")
        if ("reverse_pe" in kinds or "reverse_dwi" in kinds) and "reverse_json" not in kinds:
            raise ValueError(f"{identifier}: reversed PE image requires declared reverse_json")
        session = case.get("session")
        session = session.removeprefix("ses-") if session else None
        for kind, suffix in (("raw_t1w", "_T1w"), ("raw_dwi", "_dwi")):
            image = kinds[kind][0]
            relative = image.relative_to(root)
            if relative.parts[0] != f"sub-{subject}" or (session and f"ses-{session}" not in relative.parts):
                raise ValueError(f"{identifier}: subject/session pairing differs for {image}")
            if not image.name.endswith((suffix + ".nii", suffix + ".nii.gz")):
                raise ValueError(f"{identifier}: expected original BIDS {suffix} NIfTI")
        for key in ("session", "acquisition", "direction", "run"):
            if case.get(key) is not None and not re.fullmatch(r"[A-Za-z0-9]+", case[key].removeprefix("ses-") if key == "session" else case[key]):
                raise ValueError(f"{identifier}: invalid BIDS {key}")
    if selected:
        missing = set(selected) - identifiers
        if missing:
            raise ValueError(f"unknown selected case identifiers: {sorted(missing)}")
        cases = [case for case in cases if case["case_id"] in set(selected)]
    if not pilot and len(cases) < 10:
        raise ValueError("formal cohort requires at least 10 different subjects; use --pilot for diagnostics")
    return cases


def verify_inputs(case):
    verified = []
    for record in case["input_files"]:
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing/empty raw input: {path}")
        actual = sha256(path)
        if actual.lower() != record["sha256"].lower():
            raise ValueError(f"raw input hash mismatch: {path}")
        verified.append({**record, "size_bytes": path.stat().st_size, "actual_sha256": actual})
    return verified


def source_manifest(source):
    source = Path(source)
    if not (source / "src/fnit/cli.py").is_file():
        raise FileNotFoundError(f"FNIT source missing: {source}")
    def git(*arguments):
        try:
            result = subprocess.run(["git", *arguments], cwd=source, capture_output=True, text=True)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    files = sorted(path for path in (source / "src/fnit").rglob("*")
                   if path.is_file() and path.suffix not in {".pyc", ".pyo"}
                   and "__pycache__" not in path.parts)
    files += [path for path in (source / "pyproject.toml", source / "environment.yml") if path.is_file()]
    hashes = {str(path.relative_to(source)): sha256(path) for path in files}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"directory": str(source.resolve()), "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"), "source_sha256": hashes,
            "source_fingerprint": identity}


def require_fresh(path):
    """Atomically claim a namespace; even an empty previous run is rejected."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def check_anatomy(subject_dir, atlases):
    extra = []
    if "fs-aparc-a2009s" in atlases:
        extra.append("mri/aparc.a2009s+aseg.mgz")
    if any("+tian" in name for name in atlases):
        extra += ["label/lh.aparc.annot", "label/rh.aparc.annot"]
    if "aparc.a2009s+tian-s1" in atlases:
        extra += ["label/lh.aparc.a2009s.annot", "label/rh.aparc.a2009s.annot"]
    if any(name.startswith(("schaefer", "glasser")) for name in atlases):
        extra += ["surf/lh.sphere.reg", "surf/rh.sphere.reg"]
    files = {}
    for name in (*ANATOMY, *extra):
        path = Path(subject_dir) / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"incomplete official recon-all anatomy: {path}")
        with path.open("rb") as stream:
            stream.read(1)
        files[name] = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    done = Path(subject_dir) / "scripts/recon-all.done"
    with done.open("rb") as stream:
        stream.read(1)
    files["scripts/recon-all.done"] = {"path": str(done), "size_bytes": done.stat().st_size, "sha256": sha256(done)}
    return files


def anatomy_geometry(payload):
    """Read real MRI arrays, surfaces and annotations in the Conda child."""
    import nibabel as nib
    import numpy as np

    subject = Path(payload["subject_dir"])
    geometry = {"images": {}, "surfaces": {}, "annotations": {}, "coordinate_policy": "MRI affine is scanner RAS; FreeSurfer surface coordinates are surface RAS"}
    surfaces = {}
    for name in payload["files"]:
        path = subject / name
        if name.endswith(".mgz"):
            image = nib.load(str(path))
            data = np.asanyarray(image.dataobj)
            affine = np.asarray(image.affine)
            if data.ndim != 3 or not data.size or not np.isfinite(data).all() or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
                raise RuntimeError(f"invalid real MGZ data/geometry: {path}")
            if np.count_nonzero(data) == 0:
                raise RuntimeError(f"empty anatomy volume: {path}")
            segmentation = name != "mri/brain.mgz"
            if segmentation and (np.any(data < 0) or not np.array_equal(data, np.rint(data))):
                raise RuntimeError(f"noninteger/negative official segmentation: {path}")
            # MGH/MGZ header dimensions may be numpy.int32; keep the
            # measured dimensions but serialize standard JSON integers.
            record = {"shape": [int(value) for value in image.shape], "dtype": str(data.dtype), "scanner_ras_affine": affine.tolist(),
                      "vox2ras_tkr": image.header.get_vox2ras_tkr().tolist(), "foreground_voxels": int(np.count_nonzero(data)),
                      "full_voxel_array_read": True}
            if segmentation:
                record["labels"] = np.unique(data).astype(np.int64).tolist()
            geometry["images"][name] = record
        elif name.startswith("surf/"):
            coordinates, faces = nib.freesurfer.read_geometry(str(path))
            if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) == 0 or not np.isfinite(coordinates).all() or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or faces.min() < 0 or faces.max() >= len(coordinates):
                raise RuntimeError(f"invalid actual FreeSurfer geometry: {path}")
            geometry["surfaces"][name] = {"vertices": len(coordinates), "faces": len(faces), "full_arrays_read": True}
            surfaces[name] = (coordinates, faces)
        elif name.endswith(".annot"):
            labels, color_table, names = nib.freesurfer.read_annot(str(path), orig_ids=True)
            hemi = path.name[:2]
            white = surfaces.get(f"surf/{hemi}.white")
            if white is None or len(labels) != len(white[0]) or len(names) != len(color_table):
                raise RuntimeError(f"annotation/white vertex mismatch: {path}")
            if not np.isin(labels, np.append(color_table[:, 4], 0)).all():
                raise RuntimeError(f"annotation label absent from its color table: {path}")
            geometry["annotations"][name] = {"vertices": len(labels), "color_table_entries": len(names),
                                             "actual_labels": np.unique(labels).astype(np.int64).tolist(), "full_arrays_read": True}
    for hemi in ("lh", "rh"):
        white, pial = surfaces[f"surf/{hemi}.white"], surfaces[f"surf/{hemi}.pial"]
        if white[0].shape != pial[0].shape or not np.array_equal(white[1], pial[1]):
            raise RuntimeError(f"{hemi} white/pial vertex or ordered face correspondence differs")
    geometry.update(status="actual_images_surfaces_annotations_read", nibabel_version=nib.__version__, numpy_version=np.__version__)
    return geometry


def validate_anatomy_child(subject, files, python_executable):
    payload = {"subject_dir": str(subject), "files": list(files)}
    command = [python_executable, str(Path(__file__).resolve()), "_anatomy"]
    result = subprocess.run(command, input=json.dumps(payload), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"actual nibabel anatomy validation failed: {result.stderr.strip()}")
    geometry = json.loads(result.stdout)
    if geometry.get("status") != "actual_images_surfaces_annotations_read":
        raise RuntimeError("nibabel anatomy child did not verify actual arrays")
    return geometry


def expected_outputs(output, atlases, reverse_pe):
    output = Path(output)
    files = [output / name for name in ("five_tissue_dwi_world.nii.gz", "gmwmi_dwi_world.nii.gz",
             "fa_dwi.nii.gz", "brain_mask_dwi.nii.gz", "dwi_to_t1_world.csv",
             "dataset_description.json", "run_state.json", "preproc/eddy/data.nii.gz",
             "preproc/eddy/data.eddy_rotated_bvecs", "preproc/eddy/state.json", "preproc/raw/bids_selection.json")]
    if reverse_pe:
        files += [output / "preproc/topup" / name for name in
                  ("fieldmap_out_fieldcoef.nii.gz", "fieldmap_iout.nii.gz", "acqparams.txt", "state.json")]
    for atlas in atlases:
        directory = output / "atlases" / atlas
        files += [directory / f"connectome_{name}.csv" for name in MATRICES]
        files += [directory / name for name in ("atlas_dwi.nii.gz", "region_labels.csv", "nodes.tsv")]
    return files


def check_outputs(output, atlases, reverse_pe):
    files = expected_outputs(output, atlases, reverse_pe)
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing/empty CLI output: {path}")
    dimensions = {}
    for atlas in atlases:
        directory = Path(output) / "atlases" / atlas
        with (directory / "nodes.tsv").open(newline="") as stream:
            nodes = list(csv.DictReader(stream, delimiter="\t"))
        count = len(nodes)
        if not count or [int(node["index"]) for node in nodes] != list(range(1, count + 1)):
            raise RuntimeError(f"invalid nodes.tsv: {directory}")
        with (directory / "region_labels.csv").open(newline="") as stream:
            labels = [int(row[0]) for row in csv.reader(stream)]
        if len(labels) != count or len(set(labels)) != count:
            raise RuntimeError(f"region labels and node count differ: {directory}")
        for name in MATRICES:
            with (directory / f"connectome_{name}.csv").open(newline="") as stream:
                rows = [[float(value) for value in row] for row in csv.reader(stream)]
            if len(rows) != count or any(len(row) != count for row in rows):
                raise RuntimeError(f"matrix/node dimension mismatch: {directory}/{name}")
            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    # FNIT and the reference tck2connectome commands retain
                    # self-connections. A nonzero diagonal is a valid output.
                    if not math.isfinite(value) or value != rows[j][i]:
                        raise RuntimeError(f"nonfinite/asymmetric matrix: {directory}/{name}")
                    if name == "count" and (value < 0 or value != int(value)):
                        raise RuntimeError(f"invalid integer counts: {directory}")
        dimensions[atlas] = count
    return {"status": "complete_files_and_matrix_structure", "atlas_node_counts": dimensions,
            "self_connection_policy": "retained; no zero-diagonal requirement",
            "files": {str(path): {"size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in files},
            "scientific_parity": "not_assessed"}


def recon_command(config, case, job):
    subjects = Path(job) / "freesurfer"
    name = f"sub-{case['subject'].removeprefix('sub-')}"
    if case.get("session"):
        name += f"_ses-{case['session'].removeprefix('ses-')}"
    command = [config["recon_all"], "-sd", str(subjects), "-s", name,
               "-i", case["t1w"], "-all", "-openmp", str(config["cpu_threads"])]
    # Positional arguments are expanded by bash, never interpolated as shell code.
    launch = ["/bin/bash", "-c", 'source "$1/SetUpFreeSurfer.sh" >/dev/null && exec "$2" "${@:3}"',
              "fnit-official-recon", config["freesurfer_home"], *command]
    return command, launch, subjects / name


def recon_environment(config, subject):
    """Set the official installation before its setup script is sourced."""
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = str(config["cpu_threads"])
    environment["FREESURFER_HOME"] = config["freesurfer_home"]
    environment["SUBJECTS_DIR"] = str(subject.parent)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    if config.get("fs_license"):
        environment["FS_LICENSE"] = config["fs_license"]
    return environment


def cli_command(config, case, job, *, anatomy_subject=None):
    anatomy = anatomy_subject if anatomy_subject is not None else gpu_anatomy_subject(config, case, job)
    arguments = ["UKBConnectome_pipeline", "--bids-root", case["bids_root"], "--subject", case["subject"],
                 "--freesurfer-subject-dir", str(anatomy), "--output-dir", str(Path(job) / "connectome"),
                 "--device", config["device"], "--n-seeds", str(config["n_seeds"]), "--seed", str(config["seed"])]
    for name in ("session", "run", "acquisition", "direction"):
        if case.get(name) is not None:
            arguments += ["--" + name, case[name]]
    arguments += ["--atlas", *config["atlases"], *config["atlas_options"]]
    return arguments


def gpu_anatomy_subject(config, case, job):
    """Normal runs use their new reconstruction; private reruns bind its origin."""
    if config.get("recovery_mode") == "same_round_common_compatibility_raw_dwi_rerun":
        if __package__:
            from . import benchmark_connectome_raw_rerun as rerun
        else:
            import benchmark_connectome_raw_rerun as rerun
        return rerun.original_paths(config, case)[2]
    return recon_command(config, case, job)[2]


def ssh_command(host, port, control_path, command):
    if not host or host.startswith("-") or any(character.isspace() for character in host):
        raise ValueError("SSH host must be a hostname or user@hostname")
    words = ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", "ConnectTimeout=20",
             "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10"]
    if port is not None:
        words += ["-p", str(port)]
    if control_path:
        words += ["-S", str(control_path)]
    return [*words, host, shlex.join(list(map(str, command)))]


def host_identity():
    return {"hostname": socket.gethostname(), "user": os.environ.get("USER"),
            "python": sys.executable, "python_version": sys.version, "python_executable_sha256": sha256(sys.executable),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}


def load_recon_for_gpu(config, case, version, job):
    """Accept normal completion, or a separately bound private repair report.

    The ordinary fresh-cohort CLI does not declare recovery reports. Only the
    isolated recovery controller may supply an explicit per-case declaration;
    its helper verifies the original failure and this run's unchanged anatomy.
    """
    job = Path(job)
    if config.get("recovery_mode") == "same_round_common_compatibility_raw_dwi_rerun":
        if __package__:
            from . import benchmark_connectome_raw_rerun as rerun
        else:
            import benchmark_connectome_raw_rerun as rerun
        return rerun.load_anatomy(config, case, version, job)
    path = job / "recon_report.json"
    original = json.loads(path.read_text())
    declared = config.get("revalidated_recon_reports", {})
    key = f"{version}/{case['case_id']}"
    if key in declared:
        if config.get("recovery_mode") != "known_anatomy_int32_serialization":
            raise RuntimeError("revalidation requires the explicit private recovery mode")
        if __package__:
            from . import benchmark_connectome_raw_recovery as recovery
        else:
            import benchmark_connectome_raw_recovery as recovery
        return recovery.load_revalidated_recon(config, case, version, job, original)
    if original.get("status") != "completed":
        raise RuntimeError("fresh official recon-all report has not completed")
    return original


def worker(payload, *, anatomy_loader=None, anatomy_subject=None, extra_cli_arguments=()):
    """Execute a cohort stage; private staged drivers may bind verified anatomy.

    The default path remains fresh official reconstruction in this cohort.
    Callbacks belong to the benchmark driver, never to the FNIT runtime.
    """
    config, action = payload["config"], payload["action"]
    if sha256(Path(__file__)) != config["worker_script_sha256"]:
        raise RuntimeError("worker script differs from the coordinator frozen script")
    if action == "claim":
        for path in (config["wall_script"], config["worker_script"], config["recon_all"],
                     str(Path(config["freesurfer_home"]) / "SetUpFreeSurfer.sh")):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        frozen_sources = {label: source_manifest(source) for label, source in config["sources"].items()}
        root = require_fresh(config["run_root"])
        atomic_json(root / "cohort_config.json", config)
        atomic_json(root / "input_manifest.json", payload["manifest"])
        return {"status": "claimed_fresh_namespace", "identity": host_identity(), "path": str(root), "frozen_sources": frozen_sources,
                "wall_script_sha256": sha256(config["wall_script"])}
    case, version = payload["case"], payload["version"]
    job = Path(config["run_root"]) / version / case["case_id"]
    report_path = job / ("recon_report.json" if action == "recon" else "gpu_report.json")
    if report_path.exists() or (action == "recon" and job.exists()):
        return {"status": "failed", "case_id": case["case_id"], "version": version,
                "error": {"type": "FileExistsError", "message": f"refuse existing report/job: {job}"}}
    report = {"schema_version": SCHEMA_VERSION, "action": action, "case_id": case["case_id"],
              "subject": case["subject"], "version": version, "start_utc": utc(), "identity": host_identity(),
              "status": "running", "raw_input_provenance": case["input_files"]}
    started = time.perf_counter()
    try:
        if action == "recon":
            require_fresh(job)
            report["input_verification"] = verify_inputs(case)
            command, launch, subject = recon_command(config, case, job)
            if subject.exists() or (job / "connectome").exists():
                raise FileExistsError(f"preexisting anatomy/downstream output: {job}")
            environment = recon_environment(config, subject)
            license_path = Path(environment.get("FS_LICENSE", str(Path(config["freesurfer_home"]) / "license.txt")))
            with license_path.open("rb") as stream:
                stream.read(1)
            report["license_readable"] = True
            report["license_source"] = "explicit_argument" if config.get("fs_license") else "existing_environment_or_official_default"
            subject.parent.mkdir()
            report.update(command=command, launch_arguments=launch, cpu_threads=config["cpu_threads"],
                          executable_sha256=sha256(command[0]), setup_script_sha256=sha256(Path(config["freesurfer_home"]) / "SetUpFreeSurfer.sh"),
                          environment={key: environment[key] for key in ("FREESURFER_HOME", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "SUBJECTS_DIR", "CUDA_VISIBLE_DEVICES")})
            version_launch = [*launch[:6], "-version"]
            # launch[:6] includes setup/home/executable, but no reconstruction flags.
            version_result = subprocess.run(version_launch, env=environment, capture_output=True, text=True)
            report["freesurfer_version"] = {"returncode": version_result.returncode,
                                            "stdout": version_result.stdout.strip(), "stderr": version_result.stderr.strip()}
            if version_result.returncode or "freesurfer" not in version_result.stdout.lower():
                raise RuntimeError("official recon-all version was not verified")
            environment_keys = ("FREESURFER_HOME", "SUBJECTS_DIR", "FSFAST_HOME", "MNI_DIR", "PATH", "OMP_NUM_THREADS",
                                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
            probe_code = "import json,os; print(json.dumps({key:os.environ.get(key) for key in " + repr(environment_keys) + "}))"
            probe_launch = ["/bin/bash", "-c", 'source "$1/SetUpFreeSurfer.sh" >/dev/null && exec "$2" -c "$3"',
                            "fnit-official-environment", config["freesurfer_home"], sys.executable, probe_code]
            probe = subprocess.run(probe_launch, env=environment, capture_output=True, text=True, check=True)
            report["environment_after_official_setup"] = json.loads(probe.stdout)
            atomic_json(report_path, report)
            with (job / "recon-all.log").open("w") as log:
                command_started = time.perf_counter()
                result = subprocess.run(launch, env=environment, stdout=log, stderr=subprocess.STDOUT)
            report["recon_command_seconds"] = time.perf_counter() - command_started
            report["exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"official recon-all exited {result.returncode}; see {job}/recon-all.log")
            report["anatomy"] = check_anatomy(subject, config["atlases"])
            validation_started = time.perf_counter()
            report["anatomy_geometry"] = validate_anatomy_child(subject, report["anatomy"], config["anatomy_validation_python"])
            report["anatomy_validation_seconds"] = time.perf_counter() - validation_started
            report["raw_t1_sha256_after"] = sha256(case["t1w"])
            expected_t1 = next(item["sha256"] for item in case["input_files"] if item["kind"] == "raw_t1w")
            if report["raw_t1_sha256_after"].lower() != expected_t1.lower():
                raise RuntimeError("raw T1 changed during official reconstruction")
        elif action == "gpu":
            loader = anatomy_loader or load_recon_for_gpu
            recon = loader(config, case, version, job)
            if recon.get("staged_anatomy"):
                report["staged_anatomy"] = recon["staged_anatomy"]
                report["execution_scope"] = "staged end-to-end: this round's separately prepared fresh raw-T1 official anatomy, later frozen candidate source, new full raw-DWI output; not continuous cold pipeline"
            if recon.get("rerun"):
                report["rerun"] = recon["rerun"]
                report["execution_scope"] = "new raw-DWI namespace with this round's original raw-T1 reconstruction and declared common compatibility source; prior failures preserved"
            if recon.get("recovery"):
                report["recovery"] = recon["recovery"]
                report["execution_scope"] = "same-run raw reconstruction with separately recorded tool-validation recovery; not a pristine cold benchmark"
            output = job / "connectome"
            if output.exists() or (job / "raw_bids_wall.json").exists():
                raise FileExistsError(f"refuse reused raw-DWI output/report: {job}")
            lock_path = Path(config["gpu_lock"])
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a") as lock:
                queue_started = time.perf_counter()
                fcntl.flock(lock, fcntl.LOCK_EX)
                report["gpu_lock_queue_seconds"] = time.perf_counter() - queue_started
                source = source_manifest(config["sources"][version])
                if source["source_fingerprint"] != config["frozen_sources"][version]["source_fingerprint"]:
                    raise RuntimeError("source differs from the frozen cohort start")
                if sha256(config["wall_script"]) != config["wall_script_sha256"]:
                    raise RuntimeError("raw-DWI wall script changed after cohort start")
                report["source_before"] = source
                subject = Path(anatomy_subject(config, case, job)) if anatomy_subject is not None else gpu_anatomy_subject(config, case, job)
                report["anatomy"] = check_anatomy(subject, config["atlases"])
                if report["anatomy"] != recon["anatomy"]:
                    raise RuntimeError("official anatomy changed after reconstruction")
                report["input_verification"] = verify_inputs(case)
                command = [config["gpu_python"], config["wall_script"], "--mode", "wall", "--eddy-gp-seed", str(config["eddy_gp_seed"]),
                           "--report", str(job / "raw_bids_wall.json")]
                if config.get("gpu_uuid"):
                    command += ["--gpu-uuid", config["gpu_uuid"]]
                command += ["--", *cli_command(config, case, job, anatomy_subject=subject), *extra_cli_arguments]
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(Path(config["sources"][version]) / "src")
                if config.get("fnit_weights"):
                    environment["FNIT_WEIGHTS"] = config["fnit_weights"]
                if config.get("cuda_alloc_conf"):
                    environment["PYTORCH_CUDA_ALLOC_CONF"] = config["cuda_alloc_conf"]
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    environment[key] = str(config["gpu_cpu_threads"])
                if config.get("cuda_visible_devices") is not None:
                    environment["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
                if config.get("gpu_path_prefix"):
                    environment["PATH"] = os.pathsep.join([*config["gpu_path_prefix"], environment.get("PATH", "")])
                report.update(command=command, wall_script_sha256=sha256(config["wall_script"]),
                              environment={key: environment[key] for key in ("PATH", "PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
                if any(atlas.startswith("glasser") for atlas in config["atlases"]):
                    import shutil
                    workbench = shutil.which("wb_command", path=environment.get("PATH"))
                    if workbench is None:
                        raise FileNotFoundError("Glasser requires the declared Workbench executable on GPU PATH")
                    report["workbench_program"] = {"path": workbench, "sha256": sha256(workbench)}
                if "CUDA_VISIBLE_DEVICES" in environment:
                    report["environment"]["CUDA_VISIBLE_DEVICES"] = environment["CUDA_VISIBLE_DEVICES"]
                if "FNIT_WEIGHTS" in environment:
                    report["environment"]["FNIT_WEIGHTS"] = environment["FNIT_WEIGHTS"]
                if "PYTORCH_CUDA_ALLOC_CONF" in environment:
                    report["environment"]["PYTORCH_CUDA_ALLOC_CONF"] = environment["PYTORCH_CUDA_ALLOC_CONF"]
                atomic_json(report_path, report)
                with (job / "raw_bids_wall.log").open("w") as log:
                    command_started = time.perf_counter()
                    result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                report["gpu_command_wall_seconds"] = time.perf_counter() - command_started
                report["exit_code"] = result.returncode
                if result.returncode:
                    failed_wall_path = job / "raw_bids_wall.json"
                    if failed_wall_path.is_file():
                        failed_wall = json.loads(failed_wall_path.read_text())
                        report["wall_exception"] = failed_wall.get("exception")
                        report["wall_preprocessing"] = failed_wall.get("preprocessing")
                    raise RuntimeError(f"raw-BIDS runner exited {result.returncode}; see {job}/raw_bids_wall.log")
                wall = json.loads((job / "raw_bids_wall.json").read_text())
                check_wall_report(wall, config, case)
                check_selected_inputs(wall, case)
                if Path(wall.get("selected_inputs", {}).get("freesurfer_subject_dir", "")).resolve() != subject.resolve():
                    raise RuntimeError("raw CLI did not use this run's fresh official anatomy")
                reverse = any(item["kind"] in ("reverse_pe", "reverse_dwi") for item in case["input_files"])
                report["outputs"] = check_outputs(output, config["atlases"], reverse)
                after = source_manifest(config["sources"][version])
                if after["source_fingerprint"] != source["source_fingerprint"]:
                    raise RuntimeError("source changed during the actual raw-DWI run")
                report["source_after"] = after
                report["input_verification_after"] = verify_inputs(case)
                if sha256(config["wall_script"]) != config["wall_script_sha256"]:
                    raise RuntimeError("raw-DWI wall script changed during execution")
                if recon.get("recovery"):
                    # The original failed record and its separate repair must
                    # remain unchanged throughout the actual GPU calculation.
                    if load_recon_for_gpu(config, case, version, job).get("recovery") != recon["recovery"]:
                        raise RuntimeError("separate revalidation report changed during the actual GPU execution")
                if recon.get("rerun") and load_recon_for_gpu(config, case, version, job).get("rerun") != recon["rerun"]:
                    raise RuntimeError("original anatomy binding changed during actual GPU execution")
                if recon.get("staged_anatomy") and loader(config, case, version, job).get("staged_anatomy") != recon["staged_anatomy"]:
                    raise RuntimeError("staged anatomy binding changed during actual GPU execution")
                report["wall_report"] = str(job / "raw_bids_wall.json")
                report["raw_dwi_cli_total_runtime_seconds"] = wall["total_runtime_seconds"]
                report["gpu_memory"] = {"process": wall.get("gpu_process_memory"), "allocator": wall.get("cuda_allocator")}
                report["memory_budget"] = memory_budget(wall)
                if report["memory_budget"]["status"] == "exceeded":
                    raise RuntimeError("measured CUDA memory exceeds the strict 20,000,000,000-byte budget")
        else:
            raise ValueError(f"unknown worker action: {action}")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    report.update(end_utc=utc(), worker_wall_seconds=time.perf_counter() - started)
    atomic_json(report_path, report)
    return report


def check_wall_report(wall, config, case):
    if wall.get("mode") != "wall" or wall.get("status") != "completed" or wall.get("exit_code") != 0:
        raise RuntimeError("raw-DWI wall report did not succeed in formal wall mode")
    if wall.get("outputs", {}).get("status") != "complete":
        raise RuntimeError("raw-DWI wall report did not verify complete actual outputs")
    initial = wall.get("initial_output_state", {})
    if initial.get("output_directory_existed") is not False or initial.get("preexisting_state_files") or initial.get("preexisting_run_state"):
        raise RuntimeError("wall runner did not start from an absent output directory")
    stages = wall.get("preprocessing", [])
    if len(stages) != 1 or stages[0].get("eddy") != "completed" or stages[0].get("recon_all") != "supplied":
        raise RuntimeError("expected fresh EDDY and this cohort's freshly generated supplied anatomy")
    reverse = any(item["kind"] in ("reverse_pe", "reverse_dwi") for item in case["input_files"])
    if stages[0].get("topup") != ("completed" if reverse else "no_reverse_pe"):
        raise RuntimeError("TOPUP was reused/supplied or differs from the declared raw inputs")
    if wall.get("actual_eddy_gp_seeds") != [config["eddy_gp_seed"]]:
        raise RuntimeError("actual EDDY GP seed does not match this benchmark protocol")
    if wall.get("official_recon_all_calls"):
        raise RuntimeError("raw-DWI phase unexpectedly launched a second official recon-all")
    seconds = wall.get("total_runtime_seconds")
    if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("missing positive actual raw-DWI wall duration")


def check_selected_inputs(wall, case):
    """The CLI must have processed the exact raw acquisition in the manifest."""
    inputs = wall.get("inputs", {})
    keys = {"raw_dwi": "raw/image", "bval": "raw/bval", "bvec": "raw/bvec",
            "dataset_description": "raw/dataset_description", "reverse_pe": "raw/reverse",
            "reverse_dwi": "raw/reverse", "reverse_bval": "raw/reverse_bval"}
    for declared in case["input_files"]:
        kind = declared["kind"]
        if kind in keys:
            actual = inputs.get(keys[kind], {})
            if Path(actual.get("path", "")).resolve() != Path(declared["path"]).resolve() or actual.get("sha256", "").lower() != declared["sha256"].lower():
                raise RuntimeError(f"CLI raw selection/hash differs from manifest: {declared['path']}")
        elif kind in ("dwi_json", "reverse_json"):
            matches = [item for key, item in inputs.items() if key.startswith("raw_metadata/")
                       and Path(item.get("path", "")).resolve() == Path(declared["path"]).resolve()
                       and item.get("sha256", "").lower() == declared["sha256"].lower()]
            if not matches:
                raise RuntimeError(f"CLI inherited JSON differs from manifest: {declared['path']}")


def memory_budget(wall):
    """Observed bounds only; missing measurements never imply zero usage."""
    process = wall.get("gpu_process_memory") or {}
    allocator = wall.get("cuda_allocator") or {}
    measurements = {"process_tree": process.get("peak_process_tree_bytes")}
    # The ledger exposes these fields; tolerate older ledgers without claiming a pass.
    for key in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes", "max_memory_allocated_bytes", "max_memory_reserved_bytes"):
        if key in allocator:
            measurements[key] = allocator[key]
    values = [value for value in measurements.values() if isinstance(value, (float, int)) and math.isfinite(value) and value >= 0]
    monitor_issues = []
    for key in ("failed_samples", "unresolved_device_samples"):
        if process.get(key, 0):
            monitor_issues.append(key)
    if process.get("errors"):
        monitor_issues.append("errors")
    interval = process.get("sample_interval_seconds")
    gap = process.get("max_observed_interval_seconds")
    if isinstance(interval, (float, int)) and isinstance(gap, (float, int)) and gap > max(5., 4 * interval):
        monitor_issues.append("sampling_gap")
    if any(item.get("error") for item in allocator.get("intervals", [])):
        monitor_issues.append("allocator_errors")
    if any(value >= 20_000_000_000 for value in values):
        status = "exceeded"
    elif measurements["process_tree"] is None or len(values) < 3 or monitor_issues:
        status = "not_fully_measured"
    else:
        status = "observed_below_budget"
    return {"limit_bytes": 20_000_000_000, "measurements": measurements, "status": status,
            "monitor_issues": monitor_issues,
            "scope": "sampled process-tree maximum plus allocator ledger; not a mathematically continuous bound"}


def parent_timing(started, ended, driver_gpu_queue, lock_gpu_queue, cpu_driver_queue=0):
    full = ended - started
    queue = driver_gpu_queue + lock_gpu_queue
    if min(full, driver_gpu_queue, lock_gpu_queue, cpu_driver_queue) < 0 or queue > full + 0.001:
        raise ValueError("invalid parent monotonic timing or queue accounting")
    return {"parent_full_wall_seconds": full, "parent_full_wall_excluding_gpu_queue_seconds": max(0., full - queue),
            "gpu_driver_queue_seconds": driver_gpu_queue, "gpu_lock_queue_seconds": lock_gpu_queue,
            "cpu_driver_queue_seconds": cpu_driver_queue,
            "scope": "actual driver timer from before fresh official recon-all SSH launch through completed raw-DWI SSH/report/output checks; includes GPU queues and source/input hashing; CPU scheduler wait before this timer is separate"}


def remote(config, host_kind, payload, log_path):
    command = [config[host_kind + "_python"], config["worker_script"], "_worker"]
    arguments = ssh_command(config[host_kind + "_host"], config.get(host_kind + "_port"), config.get(host_kind + "_control_path"), command)
    result = subprocess.run(arguments, input=json.dumps(payload, separators=(",", ":")), capture_output=True, text=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{host_kind} SSH/worker failed with code {result.returncode}; see {log_path}")
    return json.loads(result.stdout)


def atomic_csv(path, cases):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    columns = ("version", "case_id", "subject", "status", "parent_full_wall_seconds", "parent_full_wall_excluding_gpu_queue_seconds",
               "gpu_driver_queue_seconds", "gpu_lock_queue_seconds", "cpu_driver_queue_seconds", "recon_command_seconds", "raw_dwi_cli_total_runtime_seconds")
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for record in cases.values():
                writer.writerow({key: record.get(key, record.get("timing", {}).get(key)) for key in columns})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_cohort(config, manifest, cases, report_dir):
    report_dir = require_fresh(report_dir)
    state = {"schema_version": SCHEMA_VERSION, "status": "claiming", "start_utc": utc(), "config": config,
             "dataset": {key: manifest[key] for key in ("dataset", "snapshot", "license", "source_url", "download_completed_utc")},
             "scope": "diagnostic_subset" if config["pilot"] else "formal_raw_cohort",
             "requested_cases_per_version": len(cases), "versions": list(config["sources"]), "cases": {},
             "scientific_parity": "not_assessed", "speedup": "not_assessed", "memory_acceptance": "not_assessed"}
    lock = threading.Lock()
    def save():
        atomic_json(report_dir / "status.json", state)
        atomic_csv(report_dir / "cases.csv", state["cases"])
    save()
    cohort_started = time.perf_counter()
    try:
        claim = remote(config, "cpu", {"action": "claim", "config": config, "manifest": manifest}, report_dir / "claim.stderr.log")
        config["frozen_sources"] = claim["frozen_sources"]
        config["wall_script_sha256"] = claim["wall_script_sha256"]
        state.update(status="running", fresh_namespace=claim)
        save()
        def update(key, **values):
            with lock:
                state["cases"][key].update(values)
                save()
        def reconstruct(key, case, version, submitted):
            started = time.perf_counter()
            update(key, status="recon_running", start_utc=utc(), cpu_driver_queue_seconds=started-submitted)
            result = remote(config, "cpu", {"action": "recon", "config": config, "case": case, "version": version}, report_dir / f"{version}-{case['case_id']}-recon.stderr.log")
            return {"key": key, "case": case, "version": version, "started": started, "cpu_queue": started-submitted, "report": result, "gpu_ready_monotonic": time.perf_counter()}
        def downstream(context, queued):
            begun = time.perf_counter()
            key, case, version = context["key"], context["case"], context["version"]
            update(key, status="gpu_running_or_remote_lock_queue")
            try:
                result = remote(config, "gpu", {"action": "gpu", "config": config, "case": case, "version": version}, report_dir / f"{version}-{case['case_id']}-gpu.stderr.log")
                context.update(gpu_report=result, timing=parent_timing(context["started"], time.perf_counter(), begun-queued,
                               result.get("gpu_lock_queue_seconds", 0.), context["cpu_queue"]))
                update(key, status=result["status"], gpu_report=result, timing=context["timing"],
                       raw_dwi_cli_total_runtime_seconds=result.get("raw_dwi_cli_total_runtime_seconds"),
                       error=result.get("error"), end_utc=utc())
                print(json.dumps({"case": key, "status": result["status"], "timing": context["timing"]}), flush=True)
                return context
            except Exception as error:
                update(key, status="failed", error={"type": type(error).__name__, "message": str(error)})
                raise
        with ThreadPoolExecutor(max_workers=config["cpu_jobs"]) as cpu_pool, ThreadPoolExecutor(max_workers=1) as gpu_pool:
            cpu_futures, gpu_futures = {}, {}
            jobs = [(version, case) for case in cases for version in config["sources"]]
            for version, case in jobs:
                key = f"{version}/{case['case_id']}"
                state["cases"][key] = {"version": version, "case_id": case["case_id"], "subject": case["subject"], "status": "cpu_queued"}
            save()
            for version, case in jobs:
                key = f"{version}/{case['case_id']}"
                cpu_futures[cpu_pool.submit(reconstruct, key, case, version, time.perf_counter())] = key
            for future in as_completed(cpu_futures):
                key = cpu_futures[future]
                try:
                    context = future.result()
                    report = context["report"]
                    if report.get("status") != "completed":
                        update(key, status="failed", error=report.get("error"), recon_report=report)
                        continue
                    update(key, status="gpu_queued", recon_report=report, recon_command_seconds=report["recon_command_seconds"])
                    gpu_futures[gpu_pool.submit(downstream, context, context["gpu_ready_monotonic"])] = key
                except Exception as error:
                    update(key, status="failed", error={"type": type(error).__name__, "message": str(error)})
            for future in as_completed(gpu_futures):
                key = gpu_futures[future]
                try:
                    future.result()
                except Exception as error:
                    update(key, status="failed", error={"type": type(error).__name__, "message": str(error)})
        counts = {version: sum(record["version"] == version and record["status"] == "completed" for record in state["cases"].values()) for version in config["sources"]}
        state["completed_cases_per_version"] = counts
        state["status"] = "completed_execution" if all(value == len(cases) for value in counts.values()) else "failed_or_incomplete"
        memory_states = [record.get("gpu_report", {}).get("memory_budget", {}).get("status") for record in state["cases"].values()]
        state["memory_acceptance"] = "observed_below_budget" if memory_states and all(value == "observed_below_budget" for value in memory_states) else "not_fully_measured_or_failed"
        state["comparison_ready"] = not config["pilot"] and set(counts) == {"baseline", "candidate"} and all(value >= 10 for value in counts.values())
    except Exception as error:
        state.update(status="failed_or_incomplete", error={"type": type(error).__name__, "message": str(error)})
    state.update(end_utc=utc(), cohort_makespan_seconds=time.perf_counter()-cohort_started)
    save()
    return 0 if state["status"] == "completed_execution" else 1


def parse_options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="read saved driver status; never resume or launch work")
    status.add_argument("--report-dir", type=Path, required=True)
    run = commands.add_parser("run", help="claim a new namespace and execute real fresh raw runs")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--run-root", required=True, help="new absolute shared-storage namespace on both remote hosts")
    run.add_argument("--report-dir", type=Path, required=True, help="new driver-local report directory")
    run.add_argument("--baseline-source", required=True)
    run.add_argument("--candidate-source")
    run.add_argument("--versions", nargs="+", choices=("baseline", "candidate"), default=["baseline", "candidate"])
    run.add_argument("--candidate-ready", action="store_true", help="explicit coordinator gate after candidate implementation and review")
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--cases", nargs="+", help="formal selection, at least 10 different subjects")
    selection.add_argument("--pilot", nargs="+", help="explicit diagnostic subset; not a ten-case formal result")
    run.add_argument("--cpu-host", default="nodecw10")
    run.add_argument("--gpu-host", default="gpucw1")
    for kind in ("cpu", "gpu"):
        run.add_argument(f"--{kind}-control-path")
        run.add_argument(f"--{kind}-port", type=int)
        run.add_argument(f"--{kind}-python", required=True)
    run.add_argument("--anatomy-validation-python", help="CPU-host Python containing nibabel/numpy; default --gpu-python on shared storage")
    run.add_argument("--worker-script", required=True, help="this script uploaded to shared remote storage")
    run.add_argument("--wall-script", required=True, help="the fixed raw_bids wall script on shared remote storage")
    run.add_argument("--freesurfer-home", required=True)
    run.add_argument("--recon-all", required=True, help="absolute official executable, not FNIT recon-all")
    run.add_argument("--fs-license", help="readable official license path; never read into logs, hashes, or reports")
    run.add_argument("--cpu-jobs", type=int, default=2)
    run.add_argument("--cpu-threads", type=int, default=8)
    run.add_argument("--gpu-cpu-threads", type=int, default=8)
    run.add_argument("--gpu-lock", default=GPU_LOCK)
    run.add_argument("--gpu-uuid")
    run.add_argument("--cuda-visible-devices")
    run.add_argument("--gpu-path-prefix", action="append", default=[], help="absolute executable directory prepended on GPU worker PATH; repeat if needed")
    run.add_argument("--fnit-weights", help="explicit local weight directory; supplied as FNIT_WEIGHTS to raw-DWI worker")
    run.add_argument("--cuda-alloc-conf", choices=("expandable_segments:True",), help="explicit CUDA allocator setting; keep identical across benchmark versions")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--n-seeds", type=int, default=100000)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--eddy-gp-seed", type=int, default=12345)
    run.add_argument("--atlas", nargs="+", choices=ATLAS_NAMES, default=["fs-aparc"])
    run.add_argument("--atlas-config", type=Path, help="JSON array of resource-option/value pairs; no bypass/precision/overwrite switches")
    args = parser.parse_args(argv)
    if args.command == "status":
        return args
    if len(set(args.versions)) != len(args.versions):
        parser.error("versions must be distinct")
    if "candidate" in args.versions and (not args.candidate_source or not args.candidate_ready):
        parser.error("candidate requires --candidate-source and the explicit --candidate-ready coordinator gate")
    if args.cpu_jobs not in (1, 2, 3, 4) or min(args.cpu_threads, args.gpu_cpu_threads) < 1:
        parser.error("CPU jobs must be 1..4 (default 2) with positive thread budgets")
    if args.n_seeds < 1 or (args.pilot is None and args.n_seeds < 100000):
        parser.error("formal runs require at least 100000 seeds; smaller counts are explicit --pilot diagnostics")
    if not 1 <= args.eddy_gp_seed < 2**32:
        parser.error("EDDY GP seed must be 1..2**32-1")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", args.device):
        parser.error("this cohort benchmark requires a declared CUDA device")
    if len(set(args.atlas)) != len(args.atlas):
        parser.error("atlas names must be distinct")
    return args


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_anatomy":
        print(json.dumps(anatomy_geometry(json.load(sys.stdin)), allow_nan=False), flush=True)
        return 0
    if argv and argv[0] == "_worker":
        report = worker(json.loads(argv[1]) if len(argv) > 1 else json.load(sys.stdin))
        print(json.dumps(report, allow_nan=False), flush=True)
        return 0
    args = parse_options(argv)
    if args.command == "status":
        state = json.loads((args.report_dir / "status.json").read_text())
        print(json.dumps({"saved_state": state, "query_scope": "saved driver artifacts only; remote jobs are not refreshed or restarted"}, indent=2))
        return 0
    manifest = json.loads(args.manifest.read_text())
    cases = validate_manifest(manifest, args.pilot or args.cases, pilot=args.pilot is not None)
    atlas_options = json.loads(args.atlas_config.read_text()) if args.atlas_config else []
    if not isinstance(atlas_options, list) or len(atlas_options) % 2:
        raise ValueError("atlas config must be option/value pairs in a JSON array")
    for i in range(0, len(atlas_options), 2):
        if atlas_options[i] not in RESOURCE_OPTIONS:
            raise ValueError(f"not an atlas resource option: {atlas_options[i]}")
        absolute_path(atlas_options[i+1], "atlas resource path")
    if "--mni-template" in atlas_options and "--tian-fnirt-coeff" in atlas_options:
        raise ValueError("MNI template and FNIRT coefficient are alternative atlas resources")
    config = vars(args).copy()
    for key in ("manifest", "report_dir", "atlas_config", "cases", "command"):
        config.pop(key, None)
    config["pilot"] = args.pilot is not None
    config["anatomy_validation_python"] = args.anatomy_validation_python or args.gpu_python
    config["sources"] = {version: getattr(args, version + "_source") for version in args.versions}
    config["atlases"] = config.pop("atlas")
    config["atlas_options"] = atlas_options
    for key in ("run_root", "worker_script", "wall_script", "freesurfer_home", "recon_all", "cpu_python", "gpu_python", "anatomy_validation_python", "gpu_lock"):
        absolute_path(config[key], key)
    for source in config["sources"].values():
        absolute_path(source, "source directory")
    if config.get("fs_license"):
        absolute_path(config["fs_license"], "FreeSurfer license path")
    if config.get("fnit_weights"):
        absolute_path(config["fnit_weights"], "FNIT weight directory")
    if not Path(config["recon_all"]).is_relative_to(Path(config["freesurfer_home"])):
        raise ValueError("official recon-all must be inside the declared FreeSurfer installation")
    config["worker_script_sha256"] = sha256(Path(__file__))
    config["manifest_sha256"] = sha256(args.manifest)
    for path in config["gpu_path_prefix"]:
        absolute_path(path, "GPU executable directory")
    return run_cohort(config, manifest, cases, args.report_dir)


if __name__ == "__main__":
    raise SystemExit(main())
