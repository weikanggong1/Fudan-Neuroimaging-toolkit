"""Explicit, content-addressed anatomy sources for the connectome pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Mapping

import nibabel as nib
import numpy as np


CORE_FILES = (
    "mri/brain.mgz", "mri/aparc+aseg.mgz", "mri/ribbon.mgz",
    "surf/lh.white", "surf/rh.white", "surf/lh.pial", "surf/rh.pial",
)
FNIT_OPTIONS = {
    "weights_dir", "assets_dir", "native_bin_dir", "threads", "profile_stages",
    "cuda_allocator_cache", "hemisphere_workers", "native_optimizations",
}
FREESURFER_OPTIONS = {"executable", "freesurfer_home", "threads"}
# Programs consumed by the existing, complete FNIT recon-all entry point.
NATIVE_PROGRAMS = (
    "fnit_n4_itk", "mri_em_register", "mris_fix_topology_fnit",
    "mris_place_surface", "mris_place_surface_white_fast", "mris_inflate",
    "mris_remove_intersection", "mrisp_paint", "mris_curvature_stats",
    "mri_label2vol", "mri_warp_convert", "mri_ca_register", "mri_convert",
    "mri_segment", "mri_edit_wm_with_aseg",
)


@dataclass(frozen=True)
class ReconSubjectResult:
    """Completed subject, stage decision, and serialisable source provenance."""

    subject_dir: Path
    stage: str
    metadata: dict


def file_fingerprint(path: str | Path) -> dict:
    """Hash file contents in bounded buffers; modification time is not identity."""
    file = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(file), "size": file.stat().st_size, "sha256": digest.hexdigest()}


def _tree_fingerprint(directory: Path, *, source: bool = False) -> dict:
    records = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or "license" in path.name.lower():
            continue
        if source and path.suffix not in {".py", ".hpp", ".cpp", ".cu", ".cuh"}:
            continue
        records[str(path.relative_to(directory))] = file_fingerprint(path)
    return records


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def resolve_recon_backend(recon_backend: str, provided: str | Path | None) -> str:
    """Resolve auto without falling back from FNIT to official FreeSurfer."""
    if recon_backend not in {"auto", "provided", "freesurfer", "fnit"}:
        raise ValueError("recon_backend must be auto, provided, freesurfer, or fnit")
    backend = ("provided" if provided is not None else "fnit") if recon_backend == "auto" else recon_backend
    if backend == "provided" and provided is None:
        raise ValueError("recon_backend='provided' requires freesurfer_subject_dir")
    if backend != "provided" and provided is not None:
        raise ValueError("freesurfer_subject_dir requires recon_backend='auto' or 'provided'")
    return backend


def _options(backend: str, options: Mapping | None) -> dict:
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("recon_options must be a mapping")
    values = dict(options or {})
    allowed = FNIT_OPTIONS if backend == "fnit" else FREESURFER_OPTIONS if backend == "freesurfer" else set()
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unsupported {backend} recon_options: {sorted(unknown)}")
    if backend == "provided":
        return values
    threads = values.setdefault("threads", 4)
    if isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 1:
        raise ValueError("recon_options['threads'] must be a positive integer")
    values["threads"] = int(threads)
    if backend == "fnit":
        for name in ("weights_dir", "assets_dir"):
            if values.get(name) is None:
                raise ValueError(f"FNIT recon-all requires recon_options['{name}']; use the FNIT asset installers")
            directory = Path(values[name]).expanduser().resolve()
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            values[name] = str(directory)
        if values.get("native_bin_dir") is not None:
            values["native_bin_dir"] = str(Path(values["native_bin_dir"]).expanduser().resolve())
        for name in ("profile_stages",):
            if name in values and not isinstance(values[name], bool):
                raise ValueError(f"recon_options['{name}'] must be bool")
        for name, allowed_values in (("cuda_allocator_cache", {"auto", "enabled", "disabled"}),
                                    ("native_optimizations", {"auto", "original"})):
            if name in values and values[name] not in allowed_values:
                raise ValueError(f"invalid recon_options['{name}']")
        workers = values.get("hemisphere_workers", 1)
        if isinstance(workers, bool) or not isinstance(workers, Integral) or workers not in (1, 2) or workers > threads:
            raise ValueError("hemisphere_workers must be 1 or 2 and no greater than threads")
    else:
        executable = values.get("executable") or shutil.which("recon-all")
        if executable is not None and Path(executable).name == str(executable):
            executable = shutil.which(str(executable))
        if executable is None:
            raise FileNotFoundError("official recon-all not found; set recon_options['executable']")
        executable = Path(executable).expanduser().resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"official recon-all is not executable: {executable}")
        values["executable"] = str(executable)
        declared_home = values.get("freesurfer_home") or os.environ.get("FREESURFER_HOME")
        if declared_home is None:
            raise ValueError("official recon-all requires recon_options['freesurfer_home'] or FREESURFER_HOME")
        home = Path(declared_home).expanduser().resolve()
        if not home.is_dir():
            raise FileNotFoundError(home)
        for name in ("SetUpFreeSurfer.sh", "FreeSurferEnv.sh"):
            if not (home / name).is_file():
                raise FileNotFoundError(f"official FreeSurfer environment script missing: {home / name}")
        values["freesurfer_home"] = str(home)
    return values


def validate_recon_configuration(
    recon_backend: str = "auto", *, freesurfer_subject_dir: str | Path | None = None,
    recon_options: Mapping | None = None,
) -> tuple[str, dict]:
    """Validate source selection and options before starting expensive DWI steps."""
    backend = resolve_recon_backend(recon_backend, freesurfer_subject_dir)
    return backend, _options(backend, recon_options)


def inspect_recon_subject(subject_dir: str | Path, *, full_fnit: bool = False,
                          require_official_done: bool = False) -> dict:
    """Read-only format/geometry checks and hashes of anatomy files actually present.

    Provided subjects need the seven core files. Surface templates additionally
    require matching annotation/sphere files, checked by their atlas builder.
    FNIT-generated subjects require all 138 outputs and the complete public report.
    This checks usable data and completion, rather than official numeric equality.
    """
    subject = Path(subject_dir).expanduser().resolve()
    if not subject.is_dir():
        raise FileNotFoundError(subject)
    required = CORE_FILES
    if full_fnit:
        from ..recon_all.expected_outputs import paths
        required = paths()
    if require_official_done:
        required += ("scripts/recon-all.done",)
    missing = [name for name in required if not (subject / name).is_file() or not (subject / name).stat().st_size]
    if missing:
        raise ValueError(f"incomplete recon-all subject directory {subject}: {missing}")
    images = {name: nib.load(subject / name) for name in CORE_FILES[:3]}
    brain = images[CORE_FILES[0]]
    if len(brain.shape) != 3 or not np.isfinite(brain.affine).all() or abs(np.linalg.det(brain.affine[:3, :3])) < 1e-10:
        raise ValueError("recon-all brain must be a 3D image with a finite invertible affine")
    for name, image in images.items():
        if image.shape != brain.shape or not np.allclose(image.affine, brain.affine, rtol=0, atol=1e-5):
            raise ValueError(f"recon-all volume geometry mismatch: {name}")
        values = np.asarray(image.dataobj)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite recon-all volume: {name}")
        if name != "mri/brain.mgz" and ((values < 0).any() or not np.equal(values, np.round(values)).all()):
            raise ValueError(f"recon-all segmentation must contain nonnegative integer labels: {name}")
    hemispheres = {}
    for hemi in ("lh", "rh"):
        reference_faces = reference_vertices = None
        hemispheres[hemi] = {}
        for surface in ("white", "pial"):
            vertices, faces = nib.freesurfer.read_geometry(str(subject / f"surf/{hemi}.{surface}"))
            if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3
                    or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
                    or not len(faces) or (faces < 0).any() or (faces >= len(vertices)).any()):
                raise ValueError(f"invalid recon-all surface: {hemi}.{surface}")
            if reference_faces is not None and (vertices.shape != reference_vertices or not np.array_equal(faces, reference_faces)):
                raise ValueError(f"recon-all white/pial vertex or face correspondence mismatch: {hemi}")
            reference_faces, reference_vertices = faces, vertices.shape
            hemispheres[hemi][surface] = {"vertices": len(vertices), "faces": len(faces)}
    report = None
    report_path = subject / "fnit-native-free-run.json"
    if full_fnit:
        if not report_path.is_file():
            raise ValueError("FNIT recon-all is missing its completion report")
        report = json.loads(report_path.read_text())
        if (report.get("status") != "complete"
                or report.get("output_validation", {}).get("status") != "passed"
                or report.get("mesh_validation", {}).get("status") != "passed"
                or report.get("output_validation", {}).get("expected") != len(required)
                or report.get("output_validation", {}).get("present") != len(required)
                or report.get("output_validation", {}).get("missing") != []
                or not report.get("stages")
                or any(stage.get("status", "complete") in {"failed", "incomplete", "running"} for stage in report["stages"])):
            raise ValueError("FNIT recon-all report is not complete with all stages, outputs and meshes passed")
    # Include annotations, surfaces and volumes so an edited supplied atlas or
    # structure invalidates downstream anatomy even when size/mtime is unchanged.
    records = {}
    for directory_name in ("mri", "surf", "label"):
        directory = subject / directory_name
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    records[str(path.relative_to(subject))] = file_fingerprint(path)
    for name in required:
        if name not in records:
            records[name] = file_fingerprint(subject / name)
    if report_path.is_file():
        records[report_path.name] = file_fingerprint(report_path)
    return {"subject_dir": str(subject), "shape": [int(value) for value in brain.shape],
            "affine": brain.affine.tolist(), "surface_geometry": hemispheres,
            "files": records, "content_sha256": _digest(records),
            "completion_scope": "FNIT complete 138 outputs and mesh report" if full_fnit else "readable core anatomy and compatible geometry",
            "numeric_equivalence": "not_assessed",
            "reconstruction_report": {name: report.get(name) for name in ("status", "total_seconds", "numeric_validation")} if report else None}


def _execution_identity(backend: str, t1: Path, options: dict, device: str) -> dict:
    identity = {"schema_version": 1, "backend": backend, "input": file_fingerprint(t1), "options": options}
    if backend == "fnit":
        identity["device"] = device
        identity["source"] = _tree_fingerprint(Path(__file__).resolve().parents[1], source=True)
        identity["resources"] = {name: _tree_fingerprint(Path(options[name])) for name in ("weights_dir", "assets_dir")}
        binaries = Path(options.get("native_bin_dir") or os.environ.get("FNIT_RECON_ALL_BIN_DIR", Path(sys.prefix) / "bin")).resolve()
        identity["native_programs"] = {name: file_fingerprint(binaries / name) for name in NATIVE_PROGRAMS if (binaries / name).is_file()}
    else:
        identity["program"] = file_fingerprint(options["executable"])
        home = options.get("freesurfer_home") or os.environ.get("FREESURFER_HOME")
        if home:
            stamp = Path(home) / "build-stamp.txt"
            if stamp.is_file():
                identity["build_stamp"] = file_fingerprint(stamp)
            identity["setup_scripts"] = {name: file_fingerprint(Path(home) / name)
                                         for name in ("SetUpFreeSurfer.sh", "FreeSurferEnv.sh")}
    return identity


def prepare_recon_subject(
    t1: str | Path | None, output_dir: str | Path, *, subject_name: str,
    freesurfer_subject_dir: str | Path | None = None, recon_backend: str = "auto",
    recon_options: Mapping | None = None, device: str = "cuda:0", overwrite: bool = False,
) -> ReconSubjectResult:
    """Produce/read a complete subject and reuse only unchanged successful results.

    auto uses the supplied subject, otherwise FNIT. Official recon-all is invoked
    only with explicit freesurfer selection. Failed/changed results are retained;
    a new attempt uses a fresh subject directory. Source subjects are never edited.
    FNIT execution uses the existing one-job batch API, isolating CUDA state in a
    fresh process; all public recon parameters retain their existing meaning.
    """
    backend, options = validate_recon_configuration(recon_backend, freesurfer_subject_dir=freesurfer_subject_dir, recon_options=recon_options)
    if backend == "provided":
        anatomy = inspect_recon_subject(freesurfer_subject_dir)
        return ReconSubjectResult(Path(anatomy["subject_dir"]), "supplied", {"resolved_backend": backend, "anatomy": anatomy})
    if not isinstance(subject_name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", subject_name) is None:
        raise ValueError("subject_name must contain only letters, digits, _, ., -")
    if t1 is None:
        raise FileNotFoundError("T1w missing; supply t1 or a completed recon-all subject directory")
    image = Path(t1).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    identity = _execution_identity(backend, image, options, device)
    fingerprint = _digest(identity)
    root = Path(output_dir).expanduser().resolve()
    state_path = root / "anatomy" / "state" / f"{backend}-{subject_name}-{fingerprint}.json"
    if not overwrite and state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
            if state.get("status") == "complete" and state.get("identity") == identity:
                anatomy = inspect_recon_subject(state["subject_dir"], full_fnit=backend == "fnit", require_official_done=backend == "freesurfer")
                if anatomy["content_sha256"] == state.get("anatomy", {}).get("content_sha256"):
                    return ReconSubjectResult(Path(state["subject_dir"]), "skipped", dict(state, resolved_backend=backend))
        except (OSError, ValueError, KeyError):
            # An unreadable/tampered result never passes as a completed cache.
            pass
    subjects_dir = root / "anatomy" / backend
    base = f"{subject_name}-{fingerprint[:12]}"
    subject = subjects_dir / base
    attempt = 1
    while subject.exists():
        attempt += 1
        subject = subjects_dir / f"{base}-attempt-{attempt}"
    subjects_dir.mkdir(parents=True, exist_ok=True)
    command = None
    if backend == "fnit":
        from ..recon_all.batch import run_recon_all_python_batch
        kwargs = {key: value for key, value in options.items() if key not in {"weights_dir", "assets_dir"}}
        run_recon_all_python_batch(
            jobs=[{"t1": image, "subject_dir": subject}],
            weights_dir=options["weights_dir"], assets_dir=options["assets_dir"],
            devices=(device,), **kwargs)
    else:
        command = [options["executable"], "-sd", str(subjects_dir), "-s", subject.name,
                   "-i", str(image), "-all", "-parallel", "-openmp", str(options["threads"])]
        environment = dict(os.environ, SUBJECTS_DIR=str(subjects_dir),
                           OMP_NUM_THREADS=str(options["threads"]),
                           ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(options["threads"]))
        home = options.get("freesurfer_home") or environment.get("FREESURFER_HOME")
        if home:
            environment["FREESURFER_HOME"] = str(home)
            environment["PATH"] = str(Path(home) / "bin") + os.pathsep + environment.get("PATH", "")
        # The official 8.x script consumes variables established by its full
        # setup (including FREESURFER), not only FREESURFER_HOME and PATH.
        # Keep the shell program constant: every MRI/resource path remains a
        # separate argv value, and inherited credentials are never printed.
        launcher = ["/bin/bash", "-c",
                    'set -e; source "$FREESURFER_HOME/SetUpFreeSurfer.sh" > /dev/null; '
                    'export PATH="$FREESURFER_HOME/bin:$PATH"; exec "$@"',
                    "fnit-freesurfer", *command]
        subprocess.run(launcher, check=True, env=environment)
        done = subject / "scripts/recon-all.done"
        if not done.is_file() or not done.stat().st_size:
            raise RuntimeError("official recon-all did not produce its completion marker")
    if _execution_identity(backend, image, options, device) != identity:
        raise RuntimeError("recon-all source, input, program or resource contents changed during execution; completion cache not published")
    anatomy = inspect_recon_subject(subject, full_fnit=backend == "fnit", require_official_done=backend == "freesurfer")
    state = {"status": "complete", "resolved_backend": backend, "identity": identity,
             "input_sha256": identity["input"]["sha256"], "execution_sha256": fingerprint,
             "subject_dir": str(subject), "anatomy": anatomy, "command": command}
    # Completion is published only after the public API and output checks succeed.
    _atomic_record(state_path, state)
    return ReconSubjectResult(subject, "completed", state)
