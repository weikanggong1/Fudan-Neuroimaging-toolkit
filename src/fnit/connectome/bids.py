"""Raw BIDS DWI/T1 preparation for the connectome pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import tempfile
import uuid
from numbers import Integral
from pathlib import Path
from typing import Mapping

from ..dmri_pipeline.bids import locate_bids_dwi, stage_bids_dwi
from ..dmri_pipeline.pipeline import _prepare_ap_only
from ..eddy import TorchEDDY
from ..eddy.ukb import prepare_ukb_eddy
from ..topup.ukb import run_ukb_topup
from .freesurfer_subject import FreeSurferSubject
from .recon_backend import (file_fingerprint, inspect_recon_subject,
                            prepare_recon_subject, validate_recon_configuration)
from .input_protection import recon_resource_paths, validate_bids_preparation_inputs


@dataclass(frozen=True)
class BIDSConnectomeInputs:
    dwi: Path
    bvals: Path
    bvecs: Path
    freesurfer_subject_dir: Path
    stages: dict[str, str]
    recon_metadata: dict | None = None


def _fingerprint(paths: tuple[Path, ...], options: dict) -> dict:
    return {
        "inputs": {str(path.resolve()): file_fingerprint(path) for path in paths},
        "options": options,
    }


def _output_fingerprints(outputs: tuple[Path, ...]) -> dict:
    """Read complete stage products, rejecting changes during the read."""
    if not outputs:
        raise ValueError("completed stage must have at least one output")
    records = {}
    for output in outputs:
        path = Path(output).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        before = path.stat()
        if not before.st_size:
            raise ValueError(f"stage output is empty: {path}")
        record = file_fingerprint(path)
        # Shared filesystems can retain coarse timestamps even after a
        # same-size write and restored mtime. Confirm bytes, not just stat.
        confirmed = file_fingerprint(path)
        after = path.stat()
        fields = ("st_size", "st_mtime_ns", "st_ctime_ns", "st_ino")
        if (record != confirmed
                or any(getattr(before, field) != getattr(after, field) for field in fields)):
            raise RuntimeError(f"stage output changed while hashing: {path}")
        records[str(path)] = record
    return records


def _reusable(state: Path, fingerprint: dict, outputs: tuple[Path, ...], *,
              require_output_hashes: bool = False) -> bool:
    """Raw stages require completed, unchanged input and output identities.

    The legacy named-atlas CLI also uses this helper without the keyword; its
    old fingerprint-only state remains readable. Raw callers set the keyword
    so a prior marker without output content hashes is always a cache miss.
    """
    try:
        if not state.is_file() or not all(path.is_file() and path.stat().st_size
                                          for path in outputs):
            return False
        recorded = json.loads(state.read_text())
        if (isinstance(recorded, dict) and recorded.get("schema_version") == 2
                and set(recorded) == {"schema_version", "status", "fingerprint", "outputs"}):
            return (recorded["status"] == "completed"
                    and recorded["fingerprint"] == fingerprint
                    and recorded["outputs"] == _output_fingerprints(outputs))
        return not require_output_hashes and recorded == fingerprint
    except (OSError, ValueError, RuntimeError):
        return False


def _invalidate_stage(state: Path) -> None:
    """Retain the old marker outside the reusable name before any writes."""
    if state.exists():
        previous = state.with_name(f"{state.stem}.prior-{uuid.uuid4().hex}{state.suffix}")
        state.replace(previous)


def _record(state: Path, fingerprint: dict, outputs: tuple[Path, ...] | None = None) -> None:
    """Publish a completion marker only after outputs and inputs are verified.

    No outputs preserves the mature named-atlas CLI helper contract. Raw
    stage callers always provide their required products and use schema 2.
    """
    value = fingerprint
    if outputs is not None:
        actual = _output_fingerprints(outputs)
        if _fingerprint(tuple(Path(path) for path in fingerprint["inputs"]),
                        fingerprint["options"]) != fingerprint:
            raise RuntimeError("stage input changed during execution; completion not published")
        value = {"schema_version": 2, "status": "completed", "fingerprint": fingerprint,
                 "outputs": actual}
    descriptor, name = tempfile.mkstemp(prefix=f".{state.name}.", suffix=".tmp", dir=state.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(state)
    finally:
        temporary.unlink(missing_ok=True)


def _recon_complete(directory: Path) -> bool:
    """Read-only usable anatomy check; a filename alone is not completion."""
    try:
        inspect_recon_subject(directory)
        return True
    except (OSError, ValueError):
        return False


def prepare_bids_connectome(
    bids_root: str | Path,
    output_dir: str | Path,
    *,
    subject: str,
    session: str | None = None,
    run: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    t1: str | Path | None = None,
    freesurfer_subject_dir: str | Path | None = None,
    recon_backend: str = "auto",
    recon_options: Mapping | None = None,
    corrected_dwi: str | Path | None = None,
    rotated_bvecs: str | Path | None = None,
    device: str = "cuda:0",
    overwrite: bool = False,
    eddy_gp_seed: int | None = None,
    readonly_inputs: tuple[Path, ...] = (),
    planned_output_paths: tuple[Path, ...] = (),
) -> BIDSConnectomeInputs:
    """Select one BIDS DWI/T1 and run only missing TOPUP, EDDY and recon-all stages.

    A reversed phase-encoding image enables TOPUP; without one, EDDY runs
    without a field map. Reuse of FNIT intermediates requires matching
    input content hashes, options and complete output content hashes. Anatomy
    source is selected explicitly via
    recon_backend; auto consumes a provided subject, otherwise uses FNIT.
    recon_options passes resource paths and backend-specific execution options.
    An external corrected DWI must be supplied together with its rotated bvec file.
    readonly_inputs protects user templates/masks/resources; planned_output_paths
    protects selected raw/T1 inputs against subsequent caller-owned outputs.
    """
    if eddy_gp_seed is not None and (
        isinstance(eddy_gp_seed, bool) or not isinstance(eddy_gp_seed, Integral)
        or not 1 <= eddy_gp_seed <= 2**32 - 1
    ):
        raise ValueError("eddy_gp_seed must be None or an integer in [1, 2**32-1]")
    if eddy_gp_seed is not None:
        eddy_gp_seed = int(eddy_gp_seed)
    if t1 is not None and freesurfer_subject_dir is not None:
        raise ValueError("t1 and freesurfer_subject_dir are alternative anatomy inputs")
    resolved_backend, resolved_options = validate_recon_configuration(
        recon_backend, freesurfer_subject_dir=freesurfer_subject_dir,
        recon_options=recon_options)
    selected = locate_bids_dwi(
        bids_root, subject=subject, session=session, run=run,
        acquisition=acquisition, direction=direction, t1=t1,
        select_t1=freesurfer_subject_dir is None,
    )
    if (corrected_dwi is None) != (rotated_bvecs is None):
        raise ValueError("corrected_dwi and rotated_bvecs must be supplied together")
    root = Path(output_dir).expanduser().resolve()
    # Protect execution resources for Python callers as well as the CLI.
    resources = recon_resource_paths(resolved_options)
    validate_bids_preparation_inputs(
        selected, root, corrected_dwi=corrected_dwi, rotated_bvecs=rotated_bvecs,
        recon_backend=resolved_backend, readonly_inputs=(*readonly_inputs, *resources),
        planned_output_paths=planned_output_paths)
    root.mkdir(parents=True, exist_ok=True)
    stages = {}

    # Finish isolated FNIT reconstruction before DWI CUDA allocations exist.
    # Otherwise the parent EDDY allocator and child reconstruction can coexist.
    name = f"sub-{subject.removeprefix('sub-')}"
    session_label = session or next((part[4:] for part in
        selected.image.relative_to(selected.root).parts if part.startswith("ses-")), None)
    if session_label is not None:
        name += f"_ses-{session_label.removeprefix('ses-')}"
    reconstruction = prepare_recon_subject(
        selected.t1w, root, subject_name=name,
        freesurfer_subject_dir=freesurfer_subject_dir,
        recon_backend=recon_backend, recon_options=recon_options,
        device=device, overwrite=overwrite,
    )
    fs_subject = reconstruction.subject_dir
    stages["recon_all"] = reconstruction.stage


    if corrected_dwi is not None:
        dwi, bvecs = Path(corrected_dwi).resolve(), Path(rotated_bvecs).resolve()
        if not dwi.is_file() or not bvecs.is_file():
            raise FileNotFoundError(dwi if not dwi.is_file() else bvecs)
        stages.update(topup="supplied", eddy="supplied")
    else:
        preproc = root / "preproc"
        raw = preproc / "raw"
        sources = (selected.image, selected.bval, selected.bvec)
        if selected.reverse is not None:
            sources += (selected.reverse,)
        if selected.reverse_bval is not None:
            sources += (selected.reverse_bval,)
        stage_state = raw / "state.json"
        stage_key = _fingerprint(sources, {
            "forward_metadata": selected.metadata,
            "reverse_metadata": selected.reverse_metadata,
        })
        staged = (raw / "AP.nii.gz", raw / "AP.bval", raw / "AP.bvec", raw / "AP.json",
                  raw / "bids_selection.json")
        if selected.reverse is not None:
            staged += (raw / "PA.nii.gz", raw / "PA.bval", raw / "PA.json")
        if not _reusable(stage_state, stage_key, staged, require_output_hashes=True) or overwrite:
            _invalidate_stage(stage_state)
            stage_bids_dwi(selected, raw, overwrite=True)
            _record(stage_state, stage_key, staged)

        topup_dir = preproc / "topup"
        topup_prepared = None
        if selected.reverse is not None:
            topup_state = topup_dir / "state.json"
            topup_key = _fingerprint((raw / "AP.nii.gz", raw / "AP.bval",
                                      raw / "AP.json", raw / "PA.nii.gz",
                                      raw / "PA.bval", raw / "PA.json"),
                                     {"pair_geometry": "fslmerge-first"})
            topup_outputs = (topup_dir / "fieldmap_out_fieldcoef.nii.gz",
                             topup_dir / "fieldmap_out_movpar.txt",
                             topup_dir / "fieldmap_iout.nii.gz",
                             topup_dir / "acqparams.txt")
            if not overwrite and _reusable(topup_state, topup_key, topup_outputs, require_output_hashes=True):
                stages["topup"] = "skipped"
            else:
                _invalidate_stage(topup_state)
                _, topup_prepared = run_ukb_topup(
                    raw, topup_dir, device=device, overwrite=True,
                    pair_geometry="fslmerge-first")
                _record(topup_state, topup_key, topup_outputs)
                stages["topup"] = "completed"
        else:
            stages["topup"] = "no_reverse_pe"

        eddy_dir = preproc / "eddy"
        dwi = eddy_dir / "data.nii.gz"
        bvecs = eddy_dir / "data.eddy_rotated_bvecs"
        eddy_state = eddy_dir / "state.json"
        eddy_sources = (raw / "AP.nii.gz", raw / "AP.bval", raw / "AP.bvec",
                        raw / "AP.json")
        if selected.reverse is not None:
            # EDDY consumes coefficients/movement/acquisition and prepares
            # its brain mask from the corrected TOPUP b0 mean. Bind them all.
            eddy_sources += topup_outputs
        eddy_key = _fingerprint(eddy_sources, {"topup": selected.reverse is not None, "gp_seed": eddy_gp_seed})
        if not overwrite and _reusable(eddy_state, eddy_key, (dwi, bvecs), require_output_hashes=True):
            stages["eddy"] = "skipped"
        else:
            _invalidate_stage(eddy_state)
            if selected.reverse is not None:
                eddy_inputs = prepare_ukb_eddy(
                    raw, topup_dir, eddy_dir, device=device, overwrite=True,
                    ref_scan_no=(topup_prepared["ap_index"]
                                 if topup_prepared is not None else None))
            else:
                eddy_inputs = _prepare_ap_only(raw, eddy_dir, overwrite=True, device=device)
            TorchEDDY(device=device).run(
                **eddy_inputs, out=eddy_dir / "data", overwrite=True, gp_seed=eddy_gp_seed)
            _record(eddy_state, eddy_key, (dwi, bvecs))
            stages["eddy"] = "completed"

    FreeSurferSubject(fs_subject)
    return BIDSConnectomeInputs(dwi, selected.bval, bvecs, fs_subject, stages,
                                reconstruction.metadata)
