"""Raw BIDS DWI/T1 preparation for the connectome pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

from ..dmri_pipeline.bids import locate_bids_dwi, stage_bids_dwi
from ..dmri_pipeline.pipeline import _prepare_ap_only
from ..eddy import TorchEDDY
from ..eddy.ukb import prepare_ukb_eddy
from ..topup.ukb import run_ukb_topup
from .freesurfer_subject import FreeSurferSubject


@dataclass(frozen=True)
class BIDSConnectomeInputs:
    dwi: Path
    bvals: Path
    bvecs: Path
    freesurfer_subject_dir: Path
    stages: dict[str, str]


def _fingerprint(paths: tuple[Path, ...], options: dict) -> dict:
    return {
        "inputs": {str(path.resolve()): [path.stat().st_size, path.stat().st_mtime_ns]
                   for path in paths},
        "options": options,
    }


def _reusable(state: Path, fingerprint: dict, outputs: tuple[Path, ...]) -> bool:
    if not state.is_file() or not all(path.is_file() and path.stat().st_size
                                      for path in outputs):
        return False
    return json.loads(state.read_text()) == fingerprint


def _record(state: Path, fingerprint: dict) -> None:
    temporary = state.with_suffix(".tmp")
    temporary.write_text(json.dumps(fingerprint, indent=2) + "\n")
    temporary.replace(state)


def _recon_complete(directory: Path) -> bool:
    files = ("mri/brain.mgz", "mri/aparc+aseg.mgz", "mri/ribbon.mgz",
             "surf/lh.white", "surf/rh.white", "surf/lh.pial", "surf/rh.pial")
    return all((directory / name).is_file() and (directory / name).stat().st_size
               for name in files)


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
    corrected_dwi: str | Path | None = None,
    rotated_bvecs: str | Path | None = None,
    device: str = "cuda:0",
    overwrite: bool = False,
) -> BIDSConnectomeInputs:
    """Select one BIDS DWI/T1 and run only missing TOPUP, EDDY and recon-all stages.

    A reversed phase-encoding image enables TOPUP; without one, EDDY runs
    without a field map. Reuse of FNIT intermediates requires matching input
    paths, sizes, modification times, and options. An external corrected DWI
    must be supplied together with its rotated bvec file.
    """
    if t1 is not None and freesurfer_subject_dir is not None:
        raise ValueError("t1 and freesurfer_subject_dir are alternative anatomy inputs")
    selected = locate_bids_dwi(
        bids_root, subject=subject, session=session, run=run,
        acquisition=acquisition, direction=direction, t1=t1,
        select_t1=freesurfer_subject_dir is None,
    )
    if (corrected_dwi is None) != (rotated_bvecs is None):
        raise ValueError("corrected_dwi and rotated_bvecs must be supplied together")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stages = {}

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
        if not _reusable(stage_state, stage_key, staged) or overwrite:
            stage_bids_dwi(selected, raw, overwrite=True)
            _record(stage_state, stage_key)

        topup_dir = preproc / "topup"
        if selected.reverse is not None:
            topup_state = topup_dir / "state.json"
            topup_key = _fingerprint((raw / "AP.nii.gz", raw / "AP.bval",
                                      raw / "AP.json", raw / "PA.nii.gz",
                                      raw / "PA.bval", raw / "PA.json"), {})
            topup_outputs = (topup_dir / "fieldmap_out_fieldcoef.nii.gz",
                             topup_dir / "fieldmap_iout.nii.gz",
                             topup_dir / "acqparams.txt")
            if not overwrite and _reusable(topup_state, topup_key, topup_outputs):
                stages["topup"] = "skipped"
            else:
                run_ukb_topup(raw, topup_dir, device=device, overwrite=True)
                _record(topup_state, topup_key)
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
            eddy_sources += (topup_dir / "fieldmap_out_fieldcoef.nii.gz",)
        eddy_key = _fingerprint(eddy_sources, {"topup": selected.reverse is not None})
        if not overwrite and _reusable(eddy_state, eddy_key, (dwi, bvecs)):
            stages["eddy"] = "skipped"
        else:
            if selected.reverse is not None:
                eddy_inputs = prepare_ukb_eddy(
                    raw, topup_dir, eddy_dir, device=device, overwrite=True)
            else:
                eddy_inputs = _prepare_ap_only(raw, eddy_dir, overwrite=True)
            TorchEDDY(device=device).run(
                **eddy_inputs, out=eddy_dir / "data", overwrite=True)
            _record(eddy_state, eddy_key)
            stages["eddy"] = "completed"

    if freesurfer_subject_dir is not None:
        fs_subject = Path(freesurfer_subject_dir).expanduser().resolve()
        if not _recon_complete(fs_subject):
            raise ValueError(f"incomplete FreeSurfer subject directory: {fs_subject}")
        stages["recon_all"] = "supplied"
    else:
        if selected.t1w is None:
            raise FileNotFoundError("BIDS T1w missing; supply --t1 or --freesurfer-subject-dir")
        name = f"sub-{subject.removeprefix('sub-')}"
        session_label = session or next((part[4:] for part in
            selected.image.relative_to(selected.root).parts if part.startswith("ses-")), None)
        if session_label is not None:
            name += f"_ses-{session_label.removeprefix('ses-')}"
        subjects_dir = root / "freesurfer"
        fs_subject = subjects_dir / name
        if _recon_complete(fs_subject) and (fs_subject / "scripts/recon-all.done").is_file() and not overwrite:
            stages["recon_all"] = "skipped"
        else:
            executable = shutil.which("recon-all")
            if executable is None:
                raise RuntimeError("official FreeSurfer recon-all is required for an unprocessed T1w")
            command = [executable, "-sd", str(subjects_dir), "-s", name]
            if not (fs_subject / "mri/orig/001.mgz").is_file():
                command += ["-i", str(selected.t1w)]
            subprocess.run([*command, "-all"], check=True)
            if not _recon_complete(fs_subject):
                raise RuntimeError(f"recon-all did not produce complete anatomy: {fs_subject}")
            stages["recon_all"] = "completed"
    FreeSurferSubject(fs_subject)
    return BIDSConnectomeInputs(dwi, selected.bval, bvecs, fs_subject, stages)
