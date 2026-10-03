"""Automatic prerequisite control-flow contracts, not processing benchmarks."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from fnit.fmri import cli, end_to_end, surface_pipeline as pipeline
from fnit.fmri.surface_volume import SurfaceVolumeStatus
from test_fmri_surface_public_contracts import public_case


def _spy_volume(monkeypatch, call):
    signature = inspect.signature(end_to_end.fMRIVolume_pipeline)
    call.__signature__ = signature
    monkeypatch.setattr(end_to_end, "fMRIVolume_pipeline", call)


def _status(case, state, reasons=()):
    return SurfaceVolumeStatus(state, case.source_t1, reasons, ())


def test_verified_volume_reused_without_computation(public_case, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("existing complete volume must not run again")
    _spy_volume(monkeypatch, unexpected)
    result = pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert result.volume_executed is False
    assert result.recon_all == public_case.recon


def test_missing_volume_runs_once_then_rechecks_and_preserves_run(public_case, monkeypatch):
    statuses = iter((_status(public_case, "missing"), _status(public_case, "ready")))
    monkeypatch.setattr(pipeline, "inspect_surface_volume", lambda *args, **kwargs: next(statuses))
    calls = []
    def compute(*args, **kwargs):
        calls.append((args, kwargs))
    _spy_volume(monkeypatch, compute)
    result = pipeline.fMRISurface_pipeline(**public_case.arguments, volume_options={
        "mni_template": "explicit-template.nii.gz", "slice_timing": True,
        "registration_backend": "fnirt", "reuse_anatomical": False})
    assert len(calls) == 1
    args, options = calls[0]
    assert args == (public_case.raw, public_case.roots)
    assert options["subject"] == "example" and options["task"] == "rest"
    assert options["t1w_image"] == public_case.source_t1
    assert options["device"] == "cpu" and options["slice_timing"] is True
    assert options["registration_backend"] == "fnirt"
    assert result.volume_executed is True


@pytest.mark.parametrize("state,error", [("partial", FileNotFoundError), ("invalid", ValueError)])
def test_bad_volume_never_computed_or_reconstructed(public_case, monkeypatch, state, error):
    monkeypatch.setattr(pipeline, "inspect_surface_volume", lambda *args, **kwargs:
                        _status(public_case, state, ("source or required input invalid",)))
    def unexpected(*args, **kwargs):
        pytest.fail("invalid/partial volume cannot reach a compute stage")
    _spy_volume(monkeypatch, unexpected)
    monkeypatch.setattr(pipeline, "prepare_surface_reconstruction", unexpected)
    with pytest.raises(error, match=state):
        pipeline.fMRISurface_pipeline(**public_case.arguments)


def test_volume_failure_stops_before_reconstruction(public_case, monkeypatch):
    monkeypatch.setattr(pipeline, "inspect_surface_volume", lambda *args, **kwargs:
                        _status(public_case, "missing"))
    def compute(*args, **kwargs):
        raise RuntimeError("real volume failed")
    _spy_volume(monkeypatch, compute)
    monkeypatch.setattr(pipeline, "prepare_surface_reconstruction", lambda *args, **kwargs: pytest.fail("reconstruction ran"))
    with pytest.raises(RuntimeError, match="real volume failed"):
        pipeline.fMRISurface_pipeline(**public_case.arguments,
                                      volume_options={"mni_template": "template.nii.gz"})


def test_require_volume_disables_automatic_compute(public_case, monkeypatch):
    monkeypatch.setattr(pipeline, "inspect_surface_volume", lambda *args, **kwargs:
                        _status(public_case, "missing"))
    with pytest.raises(FileNotFoundError, match="auto_volume=False"):
        pipeline.fMRISurface_pipeline(**public_case.arguments, auto_volume=False)


@pytest.mark.parametrize("key", ["subject", "t1w_image", "device", "bids_root", "unknown"])
def test_volume_options_cannot_change_run_identity(public_case, key):
    with pytest.raises(ValueError, match="volume_options"):
        pipeline.fMRISurface_pipeline(**public_case.arguments, volume_options={key: "other"})


@pytest.mark.parametrize("backend", ["fnit", "freesurfer", "provided"])
def test_surface_routes_three_backends_and_persists_adapter_output(public_case, monkeypatch, backend):
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(subject_dir=public_case.recon, backend=backend,
                               reused=backend == "provided", metadata={"backend": backend})
    monkeypatch.setattr(pipeline, "prepare_surface_reconstruction", prepare)
    arguments = dict(public_case.arguments, recon_all_backend=backend)
    if backend != "provided":
        arguments.pop("recon_all")
    pipeline.fMRISurface_pipeline(**arguments)
    assert calls[0][0][0] == public_case.source_t1
    assert calls[0][1]["backend"] == backend
    assert calls[0][1]["output_dir"].parent == public_case.paths.anat_dir
    assert f"desc-{backend}" in calls[0][1]["output_dir"].name


def test_cli_accepts_raw_fnit_without_recon_input(monkeypatch):
    captured = {}
    def surface(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(dtseries=Path("result.dtseries.nii"))
    monkeypatch.setattr(cli, "fMRISurface_pipeline", surface)
    assert cli.main(["surface", "--bids-root", "raw", "--derivatives-root", "out",
                     "--subject", "CON01", "--surface-assets-dir", "assets",
                     "--recon-all-backend", "fnit", "--recon-all-weights-dir", "weights",
                     "--mni-template", "template.nii.gz"]) == 0
    assert captured["recon_all"] is None
    assert captured["recon_all_backend"] == "fnit" and captured["auto_volume"] is True
    assert captured["volume_options"]["mni_template"] == "template.nii.gz"
    assert captured["recon_all_options"]["weights_dir"] == "weights"
