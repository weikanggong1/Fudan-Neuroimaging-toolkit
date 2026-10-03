"""User input preservation checks, with the numerical core never run on GPU."""
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fnit import cli
from fnit.connectome.input_protection import (validate_bids_preparation_inputs,
                                             validate_input_output_paths)
from fnit.connectome.pipeline import UKBConnectome_pipeline
from fnit.connectome.template_inputs import TemplatePair
from test_bids import _inputs, _subject
from test_paired_e2e import setup, pair_json


@pytest.mark.parametrize("namespace", ["raw", "eddy"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_cli_protects_template_before_raw_preparation(setup, tmp_path, namespace, overwrite):
    image, _ = _inputs(tmp_path / "bids")
    template = setup["output"] / "preproc" / namespace / "AP.nii.gz"
    template.parent.mkdir(parents=True)
    template.write_bytes(setup["volume_a"].volume_path.read_bytes())
    before = template.read_bytes()
    pair = TemplatePair("pair", replace(setup["volume_a"], volume_path=template), setup["volume_b"])
    path = pair_json(setup, [pair], tmp_path / "pairs.json")
    arguments = ["UKBConnectome_pipeline", "--bids-root", str(image.parents[2]),
                 "--subject", "01", "--freesurfer-subject-dir", str(setup["subject"]),
                 "--template-pairs", str(path), "--output-dir", str(setup["output"]),
                 "--device", "cpu", "--n-seeds", "8"]
    if overwrite:
        arguments.append("--overwrite")
    with pytest.raises(ValueError, match="preparation output namespace"):
        cli.main(arguments)
    assert template.read_bytes() == before
    assert not setup["calls"]
    assert not (setup["output"] / "preproc/raw/state.json").exists()


def test_python_bids_protects_template_before_preparation(setup, tmp_path):
    image, _ = _inputs(tmp_path / "bids")
    template = setup["output"] / "preproc/raw/AP.nii.gz"
    template.parent.mkdir(parents=True)
    template.write_bytes(setup["volume_a"].volume_path.read_bytes())
    before = template.read_bytes()
    pair = TemplatePair("pair", replace(setup["volume_a"], volume_path=template), setup["volume_b"])
    with pytest.raises(ValueError, match="preparation output namespace"):
        UKBConnectome_pipeline(device="cpu").run_bids(
            image.parents[2], setup["output"], subject="01", n_seeds=8,
            freesurfer_subject_dir=setup["subject"], template_pairs=[pair])
    assert template.read_bytes() == before and not setup["calls"]


def test_cli_keeps_user_template_symlink_path_in_active_namespace_readonly(setup, tmp_path):
    image, _ = _inputs(tmp_path / "bids")
    alias = setup["output"] / "preproc/raw/AP.nii.gz"
    alias.parent.mkdir(parents=True)
    source = setup["volume_a"].volume_path
    alias.symlink_to(source)
    pair = TemplatePair("pair", replace(setup["volume_a"], volume_path=alias), setup["volume_b"])
    path = pair_json(setup, [pair], tmp_path / "pairs.json")
    before = source.read_bytes()
    with pytest.raises(ValueError, match="preparation output namespace"):
        cli.main(["UKBConnectome_pipeline", "--bids-root", str(image.parents[2]),
                  "--subject", "01", "--freesurfer-subject-dir", str(setup["subject"]),
                  "--template-pairs", str(path), "--output-dir", str(setup["output"]),
                  "--device", "cpu", "--n-seeds", "8", "--overwrite"])
    assert alias.is_symlink() and source.read_bytes() == before and not setup["calls"]


def test_cli_explicit_t1_cannot_be_final_output_even_with_overwrite(setup, tmp_path):
    image, _ = _inputs(tmp_path / "bids")
    pair = TemplatePair("pair", setup["volume_a"], setup["volume_b"])
    config = pair_json(setup, [pair], tmp_path / "pairs.json")
    t1 = setup["output"] / "pairs/pair/first_atlas_dwi.nii.gz"
    t1.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.ones(setup["shape"], np.float32), np.eye(4)), t1)
    before = t1.read_bytes()
    with pytest.raises(ValueError, match="read-only input"):
        cli.main(["UKBConnectome_pipeline", "--bids-root", str(image.parents[2]),
                  "--subject", "01", "--t1", str(t1), "--recon-backend", "freesurfer",
                  "--template-pairs", str(config), "--output-dir", str(setup["output"]),
                  "--device", "cpu", "--n-seeds", "8", "--overwrite"])
    assert t1.read_bytes() == before and not setup["calls"]


def test_cli_corrected_input_from_existing_eddy_directory_remains_reusable(setup, tmp_path):
    image, _ = _inputs(tmp_path / "bids")
    folder = setup["output"] / "preproc/eddy"
    folder.mkdir(parents=True)
    corrected, rotated = folder / "data.nii.gz", folder / "data.eddy_rotated_bvecs"
    corrected.write_bytes(setup["dwi"].read_bytes())
    rotated.write_bytes(setup["bvec"].read_bytes())
    before = corrected.read_bytes(), rotated.read_bytes()
    config = pair_json(setup, [TemplatePair("pair", setup["volume_a"], setup["volume_b"])],
                       tmp_path / "pairs.json")
    arguments = ["UKBConnectome_pipeline", "--bids-root", str(image.parents[2]), "--subject", "01",
                 "--freesurfer-subject-dir", str(setup["subject"]), "--recon-backend", "provided",
                 "--corrected-dwi", str(corrected), "--rotated-bvecs", str(rotated),
                 "--template-pairs", str(config), "--output-dir", str(setup["output"]),
                 "--device", "cpu", "--n-seeds", "8", "--assignment-radius", ".1"]
    cli.main(arguments)
    cli.main(arguments)
    state = json.loads((setup["output"] / "pairs_run_state.json").read_text())
    assert state["preparation_stages"]["eddy"] == state["preparation_stages"]["topup"] == "supplied"
    assert state["cache_status"]["core"] == state["cache_status"]["template_pairs"]["pair"] == "skipped"
    assert setup["calls"]["core"] == 1
    assert before == (corrected.read_bytes(), rotated.read_bytes())


@pytest.mark.parametrize("namespace", ["raw", "topup", "eddy"])
def test_stage_namespace_directory_alias_cannot_write_into_input(tmp_path, namespace):
    inputs = tmp_path / "input"
    inputs.mkdir()
    atlas = inputs / "atlas.nii.gz"
    atlas.write_bytes(b"readonly")
    output = tmp_path / "output/preproc"
    output.mkdir(parents=True)
    (output / namespace).symlink_to(inputs, target_is_directory=True)
    raw = SimpleNamespace(image=tmp_path / "raw.nii.gz", bval=tmp_path / "raw.bval",
                          bvec=tmp_path / "raw.bvec", reverse=tmp_path / "reverse.nii.gz",
                          reverse_bval=None, t1w=None)
    with pytest.raises(ValueError, match="preparation output namespace"):
        validate_bids_preparation_inputs(raw, output.parent, readonly_inputs=(atlas,))
    assert atlas.read_bytes() == b"readonly"


def test_raw_source_alias_in_stage_directory_is_rejected(tmp_path):
    output = tmp_path / "output"
    source = output / "preproc/raw/AP.nii.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"readonly raw")
    alias = tmp_path / "sub-01_dwi.nii.gz"
    alias.symlink_to(source)
    raw = SimpleNamespace(image=alias, bval=tmp_path / "dwi.bval", bvec=tmp_path / "dwi.bvec",
                          reverse=None, reverse_bval=None, t1w=None)
    with pytest.raises(ValueError, match="preparation output namespace"):
        validate_bids_preparation_inputs(raw, output)
    assert alias.is_symlink() and source.read_bytes() == b"readonly raw"


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_second_raw_run_accepts_staged_aliases_and_keeps_sources(tmp_path, monkeypatch,
                                                               alias_kind, overwrite):
    import fnit.connectome.bids as module

    image, bvals = _inputs(tmp_path / "bids")
    bvecs = image.with_name("sub-01_dwi.bvec")
    subject = _subject(tmp_path / "freesurfer")
    originals = {path: path.read_bytes() for path in (image, bvals, bvecs)}
    calls = []
    monkeypatch.setattr(module, "_prepare_ap_only", lambda *args, **kwargs: {})

    class Eddy:
        def __init__(self, **kwargs):
            pass

        def run(self, *, out, **kwargs):
            calls.append(1)
            out.parent.mkdir(parents=True, exist_ok=True)
            (out.parent / "data.nii.gz").write_bytes(b"corrected")
            (out.parent / "data.eddy_rotated_bvecs").write_bytes(b"rotated")

    monkeypatch.setattr(module, "TorchEDDY", Eddy)
    options = dict(subject="01", freesurfer_subject_dir=subject, device="cpu")
    output = tmp_path / "output"
    module.prepare_bids_connectome(image.parents[2], output, **options)
    for source, name in ((image, "AP.nii.gz"), (bvals, "AP.bval"), (bvecs, "AP.bvec")):
        staged = output / "preproc/raw" / name
        assert staged.is_symlink()  # The production stager creates symlinks.
        if alias_kind == "hardlink":
            staged.unlink()
            os.link(source, staged)
        assert staged.samefile(source)
    second = module.prepare_bids_connectome(image.parents[2], output, overwrite=overwrite, **options)
    if alias_kind == "symlink" or overwrite:
        assert second.stages["eddy"] == ("completed" if overwrite else "skipped")
    else:
        # Manually exchanging symlinks for hardlinks may change the cache's
        # resolved-path identity. Both safe reuse and safe recomputation are
        # valid; neither may reject staging or mutate the original files.
        assert second.stages["eddy"] in ("completed", "skipped")
    assert len(calls) == 1 + int(second.stages["eddy"] == "completed")
    assert all(path.read_bytes() == before for path, before in originals.items())


@pytest.mark.parametrize("target", ["raw/bids_selection.json", "eddy/data.nii.gz",
                                    "topup/fieldmap_out_fieldcoef.nii.gz"])
def test_inplace_writer_hardlink_to_user_template_rejected(tmp_path, target):
    atlas = tmp_path / "template.nii.gz"
    atlas.write_bytes(b"readonly template")
    output = tmp_path / "output"
    destination = output / "preproc" / target
    destination.parent.mkdir(parents=True)
    os.link(atlas, destination)
    selected = SimpleNamespace(image=tmp_path / "raw.nii.gz", bval=tmp_path / "raw.bval",
                               bvec=tmp_path / "raw.bvec", reverse=tmp_path / "reverse.nii.gz",
                               reverse_bval=None, t1w=None)
    with pytest.raises(ValueError, match="read-only input via a hardlink"):
        validate_bids_preparation_inputs(selected, output, readonly_inputs=(atlas,))
    assert atlas.read_bytes() == destination.read_bytes() == b"readonly template"


def test_atomic_output_replacement_keeps_external_hardlink_source(tmp_path):
    atlas = tmp_path / "template.nii.gz"
    atlas.write_bytes(b"readonly template")
    destination = tmp_path / "output.nii.gz"
    os.link(atlas, destination)
    validate_input_output_paths([atlas], output_paths=[destination])
    temporary = tmp_path / "output.tmp"
    temporary.write_bytes(b"new output")
    temporary.replace(destination)
    assert atlas.read_bytes() == b"readonly template"
    assert destination.read_bytes() == b"new output" and not destination.samefile(atlas)


@pytest.mark.parametrize("resource", ["weights_dir", "assets_dir", "native_bin_dir"])
def test_python_preparation_protects_recon_resources(setup, tmp_path, resource):
    from fnit.connectome.bids import prepare_bids_connectome

    image, _ = _inputs(tmp_path / "bids")
    output = setup["output"]
    options = {name: tmp_path / name for name in ("weights_dir", "assets_dir", "native_bin_dir")}
    options[resource] = output / "preproc/eddy" / resource
    for directory in options.values():
        directory.mkdir(parents=True)
        (directory / "keep.txt").write_bytes(b"read-only reconstruction resource")
    before = {path: (path / "keep.txt").read_bytes() for path in options.values()}
    with pytest.raises(ValueError, match="preparation output namespace"):
        prepare_bids_connectome(
            image.parents[2], output, subject="01", device="cpu",
            t1=setup["subject"] / "mri/brain.mgz", recon_backend="fnit",
            recon_options=options)
    assert all((path / "keep.txt").read_bytes() == contents for path, contents in before.items())
    assert not (output / "preproc/raw/AP.nii.gz").exists()
    assert not (output / "anatomy").exists()


def test_official_installation_subjects_are_writable_but_resources_are_not(tmp_path):
    from fnit.connectome.input_protection import recon_resource_paths

    home = tmp_path / "freesurfer_home"
    home.mkdir()
    for name in ("SetUpFreeSurfer.sh", "FreeSurferEnv.sh", "build-stamp.txt"):
        (home / name).write_text("read-only program source")
    for name in ("bin", "average", "subjects/fsaverage", "subjects/user_subject"):
        (home / name).mkdir(parents=True)
    resources = recon_resource_paths({"freesurfer_home": home})
    assert home not in resources and home / "subjects/fsaverage" in resources
    validate_input_output_paths(resources,
        reserved_directories=(home / "subjects/user_subject/results/anatomy",))
    for directory in (home / "average", home / "subjects/fsaverage"):
        with pytest.raises(ValueError, match="preparation output namespace"):
            validate_input_output_paths(resources,
                reserved_directories=(directory / "new_outputs",))
