"""Public CLI/template/cache/matrix integration with shared numerical core stubbed.

Small readable MRI/annotation fixtures are exact unit controls, never a
scientific MRI benchmark. Source loading, geometry, cache, assignment, matrix
aggregation, CLI serialization and ownership checks use production code.
"""
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit import cli
import fnit.connectome.pipeline as pipeline
from fnit.connectome.assignment import build_connectomes
from fnit.connectome.paired_cli import load_template_pairs, load_recon_options
from fnit.connectome.template_inputs import (
    TemplatePair, TemplateSpec, preflight_template, preflight_template_pairs,
)


def volume(path, labels):
    nib.save(nib.Nifti1Image(np.asarray(labels), np.eye(4)), path)
    return path


def file_contents(directory):
    return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    shape = (5, 5, 5)
    subject = tmp_path / "subject"
    for folder in ("mri", "surf", "label"):
        (subject / folder).mkdir(parents=True)
    brain = nib.MGHImage(np.ones(shape, np.float32), np.eye(4))
    nib.save(brain, subject / "mri/brain.mgz")
    segmentation = np.full(shape, 2, np.float32)
    segmentation[:, 1, :] = 1001
    segmentation[:, 2, :] = 2001
    nib.save(nib.MGHImage(segmentation, np.eye(4)), subject / "mri/aparc+aseg.mgz")
    ribbon = np.zeros(shape, np.float32)
    ribbon[1:3, 1, 1], ribbon[1:3, 2, 1] = 3, 42
    nib.save(nib.MGHImage(ribbon, np.eye(4)), subject / "mri/ribbon.mgz")
    tkr = brain.header.get_vox2ras_tkr()
    faces = np.array([[0, 1, 2]], np.int32)
    for hemi, y in (("lh", 1), ("rh", 2)):
        positions = np.array([[1, y, 1], [2, y, 1], [1, y, 2]], float)
        vertices = positions @ tkr[:3, :3].T + tkr[:3, 3]
        for kind in ("white", "pial"):
            nib.freesurfer.write_geometry(str(subject / f"surf/{hemi}.{kind}"), vertices, faces)
        for atlas, labels, colors, names in (
            ("A", [1, 2, 1], [[0, 0, 0, 0], [100, 50, 30, 0], [20, 150, 50, 0], [40, 60, 120, 0]],
             [b"unknown", b"first", b"second", b"absent"]),
            ("B", [1, 1, 1], [[0, 0, 0, 0], [100, 50, 30, 0]], [b"unknown", b"one"]),
        ):
            nib.freesurfer.write_annot(str(subject / f"label/{hemi}.{atlas}.annot"),
                                      np.array(labels), np.array(colors), names)
    surface_a = TemplateSpec("SA", "surface", "native", left_path=subject / "label/lh.A.annot", right_path=subject / "label/rh.A.annot")
    surface_b = TemplateSpec("SB", "surface", "native", left_path=subject / "label/lh.B.annot", right_path=subject / "label/rh.B.annot")
    a = np.zeros(shape, np.int32); a[1, 1, 1], a[2, 1, 1] = 100, 200
    b = np.zeros(shape, np.int32); b[1, 2, 1], b[2, 2, 1], b[0, 0, 0] = 10, 20, 30
    volume_a = TemplateSpec("VA", "volume", "dwi", volume_path=volume(tmp_path / "A.nii.gz", a))
    volume_b = TemplateSpec("VB", "volume", "dwi", volume_path=volume(tmp_path / "B.nii.gz", b))
    dwi = volume(tmp_path / "dwi.nii.gz", np.ones((*shape, 2), np.float32))
    bval, bvec = tmp_path / "dwi.bval", tmp_path / "dwi.bvec"
    bval.write_text("0 1000\n"); bvec.write_text("0 1\n0 0\n0 0\n")
    endpoints = torch.tensor([[[1., 1, 1], [2., 2, 1]], [[2., 1, 1], [1., 2, 1]], [[1., 1, 1], [1., 1, 1]]])
    calls = Counter()
    def core(**options):
        calls["core"] += 1
        eye = torch.eye(4, dtype=torch.float64)
        tracks = pipeline.Tractogram(tuple(p.clone() for p in endpoints), endpoints.clone(),
                                     torch.tensor([10., 20., 30.]), torch.tensor([.2, .4, .6]),
                                     options["n_seeds"], endpoints[:, 0].clone())
        return dict(seg=torch.tensor(segmentation), seg_affine=eye,
                    five=torch.ones((*shape, 5)), gmwmi=torch.ones(shape),
                    transform=eye, five_affine=eye, wm_sh=torch.ones((*shape, 45)),
                    fa=torch.full(shape, .5), mask=torch.ones(shape, dtype=torch.bool),
                    tracks=tracks, weights=torch.tensor([1., 2., 3.], dtype=torch.float64),
                    dwi_affine=eye, dwi_shape=shape)
    monkeypatch.setattr(pipeline, "_compute_shared_core", core)
    return dict(shape=shape, subject=subject, surface_a=surface_a, surface_b=surface_b,
                volume_a=volume_a, volume_b=volume_b, dwi=dwi, bval=bval, bvec=bvec,
                calls=calls, output=tmp_path / "output")


def run(setup, pairs, *, radius=.1, checkpoint=None):
    return pipeline.UKBConnectome_pipeline(device="cpu")(
        setup["dwi"], setup["bval"], setup["bvec"], freesurfer_subject_dir=setup["subject"],
        n_seeds=8, dwi_to_t1_world=np.eye(4), template_pairs=pairs,
        assignment_radius=radius, checkpoint_dir=checkpoint or setup["output"] / "checkpoints")


def pair_json(setup, pairs, path):
    def spec(value):
        record = dict(name=value.name, kind=value.kind, space=value.space)
        for name in ("volume_path", "left_path", "right_path"):
            file = getattr(value, name)
            if file is not None:
                record[name] = os.path.relpath(file, path.parent)
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([dict(name=pair.name, first=spec(pair.first), second=spec(pair.second)) for pair in pairs]))
    return path


def cli_run(setup, path, *, output=None, radius=.1, overwrite=False):
    command = ["UKBConnectome_pipeline", "--template-pairs", str(path),
               "--dwi", str(setup["dwi"]), "--bvals", str(setup["bval"]), "--bvecs", str(setup["bvec"]),
               "--freesurfer-subject-dir", str(setup["subject"]), "--output-dir", str(output or setup["output"]),
               "--n-seeds", "8", "--device", "cpu", "--assignment-radius", str(radius)]
    if overwrite:
        command.append("--overwrite")
    cli.main(command)
    return Path(output or setup["output"]) / "pairs_run_state.json"


def test_rectangular_surface_surface_volume_volume_and_mixed_exact(setup):
    pairs = [TemplatePair("SS", setup["surface_a"], setup["surface_b"]),
             TemplatePair("VV", setup["volume_a"], setup["volume_b"]),
             TemplatePair("SV", setup["surface_a"], setup["volume_b"])]
    result = run(setup, pairs)
    assert result.pair_results["SS"].matrices["count"].tolist() == [[1, 1], [0, 1], [0, 0], [1, 0], [1, 0], [0, 0]]
    assert result.pair_results["VV"].matrices["count"].tolist() == [[0, 1, 0], [1, 0, 0]]
    assert result.pair_results["SV"].matrices["count"].tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for name, rows, cols in (("SS", 6, 2), ("VV", 2, 3), ("SV", 6, 3)):
        actual = result.pair_results[name]
        assert len(actual.first.nodes) == rows and len(actual.second.nodes) == cols
        assert all(matrix.shape == (rows, cols) for matrix in actual.matrices.values())
    assert result.pair_results["SS"].matrices["sift2_fbc"][0].tolist() == [3, 1]
    assert result.pair_results["VV"].matrices["mean_length"][1, 0] == 20
    assert setup["calls"]["core"] == 1


def test_pair_A_B_A_cache_and_radius_only_rebuilds_matrices(setup):
    first = TemplatePair("pair", setup["volume_a"], setup["volume_b"])
    second = TemplatePair("pair", setup["surface_a"], setup["volume_b"])
    a = run(setup, [first]); b = run(setup, [second]); restored_a = run(setup, [first])
    assert setup["calls"]["core"] == 1
    assert a.cache_status["core"] == "completed" and b.cache_status["core"] == restored_a.cache_status["core"] == "skipped"
    assert a.pair_results["pair"].cache_status == b.pair_results["pair"].cache_status == "completed"
    assert restored_a.pair_results["pair"].cache_status == "skipped"
    for name in a.pair_results["pair"].matrices:
        assert torch.equal(a.pair_results["pair"].matrices[name], restored_a.pair_results["pair"].matrices[name])
    wider = run(setup, [first], radius=.2)
    assert wider.cache_status["core"] == "skipped" and setup["calls"]["core"] == 1
    assert wider.pair_results["pair"].cache_status == "completed"
    assert all(event["status"] == "hit" for event in wider.cache_status["pair_events"] if event["stage"] == "template")


def test_identity_matches_existing_square_builder_exactly(setup):
    result = run(setup, [TemplatePair("identity", setup["volume_a"], setup["volume_a"])])
    pair = result.pair_results["identity"]
    original = build_connectomes(result.tractogram.endpoints, pair.first.labels, pair.first.affine,
        weights=result.sift2_weights, lengths=result.tractogram.lengths_mm, fa=result.tractogram.mean_fa,
        node_count=len(pair.first.nodes), radius=.1)
    assert all(torch.equal(original[name], pair.matrices[name]) for name in original)


def test_real_cli_relative_paths_three_rectangular_outputs_and_read_only_subject(setup, tmp_path, monkeypatch):
    pairs = [TemplatePair("SS", setup["surface_a"], setup["surface_b"]),
             TemplatePair("VV", setup["volume_a"], setup["volume_b"]),
             TemplatePair("SV", setup["surface_a"], setup["volume_b"])]
    path = pair_json(setup, pairs, tmp_path / "configs/pairs.json")
    before = file_contents(setup["subject"])
    monkeypatch.chdir(tmp_path)  # Pair paths must resolve against configs/, not cwd.
    state = json.loads(cli_run(setup, path).read_text())
    assert setup["calls"]["core"] == 1 and len(state["outputs"]) == 27
    for name, shape in (("SS", (6, 2)), ("VV", (2, 3)), ("SV", (6, 3))):
        folder = setup["output"] / "pairs" / name
        assert np.loadtxt(folder / "connectome_count.csv", delimiter=",").shape == shape
        assert len((folder / "rows.tsv").read_text().splitlines()) == shape[0] + 1
        assert len((folder / "columns.tsv").read_text().splitlines()) == shape[1] + 1
    assert before == file_contents(setup["subject"])


def test_cli_A_B_A_keeps_owned_outputs_and_rejects_user_edits(setup, tmp_path):
    a = pair_json(setup, [TemplatePair("A", setup["volume_a"], setup["volume_b"])], tmp_path / "a.json")
    b = pair_json(setup, [TemplatePair("B", setup["surface_a"], setup["volume_b"])], tmp_path / "b.json")
    first = json.loads(cli_run(setup, a).read_text())
    first_hashes = first["outputs"]
    cli_run(setup, b)
    returned = json.loads(cli_run(setup, a).read_text())
    assert setup["calls"]["core"] == 1 and returned["cache_status"]["template_pairs"]["A"] == "skipped"
    # pair.json records cache status; numeric files and node/atlas outputs stay identical.
    assert all(returned["outputs"][name] == digest for name, digest in first_hashes.items() if not name.endswith("pair.json"))
    assert any(name.startswith("pairs/B/") for name in returned["owned_outputs"])
    edited = setup["output"] / "pairs/A/connectome_count.csv"
    edited.write_text("user-edited\n")
    with pytest.raises(FileExistsError, match="unchanged FNIT-managed"):
        cli_run(setup, a)
    assert edited.read_text() == "user-edited\n" and setup["calls"]["core"] == 1


@pytest.mark.parametrize("bad", ["fractional", "four_dimensional", "unknown_nodes", "surface_count"])
def test_bad_templates_rejected_before_tracking_in_python_and_cli(setup, tmp_path, bad):
    if bad == "surface_count":
        # A valid annotation with four vertices cannot belong to this three-vertex subject.
        path = setup["subject"] / "label/lh.bad.annot"
        nib.freesurfer.write_annot(str(path), np.ones(4, np.int32), np.array([[0, 0, 0, 0], [1, 2, 3, 0]]), [b"unknown", b"one"])
        spec = replace(setup["surface_b"], left_path=path)
    else:
        values = np.full(setup["shape"], 1.5, np.float32) if bad == "fractional" else np.ones((*setup["shape"], 2), np.int32) if bad == "four_dimensional" else np.full(setup["shape"], 2, np.int32)
        path = volume(tmp_path / f"{bad}.nii.gz", values)
        spec = replace(setup["volume_a"], volume_path=path)
        if bad == "unknown_nodes":
            table = tmp_path / "nodes.tsv"; table.write_text("original_label\tname\n1\tonly\n")
            spec = replace(spec, nodes_tsv=table)
    pair = TemplatePair("bad", spec, setup["volume_b"])
    with pytest.raises(ValueError):
        run(setup, [pair])
    path = pair_json(setup, [pair], tmp_path / "bad.json")
    if spec.nodes_tsv is not None:
        data = json.loads(path.read_text()); data[0]["first"]["nodes_tsv"] = str(spec.nodes_tsv); path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        cli_run(setup, path)
    assert setup["calls"]["core"] == 0


def test_empty_pairs_rejected_before_core(setup):
    with pytest.raises(ValueError, match="one or more"):
        run(setup, [])
    assert not setup["calls"]


def test_modified_template_same_size_mtime_rebuilds_pair_only(setup):
    pair = TemplatePair("pair", setup["volume_a"], setup["volume_b"])
    first = run(setup, [pair])
    path = setup["volume_b"].volume_path
    # Use an uncompressed NIfTI to make the same byte count explicit.
    uncompressed = path.with_suffix("").with_suffix(".nii")
    nib.save(nib.load(path), uncompressed)
    pair = TemplatePair("pair", setup["volume_a"], replace(setup["volume_b"], volume_path=uncompressed))
    original = run(setup, [pair])
    stat = uncompressed.stat(); image = nib.load(uncompressed); labels = np.asarray(image.dataobj).copy()
    labels[1, 2, 1], labels[2, 2, 1] = 20, 10
    nib.save(nib.Nifti1Image(labels, image.affine), uncompressed)
    assert uncompressed.stat().st_size == stat.st_size
    os.utime(uncompressed, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = run(setup, [pair])
    assert setup["calls"]["core"] == 1 and changed.cache_status["core"] == "skipped"
    assert changed.pair_results["pair"].cache_status == "completed"
    assert not torch.equal(original.pair_results["pair"].matrices["count"], changed.pair_results["pair"].matrices["count"])


def test_corrupt_pair_matrix_restores_mapping_and_recomputes_only_pair(setup):
    pair = TemplatePair("pair", setup["volume_a"], setup["volume_b"])
    first = run(setup, [pair])
    markers = list((setup["output"] / "checkpoints/pairs/matrix").glob("*/complete.json"))
    assert len(markers) == 1
    marker = json.loads(markers[0].read_text())
    matrix = markers[0].parent / marker["generation"] / "count.npy"
    matrix.write_bytes(b"corrupt")
    repaired = run(setup, [pair])
    assert setup["calls"]["core"] == 1 and repaired.pair_results["pair"].cache_status == "completed"
    assert torch.equal(first.pair_results["pair"].matrices["count"], repaired.pair_results["pair"].matrices["count"])
    assert all(event["status"] == "hit" for event in repaired.cache_status["pair_events"] if event["stage"] == "template")


def test_supplied_subject_output_or_checkpoint_overlap_fails_before_core(setup, tmp_path):
    pair = TemplatePair("VV", setup["volume_a"], setup["volume_b"])
    path = pair_json(setup, [pair], tmp_path / "pairs.json")
    before = file_contents(setup["subject"])
    with pytest.raises(ValueError, match="read-only recon-all"):
        cli_run(setup, path, output=setup["subject"])
    with pytest.raises(ValueError, match="read-only recon-all"):
        run(setup, [pair], checkpoint=setup["subject"] / "cache")
    assert not setup["calls"] and file_contents(setup["subject"]) == before


def test_output_folder_symlink_into_subject_rejected(setup, tmp_path):
    pair = TemplatePair("VV", setup["volume_a"], setup["volume_b"])
    path = pair_json(setup, [pair], tmp_path / "pairs.json")
    (setup["output"] / "pairs").mkdir(parents=True)
    (setup["output"] / "pairs/VV").symlink_to(setup["subject"], target_is_directory=True)
    before = file_contents(setup["subject"])
    with pytest.raises(ValueError, match="read-only recon-all"):
        cli_run(setup, path)
    assert not setup["calls"] and file_contents(setup["subject"]) == before


def test_recon_options_file_resources_resolve_against_json_directory(tmp_path, monkeypatch):
    config = tmp_path / "config"; config.mkdir()
    path = config / "recon.json"
    path.write_text(json.dumps({"weights_dir": "weights", "assets_dir": "assets", "native_bin_dir": "bin", "threads": 4}))
    monkeypatch.chdir(tmp_path)
    options = load_recon_options(path)
    assert options["weights_dir"] == str(config / "weights") and options["native_bin_dir"] == str(config / "bin")
    assert options["threads"] == 4


def test_recon_options_inline_resources_resolve_against_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    options = load_recon_options(json.dumps({"weights_dir": "weights", "assets_dir": "assets", "executable": "./bin/recon-all"}))
    assert options["weights_dir"] == str(tmp_path / "weights")
    assert options["assets_dir"] == str(tmp_path / "assets")
    assert options["executable"] == str(tmp_path / "bin/recon-all")


@pytest.mark.parametrize("as_file", [False, True])
def test_explicit_official_command_resolves_on_path(tmp_path, monkeypatch, as_file):
    from fnit.connectome.recon_backend import validate_recon_configuration
    directory = tmp_path / "official-bin"; directory.mkdir()
    executable = directory / "recon-all"
    executable.write_text("#!/bin/sh\nexit 0\n"); executable.chmod(0o755)
    (tmp_path / "SetUpFreeSurfer.sh").write_text('. "$FREESURFER_HOME/FreeSurferEnv.sh"\n')
    (tmp_path / "FreeSurferEnv.sh").write_text('export FREESURFER="$FREESURFER_HOME"\n')
    monkeypatch.setenv("FREESURFER_HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(directory))
    inline = json.dumps({"executable": "recon-all", "threads": 4})
    value = inline
    if as_file:
        value = tmp_path / "options.json"; value.write_text(inline)
    options = load_recon_options(value)
    assert options["executable"] == "recon-all"
    backend, validated = validate_recon_configuration("freesurfer", recon_options=options)
    assert backend == "freesurfer" and validated["executable"] == str(executable)
