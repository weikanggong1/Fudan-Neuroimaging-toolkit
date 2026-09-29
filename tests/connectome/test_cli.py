"""Single-subject connectome CLI output and input-protection checks."""

from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.cli import main


def _command(tmp_path):
    inputs = {}
    for name in ("dwi", "bvals", "bvecs", "t1", "t1_segmentation", "atlas_dwi", "brain_mask"):
        path = tmp_path / name
        path.write_bytes(b"input")
        inputs[name] = path
    output = tmp_path / "result"
    argv = [
        "connectome", "--dwi", str(inputs["dwi"]),
        "--bvals", str(inputs["bvals"]), "--bvecs", str(inputs["bvecs"]),
        "--t1", str(inputs["t1"]),
        "--t1-segmentation", str(inputs["t1_segmentation"]),
        "--atlas-dwi", str(inputs["atlas_dwi"]),
        "--brain-mask", str(inputs["brain_mask"]),
        "--shell-bvals", "5", "999", "1997",
        "--output-dir", str(output),
        "--device", "cpu", "--n-seeds", "12", "--seed", "7",
    ]
    return argv, output


def test_connectome_cli_requires_explicit_seed_count(tmp_path):
    argv, _ = _command(tmp_path)
    index = argv.index("--n-seeds")
    del argv[index:index + 2]
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2


@pytest.mark.parametrize("include_mask", [True, False])
def test_connectome_cli_writes_named_outputs(tmp_path, monkeypatch, capsys, include_mask):
    from fnit import connectome

    calls = []
    class FakeConnectome:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(
                matrices={
                    "count": torch.tensor([[2, 1], [1, 3]], dtype=torch.int64),
                    "sift2_fbc": torch.tensor([[0.2, 0.1], [0.1, 0.3]]),
                    "mean_length": torch.ones(2, 2) * 20,
                    "mean_fa": torch.ones(2, 2) * 0.5,
                },
                atlas=torch.ones(2, 2, 2, dtype=torch.int32),
                five_tissue=torch.ones(2, 2, 2, 5),
                gmwmi=torch.ones(2, 2, 2),
                five_tissue_affine=torch.eye(4),
                fa=torch.ones(2, 2, 2),
                brain_mask=torch.ones(2, 2, 2, dtype=torch.bool),
                region_labels=(10, 20),
                dwi_affine=torch.eye(4),
                atlas_affine=torch.tensor([[1., 0., 0., -1.],
                                           [0., 1., 0., 0.],
                                           [0., 0., 1., 0.],
                                           [0., 0., 0., 1.]]),
                dwi_to_t1_world=torch.eye(4),
                tractogram=SimpleNamespace(seeds_attempted=12, paths=(None, None)),
            )
    monkeypatch.setattr(connectome, "UKBConnectome", FakeConnectome)
    argv, output = _command(tmp_path)
    if not include_mask:
        position = argv.index("--brain-mask")
        del argv[position:position + 2]
    main(argv)
    assert calls[1][1]["brain_mask"] == (tmp_path / "brain_mask" if include_mask else None)
    assert "seed_attempts=12 accepted_streamlines=2" in capsys.readouterr().out
    assert calls[0] == {"device": "cpu"}
    assert calls[1][1]["n_seeds"] == 12
    assert calls[1][1]["seed"] == 7
    assert np.array_equal(np.loadtxt(output / "connectome_count.csv", delimiter=","),
                          [[2, 1], [1, 3]])
    for name in ("sift2_fbc", "mean_length", "mean_fa"):
        assert np.loadtxt(output / f"connectome_{name}.csv", delimiter=",").shape == (2, 2)
    for name, file in (("atlas", "atlas_dwi.nii.gz"),
                       ("five_tissue", "five_tissue_dwi_world.nii.gz"),
                       ("gmwmi", "gmwmi_dwi_world.nii.gz"),
                       ("fa", "fa_dwi.nii.gz"),
                       ("brain_mask", "brain_mask_dwi.nii.gz")):
        image = nib.load(output / file)
        assert image.shape == ((2, 2, 2, 5) if name == "five_tissue" else (2, 2, 2))
        expected_affine = np.eye(4)
        if name == "atlas":
            expected_affine[0, 3] = -1
        assert np.array_equal(image.affine, expected_affine)
    assert np.array_equal(np.loadtxt(output / "region_labels.csv", delimiter=","), [10, 20])
    assert np.array_equal(np.loadtxt(output / "dwi_to_t1_world.csv", delimiter=","), np.eye(4))


def test_connectome_cli_never_overwrites_input(tmp_path):
    argv, output = _command(tmp_path)
    output.mkdir()
    source = output / "atlas_dwi.nii.gz"
    source.write_bytes(b"input image")
    argv[argv.index("--dwi") + 1] = str(source)
    with pytest.raises(ValueError, match="overwrite an input"):
        main([*argv, "--overwrite"])
    assert source.read_bytes() == b"input image"


def test_connectome_cli_requires_overwrite_for_existing_output(tmp_path):
    argv, output = _command(tmp_path)
    output.mkdir()
    existing = output / "connectome_count.csv"
    existing.write_text("prior result")
    with pytest.raises(FileExistsError, match="use --overwrite"):
        main(argv)
    assert existing.read_text() == "prior result"


def test_connectome_cli_subject_directory_writes_nodes(tmp_path, monkeypatch):
    from fnit import connectome
    from fnit.connectome.freesurfer_subject import ConnectomeNode

    subject = tmp_path / "subject" / "mri"
    subject.mkdir(parents=True)
    (subject / "brain.mgz").write_bytes(b"brain")
    (subject / "aparc+aseg.mgz").write_bytes(b"segmentation")
    for name in ("dwi", "bvals", "bvecs"):
        (tmp_path / name).write_bytes(b"input")

    class FakeConnectome:
        def __init__(self, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            assert kwargs["freesurfer_subject_dir"] == str(subject.parent)
            assert kwargs["atlas"] == "fs-aparc"
            assert kwargs["atlas_dwi"] is None
            return SimpleNamespace(
                matrices={
                    "count": torch.ones((1, 1), dtype=torch.int64),
                    "sift2_fbc": torch.ones((1, 1)),
                    "mean_length": torch.ones((1, 1)),
                    "mean_fa": torch.ones((1, 1)),
                },
                atlas=torch.ones((2, 2, 2), dtype=torch.int32),
                five_tissue=torch.ones((2, 2, 2, 5)),
                gmwmi=torch.ones((2, 2, 2)),
                fa=torch.ones((2, 2, 2)),
                brain_mask=torch.ones((2, 2, 2)),
                region_labels=(1,),
                nodes=(ConnectomeNode(1, 1001, "L", "ctx-lh-bankssts"),),
                five_tissue_affine=torch.eye(4),
                dwi_affine=torch.eye(4),
                atlas_affine=torch.eye(4),
                dwi_to_t1_world=torch.eye(4),
                tractogram=SimpleNamespace(seeds_attempted=1, paths=(None,)),
            )

    monkeypatch.setattr(connectome, "UKBConnectome", FakeConnectome)
    output = tmp_path / "result"
    main([
        "connectome", "--dwi", str(tmp_path / "dwi"),
        "--bvals", str(tmp_path / "bvals"), "--bvecs", str(tmp_path / "bvecs"),
        "--freesurfer-subject-dir", str(subject.parent), "--atlas", "fs-aparc",
        "--n-seeds", "1", "--device", "cpu", "--output-dir", str(output),
    ])
    assert (output / "nodes.tsv").read_text().splitlines() == [
        "index\toriginal_label\themisphere\tname",
        "1\t1001\tL\tctx-lh-bankssts",
    ]


def test_connectome_cli_subject_directory_requires_completed_files(tmp_path):
    subject = tmp_path / "subject"
    subject.mkdir()
    for name in ("dwi", "bvals", "bvecs"):
        (tmp_path / name).write_bytes(b"input")
    with pytest.raises(FileNotFoundError, match="brain.mgz"):
        main([
            "connectome", "--dwi", str(tmp_path / "dwi"),
            "--bvals", str(tmp_path / "bvals"), "--bvecs", str(tmp_path / "bvecs"),
            "--freesurfer-subject-dir", str(subject), "--n-seeds", "1",
            "--device", "cpu", "--output-dir", str(tmp_path / "result"),
        ])


def test_connectome_cli_schaefer_requires_template_inputs(tmp_path):
    subject = tmp_path / "subject" / "mri"
    subject.mkdir(parents=True)
    for name in ("brain.mgz", "aparc+aseg.mgz"):
        (subject / name).write_bytes(b"input")
    for name in ("dwi", "bvals", "bvecs"):
        (tmp_path / name).write_bytes(b"input")
    with pytest.raises(ValueError, match="--atlas-templates-dir"):
        main([
            "connectome", "--dwi", str(tmp_path / "dwi"),
            "--bvals", str(tmp_path / "bvals"), "--bvecs", str(tmp_path / "bvecs"),
            "--freesurfer-subject-dir", str(subject.parent),
            "--atlas", "schaefer200+tian-s1", "--n-seeds", "1",
            "--device", "cpu", "--output-dir", str(tmp_path / "result"),
        ])


def test_connectome_cli_rejects_ambiguous_tian_registration(tmp_path):
    subject = tmp_path / "subject" / "mri"
    subject.mkdir(parents=True)
    for name in ("brain.mgz", "aparc+aseg.mgz"):
        (subject / name).write_bytes(b"input")
    for name in ("dwi", "bvals", "bvecs"):
        (tmp_path / name).write_bytes(b"input")
    with pytest.raises(ValueError, match="exactly one"):
        main([
            "connectome", "--dwi", str(tmp_path / "dwi"),
            "--bvals", str(tmp_path / "bvals"), "--bvecs", str(tmp_path / "bvecs"),
            "--freesurfer-subject-dir", str(subject.parent),
            "--atlas", "schaefer200+tian-s1",
            "--atlas-templates-dir", str(tmp_path), "--fsaverage-dir", str(tmp_path),
            "--mni-template", str(tmp_path / "mni.nii.gz"),
            "--tian-fnirt-coeff", str(tmp_path / "warp.nii.gz"),
            "--n-seeds", "1", "--device", "cpu", "--output-dir", str(tmp_path / "result"),
        ])
