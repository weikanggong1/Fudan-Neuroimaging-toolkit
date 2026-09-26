"""Single-subject connectome CLI output and input-protection checks."""

from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.cli import main


def _command(tmp_path):
    inputs = {}
    for name in ("dwi", "bvals", "bvecs", "t1"):
        path = tmp_path / name
        path.write_bytes(b"input")
        inputs[name] = path
    output = tmp_path / "result"
    argv = [
        "connectome", "--dwi", str(inputs["dwi"]),
        "--bvals", str(inputs["bvals"]), "--bvecs", str(inputs["bvecs"]),
        "--t1", str(inputs["t1"]), "--output-dir", str(output),
        "--device", "cpu", "--n-seeds", "12", "--seed", "7",
    ]
    return argv, output


def test_connectome_cli_writes_named_outputs(tmp_path, monkeypatch, capsys):
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
                tissues=torch.ones(2, 2, 2, dtype=torch.int16),
                fa=torch.ones(2, 2, 2),
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
    main(argv)
    assert "seed_attempts=12 accepted_streamlines=2" in capsys.readouterr().out
    assert calls[0] == {"device": "cpu", "synthseg_weights": None}
    assert calls[1][1]["n_seeds"] == 12
    assert calls[1][1]["seed"] == 7
    assert np.array_equal(np.loadtxt(output / "connectome_count.csv", delimiter=","),
                          [[2, 1], [1, 3]])
    for name in ("sift2_fbc", "mean_length", "mean_fa"):
        assert np.loadtxt(output / f"connectome_{name}.csv", delimiter=",").shape == (2, 2)
    for name in ("atlas", "tissues", "fa"):
        image = nib.load(output / f"{name}_dwi.nii.gz")
        assert image.shape == (2, 2, 2)
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
