"""Focused numerical and NIfTI-contract checks for BEDPOSTX."""

import json

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.bedpostx import TorchBEDPOSTX
from fnit.probtrackx import TorchProbtrackX
from fnit.bedpostx.core import BedpostXConfig, _sample_chunk, _signal_model


def test_multishell_forward_signal():
    bvals = torch.tensor([0.0, 1000.0, 1000.0])
    bvecs = torch.tensor([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])
    theta = torch.tensor([[torch.pi / 2]])
    phi = torch.tensor([[0.0]])
    signal = _signal_model(torch.tensor([100.0]), torch.tensor([0.001]),
                           torch.tensor([0.0005]), torch.tensor([[0.6]]),
                           theta, phi, bvals, bvecs, model=2)[0]
    ball = (1 + 1000 * 0.0005**2 / 0.001)**-4
    expected = np.array([100.0, 100 * ball,
                         100 * (0.4 * ball + 0.6)], dtype=np.float32)
    np.testing.assert_allclose(signal.numpy(), expected, atol=1e-4)


def test_posterior_files_and_geometry(tmp_path):
    rng = np.random.default_rng(4)
    gradients = rng.normal(size=(16, 3)).astype(np.float32)
    gradients /= np.linalg.norm(gradients, axis=1, keepdims=True)
    bvecs = np.vstack((np.zeros((2, 3), np.float32), gradients))
    bvals = np.array([0, 0] + [1000] * 8 + [2000] * 8, dtype=np.float32)
    reference = _signal_model(
        torch.tensor([100.0]), torch.tensor([0.0015]),
        torch.tensor([0.0005]), torch.tensor([[0.6, 0.2]]),
        torch.tensor([[1.2, 0.8]]), torch.tensor([[0.4, 2.1]]),
        torch.from_numpy(bvals), torch.from_numpy(bvecs), model=2,
    )[0].numpy()
    shape = (2, 2, 2)
    mask = np.zeros(shape, np.uint8)
    mask[0, 0, 0] = mask[1, 0, 0] = 1
    data = np.zeros(shape + (bvals.size,), np.float32)
    data[mask > 0] = reference + rng.normal(0, 0.1, (2, bvals.size))
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    subject = tmp_path / "subject"
    subject.mkdir()
    nib.save(nib.Nifti1Image(data, affine), subject / "data.nii.gz")
    nib.save(nib.Nifti1Image(mask, affine), subject / "nodif_brain_mask.nii.gz")
    np.savetxt(subject / "bvals", bvals[None], fmt="%.0f")
    np.savetxt(subject / "bvecs", bvecs.T, fmt="%.8f")

    result = TorchBEDPOSTX(nfibres=2, burnin=4, njumps=6,
                           sample_every=2, chunk_size=1, seed=9)(subject)
    assert result.nvoxels == 2 and result.nsamples == 3
    assert result.output_dir == subject.with_name("subject.bedpostX")
    for fibre in (1, 2):
        for parameter in ("th", "ph", "f"):
            image = nib.load(result.output_dir / f"merged_{parameter}{fibre}samples.nii.gz")
            assert image.shape == shape + (3,)
            np.testing.assert_allclose(image.affine, affine)
            assert np.isfinite(image.get_fdata()).all()
        dyad = nib.load(result.output_dir / f"dyads{fibre}.nii.gz")
        assert dyad.shape == shape + (3,)
    first = nib.load(result.output_dir / "mean_f1samples.nii.gz").get_fdata()
    second = nib.load(result.output_dir / "mean_f2samples.nii.gz").get_fdata()
    assert np.all(first[mask > 0] >= second[mask > 0])
    assert not np.any(first[mask == 0])
    metadata = json.loads((result.output_dir / "run.json").read_text())
    assert metadata["nvoxels"] == 2 and metadata["nsamples"] == 3
    seed = np.zeros(shape, np.uint8)
    seed[0, 0, 0] = 1
    seed_path = tmp_path / "seed.nii.gz"
    nib.save(nib.Nifti1Image(seed, affine), seed_path)
    tract = TorchProbtrackX(nsamples=4, nsteps=8, batch_size=4, seed=7).run(
        result.output_dir, tmp_path / "tracking", seed=seed_path,
    )
    density = nib.load(tract.paths)
    np.testing.assert_allclose(density.affine, affine)
    assert density.shape == shape
    assert tract.accepted_streamlines > 0
    assert density.get_fdata()[0, 0, 0] > 0
    with pytest.raises(FileExistsError, match="not empty"):
        TorchBEDPOSTX(nfibres=2, burnin=4, njumps=6, sample_every=2)(subject)

    sentinel = result.output_dir / "keep.txt"
    sentinel.write_text("unrelated")
    TorchBEDPOSTX(nfibres=3, burnin=4, njumps=6,
                  sample_every=2, seed=9)(subject, overwrite=True)
    assert (result.output_dir / "merged_f3samples.nii.gz").exists()
    TorchBEDPOSTX(nfibres=2, burnin=4, njumps=6,
                  sample_every=2, seed=9)(subject, overwrite=True)
    assert sentinel.read_text() == "unrelated"
    for name in ("merged_th3samples.nii.gz", "merged_ph3samples.nii.gz",
                 "merged_f3samples.nii.gz", "mean_f3samples.nii.gz",
                 "dyads3.nii.gz"):
        assert not (result.output_dir / name).exists()

    TorchBEDPOSTX(nfibres=1, burnin=2, njumps=1, sample_every=1,
                  seed=9)(subject, overwrite=True)
    for parameter in ("th", "ph", "f"):
        image = nib.load(result.output_dir / f"merged_{parameter}1samples.nii.gz")
        assert image.shape == shape + (1,)
    tract = TorchProbtrackX(nsamples=2, nsteps=8, batch_size=2, seed=7).run(
        result.output_dir, tmp_path / "tracking_one_draw", seed=seed_path,
    )
    assert nib.load(tract.paths).shape == shape
    assert tract.accepted_streamlines > 0


def test_scalar_proposals_have_fixed_absolute_width(monkeypatch):
    from fnit.bedpostx import core

    state = {"s0": torch.tensor([100.0]), "d": torch.tensor([0.001]),
             "d_std": torch.tensor([0.0005]), "f": torch.tensor([[0.5]]),
             "th": torch.tensor([[1.0]]), "ph": torch.tensor([[0.0]])}
    monkeypatch.setattr(core, "_initial_state", lambda *args: state)
    monkeypatch.setattr(core, "_energy", lambda *args: torch.zeros(1))
    monkeypatch.setattr(core.torch, "randn", lambda shape, **kwargs: torch.ones(shape))
    monkeypatch.setattr(core.torch, "rand", lambda shape, **kwargs: torch.full(shape, 0.5))
    config = BedpostXConfig(nfibres=1, model=2, burnin=0, njumps=2, sample_every=1)
    _, d, d_std, s0 = _sample_chunk(torch.ones((1, 1)), torch.zeros(1),
                                    torch.zeros((1, 3)), config, torch.Generator())
    np.testing.assert_allclose(s0.numpy()[0], [110.0, 120.0], rtol=1e-6)
    np.testing.assert_allclose(d.numpy()[0], [0.0011, 0.0012], rtol=1e-6)
    np.testing.assert_allclose(d_std.numpy()[0], [0.00055, 0.0006], rtol=1e-6)


def test_invalid_sampling_plan():
    with pytest.raises(ValueError, match="divisible"):
        TorchBEDPOSTX(njumps=8, sample_every=3)
