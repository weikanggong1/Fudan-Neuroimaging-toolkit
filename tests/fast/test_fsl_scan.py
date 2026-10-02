"""Ordered FAST scan dependencies and RNG semantics, separate from benchmarks."""

import math

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.fast import TorchFAST
from fnit.fast.algorithm import FASTConfig, FASTTensorResult, _fsl_fractions, _fsl_moments
from fnit.fast._fsl_scan import GlibcRandom, compatibility, icm, schedule, tanaka


def test_glibc_seed_minus_one_and_continuous_stream():
    expected = [254925627, 1205188300, 366127624, 1401405153, 76053476, 1604170158,
                1302235366, 362229243, 334960208, 1882140968, 960816832, 627031785]
    random = GlibcRandom(-1)
    assert random.raw(5).tolist() + random.raw(7).tolist() == expected
    assert GlibcRandom(0).raw(5).tolist() == GlibcRandom(1).raw(5).tolist()


def test_eighteen_neighbours_have_strict_directed_wavefront_order():
    scan = schedule(torch.ones((3, 4, 5), dtype=torch.bool), (1, 1.2, 1.5))
    assert len(scan.neighbours) == 18
    for x, y, z, weight in scan.neighbours:
        assert 1 <= sum(value != 0 for value in (x, y, z)) <= 2
        assert weight > 0
        preceding = z < 0 or (z == 0 and (y < 0 or (y == 0 and x < 0)))
        level_change = x + 2 * y + 3 * z
        assert (level_change < 0) == preceding
        assert level_change != 0


def _scalar_tanaka(probabilities, energy, mask, scan, beta, iterations):
    output = probabilities.numpy().copy()
    energy = energy.numpy()
    beta = float(np.float32(beta))
    for _ in range(iterations):
        for z in range(mask.shape[2]):
            for y in range(mask.shape[1]):
                for x in range(mask.shape[0]):
                    if not mask[x, y, z]:
                        continue
                    likelihood = np.zeros(3, dtype=np.float32)
                    for component in range(3):
                        support = 0.
                        for dx, dy, dz, weight in scan.neighbours:
                            position = (x + dx, y + dy, z + dz)
                            if all(0 <= position[a] < mask.shape[a] for a in range(3)):
                                support += float(output[component][position]) * weight
                        likelihood[component] = np.float32(math.exp(beta * support - float(energy[component, x, y, z])))
                    total = (float(likelihood[0]) + float(likelihood[1])) + float(likelihood[2])
                    output[:, x, y, z] = likelihood.astype(np.float64) / total if total > 0 else 0
    return torch.from_numpy(output)


def test_wavefront_tanaka_matches_original_lexicographic_dependencies():
    generator = torch.Generator().manual_seed(72)
    shape = (3, 4, 5)
    mask = torch.ones(shape, dtype=torch.bool)
    mask[0, 0, 0] = False
    probabilities = torch.rand((3, *shape), generator=generator)
    probabilities /= probabilities.sum(dim=0)
    probabilities *= mask[None]
    energy = torch.rand((3, *shape), generator=generator) * 2
    scan = schedule(mask, (1., 1.1, 1.2))
    reference = _scalar_tanaka(probabilities, energy, mask, scan, .02, 5)
    candidate = tanaka(probabilities.clone(), energy, scan, .02, iterations=5)
    torch.testing.assert_close(candidate, reference, rtol=0, atol=6e-8)


def test_wavefront_icm_matches_single_lexicographic_pass():
    generator = torch.Generator().manual_seed(18)
    shape = (3, 4, 5)
    mask = torch.ones(shape, dtype=torch.bool)
    mask[0, 0, 0] = False
    probabilities = torch.rand((6, *shape), generator=generator)
    scan = schedule(mask, (1., 1.1, 1.2))
    labels = probabilities.argmax(dim=0).numpy().astype(np.int32)
    labels[~mask.numpy()] = 0
    pairwise = compatibility().numpy()
    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                if not mask[x, y, z]:
                    continue
                clique = np.zeros(6, dtype=np.float32)
                for dx, dy, dz, weight in scan.neighbours:
                    position = (x + dx, y + dy, z + dz)
                    if all(0 <= position[a] < shape[a] for a in range(3)) and mask[position]:
                        clique += np.float32(weight) * pairwise[:, labels[position]]
                score = probabilities[:, x, y, z].numpy() * np.exp(np.float32(.3) * clique)
                labels[x, y, z] = score.argmax()
    torch.testing.assert_close(icm(probabilities, mask, scan, .3), torch.from_numpy(labels).long())


def test_fractions_follow_float_loop_not_linspace_endpoints():
    fractions = _fsl_fractions(100, device="cpu")
    assert len(fractions) == 101
    assert fractions[-1].item() == float(np.float32(.9999993443489075))
    assert not torch.equal(fractions, torch.linspace(0, 1, 101))


def test_strict_moments_do_not_replace_nonpositive_variances():
    values = torch.ones((2, 2, 2))
    probabilities = torch.ones((3, 2, 2, 2)) / 3
    with pytest.raises(RuntimeError, match="non-positive class variance"):
        _fsl_moments(values, probabilities, torch.ones_like(values, dtype=torch.bool))


def test_default_remains_tensor_and_strict_geometry_uses_header(monkeypatch):
    assert FASTConfig().execution == "tensor"
    from fnit.fast import pipeline

    affine = np.diag((1.05, 1.15, 1.25, 1.))
    image = nib.Nifti1Image(np.ones((3, 4, 5), np.float32), affine)
    image.header.set_zooms((1.1, 1.2, 1.3))
    captured = {}

    def capture(image, mask, voxel_size, config):
        captured.update(voxel_size=voxel_size, execution=config.execution)
        raise RuntimeError("captured")

    monkeypatch.setattr(pipeline, "segment_t1", capture)
    with pytest.raises(RuntimeError, match="captured"):
        TorchFAST(execution="fsl")(image)
    np.testing.assert_array_equal(captured["voxel_size"], image.header.get_zooms())
    assert captured["execution"] == "fsl"


@pytest.mark.parametrize("execution,determinant_sign,flipped", [
    ("fsl", 1, True), ("fsl", -1, False), ("tensor", 1, False),
])
def test_strict_adapter_preserves_newimage_internal_scan_direction(
        monkeypatch, execution, determinant_sign, flipped):
    from fnit.fast import pipeline

    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5) + 1
    image = nib.Nifti1Image(data, np.diag((determinant_sign, 1., 1., 1.)))
    mask_data = data % 3 != 0
    mask = nib.Nifti1Image(mask_data.astype(np.uint8), image.affine)

    def capture(tensor, mask_tensor, voxel_size, config):
        expected = np.flip(data, 0).copy() if flipped else data
        expected_mask = np.flip(mask_data, 0).copy() if flipped else mask_data
        np.testing.assert_array_equal(tensor.numpy(), expected)
        np.testing.assert_array_equal(mask_tensor.numpy(), expected_mask)
        return FASTTensorResult(
            pve=tensor[None].expand(3, *tensor.shape),
            hard_segmentation=tensor.to(torch.int32),
            pve_segmentation=tensor.to(torch.int32),
            mixel_type=tensor.to(torch.int32), bias_field=tensor,
            restored=tensor, tissue_means=torch.ones(3),
            tissue_variances=torch.ones(3),
        )

    monkeypatch.setattr(pipeline, "segment_t1", capture)
    result = TorchFAST(execution=execution)(image, mask)
    for field in ("pve_csf", "pve_gm", "pve_wm", "hard_segmentation",
                  "pve_segmentation", "mixel_type", "bias_field", "restored"):
        np.testing.assert_array_equal(np.asarray(getattr(result, field).dataobj), data)
