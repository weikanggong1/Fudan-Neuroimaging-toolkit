"""FSL coordinate and image I/O contracts for the PyTorch FLIRT API."""

import numpy as np
import pytest
import torch

import fnit
from fnit.flirt import FLIRTResult, TorchFLIRT
from fnit.flirt.core import FSLCorrelationRatio
from fnit.synthstrip.geometry import Volume
from fnit.flirt import core as flirt_core
from fnit.flirt.coordinates import (
    flirt_to_world_affine,
    world_to_flirt_affine,
)


def _volume(shape=(7, 8, 9), vox2world=None):
    if vox2world is None:
        vox2world = np.eye(4)
    axes = np.meshgrid(
        *[np.arange(size, dtype=np.float32) for size in shape], indexing="ij"
    )
    center = (np.asarray(shape, dtype=np.float32) - 1) / 2
    data = np.exp(
        -sum((axis - center[index]) ** 2 for index, axis in enumerate(axes)) / 6
    ).astype(np.float32)
    return Volume(data, vox2world)


def test_world_and_flirt_matrix_conversions_are_exact_inverses():
    moving_affine = np.array(
        [[-2, 0, 0, 30], [0, 3, 0, -12], [0, 0, 4, 8], [0, 0, 0, 1]],
        dtype=float,
    )
    fixed_affine = np.array(
        [[1.5, 0, 0, -20], [0, 2.5, 0, 5], [0, 0, 3.5, -9], [0, 0, 0, 1]],
        dtype=float,
    )
    world = np.array(
        [[1.02, 0.03, 0, 7], [-0.01, 0.98, 0.02, -4],
         [0, 0.01, 1.04, 3], [0, 0, 0, 1]],
        dtype=float,
    )
    moving_shape, fixed_shape = (11, 12, 13), (21, 22, 23)

    flirt = world_to_flirt_affine(
        world, moving_affine, fixed_affine, moving_shape, fixed_shape
    )
    recovered = flirt_to_world_affine(
        flirt, moving_affine, fixed_affine, moving_shape, fixed_shape
    )

    np.testing.assert_allclose(recovered, world, atol=1e-12, rtol=1e-12)
    assert fnit.TorchFLIRT is TorchFLIRT
    assert fnit.FLIRTResult is FLIRTResult
    assert fnit.world_to_flirt_affine is world_to_flirt_affine


def test_flirt_result_rejects_one_path_for_image_and_matrix(tmp_path):
    volume = _volume()
    result = FLIRTResult(
        moved=volume,
        matrix=np.eye(4),
        moving_to_fixed_world=np.eye(4),
        fixed_to_moving_world=np.eye(4),
        qc={},
    )
    destination = tmp_path / "output.nii.gz"

    with pytest.raises(ValueError, match="different paths"):
        result.save(output=destination, omat=destination)

    assert not destination.exists()


def test_reference_validation_is_not_reported_as_current_input_equivalence(
        monkeypatch):
    class FakeEngine:
        cost_evaluations = 1

        def __init__(self, *args, **kwargs):
            pass

        def run(self, qsform, dof=12):
            return 0.0, np.eye(4)

    monkeypatch.setattr(flirt_core, "_DefaultFLIRTEngine", FakeEngine)
    monkeypatch.setattr(
        flirt_core,
        "_resample_output",
        lambda moving, fixed_shape, *args, **kwargs: torch.zeros(fixed_shape),
    )

    result = TorchFLIRT(device="cpu", angular_search=False)(_volume(), _volume())

    assert result.qc["reference_validation_matrix_gate_passed"] is True
    assert result.qc["validation_parameter_profile_matches_run"] is False
    assert result.qc["current_input_compared_with_fsl"] is False
    assert result.qc["validated_fsl_equivalent"] is False
    assert result.qc["complete_numerical_equivalence_claimed"] is False


def test_weighted_correlation_ratio_matches_unweighted_for_unit_weights():
    reference = torch.arange(7 * 8 * 9, dtype=torch.float32).reshape(7, 8, 9)
    moving = torch.flip(reference, dims=(0,)) + 1
    kwargs = {
        "reference_voxel_sizes": (1.0, 1.0, 1.0),
        "moving_voxel_sizes": (1.0, 1.0, 1.0),
        "bins": 16,
        "smooth_size": 1.0,
    }
    unweighted = FSLCorrelationRatio(reference, moving, **kwargs)
    weighted = FSLCorrelationRatio(
        reference,
        moving,
        reference_weight=torch.ones_like(reference),
        moving_weight=torch.ones_like(moving),
        **kwargs,
    )
    assert weighted(np.eye(4)) == pytest.approx(
        unweighted(np.eye(4)), abs=1e-7
    )


def test_flirt_rejects_weight_on_a_different_grid():
    image = _volume()
    shifted_geometry = np.eye(4)
    shifted_geometry[0, 3] = 2
    weight = _volume(vox2world=shifted_geometry)

    with pytest.raises(ValueError, match="inweight must match"):
        TorchFLIRT(device="cpu", angular_search=False)(
            image, image, inweight=weight
        )
