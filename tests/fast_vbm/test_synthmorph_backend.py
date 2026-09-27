"""Analytic geometry tests for the FastVBM SynthMorph backend."""

from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fnit._nib import new_image
from fnit._transforms import AffineTransform

from fnit.fast_vbm.synthmorph_backend import (
    SynthMorphDeformRegistration,
)


def _volume(shape=(5, 6, 7), vox2world=None, value=1.0):
    if vox2world is None:
        vox2world = np.eye(4)
    data = np.full(shape, value, dtype=np.float32)
    return new_image(data, nib.Nifti1Image(data, vox2world))


def _linear_pull_warp(moving, fixed, pull_linear):
    shape = tuple(fixed.shape[:3])
    indices = np.stack(
        np.meshgrid(*[np.arange(size) for size in shape], indexing="ij"), axis=-1
    )
    fixed_matrix = np.asarray(fixed.geom.vox2world.matrix)
    world = np.einsum("ab,...b->...a", fixed_matrix[:3, :3], indices)
    world += fixed_matrix[:3, 3]
    source_world = np.einsum("ab,...b->...a", pull_linear, world)
    displacement = (source_world - world).astype(np.float32)
    return new_image(displacement, fixed)


class _FakeSynthMorph:
    def __init__(self, pull_linear, moved_value=2.0):
        self.pull_linear = np.asarray(pull_linear)
        self.moved_value = moved_value
        self.calls = []

    def __call__(self, moving, fixed, **kwargs):
        self.calls.append((moving, fixed, kwargs))
        return SimpleNamespace(
            moved=fixed.new(
                np.full(fixed.shape[:3], self.moved_value, dtype=np.float32)
            ),
            transform=_linear_pull_warp(moving, fixed, self.pull_linear),
        )


def _run(pull_linear, moving_to_fixed=None, fixed_affine=None, moved_value=2.0):
    moving = _volume()
    fixed = _volume(vox2world=fixed_affine)
    if moving_to_fixed is None:
        moving_to_fixed = np.eye(4)
    initial = np.asarray(moving_to_fixed, dtype=np.float64)
    fake = _FakeSynthMorph(pull_linear, moved_value=moved_value)
    result = SynthMorphDeformRegistration(
        device="cpu", synthmorph=fake
    )(moving, fixed, initial)
    return result, fake, initial


def test_identity_affine_and_identity_deformation():
    result, fake, initial = _run(np.eye(3))

    np.testing.assert_allclose(result.full_pull_jacobian.data, 1, atol=1e-6)
    np.testing.assert_allclose(result.nonlinear_jacobian.data, 1, atol=1e-6)
    np.testing.assert_allclose(result.modulated_gm.data, 2, atol=1e-6)
    assert result.affine_pull_determinant == pytest.approx(1)
    np.testing.assert_allclose(fake.calls[0][2]["init"], initial)
    assert fake.calls[0][2]["mid_space"] is False
    assert result.qc["full_pull_jacobian_convention"].startswith("det(")
    assert result.qc["output_jacobian_convention"].startswith("full pull")


def test_default_backend_builds_the_official_deform_model(monkeypatch):
    captured = {}

    class FakeConstructor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "fnit.synthmorph.SynthMorph", FakeConstructor
    )
    model = SynthMorphDeformRegistration(
        weights="checkpoint.h5",
        device="cpu",
        extent=192,
        hyper=0.4,
        steps=6,
    )

    assert isinstance(model.synthmorph, FakeConstructor)
    assert captured == {
        "weights": "checkpoint.h5",
        "device": model.device,
        "model": "deform",
        "extent": 192,
        "hyper": 0.4,
        "steps": 6,
    }


def test_diagonal_affine_is_removed_from_modulation_jacobian():
    forward = np.diag([2.0, 1.0, 1.0, 1.0])
    result, _, _ = _run(np.diag([0.5, 1.0, 1.0]), forward)

    np.testing.assert_allclose(result.full_pull_jacobian.data, 0.5, atol=1e-6)
    np.testing.assert_allclose(result.nonlinear_jacobian.data, 1, atol=1e-6)
    np.testing.assert_allclose(result.modulated_gm.data, 2, atol=1e-6)
    assert result.affine_pull_determinant == pytest.approx(0.5)


def test_affine_and_nonlinear_factors_are_separated():
    forward = np.diag([2.0, 1.0, 1.0, 1.0])
    full_pull = np.diag([0.6, 0.8, 1.1])
    result, _, _ = _run(full_pull, forward)

    np.testing.assert_allclose(result.full_pull_jacobian.data, 0.528, atol=2e-6)
    np.testing.assert_allclose(result.nonlinear_jacobian.data, 1.056, atol=4e-6)
    np.testing.assert_allclose(result.modulated_gm.data, 2.112, atol=8e-6)


@pytest.mark.parametrize("handedness", [1.0, -1.0])
def test_world_derivative_handles_oblique_anisotropic_target_geometry(handedness):
    angle = np.deg2rad(27)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, handedness],
        ]
    )
    shear = np.array([[1, 0.12, -0.07], [0, 1, 0.09], [0, 0, 1]])
    fixed_affine = np.eye(4)
    fixed_affine[:3, :3] = rotation @ np.diag([0.8, 1.7, 2.4]) @ shear
    fixed_affine[:3, 3] = [8.0, -13.0, 4.5]
    pull = np.array([[1.1, 0.04, 0.0], [0.0, 0.9, -0.03], [0.02, 0.0, 1.2]])
    expected = np.linalg.det(pull)

    result, _, _ = _run(pull, fixed_affine=fixed_affine)

    np.testing.assert_allclose(
        result.full_pull_jacobian.data, expected, atol=3e-5
    )
    np.testing.assert_allclose(
        result.nonlinear_jacobian.data, expected, atol=3e-5
    )


def test_negative_folding_is_reported_without_absolute_value_or_clipping():
    result, _, _ = _run(np.diag([-0.25, 1.0, 1.0]), moved_value=4.0)

    np.testing.assert_allclose(result.full_pull_jacobian.data, -0.25, atol=1e-6)
    np.testing.assert_allclose(result.nonlinear_jacobian.data, -0.25, atol=1e-6)
    np.testing.assert_allclose(result.modulated_gm.data, -1.0, atol=1e-6)
    voxel_count = int(np.prod(result.nonlinear_jacobian.shape[:3]))
    assert result.qc["nonpositive_full_pull_jacobian_voxels"] == voxel_count
    assert result.qc["nonpositive_nonlinear_jacobian_voxels"] == voxel_count


def test_initial_affine_is_checked_before_model_call():
    moving = _volume()
    fixed = _volume()
    invalid = np.eye(4)
    invalid[3, 0] = 1
    fake = _FakeSynthMorph(np.eye(3))

    with pytest.raises(ValueError, match="homogeneous"):
        SynthMorphDeformRegistration(synthmorph=fake)(moving, fixed, invalid)
    assert fake.calls == []


def test_geometry_tagged_initial_affine_checks_source_before_model_call():
    moving = _volume()
    fixed = _volume()
    wrong = _volume(shape=(6, 6, 7))
    initial = AffineTransform(
        np.eye(4), source=wrong, target=fixed, space="world"
    )
    fake = _FakeSynthMorph(np.eye(3))

    with pytest.raises(ValueError, match="source geometry"):
        SynthMorphDeformRegistration(synthmorph=fake)(moving, fixed, initial)
    assert fake.calls == []
