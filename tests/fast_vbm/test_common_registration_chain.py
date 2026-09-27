"""Tests for the backend-independent FastVBM registration chain."""

from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

from fnit._nib import new_image

import fnit.fast_vbm.registration as registration_module
from fnit.fast_vbm.synthmorph_backend import (
    SynthMorphDeformRegistration,
)
from fnit.fnirt import TorchFNIRT
from fnit.flirt.coordinates import (
    voxel_to_fsl_scaled_mm,
    world_to_flirt_affine,
)
from fnit.fast_vbm.registration import (
    _register_gm,
    _fsl_dense_nonlinear_jacobian,
    _pull_ras_to_fsl_fields,
)


def _volume(shape=(7, 8, 9), affine=None):
    if affine is None:
        affine = np.array(
            [
                [1.2, 0.1, 0.0, -8.0],
                [0.0, 1.5, 0.2, 3.0],
                [0.0, 0.0, 1.8, -4.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    grid = np.stack(
        np.meshgrid(
            *(np.arange(size, dtype=np.float32) for size in shape),
            indexing="ij",
        ),
        axis=-1,
    )
    centre = (np.asarray(shape, dtype=np.float32) - 1) / 2
    data = np.exp(-np.square(grid - centre).sum(-1) / 12).astype(np.float32)
    return new_image(data, nib.Nifti1Image(data, affine))


def _pull_from_fsl_residual(moving, fixed, forward_fsl, residual):
    shape = tuple(fixed.shape[:3])
    target_voxels = np.stack(
        np.meshgrid(*(np.arange(size) for size in shape), indexing="ij"),
        axis=-1,
    )
    moving_world = np.asarray(moving.geom.vox2world.matrix)
    fixed_world = np.asarray(fixed.geom.vox2world.matrix)
    moving_fsl = voxel_to_fsl_scaled_mm(
        moving_world, moving.shape[:3], moving.geom.voxsize
    )
    fixed_fsl = voxel_to_fsl_scaled_mm(
        fixed_world, fixed.shape[:3], fixed.geom.voxsize
    )
    target_fsl = np.einsum(
        "ab,...b->...a", fixed_fsl[:3, :3], target_voxels
    ) + fixed_fsl[:3, 3]
    affine_pull = np.linalg.inv(forward_fsl)
    source_fsl = np.einsum(
        "ab,...b->...a", affine_pull[:3, :3], target_fsl
    ) + affine_pull[:3, 3]
    source_fsl += residual
    source_voxels = np.einsum(
        "ab,...b->...a", np.linalg.inv(moving_fsl)[:3, :3], source_fsl
    ) + np.linalg.inv(moving_fsl)[:3, 3]
    source_world = np.einsum(
        "ab,...b->...a", moving_world[:3, :3], source_voxels
    ) + moving_world[:3, 3]
    target_world = np.einsum(
        "ab,...b->...a", fixed_world[:3, :3], target_voxels
    ) + fixed_world[:3, 3]
    return new_image((source_world - target_world).astype(np.float32), fixed)


def test_synthmorph_ras_pull_conversion_recovers_fsl_residual():
    moving = _volume()
    fixed = _volume()
    forward = np.array(
        [
            [1.03, 0.02, 0.00, 1.2],
            [0.01, 0.97, 0.03, -0.7],
            [0.00, 0.01, 1.04, 0.4],
            [0.00, 0.00, 0.00, 1.0],
        ]
    )
    shape = tuple(fixed.shape[:3])
    target_voxels = np.stack(
        np.meshgrid(*(np.arange(size) for size in shape), indexing="ij"),
        axis=-1,
    )
    fixed_fsl = voxel_to_fsl_scaled_mm(
        fixed.geom.vox2world.matrix, shape, fixed.geom.voxsize
    )
    target_fsl = np.einsum(
        "ab,...b->...a", fixed_fsl[:3, :3], target_voxels
    ) + fixed_fsl[:3, 3]
    linear = np.array(
        [[0.015, -0.004, 0.002], [0.003, -0.010, 0.001], [0.0, 0.002, 0.008]]
    )
    residual = np.einsum("ab,...b->...a", linear, target_fsl)
    residual += np.array([0.2, -0.1, 0.05])
    pull = _pull_from_fsl_residual(moving, fixed, forward, residual)

    recovered, dense, recovered_fsl = _pull_ras_to_fsl_fields(
        pull, moving, fixed, forward, device=torch.device("cpu")
    )

    np.testing.assert_allclose(recovered.numpy(), residual, atol=7e-6, rtol=0)
    np.testing.assert_allclose(recovered_fsl, fixed_fsl, atol=0, rtol=0)
    affine_pull = np.linalg.inv(forward)
    affine_source_fsl = np.einsum(
        "ab,...b->...a", affine_pull[:3, :3], target_fsl
    ) + affine_pull[:3, 3]
    expected_dense = affine_source_fsl + residual - target_fsl
    np.testing.assert_allclose(
        dense.numpy(), expected_dense, atol=7e-6, rtol=0
    )


def test_common_dense_jacobian_uses_fsl_residual_and_excludes_affine():
    shape = (7, 8, 9)
    fixed_fsl = np.diag([-1.2, 1.5, 1.8, 1.0])
    fixed_fsl[0, 3] = 1.2 * (shape[0] - 1)
    axes = torch.meshgrid(
        *(torch.arange(size, dtype=torch.float32) for size in shape),
        indexing="ij",
    )
    voxels = torch.stack(axes).reshape(3, -1)
    fsl = torch.as_tensor(fixed_fsl, dtype=torch.float32)
    coordinates = (fsl[:3, :3] @ voxels + fsl[:3, 3:4]).reshape(3, *shape)
    derivative = torch.tensor(
        [[0.02, -0.01, 0.00], [0.01, 0.03, 0.01], [0.00, -0.02, -0.01]],
        dtype=torch.float32,
    )
    residual = torch.einsum("ab,bxyz->axyz", derivative, coordinates)
    jacobian = _fsl_dense_nonlinear_jacobian(
        residual.movedim(0, -1), fixed_fsl
    )
    expected = torch.linalg.det(torch.eye(3) + derivative)

    torch.testing.assert_close(
        jacobian, torch.full(shape, expected), atol=2e-6, rtol=0
    )


def test_real_backend_dispatch_uses_the_same_default_flirt_and_common_tail(
    monkeypatch,
):
    moving = _volume()
    fixed = _volume(
        shape=(8, 7, 9),
        affine=np.array(
            [
                [-1.1, 0.05, 0.0, 7.0],
                [0.0, 1.4, 0.1, -3.0],
                [0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    reference_mask = fixed.new(
        (np.asarray(fixed.data) > 0.15).astype(np.uint8)
    )
    flirt_calls = []
    synth_calls = []
    fnirt_calls = []

    class FakeFLIRT:
        def __init__(self, *, device):
            self.device = device

        def __call__(self, moving_value, fixed_value):
            flirt_calls.append(
                (
                    np.asarray(moving_value.data).copy(),
                    np.asarray(fixed_value.data).copy(),
                )
            )
            forward_world = np.array(
                [
                    [1.02, 0.01, 0.00, 0.3],
                    [0.00, 0.98, 0.02, -0.4],
                    [0.01, 0.00, 1.01, 0.2],
                    [0.00, 0.00, 0.00, 1.0],
                ]
            )
            matrix = world_to_flirt_affine(
                forward_world,
                moving_value.geom.vox2world.matrix,
                fixed_value.geom.vox2world.matrix,
                moving_value.shape[:3],
                fixed_value.shape[:3],
                moving_value.geom.voxsize,
                fixed_value.geom.voxsize,
            )
            return SimpleNamespace(
                matrix=matrix,
                moving_to_fixed_world=forward_world,
                fixed_to_moving_world=np.linalg.inv(forward_world),
                qc={"validated_fsl_equivalent": False},
            )

    def common_residual(fixed_value):
        axes = np.stack(
            np.meshgrid(
                *(
                    np.arange(size, dtype=np.float32)
                    for size in fixed_value.shape[:3]
                ),
                indexing="ij",
            ),
            axis=-1,
        )
        residual = np.empty((*fixed_value.shape[:3], 3), dtype=np.float32)
        residual[..., 0] = 0.15 + 0.012 * axes[..., 0]
        residual[..., 1] = -0.08 + 0.009 * axes[..., 1]
        residual[..., 2] = 0.04 - 0.006 * axes[..., 2]
        return residual

    class FakeSynthMorph(SynthMorphDeformRegistration):
        def __init__(self):
            def model(moving_value, fixed_value, *, init, mid_space):
                synth_calls.append(
                    {
                        "moving": np.asarray(moving_value.data).copy(),
                        "fixed": np.asarray(fixed_value.data).copy(),
                        "initial": np.asarray(init).copy(),
                        "mid_space": mid_space,
                    }
                )
                forward = world_to_flirt_affine(
                    init,
                    moving_value.geom.vox2world.matrix,
                    fixed_value.geom.vox2world.matrix,
                    moving_value.shape[:3],
                    fixed_value.shape[:3],
                    moving_value.geom.voxsize,
                    fixed_value.geom.voxsize,
                )
                residual = common_residual(fixed_value)
                return SimpleNamespace(
                    transform=_pull_from_fsl_residual(
                        moving_value, fixed_value, forward, residual
                    )
                )

            self.synthmorph = model

    class FakeFNIRT(TorchFNIRT):
        def __init__(self):
            pass

        def __call__(
            self, moving_value, fixed_value, initial,
            *, reference_mask,
        ):
            fnirt_calls.append(
                {
                    "moving": np.asarray(moving_value.data).copy(),
                    "fixed": np.asarray(fixed_value.data).copy(),
                    "initial": np.asarray(initial).copy(),
                    "reference_mask": np.asarray(reference_mask.data).copy(),
                }
            )
            forward = world_to_flirt_affine(
                initial,
                moving_value.geom.vox2world.matrix,
                fixed_value.geom.vox2world.matrix,
                moving_value.shape[:3],
                fixed_value.shape[:3],
                moving_value.geom.voxsize,
                fixed_value.geom.voxsize,
            )
            residual = common_residual(fixed_value)
            return SimpleNamespace(
                pull_transform=_pull_from_fsl_residual(
                    moving_value, fixed_value, forward, residual
                ),
                nonlinear_jacobian=fixed_value.new(
                    np.ones(fixed_value.shape[:3], dtype=np.float32)
                ),
                qc={"fsl_fnirt_numerically_equivalent": False},
            )

    monkeypatch.setattr(registration_module, "TorchFLIRT", FakeFLIRT)
    synth_result = _register_gm(
        moving,
        fixed,
        device="cpu",
        reference_mask=reference_mask,
        registration_backend="synthmorph",
        deform_model=FakeSynthMorph(),
    )
    fnirt_result = _register_gm(
        moving,
        fixed,
        device="cpu",
        reference_mask=reference_mask,
        registration_backend="fnirt",
        deform_model=FakeFNIRT(),
    )

    assert len(flirt_calls) == 2
    for index in (0, 1):
        np.testing.assert_array_equal(flirt_calls[0][index], flirt_calls[1][index])
    np.testing.assert_array_equal(synth_calls[0]["moving"], fnirt_calls[0]["moving"])
    np.testing.assert_array_equal(synth_calls[0]["fixed"], fnirt_calls[0]["fixed"])
    np.testing.assert_array_equal(synth_calls[0]["initial"], fnirt_calls[0]["initial"])
    np.testing.assert_array_equal(
        fnirt_calls[0]["reference_mask"], np.asarray(reference_mask.data)
    )
    assert synth_calls[0]["mid_space"] is False
    assert (
        synth_result.qc["pre_nonlinear_signature"]
        == fnirt_result.qc["pre_nonlinear_signature"]
    )
    np.testing.assert_array_equal(
        synth_result.warped_gm.data, fnirt_result.warped_gm.data
    )
    np.testing.assert_array_equal(
        synth_result.jacobian.data, fnirt_result.jacobian.data
    )
    np.testing.assert_array_equal(
        synth_result.modulated_gm.data, fnirt_result.modulated_gm.data
    )
    assert not np.allclose(synth_result.jacobian.data, 1.0)
