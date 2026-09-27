"""Geometry and file-contract tests for nibabel-native SynthMorph outputs."""

import nibabel as nib
import numpy as np

from fnit._nib import FNITNifti1Image
from fnit._transforms import (
    AffineTransform,
    DenseWarp,
    load_lta,
    ras_displacement_to_voxel,
    voxel_displacement_to_ras,
)
from fnit.synthmorph import apply_transform, network_space


def _image(shape=(5, 6, 7), affine=None, frames=1):
    if affine is None:
        affine = np.eye(4)
    values = np.arange(np.prod(shape) * frames, dtype=np.float32)
    data = values.reshape((*shape, frames)) if frames > 1 else values.reshape(shape)
    return FNITNifti1Image(data, affine)


def test_network_space_uses_lia_and_freesurfer_center():
    affine = np.array(
        [[1.2, 0, 0, -7], [0, 1.3, 0, 3], [0, 0, 1.4, 9], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    image = _image((10, 12, 14), affine)
    shape = (192, 192, 192)

    network_to_image, image_to_network = network_space(image, shape)
    network_affine = affine @ network_to_image

    np.testing.assert_allclose(
        image_to_network @ network_to_image, np.eye(4), atol=2e-14
    )
    np.testing.assert_allclose(
        network_affine[:3, :3],
        [[-1, 0, 0], [0, 0, 1], [0, -1, 0]],
    )
    np.testing.assert_allclose(
        nib.affines.apply_affine(network_affine, np.asarray(shape) / 2),
        nib.affines.apply_affine(affine, np.asarray(image.shape[:3]) / 2),
    )


def test_lta_round_trip_preserves_matrix_and_geometries(tmp_path):
    source = _image(
        affine=np.array(
            [[-1.2, 0, 0, 4], [0, 1.3, 0, -2], [0, 0, 1.4, 9], [0, 0, 0, 1]]
        )
    )
    target = _image(
        (7, 8, 9),
        np.array(
            [[2, 0, 0, -10], [0, 2, 0, 5], [0, 0, 2, 3], [0, 0, 0, 1]]
        ),
    )
    matrix = np.array(
        [[1.01, 0.02, 0, 3], [0, 0.99, -0.01, -4], [0, 0, 1.02, 2], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    transform = AffineTransform(matrix, source=source, target=target, space="world")
    path = tmp_path / "registration.lta"

    transform.save(path)
    loaded = load_lta(path)

    np.testing.assert_allclose(np.asarray(loaded, dtype=np.float32), matrix)
    np.testing.assert_allclose(loaded.source.affine, source.affine)
    np.testing.assert_allclose(loaded.target.affine, target.affine)
    np.testing.assert_allclose(
        loaded.convert(space="voxel").convert(space="world").matrix,
        matrix,
        atol=1e-12,
    )


def test_voxel_and_ras_displacements_are_inverse_conversions():
    source_affine = np.array(
        [[-1.2, 0.1, 0, 7], [0, 1.4, 0.2, -3], [0, 0, 1.6, 5], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    target_affine = np.array(
        [[2, 0, 0, -4], [0, 1.8, 0, 6], [0, 0, 2.2, -8], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    source = _image((7, 8, 9), source_affine)
    target = _image((4, 5, 6), target_affine)
    displacement = np.linspace(-0.4, 0.6, 4 * 5 * 6 * 3, dtype=np.float32)
    displacement = displacement.reshape(4, 5, 6, 3)

    ras = voxel_displacement_to_ras(displacement, source, target)
    recovered = ras_displacement_to_voxel(ras, source, target)

    np.testing.assert_allclose(recovered, displacement, atol=4e-6, rtol=0)


def test_apply_nearest_uses_surfa_half_up_ties():
    data = np.arange(3, dtype=np.float32).reshape(3, 1, 1)
    image = FNITNifti1Image(data, np.eye(4))
    target = _image((3, 1, 1))

    forward = np.eye(4)
    forward[0, 3] = -0.5
    affine = AffineTransform(
        forward, source=image, target=target, space="world"
    )
    affine_result = apply_transform(
        image, affine, method="nearest", fill=-9
    )
    assert np.asarray(affine_result.dataobj)[0, 0, 0] == 1

    displacement = np.zeros((*target.shape[:3], 3), dtype=np.float32)
    displacement[..., 0] = 0.5
    warp = DenseWarp(
        displacement, source=image, target=target
    )
    warp_result = apply_transform(
        image, warp, method="nearest", fill=-9
    )
    assert np.asarray(warp_result.dataobj)[0, 0, 0] == 1


def test_apply_nearest_uses_surfa_upper_boundary_domain():
    data = np.arange(3, dtype=np.float32).reshape(3, 1, 1)
    image = FNITNifti1Image(data, np.eye(4))
    target = _image((3, 1, 1))

    for pull_shift, expected in ((0.25, 2), (1.0, -9)):
        forward = np.eye(4)
        forward[0, 3] = -pull_shift
        affine = AffineTransform(
            forward, source=image, target=target, space="world"
        )
        result = apply_transform(
            image, affine, method="nearest", fill=-9
        )
        assert np.asarray(result.dataobj)[2, 0, 0] == expected


def test_apply_transform_handles_affine_dense_and_header_only():
    image = _image((4, 5, 6), frames=2)
    target = _image((4, 5, 6))
    identity = AffineTransform(
        np.eye(4), source=image, target=target, space="world"
    )

    affine_result = apply_transform(image, identity, method="nearest")
    np.testing.assert_array_equal(affine_result.dataobj, image.dataobj)
    np.testing.assert_allclose(affine_result.affine, target.affine)

    warp = DenseWarp(
        np.zeros((*target.shape[:3], 3), dtype=np.float32),
        source=image,
        target=target,
    )
    warp_result = apply_transform(image, warp, method="nearest")
    np.testing.assert_array_equal(warp_result.dataobj, image.dataobj)

    translated = np.eye(4)
    translated[0, 3] = 3
    header = apply_transform(
        image,
        AffineTransform(translated, source=image, target=target, space="world"),
        header_only=True,
    )
    np.testing.assert_array_equal(header.dataobj, image.dataobj)
    np.testing.assert_allclose(header.affine, translated @ image.affine)
