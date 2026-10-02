"""Public SynthMorph world-chain dispatch and image-contract coverage."""

from dataclasses import FrozenInstanceError

import nibabel as nib
import numpy as np
import pytest

from fnit._transforms import AffineTransform
from fnit._world_resampling import WorldTransformChain as SharedWorldTransformChain
from fnit.synthmorph import WorldTransformChain, apply_transform
from fnit.synthmorph import pipeline as synthmorph_pipeline


def _image(frames=1):
    shape = (2, 3, 4, frames)
    data = np.linspace(-3, 3, np.prod(shape), dtype=np.float32).reshape(shape)
    data.flat[0] = np.float32(-0.0)
    affine = np.diag([1.25, 1.5, 2.0, 1.0])
    affine[:3, 3] = [-2, 4, 8]
    image = nib.Nifti1Image(data, affine)
    image.header.set_zooms((1.25, 1.5, 2.0, 0.735))
    image.header.set_xyzt_units("mm", "sec")
    image.header["toffset"] = 0.4
    image.set_qform(affine, code=0)
    image.set_sform(affine, code=4)
    image.header.extensions.append(nib.nifti1.Nifti1Extension(6, b"contract metadata"))
    return image


def _reject_materialization(*args, **kwargs):
    raise AssertionError("world-chain dispatch must precede legacy image materialization")


def test_world_transform_chain_export_and_frozen_contract():
    assert WorldTransformChain is SharedWorldTransformChain
    reference = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), np.eye(4))
    chain = WorldTransformChain(reference, np.eye(4))
    assert chain.pre_affine_pull_ras is None
    assert chain.motion_pull_world is None
    assert chain.coordinate_precision == "float64"
    with pytest.raises(FrozenInstanceError):
        chain.coordinate_precision = "fmriprep"


@pytest.mark.parametrize("frames", [1, 4])
@pytest.mark.parametrize(
    "method, chunk, boundary, precision",
    [
        ("linear", None, "grid-constant", "float64"),
        ("nearest", np.int64(2), "periodic", "float64"),
        ("spline", 3, "grid-constant", "fmriprep"),
    ],
)
def test_world_chain_forwards_composition_and_preserves_default_image(
    monkeypatch, frames, method, chunk, boundary, precision,
):
    source = _image(frames)
    reference = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), np.eye(4))
    affine = np.eye(4)
    affine[:3, 3] = [0.2, -0.3, 0.4]
    motion = np.broadcast_to(np.eye(4), (frames, 4, 4)).copy()
    motion[:, 0, 3] = np.arange(frames) * 0.1
    pull = nib.Nifti1Image(np.zeros((2, 3, 4, 3), dtype=np.float32), reference.affine)
    mask = nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.uint8), reference.affine)
    chain = WorldTransformChain(reference, affine, pull, motion, precision)
    sampled_image = _image(frames)
    before_bits = np.asarray(sampled_image.dataobj).view(np.uint32).copy()
    before_header = sampled_image.header.binaryblock
    before_extensions = [extension.get_content() for extension in sampled_image.header.extensions]
    received = {}

    def sample(image, target, reference_to_source_world, **kwargs):
        received.update(kwargs)
        assert image is source
        assert target is chain.reference
        np.testing.assert_array_equal(reference_to_source_world, affine)
        return sampled_image

    monkeypatch.setattr(synthmorph_pipeline, "_load", _reject_materialization)
    monkeypatch.setattr(synthmorph_pipeline, "resample_world_image", sample)
    actual = apply_transform(
        source, chain, method=method, boundary=boundary, output_mask=mask,
        frame_chunk_size=chunk, spatial_chunk_size=17, device="cpu",
    )
    assert received["pre_affine_pull_ras"] is chain.pre_affine_pull_ras
    assert received["motion_pull_world"] is chain.motion_pull_world
    np.testing.assert_array_equal(received["motion_pull_world"], motion)
    assert received["output_mask"] is mask
    assert received["interpolation"] == method
    assert received["boundary"] == boundary
    assert received["coordinate_precision"] == precision
    assert received["spatial_chunk_size"] == 17
    assert received["batch_size"] == (8 if chunk is None else int(chunk))
    assert received["device"] == "cpu"
    # Returning this exact object also avoids an extra header reconstruction.
    assert actual is sampled_image
    np.testing.assert_array_equal(np.asarray(actual.dataobj).view(np.uint32), before_bits)
    assert actual.header.binaryblock == before_header
    assert [extension.get_content() for extension in actual.header.extensions] == before_extensions
    assert actual.shape == (2, 3, 4, frames)
    assert actual.header.get_zooms()[-1] == np.float32(0.735)
    assert actual.header.get_xyzt_units() == ("mm", "sec")


def test_world_chain_passes_unread_paths_and_default_batch_to_shared_sampler(monkeypatch):
    source_path = "unread_source.nii.gz"
    reference_path = "unread_reference.nii.gz"
    chain = WorldTransformChain(reference_path, np.eye(4))
    sampled_image = _image()

    def sample(source, reference, matrix, **kwargs):
        assert source == source_path
        assert reference == reference_path
        np.testing.assert_array_equal(matrix, np.eye(4))
        assert kwargs["batch_size"] == 8
        assert kwargs["device"] == "cpu"
        assert kwargs["pre_affine_pull_ras"] is None
        assert kwargs["motion_pull_world"] is None
        assert kwargs["output_mask"] is None
        assert kwargs["boundary"] == "grid-constant"
        assert kwargs["coordinate_precision"] == "float64"
        return sampled_image

    monkeypatch.setattr(synthmorph_pipeline, "_load", _reject_materialization)
    monkeypatch.setattr(synthmorph_pipeline, "resample_world_image", sample)
    assert apply_transform(source_path, chain) is sampled_image


@pytest.mark.parametrize("dtype", ["float64", "int16"])
def test_world_chain_nondefault_dtype_changes_only_output_cast(monkeypatch, dtype):
    image = _image(frames=3)
    reference = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), image.affine)
    chain = WorldTransformChain(reference, np.eye(4))
    header_before = image.header.binaryblock
    monkeypatch.setattr(synthmorph_pipeline, "_load", _reject_materialization)
    monkeypatch.setattr(synthmorph_pipeline, "resample_world_image", lambda *a, **k: image)
    actual = apply_transform(image, chain, dtype=dtype)
    np.testing.assert_array_equal(actual.dataobj, np.asarray(image.dataobj).astype(dtype))
    np.testing.assert_array_equal(actual.affine, image.affine)
    np.testing.assert_array_equal(actual.get_qform(), image.get_qform())
    np.testing.assert_array_equal(actual.get_sform(), image.get_sform())
    assert actual.header["qform_code"] == image.header["qform_code"]
    assert actual.header["sform_code"] == image.header["sform_code"]
    assert actual.header.get_zooms() == image.header.get_zooms()
    assert actual.header.get_xyzt_units() == image.header.get_xyzt_units()
    assert actual.header["toffset"] == image.header["toffset"]
    assert actual.header.extensions[0].get_content() == image.header.extensions[0].get_content()
    assert actual.get_data_dtype() == np.dtype(dtype)
    assert image.header.binaryblock == header_before
    assert image.get_data_dtype() == np.dtype(np.float32)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"header_only": True}, "header_only"),
        ({"fill": 1}, "fill=0"),
        ({"fill": -1}, "fill=0"),
        ({"fill": None}, "fill=0"),
        ({"fill": np.nan}, "fill=0"),
        ({"method": "cubic"}, "method"),
        ({"frame_chunk_size": 0}, "frame_chunk_size"),
        ({"frame_chunk_size": -1}, "frame_chunk_size"),
        ({"frame_chunk_size": 1.5}, "frame_chunk_size"),
        ({"frame_chunk_size": True}, "frame_chunk_size"),
    ],
)
def test_world_chain_rejects_unsupported_parameters_before_loading(
    monkeypatch, arguments, message,
):
    image = _image()
    reference = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), image.affine)
    chain = WorldTransformChain(reference, np.eye(4))
    monkeypatch.setattr(synthmorph_pipeline, "_load", _reject_materialization)
    monkeypatch.setattr(synthmorph_pipeline, "resample_world_image", _reject_materialization)
    with pytest.raises(ValueError, match=message):
        apply_transform(image, chain, **arguments)


@pytest.mark.parametrize(
    "arguments",
    [{"boundary": "periodic"}, {"output_mask": "mask.nii.gz"}, {"spatial_chunk_size": 17}],
)
def test_world_only_options_are_not_silently_ignored_for_affine(monkeypatch, arguments):
    image = _image()
    affine = AffineTransform(np.eye(4), source=image, target=image, space="world")
    monkeypatch.setattr(synthmorph_pipeline, "_load", _reject_materialization)
    with pytest.raises(ValueError, match="require WorldTransformChain"):
        apply_transform(image, affine, **arguments)
