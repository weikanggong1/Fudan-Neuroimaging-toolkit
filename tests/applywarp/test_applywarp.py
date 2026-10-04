"""Coordinate, interpolation, header, and CLI tests for TorchApplyWarp."""

import numpy as np
import nibabel as nib
import pytest
import torch

import fnit
from fnit import cli
from fnit.applywarp import ApplyWarpResult, TorchApplyWarp


def _image(data, affine):
    image = nib.Nifti1Image(np.asarray(data), np.asarray(affine, dtype=float))
    image.set_qform(affine, 2)
    image.set_sform(affine, 4)
    return image


def _warp(field, affine, intent=0):
    image = _image(np.asarray(field, dtype=np.float32), affine)
    image.header["intent_code"] = intent
    return image


def _zero_field(shape):
    return np.zeros((*shape, 3), dtype=np.float32)


def _cubic_coefficients(field_shape, knot_spacing, values, embedded_affine=None):
    coefficient_shape = tuple(
        size if spacing == 1 else int(np.ceil((size + 1) / spacing)) + 2
        for size, spacing in zip(field_shape, knot_spacing)
    )
    data = np.zeros((*coefficient_shape, 3), dtype=np.float32)
    data[...] = values
    image = nib.Nifti1Image(data, np.eye(4))
    image.header["intent_code"] = 2007
    image.header["intent_p1"] = 1
    image.header["intent_p2"] = 1
    image.header["intent_p3"] = 1
    image.header["pixdim"][1:4] = knot_spacing
    image.header["qoffset_x"] = field_shape[0]
    image.header["qoffset_y"] = field_shape[1]
    image.header["qoffset_z"] = field_shape[2]
    image.set_sform(
        np.eye(4) if embedded_affine is None else embedded_affine, code=1
    )
    return image


@pytest.mark.parametrize(
    "affine,source_slice,target_slice",
    [
        (np.diag([-2.0, 3.0, 4.0, 1.0]), slice(None, -1), slice(1, None)),
        (np.diag([2.0, 3.0, 4.0, 1.0]), slice(1, None), slice(None, -1)),
    ],
)
def test_relative_x_displacement_obeys_fsl_storage_handedness(
    affine, source_slice, target_slice
):
    shape = (6, 5, 4)
    ramp = np.broadcast_to(
        np.arange(shape[0], dtype=np.float32)[:, None, None], shape
    ).copy()
    field = _zero_field(shape)
    field[..., 0] = 2.0

    result = TorchApplyWarp("cpu")(
        _image(ramp, affine),
        _image(np.zeros(shape, dtype=np.float32), affine),
        warp=_warp(field, affine, 2006),
    )

    expected = np.zeros(shape, dtype=np.float32)
    expected[source_slice] = ramp[target_slice]
    np.testing.assert_allclose(
        np.asarray(result.image.dataobj), expected, atol=1e-6, rtol=0
    )
    assert result.qc["warp_convention"] == "relative"
    assert result.qc["warp_convention_source"] == "FSL intent 2006"


def test_premat_and_postmat_use_fsl_forward_order():
    shape = (8, 5, 4)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    ramp = np.broadcast_to(
        np.arange(shape[0], dtype=np.float32)[:, None, None], shape
    ).copy()
    translate = np.eye(4)
    translate[0, 3] = 1.0

    result = TorchApplyWarp("cpu")(
        _image(ramp, affine),
        _image(np.zeros(shape, dtype=np.float32), affine),
        warp=_warp(_zero_field(shape), affine, 2006),
        premat=translate,
        postmat=translate,
    )

    expected = np.zeros(shape, dtype=np.float32)
    expected[2:] = ramp[:-2]
    np.testing.assert_allclose(
        np.asarray(result.image.dataobj), expected, atol=1e-6, rtol=0
    )


def test_nearest_uses_fsl_positive_half_up_ties():
    shape = (5, 4, 3)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    ramp = np.broadcast_to(
        np.arange(shape[0], dtype=np.float32)[:, None, None], shape
    ).copy()
    coordinates = np.indices(shape, dtype=np.float32)
    field = np.stack(
        (
            np.full(shape, 0.5, dtype=np.float32),
            coordinates[1],
            coordinates[2],
        ),
        axis=-1,
    )

    result = TorchApplyWarp("cpu")(
        _image(ramp, affine),
        _image(np.zeros(shape, dtype=np.float32), affine),
        warp=_warp(field, affine),
        warp_convention="absolute",
        interpolation="nearest",
    )

    np.testing.assert_array_equal(np.asarray(result.image.dataobj), 1)


@pytest.mark.parametrize(
    "affine,source_slice,target_slice",
    [
        (np.diag([-1.0, 1.0, 1.0, 1.0]), slice(1, None), slice(None, -1)),
        (np.diag([1.0, 1.0, 1.0, 1.0]), slice(None, -1), slice(1, None)),
    ],
)
def test_cubic_coefficients_decode_residual_and_embedded_affine(
    affine, source_slice, target_slice
):
    shape = (8, 7, 6)
    ramp = np.broadcast_to(
        np.arange(shape[0], dtype=np.float32)[:, None, None], shape
    ).copy()
    embedded_affine = np.eye(4)
    embedded_affine[0, 3] = 1.0
    coefficients = _cubic_coefficients(
        shape, (4, 4, 4), (0.0, 0.0, 0.0), embedded_affine
    )

    result = TorchApplyWarp("cpu")(
        _image(ramp, affine),
        _image(np.zeros(shape, dtype=np.float32), affine),
        warp=coefficients,
    )

    expected = np.zeros_like(ramp)
    expected[source_slice] = ramp[target_slice]
    np.testing.assert_allclose(
        np.asarray(result.image.dataobj), expected, atol=1e-6, rtol=0
    )
    assert result.qc["warp_representation"] == "FNIRT cubic spline coefficients"
    assert result.qc["warp_convention_source"] == "FSL coefficient intent 2007"


def test_reference_header_and_input_dtype_define_output_contract():
    input_shape = (7, 6, 5)
    reference_shape = (5, 4, 3)
    input_affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    reference_affine = np.array(
        [[-2, 0, 0, 10], [0, 3, 0, -5], [0, 0, 4, 7], [0, 0, 0, 1]],
        dtype=float,
    )
    moving = _image(np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float64), input_affine)
    reference = _image(np.zeros(reference_shape, dtype=np.float32), reference_affine)
    field = _warp(_zero_field(reference_shape), reference_affine, 2006)

    result = TorchApplyWarp("cpu")(moving, reference, warp=field)

    assert isinstance(result, ApplyWarpResult)
    assert fnit.TorchApplyWarp is TorchApplyWarp
    assert result.image.shape == reference_shape
    assert result.image.get_data_dtype() == np.dtype("float64")
    np.testing.assert_allclose(result.image.affine, reference.affine)
    np.testing.assert_allclose(result.image.get_qform(), reference.get_qform())
    np.testing.assert_allclose(result.image.get_sform(), reference.get_sform())
    assert int(result.image.header["qform_code"]) == 2
    assert int(result.image.header["sform_code"]) == 4


def test_small_integer_input_defaults_to_float_but_explicit_dtype_is_honoured():
    shape = (4, 4, 4)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    moving = _image(np.ones(shape, dtype=np.uint8), affine)
    reference = _image(np.zeros(shape, dtype=np.float32), affine)

    default = TorchApplyWarp("cpu")(moving, reference)
    forced = TorchApplyWarp("cpu")(moving, reference, output_dtype="short")

    assert default.image.get_data_dtype() == np.dtype("float32")
    assert forced.image.get_data_dtype() == np.dtype("int16")


def test_explicit_integer_dtype_uses_fsl_truncation():
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    data = np.array([1.9, -1.9], dtype=np.float32)[:, None, None]

    result = TorchApplyWarp("cpu")(
        _image(data, affine), _image(np.zeros_like(data), affine), output_dtype="short"
    )

    np.testing.assert_array_equal(
        np.asarray(result.image.dataobj)[:, 0, 0], np.array([1, -1])
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_explicit_char_keeps_fsl_direct_cast_without_saturation(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # Includes the public T1's observed maximum (2779), as well as the uint8
    # boundary. NEWIMAGE's direct cast truncates, then retains the low byte.
    shape = (2, 2, 2)
    values = np.array([0, 1.9, 254.9, 255.9, 256.1, 257.9, 512.3, 2779],
                      dtype=np.float32).reshape(shape)
    affine = np.diag([-1., 1., 1., 1.])
    result = TorchApplyWarp(device)(
        _image(values, affine), _image(np.zeros(shape, np.float32), affine),
        interpolation="nearest", output_dtype="char",
    )
    expected = np.array([0, 1, 254, 255, 0, 1, 0, 219], dtype=np.uint8).reshape(shape)
    assert result.image.get_data_dtype() == np.dtype("uint8")
    np.testing.assert_array_equal(np.asarray(result.image.dataobj), expected)


def test_many_frame_integer_default_matches_fsl_range_rule():
    shape = (2, 2, 2, 11)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    data = np.zeros(shape, dtype=np.int16)
    data[..., 1] = 1
    data[..., 10] = 200

    result = TorchApplyWarp("cpu")(
        _image(data, affine), _image(np.zeros(shape[:3]), affine)
    )

    assert result.image.get_data_dtype() == np.dtype("float32")


@pytest.mark.parametrize("intent", [2008, 2009])
def test_fnirt_coefficients_are_explicitly_rejected(intent):
    shape = (4, 4, 4)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    image = _image(np.zeros(shape, dtype=np.float32), affine)
    coefficient = _warp(_zero_field(shape), affine, intent)

    with pytest.raises(NotImplementedError, match="cubic"):
        TorchApplyWarp("cpu")(image, image, warp=coefficient)


@pytest.mark.parametrize("interpolation", ["sinc", "spline", "cubic"])
def test_unimplemented_image_interpolation_is_explicitly_rejected(interpolation):
    shape = (4, 4, 4)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    image = _image(np.zeros(shape, dtype=np.float32), affine)

    with pytest.raises(ValueError, match="trilinear or nearest"):
        TorchApplyWarp("cpu")(image, image, interpolation=interpolation)


def test_cli_writes_reference_grid_output(tmp_path):
    shape = (5, 4, 3)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    moving = tmp_path / "moving.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    warp = tmp_path / "warp.nii.gz"
    output = tmp_path / "warped.nii.gz"
    nib.save(_image(np.ones(shape, dtype=np.float32), affine), moving)
    nib.save(_image(np.zeros(shape, dtype=np.float32), affine), reference)
    nib.save(_warp(_zero_field(shape), affine, 2006), warp)

    cli.main([
        "applywarp", "--in", str(moving), "--ref", str(reference),
        "--warp", str(warp), "--out", str(output), "--device", "cpu",
        "--datatype", "float",
    ])

    saved = nib.load(output)
    assert saved.shape == shape
    assert saved.get_data_dtype() == np.dtype("float32")
    np.testing.assert_allclose(saved.affine, affine)
    np.testing.assert_array_equal(np.asarray(saved.dataobj), 1)


def _unchunked_output(plan, image, reference, output_dtype=None):
    """The pre-chunk channel sampler, including its out-of-place mask."""
    import fnit.applywarp.core as core

    data = core._input_data(image)
    frames = torch.as_tensor(
        np.moveaxis(data, -1, 0).copy(), dtype=torch.float32, device=plan.device,
    )
    if plan.interpolation == "trilinear":
        sampled = torch.nn.functional.grid_sample(
            frames[None], plan._grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )[0]
    else:
        indices = torch.floor(plan._coordinates + 0.5).long()
        for axis, size in enumerate(frames.shape[1:]):
            indices[axis].clamp_(0, size - 1)
        flat = (
            indices[0] * frames.shape[2] * frames.shape[3]
            + indices[1] * frames.shape[3] + indices[2]
        ).reshape(-1)
        sampled = frames.reshape(frames.shape[0], -1)[:, flat].reshape(
            frames.shape[0], *reference.shape[:3],
        )
    sampled = sampled * plan._valid.to(sampled.dtype)[None]
    result = np.moveaxis(sampled.detach().cpu().numpy(), 0, -1)
    if image.ndim == 3:
        result = result[..., 0]
    dtype = core._resolve_dtype(image, result, output_dtype)
    return core._output_image(reference, image, result, dtype)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("interpolation", ["trilinear", "nearest"])
@pytest.mark.parametrize("kind", ["dense", "coefficient"])
@pytest.mark.parametrize("frame_chunk_size", [1, 3, 64])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_frame_chunks_preserve_all_voxel_bits_and_output_contract(
    device, interpolation, kind, frame_chunk_size, dtype,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    shape = (7, 6, 5)
    affine = np.diag([-1.0, 1.2, 1.4, 1.0])
    values = np.arange(np.prod(shape) * 11).reshape(*shape, 11)
    data = ((values % 113) - 90).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        data[2, 2, 2, 0] = -0.0
    image = _image(data, affine)
    image.header.set_zooms((*image.header.get_zooms()[:3], 2.7))
    image.header.set_xyzt_units("mm", "sec")
    reference = _image(np.zeros(shape, dtype=np.float32), affine)
    reference.header.extensions.append(nib.nifti1.Nifti1Extension(6, b"chunk contract"))
    if kind == "coefficient":
        warp = _cubic_coefficients(shape, (2, 2, 2), (0.7, -0.25, 0.2))
    else:
        field = np.full((*shape, 3), (0.7, -0.25, 0.2), dtype=np.float32)
        warp = _warp(field, affine, 2006)
    plan = TorchApplyWarp(device, frame_chunk_size=frame_chunk_size).prepare(
        image, reference, warp=warp, interpolation=interpolation,
    )
    expected = _unchunked_output(plan, image, reference)
    actual = plan.apply(image)
    expected_data = np.ascontiguousarray(expected.dataobj)
    actual_data = np.ascontiguousarray(actual.image.dataobj)
    np.testing.assert_array_equal(actual_data.view(np.uint8), expected_data.view(np.uint8))
    assert actual.image.header.binaryblock == expected.header.binaryblock
    assert actual.image.header.extensions == expected.header.extensions
    np.testing.assert_array_equal(actual.image.affine, expected.affine)
    np.testing.assert_array_equal(actual.valid_mask, plan._valid_mask)
    assert actual.image.shape == (*shape, 11)
    assert actual.image.header.get_zooms()[3] == image.header.get_zooms()[3]
    assert actual.image.header.get_xyzt_units() == expected.header.get_xyzt_units()


def test_chunked_nearest_reuses_indices_and_preserves_negative_zero(monkeypatch):
    import fnit.applywarp.core as core

    shape = (5, 4, 3)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    image = _image(-np.ones((*shape, 5), dtype=np.float32), affine)
    reference = _image(np.zeros(shape, dtype=np.float32), affine)
    field = _zero_field(shape)
    field[..., 0] = 0.25
    plan = TorchApplyWarp("cpu", frame_chunk_size=2).prepare(
        image, reference, warp=_warp(field, affine, 2006), interpolation="nearest",
    )
    expected = _unchunked_output(plan, image, reference)
    monkeypatch.setattr(
        core, "_nearest_indices", lambda *args: pytest.fail("nearest indices rebuilt"),
    )
    result = plan.apply(image, frame_chunk_size=3)
    actual = np.asarray(result.image.dataobj)
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(np.uint8),
        np.ascontiguousarray(expected.dataobj).view(np.uint8),
    )
    invalid = ~result.valid_mask
    assert invalid.any()
    assert np.signbit(actual[invalid]).all()


def test_chunked_apply_decodes_input_once_and_uses_channel_chunks(monkeypatch):
    import fnit.applywarp.core as core

    class CountingProxy:
        def __init__(self, data):
            self.data = data
            self.shape = data.shape
            self.ndim = data.ndim
            self.calls = 0

        def __array__(self, dtype=None):
            self.calls += 1
            return np.asarray(self.data, dtype=dtype)

    data = np.arange(4 * 5 * 6 * 7, dtype=np.float32).reshape(4, 5, 6, 7)
    image = _image(data, np.eye(4))
    proxy = CountingProxy(data)
    image._dataobj = proxy
    reference = _image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4))
    channels = []
    original = core.F.grid_sample

    def record_sample(frames, grid, **kwargs):
        assert frames.shape[0] == 1
        channels.append(frames.shape[1])
        return original(frames, grid, **kwargs)

    monkeypatch.setattr(core.F, "grid_sample", record_sample)
    result = TorchApplyWarp("cpu", frame_chunk_size=3)(image, reference)
    assert proxy.calls == 1
    assert channels == [3, 3, 1]
    assert result.image.shape == data.shape


@pytest.mark.parametrize("frame_chunk_size", [0, -1, 1.5, "2", True])
def test_frame_chunk_size_requires_a_positive_integer(frame_chunk_size):
    with pytest.raises(ValueError, match="frame_chunk_size"):
        TorchApplyWarp("cpu", frame_chunk_size=frame_chunk_size)


def test_automatic_cuda_chunk_policy_respects_resident_tensors(monkeypatch):
    from dataclasses import replace
    import fnit.applywarp.core as core

    image = _image(np.ones((4, 5, 6), dtype=np.float32), np.eye(4))
    plan = replace(TorchApplyWarp("cpu").prepare(image, image), device=torch.device("cuda"))
    monkeypatch.setattr(core.torch.cuda, "memory_allocated", lambda device: 0)
    assert plan._frame_chunk(490, (4, 5, 6), None) == 490
    assert plan._frame_chunk(490, (4, 5, 6), 16) == 16
    # The complete real benchmark geometry needs about 1.09 GB by the
    # conservative estimate and should not be split at an arbitrary 64 cap.
    plan = replace(plan, reference_geometry=replace(plan.reference_geometry, shape=(91, 109, 91)))
    input_shape = (104, 104, 72)
    bytes_per_frame = 4 * (np.prod(input_shape) + 2 * np.prod(plan.reference_geometry.shape))
    assert plan._frame_chunk(105, input_shape, None) == 105
    assert plan._frame_chunk(105, input_shape, 32) == 32
    # Larger series remain bounded even when all frames would fit on an H100.
    estimated_chunk = core._FRAME_BUFFER_BUDGET_BYTES // bytes_per_frame
    assert plan._frame_chunk(1000, input_shape, None) == estimated_chunk
    assert estimated_chunk * bytes_per_frame <= 8 * 1024 ** 3
    assert (estimated_chunk + 1) * bytes_per_frame > 8 * 1024 ** 3
    # Resident geometry/other tensors can become the tighter limit.
    monkeypatch.setattr(core.torch.cuda, "memory_allocated", lambda device: 12_000_000_000)
    remaining = 20_000_000_000 - 12_000_000_000 - 512 * 1024 ** 2
    assert plan._frame_chunk(1000, input_shape, None) == remaining // bytes_per_frame
    with pytest.raises(RuntimeError, match="20 GB"):
        plan._frame_chunk(1000, input_shape, 1000)
    monkeypatch.setattr(core.torch.cuda, "memory_allocated", lambda device: 19_900_000_000)
    with pytest.raises(RuntimeError, match="20 GB"):
        plan._frame_chunk(490, (4, 5, 6), None)
