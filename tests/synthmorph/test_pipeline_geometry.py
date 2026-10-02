"""Geometry and file-contract tests for nibabel-native SynthMorph outputs."""

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit._nib import FNITNifti1Image
from fnit._transforms import (
    AffineTransform,
    DenseWarp,
    load_lta,
    ras_displacement_to_voxel,
    voxel_displacement_to_ras,
)
from fnit.synthmorph import apply_transform, network_space
from fnit.synthmorph import pipeline as synthmorph_pipeline
from fnit.synthmorph.pipeline import _resampled_image


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


@pytest.mark.parametrize("method", ["linear", "nearest"])
@pytest.mark.parametrize("kind", ["affine", "dense"])
@pytest.mark.parametrize("fill", [-7, None])
@pytest.mark.parametrize("chunk_size", [1, 2, None])
def test_apply_frame_chunks_match_previous_cpu_sampler(method, kind, fill, chunk_size):
    # Mixed voxel geometries, frames and noninteger locations exercise both
    # interpolation and whole-sample fill. The previous one-shot sampling
    # path is retained by registration and provides the dataflow reference.
    random = np.random.default_rng(4201)
    source_affine = np.diag([1.2, 1.4, 1.6, 1.0])
    target_affine = np.diag([1.1, 1.3, 1.5, 1.0])
    source_affine[:3, 3] = [-2, 1, 3]
    target_affine[:3, 3] = [-1.7, 1.4, 3.2]
    data = random.normal(size=(7, 8, 9, 5)).astype(np.float64)
    image = FNITNifti1Image(data, source_affine)
    image.header.set_zooms((1.2, 1.4, 1.6, 2.5))
    image.header["toffset"] = 0.7
    target = _image((6, 7, 8), target_affine)
    if kind == "affine":
        forward = np.eye(4)
        forward[:3, :3] = [[1.01, 0.01, 0], [0, 0.99, -0.02], [0.01, 0, 1]]
        forward[:3, 3] = [0.4, -0.2, 0.1]
        transformation = AffineTransform(
            forward, source=image, target=target, space="world"
        )
        pull = np.linalg.inv(image.affine) @ np.linalg.inv(forward) @ target.affine
    else:
        displacement = random.uniform(-0.4, 0.4, size=(*target.shape[:3], 3))
        transformation = DenseWarp(displacement, source=image, target=target)
        pull = torch.as_tensor(ras_displacement_to_voxel(
            transformation.dataobj, image, target
        )).permute(3, 0, 1, 2)[None]
    previous = _resampled_image(
        image, pull, target, "cpu", method=method, fill=fill,
        surfa_nearest_rule=True,
    )
    actual = apply_transform(
        image, transformation, method=method, fill=fill,
        frame_chunk_size=chunk_size,
    )
    np.testing.assert_array_equal(actual.dataobj, previous.dataobj)
    np.testing.assert_array_equal(actual.affine, previous.affine)
    np.testing.assert_array_equal(actual.header.binaryblock, previous.header.binaryblock)


@pytest.mark.parametrize("header_only", [False, True])
def test_apply_decodes_image_proxy_once(header_only):
    data = np.arange(4 * 5 * 6 * 3, dtype=np.float32).reshape(4, 5, 6, 3)
    image = FNITNifti1Image(data, np.eye(4))

    class CountingProxy:
        shape = data.shape
        reads = 0

        def __array__(self, dtype=None):
            self.reads += 1
            return np.asarray(data, dtype=dtype)

    proxy = CountingProxy()
    image._dataobj = proxy
    target = _image((4, 5, 6))
    affine = AffineTransform(np.eye(4), source=image, target=target, space="world")
    actual = apply_transform(
        image, affine, method="nearest", header_only=header_only,
        frame_chunk_size=1,
    )
    assert proxy.reads == 1
    np.testing.assert_array_equal(actual.dataobj, data)


def test_apply_preserves_singleton_frame_dimension():
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6, 1)
    image = FNITNifti1Image(data, np.eye(4))
    target = _image((4, 5, 6))
    affine = AffineTransform(np.eye(4), source=image, target=target, space="world")
    actual = apply_transform(image, affine, method="nearest", frame_chunk_size=1)
    assert actual.shape == data.shape
    np.testing.assert_array_equal(actual.dataobj, data)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_apply_rejects_invalid_frame_chunk_size(chunk_size):
    image = _image()
    affine = AffineTransform(np.eye(4), source=image, target=image, space="world")
    with pytest.raises(ValueError, match="frame_chunk_size"):
        apply_transform(image, affine, frame_chunk_size=chunk_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("method", ["linear", "nearest"])
@pytest.mark.parametrize("kind", ["affine", "dense"])
def test_apply_cuda_chunks_match_cuda_one_shot(method, kind):
    image = _image((7, 8, 9), frames=5)
    target = _image((6, 7, 8))
    displacement = np.zeros((*target.shape[:3], 3), dtype=np.float32)
    displacement[..., 0] = 0.3
    displacement[..., 1] = -0.2
    if kind == "dense":
        transformation = DenseWarp(displacement, source=image, target=target)
    else:
        forward = np.eye(4)
        forward[:3, :3] = [[1.01, 0.01, 0], [0, 0.99, -0.02], [0.01, 0, 1]]
        forward[:3, 3] = [-0.3, 0.2, 0.1]
        transformation = AffineTransform(
            forward, source=image, target=target, space="world"
        )
    one_shot = apply_transform(
        image, transformation, method=method, device="cuda", frame_chunk_size=5,
    )
    chunked = apply_transform(
        image, transformation, method=method, device="cuda", frame_chunk_size=2,
    )
    np.testing.assert_array_equal(chunked.dataobj, one_shot.dataobj)
    np.testing.assert_array_equal(chunked.affine, one_shot.affine)
    if method == "nearest":
        cpu = apply_transform(image, transformation, method=method)
        np.testing.assert_array_equal(chunked.dataobj, cpu.dataobj)


@pytest.mark.parametrize("method", ["linear", "nearest"])
@pytest.mark.parametrize("frames, expected", [(32, [32]), (33, [32, 1]), (65, [32, 32, 1])])
def test_apply_cpu_default_chunks_at_32_frames(monkeypatch, method, frames, expected):
    image = _image((3, 4, 5), frames=frames)
    target = _image((3, 4, 5))
    affine = AffineTransform(np.eye(4), source=image, target=target, space="world")
    previous = _resampled_image(
        image, np.eye(4), target, "cpu", method=method, surfa_nearest_rule=True,
    )
    chunks = []
    sample = synthmorph_pipeline._sample_prepared

    def record_sample(tensor, plan, fill):
        chunks.append(tensor.shape[1])
        return sample(tensor, plan, fill)

    monkeypatch.setattr(synthmorph_pipeline, "_sample_prepared", record_sample)
    actual = apply_transform(image, affine, method=method)
    assert chunks == expected
    np.testing.assert_array_equal(actual.dataobj, previous.dataobj)
    np.testing.assert_array_equal(actual.header.binaryblock, previous.header.binaryblock)


@pytest.fixture
def mocked_cuda_frame_sampler(monkeypatch):
    # Exercise allocation decisions without a GPU: leave sampling on CPU and
    # replace only the CUDA transfer/sampler/allocation APIs with observables.
    # The allocation query must follow coordinate residency, even on errors.
    state = {"available": 0, "chunks": [], "events": []}
    as_tensor = torch.as_tensor

    class CoordinatePlan:
        def to(self, device):
            assert device.type == "cuda"
            state["events"].append("plan_to_cuda")
            return self

    def prepare(*args, **kwargs):
        assert kwargs["device"] == "cpu"
        assert kwargs["dtype"] == torch.float32
        state["events"].append("prepare_cpu")
        return CoordinatePlan()

    def memory_allocated(device):
        assert device.type == "cuda"
        assert state["events"] == ["prepare_cpu", "plan_to_cuda"]
        state["events"].append("allocation_query")
        return 20_000_000_000 - 512 * 1024 ** 2 - state["available"]

    def sample(tensor, plan, fill):
        assert state["events"][-1] == "allocation_query"
        count = tensor.shape[1]
        state["chunks"].append(count)
        return torch.zeros((1, count, 2, 3, 4), dtype=torch.float32)

    def host_as_tensor(data, *, device):
        assert device.type == "cuda"
        return as_tensor(data)

    monkeypatch.setattr(synthmorph_pipeline, "_prepare_transform", prepare)
    monkeypatch.setattr(synthmorph_pipeline, "_sample_prepared", sample)
    monkeypatch.setattr(torch.cuda, "memory_allocated", memory_allocated)
    monkeypatch.setattr(torch, "as_tensor", host_as_tensor)
    return state


@pytest.mark.parametrize(
    "requested, available_frames, expected",
    [
        (None, 3, [3, 2]),
        (None, 1, [1, 1, 1, 1, 1]),
        (3, 3, [3, 2]),
        # Clamp an explicit request to the actual frame count before checking.
        (99, 5, [5]),
    ],
)
def test_apply_cuda_budget_uses_resident_plan_and_frame_count(
    mocked_cuda_frame_sampler, requested, available_frames, expected,
):
    image = _image((2, 3, 4), frames=5)
    target = _image((2, 3, 4))
    frame_bytes = 4 * (24 + 2 * 24)
    state = mocked_cuda_frame_sampler
    state["available"] = frame_bytes * available_frames
    synthmorph_pipeline._resampled_frames(
        image, np.asarray(image.dataobj), np.eye(4), target, "cuda:0",
        method="linear", fill=0, frame_chunk_size=requested,
    )
    assert state["chunks"] == expected


@pytest.mark.parametrize(
    "requested, available_frames, short_by",
    [(None, 1, 1), (None, 0, 0), (3, 3, 1), (99, 4, 0)],
)
def test_apply_cuda_rejects_frame_buffers_above_remaining_budget(
    mocked_cuda_frame_sampler, requested, available_frames, short_by,
):
    image = _image((2, 3, 4), frames=5)
    target = _image((2, 3, 4))
    state = mocked_cuda_frame_sampler
    state["available"] = 4 * (24 + 2 * 24) * available_frames - short_by
    with pytest.raises(RuntimeError, match="CUDA.*20 GB|CUDA 20 GB"):
        synthmorph_pipeline._resampled_frames(
            image, np.asarray(image.dataobj), np.eye(4), target, "cuda:0",
            method="linear", fill=0, frame_chunk_size=requested,
        )
    assert state["chunks"] == []


@pytest.mark.parametrize("requested, expected", [(None, [15, 5]), (20, [20])])
def test_apply_cuda_auto_caps_changing_buffers_at_8_gib(
    mocked_cuda_frame_sampler, requested, expected,
):
    # A writable zero-stride view supplies a large source geometry without
    # allocating/reading large data. Mock sampling emits only a small target.
    value = np.zeros(1, dtype=np.float32)
    data = np.lib.stride_tricks.as_strided(
        value, shape=(512, 512, 512, 20), strides=(0, 0, 0, 0), writeable=True,
    )
    image = FNITNifti1Image(data, np.eye(4))
    target = _image((2, 3, 4))
    state = mocked_cuda_frame_sampler
    state["available"] = 12 * 1024 ** 3
    synthmorph_pipeline._resampled_frames(
        image, data, np.eye(4), target, "cuda:0",
        method="linear", fill=0, frame_chunk_size=requested,
    )
    assert state["chunks"] == expected


def test_apply_cuda_auto_rejects_single_frame_above_8_gib(mocked_cuda_frame_sampler):
    value = np.zeros(1, dtype=np.float32)
    data = np.lib.stride_tricks.as_strided(
        value, shape=(1536, 1536, 1536, 1), strides=(0, 0, 0, 0), writeable=True,
    )
    image = FNITNifti1Image(data, np.eye(4))
    target = _image((2, 3, 4))
    state = mocked_cuda_frame_sampler
    state["available"] = 16 * 1024 ** 3
    with pytest.raises(RuntimeError, match="automatic CUDA.*cannot fit one frame"):
        synthmorph_pipeline._resampled_frames(
            image, data, np.eye(4), target, "cuda:0",
            method="linear", fill=0, frame_chunk_size=None,
        )
    assert state["chunks"] == []
