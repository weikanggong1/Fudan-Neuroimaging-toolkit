"""CPU image semantics and device isolation regressions."""
import numpy as np
import nibabel as nib
import pytest
import torch

from fnit._nib import FNITNifti1Image
from fnit._transforms import AffineTransform, DenseWarp
from fnit.synthmorph import SynthMorph, apply_transform
from fnit.synthmorph import models, pipeline, spatial


def test_joint_cpu_barycenter_known_point_and_empty_feature():
    features = torch.zeros(1, 2, 5, 7, 9)
    features[0, 0, 1, 5, 7] = 11
    before = features.clone()
    centers, mass = models._cpu_joint_barycenter(features, (192, 256, 192))
    expected = torch.tensor([[[-192 / 5, 512 / 7, 64], [0, 0, 0]]])
    torch.testing.assert_close(centers, expected, rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(mass, torch.tensor([[11., 0.]]), rtol=0, atol=0)
    assert torch.equal(features, before)
    assert torch.isfinite(centers).all()
    alternate = features.contiguous(memory_format=torch.channels_last_3d)
    second_centers, second_mass = models._cpu_joint_barycenter(alternate, (192, 256, 192))
    assert torch.equal(centers, second_centers)
    assert torch.equal(mass, second_mass)


@pytest.mark.parametrize('device,mid_space,expected', [
    ('cpu', True, 'joint_cpu'), ('cpu', False, 'established'),
    ('cuda:0', True, 'established'),
])
def test_joint_reduction_dispatch_leaves_cuda_and_linear_routes(monkeypatch, device, mid_space, expected):
    class Input:
        shape = (1, 1, 16, 16, 16)
        def __init__(self):
            self.device = torch.device(device)
        def __getitem__(self, key):
            return self
    class StopBeforeDeviceAllocation(Exception):
        pass
    calls = []
    def centers(role):
        def compute(*args):
            calls.append(role)
            return torch.zeros(1, 1, 3), torch.ones(1, 1)
        return compute
    monkeypatch.setattr(models, 'FeatureDetector', lambda weights: torch.nn.Identity())
    monkeypatch.setattr(models, 'barycenter', centers('established'))
    monkeypatch.setattr(models, '_cpu_joint_barycenter', centers('joint_cpu'))
    def stop(*args):
        raise StopBeforeDeviceAllocation
    monkeypatch.setattr(models, 'fit_affine', stop)
    network = models.AffineNetwork('/unused.h5')
    with pytest.raises(StopBeforeDeviceAllocation):
        network(Input(), Input(), mid_space=mid_space)
    assert calls == [expected, expected]


def test_cpu_scaled_nifti_decode_matches_materialized_image(tmp_path):
    raw = (np.arange(4096, dtype=np.int16) - 1000).reshape(16, 16, 16)
    stored = nib.Nifti1Image(raw, np.eye(4))
    stored.header.set_slope_inter(np.float32(0.037421), np.float32(3.19237))
    filename = tmp_path / 'scaled.nii.gz'
    nib.save(stored, filename)
    proxy = nib.load(filename)
    decoded = np.asanyarray(proxy.dataobj)
    materialized = nib.Nifti1Image(decoded.copy(), proxy.affine, proxy.header.copy())
    direct32 = np.array(proxy.dataobj, dtype=np.float32, copy=True)
    # Exercise actual ArrayProxy scaling, including values where requesting
    # float32 rounds the slope/intercept arithmetic before the final cast.
    assert np.count_nonzero(direct32 != decoded.astype(np.float32)) > 0
    assert torch.equal(pipeline._tensor(proxy, 'cpu'),
                       pipeline._tensor(materialized, 'cpu'))
    assert torch.equal(pipeline._tensor_frames(proxy, 'cpu'),
                       pipeline._tensor_frames(materialized, 'cpu'))
    # Decoding needs no CUDA allocation, so the unchanged GPU route is
    # checked even on a CPU runner.
    assert np.array_equal(pipeline._image_data(proxy, 'cuda:0'), direct32)


@pytest.mark.parametrize('model', ['affine', 'rigid'])
def test_cpu_registration_images_apply_the_returned_affines(monkeypatch, model):
    class RoundedPair:
        def __call__(self, moving, fixed):
            # FP32 estimation and composition need not make paired matrices
            # exact inverses. A visible offset exercises which affine the
            # public image output actually consumes.
            forward = torch.eye(4)
            backward = torch.eye(4)
            forward[0, 3] = 0.21
            backward[0, 3] = -0.2
            return forward, backward

    monkeypatch.setattr(pipeline, 'resolve_weights', lambda *args: '/unused.h5')
    monkeypatch.setattr(models, 'SynthMorphNetwork', lambda **kwargs: RoundedPair())
    values = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
    moving = FNITNifti1Image(values, np.eye(4))
    fixed = FNITNifti1Image(values[::-1].copy(), np.eye(4))
    result = SynthMorph(device='cpu', model=model, extent=192)(moving, fixed)
    if model == 'rigid':
        # LIA reverses the first native axis. The declared forward affine
        # must invert its own +0.21 network pull, preserving that direction
        # even though the independently rounded reciprocal predicts -0.20.
        np.testing.assert_allclose(result.transform.matrix[0, 3], .21, atol=5e-6)
        np.testing.assert_allclose(result.inverse.matrix[0, 3], -.2, atol=5e-6)
    for source, transformation, image in (
        (moving, result.transform, result.moved),
        (fixed, result.inverse, result.fixed_moved),
    ):
        reapplied = apply_transform(source, transformation, device='cpu')
        np.testing.assert_array_equal(np.asarray(image.dataobj), np.asarray(reapplied.dataobj))
        assert image.header.binaryblock == reapplied.header.binaryblock


@pytest.mark.parametrize('device,configure,expected', [
    ('cpu', True, False), ('cuda:0', False, False), ('cuda:0', True, True),
])
def test_constructor_precision_is_device_scoped(monkeypatch, device, configure, expected):
    # No model or GPU is needed to check this process-global constructor policy.
    monkeypatch.setattr(pipeline, 'resolve_weights', lambda *args: '/unused.h5')
    monkeypatch.setattr(models, 'SynthMorphNetwork', lambda **kwargs: object())
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        SynthMorph(device=device, model='affine', configure_precision=configure)
        assert torch.backends.cuda.matmul.allow_tf32 is expected
        assert torch.backends.cudnn.allow_tf32 is expected
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


@pytest.mark.parametrize('kind', ['affine', 'dense'])
@pytest.mark.parametrize('shift,expected', [(.25, 12.), (1., -7.)])
def test_cpu_final_linear_accepts_last_center_band(kind, shift, expected):
    image = FNITNifti1Image(np.arange(10, 13, dtype=np.float32).reshape(3, 1, 1), np.eye(4))
    if kind == 'affine':
        matrix = np.eye(4); matrix[0, 3] = -shift
        transformation = AffineTransform(matrix, source=image, target=image, space='world')
    else:
        data = np.zeros((3, 1, 1, 3), dtype=np.float32); data[..., 0] = shift
        transformation = DenseWarp(data, source=image, target=image)
    result = apply_transform(image, transformation, fill=-7, device='cpu')
    assert np.asarray(result.dataobj)[2, 0, 0] == expected


def test_network_sampler_retains_neurite_closed_center_domain():
    image = torch.arange(10, 13, dtype=torch.float32).reshape(1, 1, 3, 1, 1)
    pull = torch.eye(4); pull[0, 3] = .25
    result = spatial.transform(image, pull, fill_value=-7)
    assert result[0, 0, 2, 0, 0].item() == -7


@pytest.mark.parametrize('steps', [5, 7])
def test_cpu_integration_reuses_grid_and_is_bitwise_unchanged(monkeypatch, steps):
    generator = torch.Generator().manual_seed(401)
    velocity = torch.randn((1, 3, 13, 11, 9), generator=generator) * .2
    previous = velocity / 2 ** steps
    for _ in range(steps):
        previous = previous + spatial.transform(previous, previous, fill_value=None)
    original_grid = spatial.grid
    calls = []
    def count_grid(*args, **kwargs):
        calls.append(1)
        return original_grid(*args, **kwargs)
    monkeypatch.setattr(spatial, 'grid', count_grid)
    actual = spatial.integrate(velocity, steps)
    assert len(calls) == 1
    assert torch.equal(actual.view(torch.int32), previous.view(torch.int32))


@pytest.mark.parametrize('center', [0.5, 127.5])
@pytest.mark.parametrize('direction', [-1, 0, 1])
def test_cpu_surfa_nearest_rounds_float_coordinate_before_half_tie(center, direction):
    from fnit.synthmorph.spatial import surfa_nearest
    coordinate = torch.tensor(center, dtype=torch.float32)
    if direction:
        coordinate = torch.nextafter(coordinate, torch.tensor(float('inf')*direction))
    matrix = torch.eye(4, dtype=torch.float32)
    matrix[0, 3] = coordinate
    image = torch.arange(260, dtype=torch.float32).reshape(1, 1, 260, 1, 1)
    result = surfa_nearest(image, matrix, shape=(1, 1, 1))
    # C libc round receives the exact float32 value promoted to double.
    expected = np.floor(np.float64(coordinate.item()) + .5)
    assert result.item() == expected


def test_legacy_prepared_nearest_policy_remains_explicit_for_cuda_plans():
    from fnit.synthmorph.spatial import _prepare_transform, _sample_prepared
    coordinate = torch.nextafter(torch.tensor(.5), torch.tensor(-float('inf')))
    matrix = torch.eye(4);matrix[0, 3] = coordinate
    image = torch.arange(2, dtype=torch.float32).reshape(1, 1, 2, 1, 1)
    options = dict(device='cpu', dtype=torch.float32, shape=(1, 1, 1),
                   method='nearest', surfa_nearest_rule=True)
    legacy = _prepare_transform(matrix, image.shape[2:], **options)
    strict = _prepare_transform(matrix, image.shape[2:], **options,
                                surfa_nearest_half_up=True)
    assert _sample_prepared(image, legacy).item() == 1
    assert _sample_prepared(image, strict).item() == 0


def test_cpu_linear_affine_coordinates_do_not_cancel_at_fill_boundary():
    image = FNITNifti1Image(np.arange(101, dtype=np.float32).reshape(101, 1, 1), np.eye(4))
    pull = np.eye(4, dtype=np.float32)
    pull[0, 0] = .1
    pull[0, 3] = np.float32(-10.000001)
    result = pipeline._resampled_image(image, pull, image, 'cpu', fill=-7)
    assert np.asarray(result.dataobj)[100, 0, 0] == -7
    # Network sampling keeps the old affine -> displacement -> coordinates.
    tensor = torch.from_numpy(np.asarray(image.dataobj))[None, None]
    previous = spatial.transform(tensor, torch.from_numpy(pull), fill_value=-7)
    assert previous[0, 0, 100, 0, 0].item() == 0


@pytest.mark.parametrize('device,has_init,preview_translation', [
    ('cpu', False, 14.), ('cpu', True, 0.), ('cuda:0', True, 14.),
])
def test_debug_input_geometry_after_initial_alignment(device, has_init, preview_translation):
    moving_affine = np.eye(4); moving_affine[0, 3] = 14.
    moving = FNITNifti1Image(np.zeros((3, 3, 3), dtype=np.float32), moving_affine)
    fixed = FNITNifti1Image(np.zeros((3, 3, 3), dtype=np.float32), np.eye(4))
    inputs = (torch.arange(27.).reshape(1, 1, 3, 3, 3), torch.ones(1, 1, 3, 3, 3))
    first, second = pipeline._network_input_images(
        inputs, moving, fixed, np.eye(4), np.eye(4), device, has_init=has_init)
    assert first.affine[0, 3] == preview_translation
    assert np.array_equal(second.affine, fixed.affine)
    assert np.array_equal(np.asarray(first.dataobj), inputs[0][0, 0].numpy())
