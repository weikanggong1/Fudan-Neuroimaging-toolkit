"""CPU image semantics and device isolation regressions."""
import errno
import numpy as np
import nibabel as nib
import pytest
import torch

from fnit._nib import FNITNifti1Image
from fnit._transforms import AffineTransform, DenseWarp
from fnit.synthmorph import SynthMorph, apply_transform
from fnit.synthmorph import models, pipeline, spatial


def small_detector():
    detector = models.FeatureDetector.__new__(models.FeatureDetector)
    torch.nn.Module.__init__(detector)
    detector.layers = torch.nn.ModuleList([
        torch.nn.Conv3d(1 if index == 0 else 2, 2, 3, padding=1)
        for index in range(9)])
    return detector.eval()


def test_cpu_joint_features_use_explicit_kernel_without_mutating_parameters(monkeypatch):
    detector = small_detector()
    parameters = [(layer.weight.detach().clone(), layer.weight.stride())
                  for layer in detector.layers]
    backend_before = torch.backends.mkldnn.enabled
    calls = []
    original = torch.mkldnn_convolution
    def observed(image, weight, *args):
        calls.append((image.is_contiguous(memory_format=torch.channels_last_3d),
                      weight.is_contiguous(memory_format=torch.channels_last_3d)))
        return original(image, weight, *args)
    monkeypatch.setattr(torch, 'mkldnn_convolution', observed)
    observed_features = []
    hook = detector.register_forward_hook(lambda module, arguments, result: observed_features.append(result))
    with torch.inference_mode():
        result = detector(torch.ones(1, 1, 16, 16, 16), cpu_joint_inference=True)
    hook.remove()
    assert len(observed_features) == 1 and observed_features[0] is result
    assert result.shape == (1, 2, 1, 1, 1)
    assert calls == [(True, True)] * 9
    assert torch.backends.mkldnn.enabled == backend_before
    for layer, (before, stride) in zip(detector.layers, parameters):
        assert torch.equal(layer.weight, before) and layer.weight.stride() == stride


def test_cpu_joint_features_keep_established_training_and_gradients(monkeypatch):
    from fnit.synthmorph import _cpu_features
    detector = torch.nn.Conv3d(2, 3, 3, padding=1)
    image = torch.arange(128, dtype=torch.float32).reshape(1, 2, 4, 4, 4).requires_grad_()
    def unexpected(*args):
        raise AssertionError('inference kernel called while gradients enabled')
    monkeypatch.setattr(torch, 'mkldnn_convolution', unexpected)
    result = _cpu_features.detector_features(detector, image)
    assert torch.equal(result, detector(image))
    result.sum().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_cpu_joint_features_respect_disabled_mkldnn_policy(monkeypatch):
    from fnit.synthmorph import _cpu_features
    monkeypatch.setattr(torch.backends.mkldnn, 'enabled', False)
    def unexpected(*args):
        raise AssertionError('disabled oneDNN policy overridden')
    monkeypatch.setattr(torch, 'mkldnn_convolution', unexpected)
    image = torch.ones(1, 1, 4, 4, 4)
    with torch.inference_mode():
        assert _cpu_features.detector_features(torch.nn.Identity(), image) is image
    assert not torch.backends.mkldnn.enabled


@pytest.mark.parametrize('kind', ['leaf_forward', 'leaf_pre', 'global_forward', 'global_pre'])
def test_cpu_joint_observers_keep_original_layer_calls_and_outer_hook(kind, monkeypatch):
    detector = small_detector()
    image = torch.ones(1, 1, 16, 16, 16)
    layer_calls, detector_calls = [], []
    def forward(module, arguments, output):
        if type(module) is torch.nn.Conv3d:
            layer_calls.append(module)
            return output + .125
    def pre(module, arguments):
        if type(module) is torch.nn.Conv3d:
            layer_calls.append(module)
            return (arguments[0] + .125,)
    callbacks = {
        'leaf_forward': lambda: detector.layers[0].register_forward_hook(forward),
        'leaf_pre': lambda: detector.layers[0].register_forward_pre_hook(pre),
        'global_forward': lambda: torch.nn.modules.module.register_module_forward_hook(forward),
        'global_pre': lambda: torch.nn.modules.module.register_module_forward_pre_hook(pre)}
    hook = callbacks[kind]()
    outer = detector.register_forward_hook(lambda *args: detector_calls.append(True))
    def unexpected(*args):
        raise AssertionError('explicit kernel bypassed observed layer calls')
    monkeypatch.setattr(torch, 'mkldnn_convolution', unexpected)
    try:
        with torch.inference_mode():
            expected = detector(image)
            observed = detector(image, cpu_joint_inference=True)
        assert torch.equal(observed, expected)
        assert len(detector_calls) == 2
        assert len(layer_calls) == (18 if kind.startswith('global') else 2)
    finally:
        hook.remove(); outer.remove()


@pytest.mark.parametrize('kind', ['training', 'autocast'])
def test_cpu_joint_features_keep_training_and_autocast_in_no_grad(kind, monkeypatch):
    detector = small_detector()
    if kind == 'training':
        detector.train()
    def unexpected(*args):
        raise AssertionError('explicit inference kernel selected in caller training/autocast policy')
    monkeypatch.setattr(torch, 'mkldnn_convolution', unexpected)
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=kind == 'autocast'):
        image = torch.ones(1, 1, 16, 16, 16)
        expected = detector(image)
        observed = detector(image, cpu_joint_inference=True)
    assert observed.dtype == expected.dtype and torch.equal(observed, expected)


def test_cpu_joint_guard_rejects_leaf_backward_hooks_and_custom_layer():
    from fnit.synthmorph._cpu_features import supported_inference
    detector = small_detector()
    image = torch.ones(1, 1, 16, 16, 16)
    with torch.inference_mode():
        assert supported_inference(detector, image)
        hook = detector.layers[0].register_full_backward_pre_hook(lambda *args: None)
        try:
            assert not supported_inference(detector, image)
        finally:
            hook.remove()
        class CustomConv(torch.nn.Conv3d):
            pass
        detector.layers[0] = CustomConv(1, 2, 3, padding=1).eval()
        assert not supported_inference(detector, image)


def test_cpu_joint_guard_keeps_differentiable_published_affine_route(monkeypatch):
    network = models.AffineNetwork.__new__(models.AffineNetwork)
    torch.nn.Module.__init__(network)
    network.detector = torch.nn.Identity()
    network.rigid = False
    network.eval()
    def unexpected(*args):
        raise AssertionError('new inference-only affine helper called with gradients')
    for name in ('_cpu_joint_barycenter', '_cpu_joint_fit_affine', '_cpu_joint_inverse',
                 '_cpu_joint_matrix_sqrt', '_cpu_joint_center_affine'):
        monkeypatch.setattr(models, name, unexpected)
    generator = torch.Generator().manual_seed(7)
    moving = (torch.rand(1, 64, 3, 3, 3, generator=generator) + 1).requires_grad_()
    fixed = moving.detach().clone()
    forward, inverse = network(moving, fixed, half_res=False, mid_space=True)
    (forward.sum() + inverse.sum()).backward()
    assert moving.grad is not None and torch.isfinite(moving.grad).all()


def test_cpu_joint_raw_preprocessing_interpolates_scalar_mask_and_strided_volume():
    from fnit.synthmorph._cpu_preprocessing import network_transform
    pull = torch.eye(4)
    pull[:3, 3] = torch.tensor([.25, .5, .75])
    values = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
    mask = torch.zeros_like(values)
    mask[..., 1, 1, 1] = 1
    strided = values.transpose(2, 4)
    assert not strided.is_contiguous()
    before = strided.clone()
    with torch.inference_mode():
        assert network_transform(values, pull, shape=(1, 1, 1)).item() == 2.75
        assert network_transform(mask, pull, shape=(1, 1, 1)).item() == .09375
        assert network_transform(strided, pull, shape=(1, 1, 1)).item() == 4.25
        batch = torch.cat((values, values + 10), dim=0)
        assert torch.equal(network_transform(batch, pull, shape=(1, 1, 1)).flatten(),
                           torch.tensor([2.75, 12.75]))
    assert torch.equal(strided, before)


@pytest.mark.parametrize('coordinate,fill,expected', [(2., -7, 2.), (2.25, -7, -7.),
                                                    (2.25, None, 2.), (-.25, -7, -7.)])
def test_cpu_joint_raw_preprocessing_keeps_closed_centers_and_singleton_axes(coordinate, fill, expected):
    from fnit.synthmorph._cpu_preprocessing import network_transform
    values = torch.arange(3, dtype=torch.float32).reshape(1, 1, 3, 1, 1)
    pull = torch.eye(4)
    pull[0, 3] = coordinate
    with torch.inference_mode():
        assert network_transform(values, pull, shape=(1, 1, 1), fill_value=fill).item() == expected


def test_cpu_joint_raw_preprocessing_retains_established_gradients(monkeypatch):
    from fnit.synthmorph import _cpu_preprocessing
    values = torch.arange(27, dtype=torch.float32).reshape(1, 1, 3, 3, 3).requires_grad_()
    pull = torch.eye(4)
    pull[:3, 3] = .25
    calls = []
    original = _cpu_preprocessing.transform
    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(_cpu_preprocessing, 'transform', observed)
    actual = _cpu_preprocessing.network_transform(values, pull)
    expected = spatial.transform(values, pull)
    assert calls == [True]
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(values.grad).all()


def test_cpu_inner_sum_keeps_packet_streams_and_scalar_tail():
    values = torch.zeros(2, 35)
    values[0, [0, 8, 16, 32, 34]] = torch.tensor([1e8, -1e8, 1., 2., 4.])
    values[1, 34] = 9.
    before = values.clone()
    # Four packet streams join before the remaining packet and scalar tail.
    assert torch.equal(models._cpu_inner_sum(values), torch.tensor([7., 9.]))
    assert torch.equal(values, before)
    assert models._cpu_inner_sum(torch.tensor([[3.]])).item() == 3.


def test_cpu_joint_confidence_uses_four_spatial_streams():
    features = torch.ones(1, 1, 2, 2, 2)
    features.flatten()[0] = 2 ** 24
    _, mass = models._cpu_joint_barycenter(features, (64, 64, 64))
    # The first stream loses its unit increment; the other three streams
    # each contribute two, then join in their declared left-to-right order.
    assert mass.item() == 2 ** 24 + 6


def test_cpu_joint_inverse_pivot_batch_singular_and_autograd():
    matrix = torch.tensor([[[0., 2., 0., 1.], [3., 0., 1., 0.],
                            [0., 1., 4., 2.], [0., 0., 0., 1.]],
                           [[2., 0., 1., 2.], [1., 3., 0., 1.],
                            [0., 0., 2., 1.], [0., 0., 0., 1.]]], dtype=torch.float64,
                          requires_grad=True)
    before = matrix.detach().clone()
    actual = models._cpu_joint_inverse(matrix)
    expected = torch.linalg.inv(matrix)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual.square().sum().backward()
    assert torch.isfinite(matrix.grad).all()
    assert torch.equal(matrix.detach(), before)
    with pytest.raises(torch.linalg.LinAlgError):
        models._cpu_joint_inverse(torch.zeros(4, 4))


def test_cpu_joint_square_root_retains_autograd_without_build(monkeypatch):
    # The native inference adapter must never detach a differentiable input.
    from fnit.synthmorph import _cpu_eigen
    def unexpected(*args):
        raise AssertionError('native adapter called with gradients')
    monkeypatch.setattr(_cpu_eigen, 'affine_sqrt', unexpected)
    matrix = torch.diag(torch.tensor([4., 9., 16., 1.])).requires_grad_()
    root = models._cpu_joint_matrix_sqrt(matrix)
    torch.testing.assert_close(root, torch.diag(torch.tensor([2., 3., 4., 1.])))
    root.sum().backward()
    assert torch.isfinite(matrix.grad).all()


@pytest.mark.parametrize('shape,half,translation', [
    (96, [[1.0348796844482422, .02605953812599182, .028243273496627808, .07283739745616913],
          [-.024887695908546448, 1.050148367881775, -.001133352518081665, 1.7045643329620361],
          [-.0027295947074890137, .03678622841835022, 1.0537726879119873, .41588330268859863],
          [0., 0., 0., 1.]], [-4.163330078125, .5585136413574219, -3.756011962890625]),
    (128, [[1.0349048376083374, .026067331433296204, .027138739824295044, .07573167979717255],
           [-.02547430992126465, 1.0501582622528076, -.002644747495651245, 1.7261826992034912],
           [-.0014805793762207031, .04075095057487488, 1.056070327758789, .42930060625076294],
           [0., 0., 0., .9999999403953552]], [-5.519309997558594, .326690673828125, -5.624839782714844]),
])
def test_cpu_joint_center_composition_matches_original_real_half_affines(shape, half, translation):
    # Original ComposeTransform golden values on public ds003138 real-data
    # half-affines: center_stages SHA22b8194e914d18059735f24777376a7600658aba61e8ca25505804b5998ca5ca.
    matrix = torch.tensor(half).unsqueeze(0).requires_grad_()
    before = matrix.detach().clone()
    result = models._cpu_joint_center_affine(matrix, (shape,) * 3)
    expected = matrix.detach().clone()
    expected[0, :3, 3] = torch.tensor(translation)
    expected[0, 3] = torch.tensor([0., 0., 0., 1.])
    assert torch.equal(result, expected)
    assert torch.equal(matrix.detach(), before)
    result.sum().backward()
    assert torch.isfinite(matrix.grad).all()


def test_cpu_eigen_missing_dependency_reports_the_conda_requirement(monkeypatch, tmp_path):
    from fnit.synthmorph import _cpu_eigen
    monkeypatch.setenv('FNIT_EIGEN_INCLUDE', str(tmp_path / 'missing'))
    with pytest.raises(RuntimeError, match='Eigen from the FNIT Conda environment'):
        _cpu_eigen._build_inputs()


@pytest.mark.parametrize('code,failures,expected_calls', [(errno.ENOLCK, 1, 2), (errno.ENOLCK, 9, 3), (errno.EPERM, 1, 1)])
def test_cpu_build_lock_retries_only_bounded_enolck(monkeypatch, tmp_path, code, failures, expected_calls):
    import fcntl
    import time
    from fnit.synthmorph import _cpu_eigen
    calls = []
    def flock(*args):
        calls.append(1)
        if len(calls) <= failures:
            raise OSError(code, 'fixture lock failure')
    monkeypatch.setattr(fcntl, 'flock', flock)
    monkeypatch.setattr(time, 'sleep', lambda seconds: None)
    with (tmp_path / 'cache.lock').open('w') as lock:
        if failures > 2 or code != errno.ENOLCK:
            with pytest.raises(OSError): _cpu_eigen._cache_lock(lock)
        else:
            _cpu_eigen._cache_lock(lock)
    assert len(calls) == expected_calls


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
    class Detector(torch.nn.Identity):
        def forward(self, image, **kwargs):
            return image
    monkeypatch.setattr(models, 'FeatureDetector', lambda weights: Detector())
    monkeypatch.setattr(models, '_cpu_joint_inference_enabled',
                        lambda module, moving, fixed, enabled: enabled and moving.device.type == 'cpu')
    monkeypatch.setattr(models, 'barycenter', centers('established'))
    monkeypatch.setattr(models, '_cpu_joint_barycenter', centers('joint_cpu'))
    def stop(*args):
        raise StopBeforeDeviceAllocation
    monkeypatch.setattr(models, 'fit_affine', stop)
    monkeypatch.setattr(models, '_cpu_joint_fit_affine', stop)
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
