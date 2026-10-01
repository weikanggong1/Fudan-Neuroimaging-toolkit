"""Source Point rotation arithmetic and native buffer boundary regressions."""
import numpy as np
import pytest

from fnit.msm import _fastpd_native


def rotations(prior, centre):
    prior = np.asarray(prior, np.float64).reshape(-1, 3)
    centre = np.asarray(centre, np.float64)
    result = _fastpd_native.source_rotation_matrices(prior, centre, len(prior))
    assert isinstance(result, bytes)
    return np.frombuffer(result, np.float64).reshape(-1, 3, 3)


def test_point_rotations_keep_same_opposite_and_zero_source_branches():
    prior = [[2, 0, 0], [-2, 0, 0], [0, 0, 0], [1, 1e-9, 0]]
    actual = rotations(prior, [5, 0, 0])
    expected = np.array([np.eye(3), -np.eye(3), -np.eye(3), np.eye(3)])
    # The source explicitly returns -I for a zero normalized cross product.
    # Arbitrarily choosing an orthogonal antipodal axis changes its proposals.
    assert np.array_equal(actual, expected)
    assert np.array_equal(rotations(prior, [0, 0, 0]),
                          np.broadcast_to(-np.eye(3), actual.shape))


def test_point_rotations_map_unscaled_source_direction_to_target():
    targets = np.array([[2, 1, -3], [-3, 4, 2], [5, -8, 11]], np.float64)
    centre = np.array([7, -3, 2], np.float64)
    matrices = rotations(targets, centre)
    for matrix, target in zip(matrices, targets):
        assert np.allclose(matrix@(centre/np.linalg.norm(centre)),
                           target/np.linalg.norm(target), atol=2e-15, rtol=0)
        assert np.allclose(matrix.T@matrix, np.eye(3), atol=2e-15, rtol=0)


def test_point_rotations_preserve_pinned_source_rodriques_rounding():
    # Nonprivate coordinates evaluated with the actual pinned C++ SDK. This
    # catches fused cross products, tree/FMA matrix products and CUDA libm
    # substitutions that change zero-label movement at shared mesh vertices.
    expected = np.array([float.fromhex(value) for value in [
        '0x1.be6837f3d6b4cp-3', '-0x1.13573c4edb9aap-2', '0x1.e0598b12628ffp-1',
        '0x1.3a2b184f7593ap-1', '0x1.9211260452eddp-1', '0x1.51efc48e9388ep-4',
        '-0x1.849222972b51dp-1', '0x1.1d8a336f60919p-1', '0x1.583b10a7a78aap-2',
    ]], np.float64).reshape(1, 3, 3)
    actual = rotations([[2, 1, -3]], [7, -3, 2])
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_point_rotations_accept_unaligned_readonly_buffers():
    prior = np.array([[2, 1, -3], [1, 1e-4, 0]], np.float64)
    centre = np.array([7, -3, 2], np.float64)
    expected = _fastpd_native.source_rotation_matrices(prior, centre, len(prior))
    assert _fastpd_native.source_rotation_matrices(
        memoryview(b'x'+prior.tobytes())[1:],
        memoryview(b'x'+centre.tobytes())[1:], len(prior)) == expected


@pytest.mark.parametrize('count,prior,centre', [
    (-1, b'', b'0'*24), (1, b'', b'0'*24), (0, b'', b''),
    (1, b'0'*24, b'0'*32), (2**62, b'', b'0'*24),
])
def test_point_rotations_reject_inconsistent_buffer_contract(count, prior, centre):
    with pytest.raises(ValueError, match='float64 prior'):
        _fastpd_native.source_rotation_matrices(prior, centre, count)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('field', ['centre', 'prior'])
def test_point_rotations_reject_nonfinite_coordinates(field, value):
    prior = np.array([[2, 1, -3]], np.float64)
    centre = np.array([7, -3, 2], np.float64)
    (prior if field == 'prior' else centre).flat[0] = value
    with pytest.raises(ValueError, match='finite'):
        _fastpd_native.source_rotation_matrices(prior, centre, 1)


def test_point_rotations_empty_point_set_returns_empty_bytes():
    assert _fastpd_native.source_rotation_matrices(b'', np.ones(3), 0) == b''
