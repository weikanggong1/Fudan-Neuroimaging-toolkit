import numpy as np

from fnit.recon_all.place_surface_self_repulsion import vertex_buckets_current


def _dict_reference(vertices, ripped, resolution):
    xyz = np.asarray(vertices, dtype=np.float32)
    spacing = np.float32(resolution)
    key = np.float32(np.float32(xyz / spacing) + np.float32(1000)).astype(np.int32)
    buckets = {}
    for vertex in range(len(key)):
        if not ripped[vertex]:
            buckets.setdefault(tuple(key[vertex]), []).append(vertex)
    offsets = np.zeros(len(key) + 1, dtype=np.int32)
    flat = []
    for vertex in range(len(key)):
        if not ripped[vertex]:
            flat.extend(buckets.get(tuple(key[vertex]), ()))
        offsets[vertex + 1] = len(flat)
    return offsets, np.asarray(flat, dtype=np.int32)


def test_scaled_current_buckets_preserve_python_order_and_membership():
    rng = np.random.default_rng(11)
    vertices = rng.normal(0.0, 30.0, size=(257, 3)).astype(np.float32)
    # Include coordinates near integer bucket boundaries and negative values.
    vertices[:4] = np.array(
        [[-1000.0, -999.5, 0.0], [-999.9999, -999.5, 0.0],
         [0.0, 0.0, 0.0], [0.99999, 0.0, 0.0]], dtype=np.float32)
    ripped = rng.random(len(vertices)) < 0.17
    for resolution in (0.25, 0.7, 1.0, 1.75, 3.0):
        expected = _dict_reference(vertices, ripped, resolution)
        actual = vertex_buckets_current(vertices, ripped, resolution=resolution)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


def test_scaled_current_buckets_reject_nonpositive_resolution():
    vertices = np.zeros((1, 3), dtype=np.float32)
    ripped = np.zeros(1, dtype=np.bool_)
    for resolution in (0.0, -1.0, np.inf, np.nan):
        try:
            vertex_buckets_current(vertices, ripped, resolution=resolution)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid resolution was accepted")
