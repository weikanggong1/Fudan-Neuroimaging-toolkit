"""Pinned FreeSurfer 8.2 first-epoch float32 quadratic-fit regression."""

import math

import numpy as np

from fnit.recon_all.sphere_standard_line_search import (
    _quadratic_candidate_allowed, _quadratic_fit_float32, first_epoch_line_search,
    first_epoch_sse,
)


def test_first_epoch_vnl_quadratic_fit_both_hemispheres():
    left_dt = [0.004935609672628332, 0.009871219345256664,
               0.014806829017884996]
    left_sse = [2625053.012899546, 2623582.211206758,
                2623155.068984963]
    assert _quadratic_fit_float32(left_dt, left_sse) == (
        21495808.0, -306176.0, 2627632.0)

    right_dt = [2.08044813469088, 4.16089626938176, 6.24134440407264]
    right_sse = [2631386.269562474, 2605009.44653865,
                 2582398.5766699687]
    assert _quadratic_fit_float32(right_dt, right_sse) == (
        436.0, -7688.0, 2661536.0)


def test_default_lh_quadratic_prediction_uses_native_fzero_gate():
    # This coefficient came from the frozen LH avg=64 checkpoint.
    assert not _quadratic_candidate_allowed(
        4.759931471198797e-08, 2558.2783203125, 3146.0320535722835)
    assert _quadratic_candidate_allowed(
        0.24169921875, 905.4383544921875, 322.8860656758059)


def test_full_default_rh_second_step_native_sse_reproduces_native_dt():
    # Native logSSE bracket at the first divergent continuous update.
    dt = [2684.776483805354, 5369.552967610708, 8054.3294514160625]
    sse = [911608.352910, 1022486.083387, 1420187.814861]
    a, b, _ = _quadratic_fit_float32(dt, sse)
    assert (a, b) == (0.019896268844604492, -59.47265625)
    assert float(np.float32(-b / a)) == 2989.13623046875


def test_first_epoch_sse_uses_native_float32_distance_weight():
    vertices = np.asarray([[100, 0, 0], [0, 100, 0]], np.float32)
    result = first_epoch_sse(
        vertices, np.empty((0, 3), np.int32),
        np.asarray([0, 1, 2], np.int64), np.asarray([1, 0], np.int32),
        np.zeros(2, np.float32), np.empty(0, np.float32),
        np.float32(4 * math.pi * 10000), 0.1)
    arc = float(np.float32(np.float32(math.pi / 2) * np.float32(100)))
    expected = float(np.float32(0.1)) * (2 * arc * arc)
    assert result["weighted_distance"] == expected
    assert result["total"] == expected


def test_distance_sse_sqrt_returns_float_before_double_accumulation():
    # Independently compiled with the fixed source TU's headers and compiler:
    # sqrt(float ratio) returns float, not a double square root.
    vertices = np.asarray([[100, 0, 0], [0, 100, 0]], np.float32)
    original_total = np.float32(float.fromhex("0x1.140f74p+16"))
    result = first_epoch_sse(
        vertices, np.empty((0, 3), np.int32),
        np.asarray([0, 1, 2], np.int64), np.asarray([1, 0], np.int32),
        np.zeros(2, np.float32), np.empty(0, np.float32), original_total)
    assert result["distance_scale"] == float.fromhex("0x1.7ff6p-1")
    ratio = np.float32(original_total / np.float32(4 * math.pi * 10000))
    assert result["distance_scale"] != math.sqrt(float(ratio))


def test_full_default_rh_sixth_native_sse_reproduces_native_dt():
    dt = [1082.1390482941101, 2164.2780965882203, 3246.4171448823304]
    sse = [768850.990010, 764907.671084, 764608.968844]
    a, b, c = _quadratic_fit_float32(dt, sse)
    assert (a, b, c) == (0.0015506744384765625, -4.3359375, 776408.0)
    assert float(np.float32(-b / a)) == 2796.162353515625


def test_line_search_uses_native_vertex_order_for_mean_gradient():
    n = 128
    vertices = np.zeros((n, 3), np.float32)
    vertices[:, 0] = 100
    gradient = np.zeros_like(vertices)
    gradient[:, 1] = np.asarray(
        [10.0 ** (-2 - i % 8) for i in range(n)], np.float32)
    sq = np.float32(np.float32(gradient[:, 0] ** 2 + gradient[:, 1] ** 2)
                    + gradient[:, 2] ** 2)
    lengths = np.sqrt(sq.astype(np.float64))
    native_sum = 0.0
    for length in lengths:
        native_sum += float(length)
    result = first_epoch_line_search(
        vertices, gradient, np.empty((0, 3), np.int32),
        np.zeros(n + 1, np.int64), np.empty(0, np.int32),
        np.empty(0, np.float32), np.empty(0, np.float32), np.float32(1),
        objective=lambda xyz: {"total": float(np.sum(
            xyz[:, 1].astype(np.float64) ** 2))})
    assert result["mean_delta"] == native_sum / n
    assert result["mean_delta"] != float(np.sum(lengths, dtype=np.float64) / n)
