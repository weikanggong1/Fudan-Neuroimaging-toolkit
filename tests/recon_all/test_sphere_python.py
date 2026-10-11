"""Small source-operation checks; frozen-subject parity lives in validation/."""

import numpy as np
import pytest

from fnit.recon_all.sphere_python import (
    inflate_before_quick_sphere,
    initial_scale,
    project_before_quick_sphere,
    project_radially,
    scale_about_bbox,
)
from fnit.recon_all.sphere_quick_python import _quadratic_minimum32


def test_native_first_sphere_line_fit_from_logged_bracket():
    dt = 1193.26928947152
    predicted = _quadratic_minimum32(
        (dt / 2, dt, dt * 1.5),
        (11566.981437, 11555.797034, 11548.594136),
    )
    assert predicted == float(np.float32(2204.0380859375))


def test_bbox_scale_uses_bbox_center():
    points = np.array([[1, 2, 3], [5, 6, 7]], np.float32)
    np.testing.assert_array_equal(
        scale_about_bbox(points, np.float32(0.5)),
        np.array([[2, 3, 4], [4, 5, 6]], np.float32),
    )


def test_projection_centers_and_sets_radius():
    points = np.array([[10, 0, 0], [-10, 0, 0], [0, 10, 0], [0, -10, 0]], np.float32)
    result = project_radially(points)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 100, atol=1e-5)


def test_initial_bbox_default_preserves_archived_quick_integer_rule():
    points = np.array([[0, 0, 0], [0, 215.832, 0]], np.float32)
    scaled = initial_scale(points)
    np.testing.assert_allclose(
        np.ptp(scaled[:, 1]), np.float32(215.832) * np.float32(75 / 215),
        rtol=1e-6,
    )


def test_source_float_bbox_cap_keeps_fractional_span():
    points = np.array([[0, 0, 0], [0, 215.832, 0]], np.float32)
    scaled = initial_scale(points, bbox_mode="source_float")
    # Fixed native translation unit uses abs(float): exactly 75 mm after cap.
    assert np.ptp(scaled[:, 1]) == np.float32(75)
    assert not np.array_equal(scaled, initial_scale(points))


def test_standard_projection_selects_float_mode_without_changing_quick():
    from fnit.recon_all.sphere_standard_python import project_before_standard_unfold

    points = np.array([[-48.7009315, -125.717445, -67.807350],
                       [48.7009315, 125.717445, 67.807350],
                       [11.17277, 71.11599, -38.12054]], np.float32)
    np.testing.assert_array_equal(project_before_standard_unfold(points, bbox_mode="source_float"),
                                  project_radially(initial_scale(points, bbox_mode="source_float")))
    np.testing.assert_array_equal(project_before_standard_unfold(points),
                                  project_radially(initial_scale(points, bbox_mode="source_float")))
    np.testing.assert_array_equal(project_before_standard_unfold(points, bbox_mode="legacy_integer"),
                                  project_radially(initial_scale(points)))
    np.testing.assert_array_equal(project_before_quick_sphere(points),
                                  project_before_quick_sphere(initial_scale(points, bbox_mode="legacy_integer")))


@pytest.mark.parametrize("arguments, expected", [([], "source_float"),
                                                (["--bbox-mode", "legacy_integer"], "legacy_integer")])
def test_standard_api_and_cli_default_use_verified_float_rule(monkeypatch, capsys, arguments, expected):
    import inspect
    from fnit.recon_all import sphere_standard_run

    assert inspect.signature(sphere_standard_run.run_standard_sphere).parameters["bbox_mode"].default == "source_float"
    observed = {}

    def report_only(*args, **kwargs):
        observed.update(kwargs)
        return {"bbox_mode": kwargs["bbox_mode"]}

    monkeypatch.setattr(sphere_standard_run, "run_standard_sphere", report_only)
    sphere_standard_run.main(["lh.inflated", "lh.smoothwm", "lh.sphere", *arguments])
    assert observed["bbox_mode"] == expected
    assert observed["require_unfolded"] is False
    capsys.readouterr()


@pytest.mark.parametrize("mode", ["", "float", None])
def test_bbox_mode_rejects_unrecognized_semantics(mode):
    with pytest.raises(ValueError, match="bbox_mode"):
        initial_scale(np.zeros((3, 3), np.float32), bbox_mode=mode)


@pytest.mark.parametrize("points", [[], [[1, 2]], [[np.nan, 0, 0]], [[0, np.inf, 1]]])
def test_initial_scale_invalid_input_fails_before_projection(points):
    with pytest.raises(ValueError, match="vertices"):
        initial_scale(np.asarray(points), bbox_mode="source_float")


def test_zero_inflation_updates_equal_direct_projection():
    points = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
    np.testing.assert_array_equal(
        inflate_before_quick_sphere(points, faces, iterations=0),
        project_before_quick_sphere(points),
    )
