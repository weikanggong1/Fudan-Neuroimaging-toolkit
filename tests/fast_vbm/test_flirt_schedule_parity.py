"""Component gates for FLIRT's default schedule and MISCMATHS float scalars.

These isolate source-defined boundaries; they are not real-data benchmarks.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fnit.flirt import core, search


def _engine(execution="reference"):
    engine = object.__new__(core._DefaultFLIRTEngine)
    engine.execution = execution
    engine.level = SimpleNamespace(centre=np.array([16.98656918, 16, 17]))
    engine.bound_guess = (10.0, 1.0)
    engine.requested_scale = 4.0
    engine.angular_search = True
    return engine


def _affine():
    # Existing independent decompose_aff oracle, test_fsl_flirt_exact_target.py.
    return np.array([
        [1.03870052035, .0581360013514, .0208731747207, -.742409058073],
        [-.0332339779434, .967363767006, .06686295822, -2.34995929637],
        [-.0399591258423, -.0507381279885, 1.01791154105, 1.88608230552],
        [0, 0, 0, 1],
    ])


def _oracle_parameters():
    return np.array([float.fromhex(value) for value in (
        "0x1.99999a0000000p-5", "-0x1.47ae140000000p-5", "0x1.eb851c0000000p-6",
        "0x1.33333333529d0p+0", "-0x1.2666666664498p+1", "0x1.66666666ed840p-1",
        "0x1.0a3d700000000p+0", "0x1.f0a3d80000000p-1", "0x1.051eb80000000p+0",
        "0x1.eb851e0000000p-6", "-0x1.47ae160000000p-6", "0x1.eb851e0000000p-7",
    )])


@pytest.mark.parametrize("execution", ("reference", "batched"))
def test_default_measurecost_rebuilds_and_stores_the_zero_perturbation_matrix(execution):
    # flirt.cc:2272--2283: decompose12, add zero, compose12, measure, store.
    engine = _engine(execution)
    original = _affine()
    expected = core.fsl_affine_from_parameters(
        torch.from_numpy(_oracle_parameters()), engine.level.centre, 12,
    ).numpy()
    evaluated = []

    def measure(rows):
        evaluated.extend(matrix.copy() for _, matrix in rows)
        return [(9.0, matrix.copy()) for _, matrix in rows]

    engine._measure = measure
    value, output = engine._schedule_measure([(0.0, original)])[0]

    assert value == 9.0
    assert not np.array_equal(expected, original)
    np.testing.assert_array_equal(evaluated[0], expected)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(original, _affine())


@pytest.mark.parametrize("execution", ("reference", "batched"))
@pytest.mark.parametrize("dof", (6, 12))
def test_zero_schedule_optimise_decomposes_twice_but_angular_refinement_once(
    monkeypatch, execution, dof,
):
    # usroptimise:2366/2373 then optimise_strategy1:1032. 8mm candidates call
    # optimise_strategy1 directly (1161), so retain the single decomposition.
    engine = _engine(execution)
    original = _affine()
    oracle_rebuilt = core.fsl_affine_from_parameters(
        torch.from_numpy(_oracle_parameters()), engine.level.centre, 12,
    ).numpy()
    decompose = core.fsl_parameters_from_affine
    inputs = []

    def capture(matrix, centre):
        inputs.append(matrix.copy())
        return decompose(matrix, centre)

    monkeypatch.setattr(core, "fsl_parameters_from_affine", capture)
    engine._schedule_optimize([(0.0, original)], dof, 0)

    assert len(inputs) == 2
    np.testing.assert_array_equal(inputs[0], original)
    np.testing.assert_array_equal(inputs[1], oracle_rebuilt)
    inputs.clear()
    engine._optimize([(0.0, original)], dof, 0)
    assert len(inputs) == 1
    np.testing.assert_array_equal(inputs[0], original)


def test_angular_threshold_keeps_float_rounding_and_strict_ties():
    # flirt.cc:790--822. A double threshold wrongly includes the second point.
    costs = np.full(27, .8291944861412048, dtype=np.float32)
    costs[0] = .3751913905143738
    costs[1] = .4659920036792755
    threshold = core._search_cost_threshold(costs)

    assert threshold == np.float32(.4659920036792755)
    np.testing.assert_array_equal(np.flatnonzero(costs < threshold), [0])


@pytest.mark.parametrize("minimum", (0.0, .1, -.1))
def test_angular_threshold_fallback_narrows_the_double_literal_products(minimum):
    costs = np.full(27, minimum, dtype=np.float32)
    value = float(costs[0])
    # flirt.cc:796 uses double 1.0001/.9999 before assignment to float.
    expected = np.float32(max(value * 1.0001, value * .9999))
    assert core._search_cost_threshold(costs) == expected


@pytest.mark.parametrize("angular_search, expected_angle", (
    (True, .15707963705062866),
    (False, .05999999865889549),
))
def test_schedule_rotation_perturbations_follow_fine_samples_or_scaled_tolerance(
    angular_search, expected_angle,
):
    # set_perturbations:1201--1206; defaultschedule.h:60--69 uses absolute
    # scale offsets even for a global six-DOF run (rel delta is separate).
    engine = _engine()
    engine.angular_search = angular_search
    perturbations = np.stack(engine._schedule_perturbations())

    for axis in range(3):
        assert perturbations[2 * axis, axis] == expected_angle
        assert perturbations[2 * axis + 1, axis] == -expected_angle
    np.testing.assert_array_equal(perturbations[6:, 6], [.1, -.1, .2, -.2])
    assert np.count_nonzero(perturbations) == 10


def test_absolute_scale_schedule_trials_still_return_six_dof_rigid_matrices():
    engine = _engine()
    for perturbation in engine._schedule_perturbations()[6:]:
        _, matrix = engine._optimize([(0.0, _affine())], 6, 0, perturbation)[0]
        # make_rot intentionally stores sin/cos as float, so rigidity has its
        # float rounding error; an absolute .1/.2 scale must nevertheless vanish.
        np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3),
                                   atol=5e-7, rtol=0)
        assert abs(np.linalg.det(matrix[:3, :3]) - 1) < 5e-7


@pytest.mark.parametrize("cooperative", (False, True))
def test_brent_min_distance_multiplies_the_double_point_one_literal(monkeypatch, cooperative):
    # optimise.cc:191: float min_dist = 0.1 * unittol. The previous float(.1)
    # product differs by one ULP and therefore emits a different trial point.
    unit_tolerance = np.float32(.04)
    bracket = (np.float32(0), np.float32(.8 * float(unit_tolerance)),
               np.float32(2 * float(unit_tolerance)), np.float32(2),
               np.float32(1), np.float32(2))
    emitted = []

    def cost(point):
        emitted.append(float(point[0]))
        return 2.0

    monkeypatch.setattr(core, "_next_point", lambda *args: 0.0)
    if cooperative:
        def bounds(*args):
            if False:
                yield None
            return bracket

        monkeypatch.setattr(search, "_bound_trials", bounds)
        monkeypatch.setattr(search, "_next_point", lambda *args: 0.0)
        search.evaluate_trials([search._dimension_trials(
            np.zeros(1), np.ones(1), np.array([.04]), 1, 1.0, 10.0,
        )], lambda rows: [cost(row) for row in rows])
    else:
        monkeypatch.setattr(core, "_initial_bound", lambda *args: bracket)
        core._optimize_one_dimension(
            np.zeros(1), np.ones(1), np.array([.04]), cost, 1, 1.0, 10.0,
        )

    assert emitted[-1] == .003999999724328518


@pytest.mark.parametrize("cooperative", (False, True))
def test_coordinate_stopping_narrows_the_average_to_float(monkeypatch, cooperative):
    # optimise.cc:290--291 narrows 0.99999998 to 1.0 and continues one cycle.
    calls = []

    def step(point):
        calls.append(point.copy())
        return (np.array([.99999998]), 1.0)

    if cooperative:
        def dimension(point, *args):
            if False:
                yield None
            return step(point)

        monkeypatch.setattr(search, "_dimension_trials", dimension)
        search.evaluate_trials([search.coordinate_trials(
            np.zeros(1), np.ones(1), maximum_iterations=4,
        )], lambda rows: [1.0] * len(rows))
    else:
        monkeypatch.setattr(core, "_optimize_one_dimension", lambda point, *args: step(point))
        core.fsl_coordinate_optimize(np.zeros(1), np.ones(1), lambda point: 1.0)

    assert len(calls) == 2


def test_rms_pruning_uses_the_official_float_return():
    first = np.eye(4)
    first[0, 3] = 7.9999998
    # MISCMATHS rms_deviation returns float: this candidate is not < 8mm.
    assert core._rms_deviation(first, np.eye(4)) == 8.0
