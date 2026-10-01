import math
from pathlib import Path

import torch
import pytest

from fnit.fnirt.spline import (
    BendingOperator,
    fsl_control_shape,
    zoom_coefficients,
)


ORACLE = Path(__file__).with_name("data") / "fsl_basisfield_2203_1_oracle.txt"


def _records():
    records = {}
    for line in ORACLE.read_text().splitlines():
        fields = line.split()
        records[fields[0]] = fields[1:]
    return records


def _deterministic(count):
    return torch.tensor(
        [
            math.sin(0.17 * (index + 1))
            + 0.03 * ((index % 7) - 3)
            for index in range(count)
        ],
        dtype=torch.float64,
    )


def _from_fsl_vector(values, shape):
    # NEWMAT vectors run x fastest, then y and z.
    return values.reshape(shape[2], shape[1], shape[0]).permute(2, 1, 0)[None]


def _to_fsl_vector(values):
    return values[0].permute(2, 1, 0).reshape(-1)


def test_zoomfield_matches_fsl_basisfield_2203_1_oracle():
    records = _records()
    old_shape = fsl_control_shape((7, 8, 6), (3, 3, 3))
    coefficients = _from_fsl_vector(
        _deterministic(math.prod(old_shape)), old_shape
    )
    actual = zoom_coefficients(
        coefficients,
        (13, 15, 11),
        (3, 3, 3),
        (4.0, 4.0, 4.0),
        (2.0, 2.0, 2.0),
    )
    expected = torch.tensor(
        [float(value) for value in records["ZOOM_COEF"][1:]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        _to_fsl_vector(actual), expected, atol=2e-12, rtol=2e-12
    )


@pytest.mark.parametrize("execution", ["reference", "optimized"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_bending_energy_and_gradient_match_fsl_basisfield_2203_1_oracle(execution, device):
    records = _records()
    shape = (8, 9, 7)
    spacing = (3, 3, 3)
    control_shape = fsl_control_shape(shape, spacing)
    coefficients = _from_fsl_vector(
        _deterministic(math.prod(control_shape)), control_shape
    ).to(device)
    operator = BendingOperator(
        shape,
        spacing,
        (2.0, 2.5, 3.0),
        device=device,
        dtype=torch.float64,
        execution=execution,
    )
    expected_energy = float(records["BEND_ENERGY"][0])
    expected_gradient = torch.tensor(
        [float(value) for value in records["BEND_GRAD"][1:]],
        dtype=torch.float64,
        device=device,
    )
    torch.testing.assert_close(
        operator.energy(coefficients),
        torch.tensor(expected_energy, dtype=torch.float64, device=device),
        atol=5e-14,
        rtol=5e-14,
    )
    torch.testing.assert_close(
        _to_fsl_vector(2 * operator.normal(coefficients)),
        expected_gradient,
        atol=5e-14,
        rtol=5e-14,
    )
