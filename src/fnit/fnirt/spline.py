"""Cubic B-spline operators used by the PyTorch FNIRT implementation.

The grid layout follows FSL ``basisfield`` 2203.1.  For cubic splines with a
knot spacing greater than one voxel, coefficient zero is centred one knot
before voxel zero and the coefficient count is ``ceil((size + 1) / ksp) + 2``.
The routines are separable and keep the dense image field on the selected
PyTorch device while never materialising the full design matrix.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


_SECOND_DERIVATIVES = (
    ((2, 0, 0), 1.0),
    ((0, 2, 0), 1.0),
    ((0, 0, 2), 1.0),
    ((1, 1, 0), math.sqrt(2.0)),
    ((1, 0, 1), math.sqrt(2.0)),
    ((0, 1, 1), math.sqrt(2.0)),
)


def fsl_control_shape(shape: Sequence[int], knot_spacing: Sequence[int]):
    """Return the cubic coefficient-grid size used by FSL ``splinefield``."""
    if len(shape) != 3 or len(knot_spacing) != 3:
        raise ValueError("shape and knot_spacing must contain three values")
    result = []
    for size, spacing in zip(shape, knot_spacing):
        size, spacing = int(size), int(spacing)
        if size < 1 or spacing < 1:
            raise ValueError("shape and knot spacing must be positive")
        if spacing == 1:
            result.append(size)
        else:
            result.append(math.ceil((size + 1) / spacing) + 2)
    return tuple(result)


def _cubic_weights(fraction: torch.Tensor, derivative: int):
    one_minus = 1 - fraction
    if derivative == 0:
        return torch.stack(
            (
                one_minus.pow(3) / 6,
                (3 * fraction.pow(3) - 6 * fraction.square() + 4) / 6,
                (
                    -3 * fraction.pow(3)
                    + 3 * fraction.square()
                    + 3 * fraction
                    + 1
                )
                / 6,
                fraction.pow(3) / 6,
            ),
            dim=1,
        )
    if derivative == 1:
        return torch.stack(
            (
                -0.5 * one_minus.square(),
                1.5 * fraction.square() - 2 * fraction,
                -1.5 * fraction.square() + fraction + 0.5,
                0.5 * fraction.square(),
            ),
            dim=1,
        )
    if derivative == 2:
        return torch.stack(
            (
                one_minus,
                3 * fraction - 2,
                1 - 3 * fraction,
                fraction,
            ),
            dim=1,
        )
    raise ValueError("only cubic derivatives 0, 1 and 2 are supported")


def cubic_bspline_basis(
    positions: torch.Tensor,
    knot_spacing: int,
    control_points: int,
    *,
    derivative: int = 0,
    voxel_size: float = 1.0,
):
    """Return an FSL-aligned cubic basis or spatial derivative matrix.

    Rows index voxel positions and columns index coefficients.  Derivatives
    are with respect to physical millimetres, matching ``basisfield``.
    """
    spacing = int(knot_spacing)
    control_points = int(control_points)
    if spacing < 1 or control_points < 1 or voxel_size <= 0:
        raise ValueError("invalid B-spline geometry")
    coordinate = positions / float(spacing)
    integer = torch.floor(coordinate).to(torch.long)
    fraction = coordinate - integer.to(coordinate.dtype)
    weights = _cubic_weights(fraction, derivative)
    weights = weights / float(spacing * voxel_size) ** derivative

    # FSL special-cases unit knot spacing: coefficient c is centred at c.
    # At larger spacings coefficient c is centred at (c - 1) * spacing.
    indices = integer[:, None] + torch.arange(
        4, device=positions.device, dtype=torch.long
    )[None]
    if spacing == 1:
        indices = indices - 1
    valid = (indices >= 0) & (indices < control_points)
    basis = positions.new_zeros((positions.numel(), control_points))
    basis.scatter_add_(1, indices.clamp(0, control_points - 1), weights * valid)
    return basis


def spline_bases(
    shape: Sequence[int],
    knot_spacing: Sequence[int],
    voxel_sizes: Sequence[float],
    *,
    device,
    dtype,
    derivatives=(0, 0, 0),
    positions=None,
):
    control_shape = fsl_control_shape(shape, knot_spacing)
    if positions is None:
        positions = tuple(
            torch.arange(int(size), device=device, dtype=dtype) for size in shape
        )
    return tuple(
        cubic_bspline_basis(
            axis,
            spacing,
            controls,
            derivative=derivative,
            voxel_size=voxel_size,
        )
        for axis, spacing, controls, derivative, voxel_size in zip(
            positions,
            knot_spacing,
            control_shape,
            derivatives,
            voxel_sizes,
        )
    )


def expand_coefficients(coefficients: torch.Tensor, bases):
    """Apply a separable 3-D B-spline design matrix."""
    if coefficients.ndim != 4:
        raise ValueError("coefficients must have shape [channels, Cx, Cy, Cz]")
    bx, by, bz = bases
    # Spell out the separable contractions.  A four-operand einsum is free to
    # materialise a dense 3-D design intermediate, which is prohibitive for a
    # full MNI grid and is not how basisfield evaluates a spline field.
    field = torch.einsum("xi,cijk->cxjk", bx, coefficients)
    field = torch.einsum("yj,cxjk->cxyk", by, field)
    return torch.einsum("zk,cxyk->cxyz", bz, field)


def adjoint_field(field: torch.Tensor, bases):
    """Apply the transpose of :func:`expand_coefficients`."""
    if field.ndim != 4:
        raise ValueError("field must have shape [channels, X, Y, Z]")
    bx, by, bz = bases
    coefficients = torch.einsum("zk,cxyz->cxyk", bz, field)
    coefficients = torch.einsum("yj,cxyk->cxjk", by, coefficients)
    return torch.einsum("xi,cxjk->cijk", bx, coefficients)


def design_diagonal(weight: torch.Tensor, bases):
    """Return ``diag(B.T @ diag(weight) @ B)`` without constructing ``B``."""
    if weight.ndim != 3:
        raise ValueError("weight must have shape [X, Y, Z]")
    bx, by, bz = bases
    diagonal = torch.einsum("xi,xyz->iyz", bx.square(), weight)
    diagonal = torch.einsum("yj,iyz->ijz", by.square(), diagonal)
    return torch.einsum("zk,ijz->ijk", bz.square(), diagonal)


def zoom_matrix(
    new_size: int,
    new_knot_spacing: int,
    new_control_points: int,
    old_knot_spacing: int,
    old_control_points: int,
    *,
    device,
    dtype,
):
    """Return the 1-D coefficient transform used by FSL ``ZoomField``.

    ``basisfield`` forms ``pinv(A_new.T A_new) A_new.T A_old``.  The
    calculation is intentionally performed in double precision, as in the
    upstream NEWMAT/Armadillo implementation, before conversion to the
    requested coefficient dtype.
    """
    size = int(new_size)
    if size < 1:
        raise ValueError("new_size must be positive")
    work_dtype = torch.float64
    positions = torch.arange(size, device=device, dtype=work_dtype)
    new_basis = cubic_bspline_basis(
        positions,
        int(new_knot_spacing),
        int(new_control_points),
    )
    old_basis = cubic_bspline_basis(
        positions,
        int(old_knot_spacing),
        int(old_control_points),
    )
    gram = new_basis.T @ new_basis
    transform = torch.linalg.pinv(gram) @ new_basis.T @ old_basis
    return transform.to(dtype=dtype)


def zoom_coefficients(
    coefficients: torch.Tensor,
    new_shape: Sequence[int],
    new_knot_spacing: Sequence[int],
    old_voxel_sizes: Sequence[float],
    new_voxel_sizes: Sequence[float],
    *,
    old_knot_spacing: Sequence[int] | None = None,
):
    """Apply FSL ``splinefield::ZoomField`` to 3-D coefficients.

    Changing voxel size is represented upstream by a ``fake_old_ksp`` equal
    to ``round(old_voxel_size / new_voxel_size * new_ksp)``.  One
    least-squares coefficient transform is then applied along each axis.
    ``old_knot_spacing`` is needed for an ``--inwarp`` handoff that changes
    knot spacing without changing voxel size. FSL rejects a simultaneous
    voxel-size and knot-spacing change, so callers must perform those two
    transitions separately.
    """
    if coefficients.ndim != 4:
        raise ValueError("coefficients must have shape [channels, Cx, Cy, Cz]")
    if not (
        len(new_shape)
        == len(new_knot_spacing)
        == len(old_voxel_sizes)
        == len(new_voxel_sizes)
        == 3
    ):
        raise ValueError("all spline geometries must contain three values")
    new_shape = tuple(int(value) for value in new_shape)
    new_knot_spacing = tuple(int(value) for value in new_knot_spacing)
    if old_knot_spacing is None:
        old_knot_spacing = new_knot_spacing
    else:
        old_knot_spacing = tuple(int(value) for value in old_knot_spacing)
    old_voxel_sizes = tuple(float(value) for value in old_voxel_sizes)
    new_voxel_sizes = tuple(float(value) for value in new_voxel_sizes)
    if len(old_knot_spacing) != 3:
        raise ValueError("old_knot_spacing must contain three values")
    if any(
        value < 1
        for value in new_shape + new_knot_spacing + old_knot_spacing
    ):
        raise ValueError("matrix size and knot spacing must be positive")
    if any(value <= 0 for value in old_voxel_sizes + new_voxel_sizes):
        raise ValueError("voxel sizes must be positive")

    voxel_size_changed = old_voxel_sizes != new_voxel_sizes
    if voxel_size_changed and old_knot_spacing != new_knot_spacing:
        raise ValueError(
            "FSL ZoomField cannot change voxel size and knot spacing together"
        )

    # For a voxel-size change FSL represents the old grid on the new voxel
    # lattice through fake_old_ksp. For a pure knot-spacing change it uses the
    # actual old knot spacing directly.
    fake_old_spacing = []
    for old_voxel, new_voxel, new_spacing, old_spacing in zip(
        old_voxel_sizes,
        new_voxel_sizes,
        new_knot_spacing,
        old_knot_spacing,
    ):
        exact = (
            old_voxel / new_voxel * new_spacing
            if voxel_size_changed
            else old_spacing
        )
        rounded = int(math.floor(exact + 0.5))
        if abs(exact - rounded) > 1e-6:
            raise ValueError(
                "FSL ZoomField cannot represent this voxel-size change with "
                "an integer fake knot spacing"
            )
        fake_old_spacing.append(rounded)

    new_control_shape = fsl_control_shape(new_shape, new_knot_spacing)
    matrices = tuple(
        zoom_matrix(
            size,
            new_spacing,
            new_controls,
            old_spacing,
            old_controls,
            device=coefficients.device,
            dtype=coefficients.dtype,
        )
        for size, new_spacing, new_controls, old_spacing, old_controls in zip(
            new_shape,
            new_knot_spacing,
            new_control_shape,
            fake_old_spacing,
            coefficients.shape[1:],
        )
    )
    transformed = torch.einsum("ai,nijk->najk", matrices[0], coefficients)
    transformed = torch.einsum("bj,najk->nabk", matrices[1], transformed)
    return torch.einsum("ck,nabk->nabc", matrices[2], transformed)


def fit_field_coefficients(
    field: torch.Tensor,
    knot_spacing: Sequence[int],
    voxel_sizes: Sequence[float],
    *,
    dtype=None,
):
    """Fit a dense field with ``splinefield::Set``'s separable solver.

    FSL stabilizes the two end coefficients in each dimension with
    ``A.T A + 0.005^2 S.T S`` before applying the fit successively along x,
    y and z.  ``voxel_sizes`` is accepted to make the field geometry explicit;
    the zero-order design itself is expressed in voxel coordinates.
    """
    if field.ndim != 4:
        raise ValueError("field must have shape [channels, X, Y, Z]")
    if len(knot_spacing) != 3 or len(voxel_sizes) != 3:
        raise ValueError("spline geometry must contain three values")
    shape = tuple(int(value) for value in field.shape[1:])
    controls = fsl_control_shape(shape, knot_spacing)
    work = field.to(torch.float64)
    matrices = []
    for size, spacing, count in zip(shape, knot_spacing, controls):
        positions = torch.arange(size, device=field.device, dtype=torch.float64)
        design = cubic_bspline_basis(positions, int(spacing), int(count))
        stabilizer = design.new_zeros((size, count))
        stabilizer[0, 0] = 2.0
        stabilizer[0, 1] = -1.0
        stabilizer[0, -1] = -1.0
        stabilizer[-1, 0] = -1.0
        stabilizer[-1, -2] = -1.0
        stabilizer[-1, -1] = 2.0
        gram = design.T @ design + (0.005**2) * (
            stabilizer.T @ stabilizer
        )
        matrices.append(torch.linalg.solve(gram, design.T))
    # Keep the contraction explicitly separable.  A four-operand einsum can
    # choose a dense 3-D design intermediate for a full MNI grid.
    coefficients = torch.einsum("ai,nijk->najk", matrices[0], work)
    coefficients = torch.einsum("bj,najk->nabk", matrices[1], coefficients)
    coefficients = torch.einsum("ck,nabk->nabc", matrices[2], coefficients)
    return coefficients.to(dtype=field.dtype if dtype is None else dtype)


class BendingOperator:
    """Analytic FSL cubic-spline bending-energy operator."""

    def __init__(
        self,
        shape: Sequence[int],
        knot_spacing: Sequence[int],
        voxel_sizes: Sequence[float],
        *,
        device,
        dtype,
        execution="reference",
    ):
        if execution not in ("reference", "optimized"):
            raise ValueError("execution must be reference or optimized")
        self.execution = execution
        self.shape = tuple(int(value) for value in shape)
        self.knot_spacing = tuple(int(value) for value in knot_spacing)
        self.voxel_sizes = tuple(float(value) for value in voxel_sizes)
        self.control_shape = fsl_control_shape(self.shape, self.knot_spacing)
        self._diagonal = None
        self._normal_grams = None
        self.operators = []
        for derivatives, multiplier in _SECOND_DERIVATIVES:
            # basisfield computes regularisation over the complete support of
            # every spline, including support outside the image FOV.  For a
            # cubic kernel the sampled support has 4*ksp-1 points.  Coefficient
            # zero is centred at -ksp (at zero for unit spacing).
            full_positions = []
            for controls, spacing in zip(
                self.control_shape, self.knot_spacing
            ):
                kernel_size = 4 * spacing - 1
                full_size = (controls - 1) * spacing + 1 + kernel_size - 1
                centre_zero = 0 if spacing == 1 else -spacing
                start = centre_zero - kernel_size // 2
                full_positions.append(
                    torch.arange(full_size, device=device, dtype=dtype) + start
                )
            bases = spline_bases(
                self.shape,
                self.knot_spacing,
                self.voxel_sizes,
                device=device,
                dtype=dtype,
                derivatives=derivatives,
                positions=tuple(full_positions),
            )
            self.operators.append((bases, multiplier))

    def forward(self, coefficients: torch.Tensor):
        return tuple(
            multiplier * expand_coefficients(coefficients, bases)
            for bases, multiplier in self.operators
        )

    def adjoint(self, fields):
        result = None
        for field, (bases, multiplier) in zip(fields, self.operators):
            value = multiplier * adjoint_field(field, bases)
            result = value if result is None else result + value
        return result

    def normal(self, coefficients: torch.Tensor):
        if self.execution == "optimized":
            if self._normal_grams is None:
                # (Bx⊗By⊗Bz).T(Bx⊗By⊗Bz) is the tensor product of
                # three small Gram matrices. No wide dense derivative fields
                # are needed by a PCG Hessian-vector product. All remain double.
                self._normal_grams = tuple(
                    (tuple(basis.T @ basis for basis in bases), multiplier)
                    for bases, multiplier in self.operators
                )
            result = None
            for grams, multiplier in self._normal_grams:
                value = multiplier**2 * expand_coefficients(coefficients, grams)
                result = value if result is None else result + value
            return result
        return self.adjoint(self.forward(coefficients))

    def energy(self, coefficients: torch.Tensor):
        if (self.execution == "optimized" and coefficients.device.type == "cpu"
                and not coefficients.requires_grad
                and all(not basis.requires_grad for bases, _ in self.operators for basis in bases)):
            # Retain every dense expansion, elementwise multiplication,
            # square, reduction and six-term sum in their original order.
            # Only one derivative field is live, and its storage is reused
            # for the multiplier and square. Gram-dot energy would change
            # reduction roundings and can alter LM acceptance decisions.
            result = coefficients.new_zeros(())
            for bases, multiplier in self.operators:
                field = expand_coefficients(coefficients, bases)
                if field.dtype in (torch.float32, torch.float64):
                    from ._bending_cpu import multiply_square_
                    multiply_square_(field, multiplier)
                else:
                    field.mul_(multiplier)
                    field.square_()
                result = result + field.sum()
            return result
        return sum(
            (field.square().sum() for field in self.forward(coefficients)),
            coefficients.new_zeros(()),
        )

    def diagonal(self):
        # Geometry is fixed for this level; every LM linearisation uses the
        # same diagonal. Keep its original contractions and roundings.
        if self._diagonal is not None:
            return self._diagonal
        result = None
        ones = torch.ones(
            tuple(basis.shape[0] for basis in self.operators[0][0]),
            device=self.operators[0][0][0].device,
            dtype=self.operators[0][0][0].dtype,
        )
        for bases, multiplier in self.operators:
            value = multiplier**2 * design_diagonal(ones, bases)
            result = value if result is None else result + value
        self._diagonal = result
        return result


__all__ = [
    "BendingOperator",
    "adjoint_field",
    "cubic_bspline_basis",
    "design_diagonal",
    "expand_coefficients",
    "fit_field_coefficients",
    "fsl_control_shape",
    "spline_bases",
    "zoom_coefficients",
    "zoom_matrix",
]
