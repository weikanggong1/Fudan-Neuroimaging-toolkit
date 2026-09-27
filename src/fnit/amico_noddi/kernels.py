"""AMICO 2.0.3-compatible NODDI kernel construction.

The response equations and SH/LUT sequence follow AMICO 2.0.3.  The bundled
500-direction tables are redistributed under the AMICO Software License
Agreement in ``licenses/AMICO-2.0.txt``.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import warnings

import numpy as np
from scipy.special import erf, erfi, lpmv

_GAMMA = 2.675987e8
_LMAX = 12
_NSH = (_LMAX + 1) * (_LMAX + 2) // 2
_NDIRS = 500


def load_raw_bvecs(path, count):
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1 and values.size == 3:
        values = values.reshape(3, 1)
    if values.ndim != 2:
        raise ValueError("b-vectors must be a 3xN or Nx3 matrix")
    if values.shape[0] != 3 and values.shape[1] == 3:
        values = values.T
    if values.shape != (3, count) or not np.isfinite(values).all():
        raise ValueError("b-vectors must be a finite 3xN matrix matching the DWI")
    return values


def _asset(name):
    return files(__package__).joinpath("assets", name)


@lru_cache(maxsize=1)
def direction_assets():
    with _asset("grad_500.npy").open("rb") as stream:
        gradients = np.load(stream)
    with _asset("ndirs_500.bin").open("rb") as stream:
        directions = np.frombuffer(stream.read(), dtype=np.float64).reshape(_NDIRS, 3)
    with _asset("htable_ndirs_500.bin").open("rb") as stream:
        table = np.frombuffer(stream.read(), dtype=np.int16)
    return gradients, directions, table


def amico_scheme(bvals, bvecs, b0_threshold=100.0, b_step=100.0):
    """Reproduce ``fsl2scheme`` text precision and ``Scheme`` hemisphere rules."""
    bvals = np.asarray(bvals, dtype=np.float64)
    if float(b_step) > 1:
        bvals = np.round(bvals / float(b_step)) * float(b_step)
    bvals = np.round(bvals, 6)
    bvecs = np.round(np.asarray(bvecs, dtype=np.float64).T, 6)
    raw = np.column_stack((bvecs, bvals))
    raw[raw[:, 1] < 0, :3] *= -1
    b0 = bvals <= float(b0_threshold)
    if not np.any(b0):
        raise ValueError(
            f"NODDI requires at least one b0 volume with b <= {b0_threshold:g}"
        )
    if np.all(b0):
        raise ValueError("NODDI requires at least one diffusion-weighted volume")
    shells = []
    for value in bvals[~b0]:
        if value not in shells:
            shells.append(float(value))
    return raw, b0, np.asarray(shells, dtype=np.float64)


def principal_directions(signal, raw):
    """Use the same DIPY OLS tensor fit as AMICO 2.0.3."""
    from dipy.core.gradients import gradient_table
    from dipy.reconst import dti

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = gradient_table(raw[:, 3], bvecs=raw[:, :3])
        model = dti.TensorModel(table, fit_method="OLS")
        directions = np.squeeze(
            model.fit(np.asarray(signal, dtype=np.float64)).directions
        )
    return np.asarray(directions, dtype=np.float64).reshape(-1, 3)


def direction_indices(directions):
    """Map principal directions to the same 500-entry LUT as AMICO."""
    _, _, table = direction_assets()
    directions = np.asarray(directions, dtype=np.float64).copy()
    directions[directions[:, 1] < 0] *= -1
    phi = np.fmod(np.arctan2(directions[:, 1], directions[:, 0]), 2 * np.pi)
    negative = phi < 0
    phi[negative] = np.fmod(phi[negative] + 2 * np.pi, 2 * np.pi)
    upper = phi > np.pi
    theta = np.empty(phi.shape, dtype=np.float64)
    phi[upper] = np.fmod(
        np.arctan2(-directions[upper, 1], -directions[upper, 0]), 2 * np.pi
    )
    radius = np.hypot(directions[:, 0], directions[:, 1])
    theta[upper] = np.arctan2(radius[upper], -directions[upper, 2])
    theta[~upper] = np.arctan2(radius[~upper], directions[~upper, 2])
    row = np.floor(theta / np.pi * 180 + 0.5).astype(np.int64)
    column = np.floor(phi / np.pi * 180 + 0.5).astype(np.int64)
    if np.any((row < 0) | (row > 180) | (column < 0) | (column > 180)):
        raise RuntimeError("principal direction produced an invalid AMICO LUT index")
    return table[row * 181 + column].astype(np.int64)


def _protocol(shells):
    gradients, _, _ = direction_assets()
    bvals = np.repeat(shells, _NDIRS)
    grad = np.tile(gradients, (len(shells), 1)).astype(np.float64, copy=True)
    for index in range(grad.shape[0]):
        grad[index] /= np.linalg.norm(grad[index])
    maximum = np.max(shells)
    pulse = np.power(3 * maximum * 1e6 / (2 * _GAMMA**2 * 0.04**2), 1 / 3)
    strength = np.sqrt(bvals / maximum) * 0.04
    delta = np.full_like(bvals, pulse)
    return grad, strength, delta, delta.copy()


def _legendre_gaussian_integral(values, order=6):
    exact = values > 0.05
    approx = ~exact
    integrals = np.zeros((len(values), order + 1))
    root = np.sqrt(values[exact])
    integrals[exact, 0] = np.sqrt(np.pi) * erf(root) / root
    inverse = 1.0 / values[exact]
    exponential = -np.exp(-values[exact])
    for index in range(1, order + 1):
        integrals[exact, index] = (
            exponential + (index - 0.5) * integrals[exact, index - 1]
        ) * inverse
    result = np.zeros_like(integrals)
    coefficients = (
        (1,),
        (-0.5, 1.5),
        (0.375, -3.75, 4.375),
        (-0.3125, 6.5625, -19.6875, 14.4375),
        (0.2734375, -9.84375, 54.140625, -93.84375, 50.2734375),
        (-63 / 256, 3465 / 256, -30030 / 256, 90090 / 256, -109395 / 256, 46189 / 256),
        (
            231 / 1024,
            -18018 / 1024,
            225225 / 1024,
            -1021020 / 1024,
            2078505 / 1024,
            -1939938 / 1024,
            676039 / 1024,
        ),
    )
    for index, row in enumerate(coefficients):
        for power, coefficient in enumerate(row):
            result[exact, index] += coefficient * integrals[exact, power]
    x = values[approx]
    x2, x3 = x**2, x**3
    x4, x5, x6 = x3 * x, x3 * x2, x3 * x3
    result[approx, 0] = 2 - 2 * x / 3 + x2 / 5 - x3 / 21 + x4 / 108
    result[approx, 1] = -4 * x / 15 + 4 * x2 / 35 - 2 * x3 / 63 + 2 * x4 / 297
    result[approx, 2] = 8 * x2 / 315 - 8 * x3 / 693 + 4 * x4 / 1287
    result[approx, 3] = -16 * x3 / 9009 + 16 * x4 / 19305
    result[approx, 4] = 32 * x4 / 328185
    result[approx, 5] = -64 * x5 / 14549535
    result[approx, 6] = 128 * x6 / 760543875
    return result


def _watson_coefficients(kappa):
    c = np.zeros(7)
    c[0] = 2 * np.sqrt(np.pi)
    sk = np.sqrt(kappa)
    powers = [kappa**index for index in range(8)]
    skp = [sk * powers[index] for index in range(8)]
    erfi_value = erfi(sk)
    inverse_erfi = 1 / erfi_value
    exponential = np.exp(kappa)
    dawson = 0.5 * np.sqrt(np.pi) * erfi_value / exponential
    if kappa > 0.1:
        c[1] = (
            np.sqrt(5)
            * (3 * sk - (3 + 2 * kappa) * dawson)
            * exponential
            * inverse_erfi
            / kappa
        )
        c[2] = (
            0.375
            * ((105 + 60 * kappa + 12 * powers[2]) * dawson - 105 * sk + 10 * skp[1])
            * exponential
            * inverse_erfi
            / powers[2]
        )
        c[3] = (
            (
                (-3465 - 1890 * kappa - 420 * powers[2] - 40 * powers[3]) * dawson
                + 3465 * sk
                - 420 * skp[1]
                + 84 * skp[2]
            )
            * np.sqrt(13 * np.pi)
            / (64 * powers[3] * dawson)
        )
        c[4] = (
            (
                (
                    675675
                    + 360360 * kappa
                    + 83160 * powers[2]
                    + 10080 * powers[3]
                    + 560 * powers[4]
                )
                * dawson
                - 675675 * sk
                + 90090 * skp[1]
                - 23100 * skp[2]
                + 744 * skp[3]
            )
            * np.sqrt(17)
            * exponential
            * inverse_erfi
            / (512 * powers[4])
        )
        c[5] = (
            (
                (
                    -43648605
                    - 22972950 * kappa
                    - 5405400 * powers[2]
                    - 720720 * powers[3]
                    - 55440 * powers[4]
                    - 2016 * powers[5]
                )
                * dawson
                + 43648605 * sk
                - 6126120 * skp[1]
                + 1729728 * skp[2]
                - 82368 * skp[3]
                + 5104 * skp[4]
            )
            * np.sqrt(21 * np.pi)
            / (4096 * powers[5] * dawson)
        )
        c[6] = (
            5
            * (
                (
                    7027425405
                    + 3666482820 * kappa
                    + 872972100 * powers[2]
                    + 122522400 * powers[3]
                    + 10810800 * powers[4]
                    + 576576 * powers[5]
                    + 14784 * powers[6]
                )
                * dawson
                - 7027425405 * sk
                + 1018467450 * skp[1]
                - 302630328 * skp[2]
                + 17153136 * skp[3]
                - 1553552 * skp[4]
                + 25376 * skp[5]
            )
            * exponential
            * inverse_erfi
            / (16384 * powers[6])
        )
    if kappa > 30:
        q = np.log(kappa) - np.log(30)
        q2, q3, q4, q5, q6 = q**2, q**3, q**4, q**5, q**6
        c[1:] = (
            7.52308
            + 0.411538 * q
            - 0.214588 * q2
            + 0.0784091 * q3
            - 0.023981 * q4
            + 0.00731537 * q5
            - 0.0026467 * q6,
            8.93718
            + 1.62147 * q
            - 0.733421 * q2
            + 0.191568 * q3
            - 0.0202906 * q4
            - 0.00779095 * q5
            + 0.00574847 * q6,
            8.87905
            + 3.35689 * q
            - 1.15935 * q2
            + 0.0673053 * q3
            + 0.121857 * q4
            - 0.066642 * q5
            + 0.0180215 * q6,
            7.84352
            + 5.03178 * q
            - 1.0193 * q2
            - 0.426362 * q3
            + 0.328816 * q4
            - 0.0688176 * q5
            - 0.0229398 * q6,
            6.30113
            + 6.09914 * q
            - 0.16088 * q2
            - 1.05578 * q3
            + 0.338069 * q4
            + 0.0937157 * q5
            - 0.106935 * q6,
            4.65678
            + 6.30069 * q
            + 1.13754 * q2
            - 1.38393 * q3
            - 0.0134758 * q4
            + 0.331686 * q5
            - 0.105954 * q6,
        )
    if kappa <= 0.1:
        c[1] = (4 * kappa / 3 + 8 * powers[2] / 63) * np.sqrt(np.pi / 5)
        c[2] = (8 * powers[2] / 21 + 32 * powers[3] / 693) * np.sqrt(np.pi) * 0.2
        c[3] = (16 * powers[3] / 693 + 32 * powers[4] / 10395) * np.sqrt(np.pi / 13)
        c[4] = 32 * powers[4] / 19305 * np.sqrt(np.pi / 17)
        c[5] = 64 * np.sqrt(np.pi / 21) * powers[5] / 692835
        c[6] = 128 * np.sqrt(np.pi) * powers[6] / 152108775
    return c


def _intracellular_signal(protocol, d_par, kappa):
    gradients, strength, delta, small_delta = protocol
    diffusion = d_par * 1e-6
    q2 = (_GAMMA * small_delta * strength) ** 2
    parallel = -q2 * (delta - small_delta / 3) * diffusion
    weighted = _legendre_gaussian_integral(-parallel, 6)
    coefficients = _watson_coefficients(kappa)
    cosine = gradients[:, 2].copy()
    outside = np.abs(cosine) > 1
    cosine[outside] /= np.abs(cosine[outside])
    harmonics = np.zeros((len(cosine), 7))
    for index in range(7):
        harmonics[:, index] = np.sqrt((index + 0.25) / np.pi)
        harmonics[:, index] *= lpmv(0, 2 * index, cosine)
    signal = np.sum(weighted * coefficients[None] * harmonics, axis=1)
    signal[signal <= 0] = np.min(signal[signal > 0]) * 0.1
    return 0.5 * signal


def _extracellular_signal(protocol, d_par, kappa, volume_fraction):
    gradients, strength, delta, small_delta = protocol
    parallel = d_par * 1e-6
    perpendicular = parallel * (1 - volume_fraction)
    difference = parallel - perpendicular
    if kappa < 1e-5:
        base = parallel + 2 * perpendicular
        k2 = kappa * kappa
        axial = base / 3 + 4 * difference * kappa / 45 + 8 * difference * k2 / 945
        radial = base / 3 - 2 * difference * kappa / 45 - 4 * difference * k2 / 945
    else:
        root = np.sqrt(kappa)
        dawson = 0.5 * np.exp(-kappa) * np.sqrt(np.pi) * erfi(root)
        factor = root / dawson
        axial = (-difference + 2 * perpendicular * kappa + difference * factor) / (
            2 * kappa
        )
        radial = (
            difference + 2 * (parallel + perpendicular) * kappa - difference * factor
        ) / (4 * kappa)
    q2 = (_GAMMA * small_delta * strength) ** 2
    b_si = (delta - small_delta / 3) * q2
    cosine2 = gradients[:, 2] ** 2
    return np.exp(-b_si * ((axial - radial) * cosine2 + radial))


def _isotropic_signal(protocol, diffusivity):
    _, strength, delta, small_delta = protocol
    q2 = (_GAMMA * small_delta * strength) ** 2
    return np.exp(-(delta - small_delta / 3) * q2 * diffusivity * 1e-6)


@lru_cache(maxsize=1)
def _rotation_auxiliary():
    from dipy.core.geometry import cart2sphere
    from dipy.reconst.shm import real_sh_descoteaux

    gradients, directions, _ = direction_assets()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, theta, phi = cart2sphere(gradients[:, 0], gradients[:, 1], gradients[:, 2])
        basis = real_sh_descoteaux(_LMAX, theta, phi)[0]
        fit = np.dot(np.linalg.pinv(np.dot(basis.T, basis)), basis.T)
        rotated = np.empty((_NDIRS, _NSH), dtype=np.float64)
        for index in range(_NDIRS):
            _, theta, phi = cart2sphere(*directions[index])
            rotated[index] = real_sh_descoteaux(_LMAX, theta, phi)[0].reshape(-1)
    constants = np.empty(_NSH, dtype=np.float64)
    m0 = np.empty(_NSH, dtype=np.int32)
    index = 0
    for order in range(0, _LMAX + 1, 2):
        constant = np.sqrt(4 * np.pi / (2 * order + 1))
        center = int((order * order + order + 2) / 2 - 1)
        for _ in range(-order, order + 1):
            constants[index] = constant
            m0[index] = center
            index += 1
    return fit, rotated, constants, m0


def _subject_basis(raw, shells, b0):
    from dipy.core.geometry import cart2sphere
    from dipy.reconst.shm import real_sh_descoteaux

    dwi_indices = np.flatnonzero(~b0)
    output_indices = np.empty(len(dwi_indices), dtype=np.int32)
    basis = np.zeros((len(dwi_indices), _NSH * len(shells)), dtype=np.float32)
    offset = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for shell_index, shell in enumerate(shells):
            indices = np.flatnonzero(raw[:, 3] == shell)
            count = len(indices)
            output_indices[offset : offset + count] = indices
            _, theta, phi = cart2sphere(
                raw[indices, 0], raw[indices, 1], raw[indices, 2]
            )
            values = real_sh_descoteaux(_LMAX, theta, phi)[0]
            basis[
                offset : offset + count, shell_index * _NSH : (shell_index + 1) * _NSH
            ] = values
            offset += count
    return output_indices, basis


def _rotate(signal, shell_count, isotropic=False):
    fit, ylm_rot, constants, m0 = _rotation_auxiliary()
    if isotropic:
        result = np.zeros(_NSH * shell_count, dtype=np.float32)
        for shell in range(shell_count):
            coefficients = np.dot(fit, signal[shell * _NDIRS : (shell + 1) * _NDIRS])
            result[shell * _NSH : (shell + 1) * _NSH] = coefficients.astype(np.float32)
        return result
    result = np.zeros((_NDIRS, _NSH * shell_count), dtype=np.float32)
    for shell in range(shell_count):
        coefficients = np.dot(fit, signal[shell * _NDIRS : (shell + 1) * _NDIRS])
        block = constants * coefficients[m0]
        result[:, shell * _NSH : (shell + 1) * _NSH] = ylm_rot * block
    return result


def _resample(rotated, volume_count, indices, basis, isotropic=False):
    if isotropic:
        result = np.ones(volume_count, dtype=np.float32)
        for _ in range(_NDIRS):
            result[indices] = np.dot(basis, rotated).astype(np.float32)
        return result
    result = np.ones((_NDIRS, volume_count), dtype=np.float32)
    for index in range(_NDIRS):
        result[index, indices] = np.dot(basis, rotated[index]).astype(np.float32)
    return result


def build_noddi_kernels(
    bvals,
    bvecs,
    odis,
    vfs,
    d_par=1.7e-3,
    d_iso=3e-3,
    b0_threshold=100.0,
    b_step=100.0,
):
    raw, b0, shells = amico_scheme(
        bvals, bvecs, b0_threshold=b0_threshold, b_step=b_step
    )
    protocol = _protocol(shells)
    output_indices, basis = _subject_basis(raw, shells, b0)
    kappas = 1 / np.tan(np.asarray(odis, dtype=np.float64) * np.pi / 2)
    vfs = np.asarray(vfs, dtype=np.float64)
    atoms = len(kappas) * len(vfs)
    wm = np.zeros((atoms, _NDIRS, len(bvals)), dtype=np.float32)
    icvf = np.zeros(atoms, dtype=np.float32)
    kappa_values = np.zeros(atoms, dtype=np.float32)
    atom = 0
    for kappa in kappas:
        intra = _intracellular_signal(protocol, d_par, kappa)
        for vf in vfs:
            extra = _extracellular_signal(protocol, d_par, kappa, vf)
            signal = vf * intra + (1 - vf) * extra
            wm[atom] = _resample(
                _rotate(signal, len(shells)), len(bvals), output_indices, basis
            )
            icvf[atom] = vf
            kappa_values[atom] = kappa
            atom += 1
    iso = _resample(
        _rotate(_isotropic_signal(protocol, d_iso), len(shells), isotropic=True),
        len(bvals),
        output_indices,
        basis,
        isotropic=True,
    )
    norms = np.zeros((int((~b0).sum()), atoms), dtype=np.float64)
    for atom in range(atoms):
        norms[:, atom] = 1 / np.linalg.norm(wm[atom, 0, ~b0])
    return {
        "wm": wm,
        "iso": iso,
        "kappa": kappa_values,
        "icvf": icvf,
        "norms": norms,
        "b0": b0,
        "raw": raw,
        "shells": shells,
    }


__all__ = [
    "amico_scheme",
    "build_noddi_kernels",
    "direction_assets",
    "direction_indices",
    "load_raw_bvecs",
    "principal_directions",
]
