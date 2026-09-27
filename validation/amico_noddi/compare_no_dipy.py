"""Compare FNIT's internal AMICO primitives with DIPY 1.12.1.

DIPY is a validation oracle in this script. It is not imported by the FNIT
runtime package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import scipy

from fnit.amico_noddi import kernels


def _unit_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _fibonacci_directions(count):
    index = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(1.0 - z * z)
    phi = index * (np.pi * (3.0 - np.sqrt(5.0)))
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def _tensor_signal(bvals, bvecs, principal):
    principal = np.asarray(principal, dtype=np.float64)
    principal /= np.linalg.norm(principal)
    helper = np.array([principal[1], -principal[0], 0.0])
    if np.linalg.norm(helper) < 1e-8:
        helper = np.array([1.0, 0.0, 0.0])
    helper /= np.linalg.norm(helper)
    third = np.cross(principal, helper)
    rotation = np.column_stack((principal, helper, third))
    tensor = rotation @ np.diag([1.7e-3, 0.45e-3, 0.3e-3]) @ rotation.T
    return np.exp(-bvals * np.einsum("ni,ij,nj->n", bvecs, tensor, bvecs))


def compare():
    from dipy import __version__ as dipy_version
    from dipy.core.geometry import cart2sphere
    from dipy.core.gradients import gradient_table
    from dipy.reconst import dti
    from dipy.reconst.shm import real_sh_descoteaux

    directions = _unit_rows(_fibonacci_directions(73))
    _, theta, phi = cart2sphere(directions[:, 0], directions[:, 1], directions[:, 2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sh_reference = real_sh_descoteaux(12, theta, phi)[0]
    sh_candidate = kernels._real_sh_descoteaux(theta, phi)

    bvecs = np.vstack((np.zeros((5, 3)), _fibonacci_directions(60)))
    bvals = np.r_[np.zeros(5), np.full(30, 1000.0), np.full(30, 2000.0)]
    signal = np.stack(
        [
            _tensor_signal(bvals, bvecs, (0.8, -0.3, 0.5196152422706632)),
            _tensor_signal(bvals, bvecs, (-0.25, 0.9, 0.3570714214271425)),
        ]
    )
    raw = np.column_stack((bvecs, bvals))
    principal_candidate = kernels.principal_directions(signal, raw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = gradient_table(bvals, bvecs=bvecs)
        principal_reference = np.squeeze(
            dti.TensorModel(table, fit_method="OLS").fit(signal).directions
        ).reshape(-1, 3)
    principal_error = np.minimum(
        np.linalg.norm(principal_candidate - principal_reference, axis=1),
        np.linalg.norm(principal_candidate + principal_reference, axis=1),
    )

    asset_gradients, asset_directions, _ = kernels.direction_assets()
    _, asset_theta, asset_phi = cart2sphere(
        asset_gradients[:, 0], asset_gradients[:, 1], asset_gradients[:, 2]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        basis_reference = real_sh_descoteaux(12, asset_theta, asset_phi)[0]
    fit_reference = np.linalg.pinv(basis_reference.T @ basis_reference) @ basis_reference.T
    rotated_reference = np.empty((len(asset_directions), basis_reference.shape[1]))
    for index, direction in enumerate(asset_directions):
        _, angle_theta, angle_phi = cart2sphere(*direction)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rotated_reference[index] = real_sh_descoteaux(
                12, angle_theta, angle_phi
            )[0].reshape(-1)
    fit_candidate, rotated_candidate, _, _ = kernels._rotation_auxiliary()

    source = Path(kernels.__file__).resolve()
    return {
        "schema_version": 2,
        "feature": "AMICO-NODDI internal DIPY removal",
        "fnit_version": "0.14.0",
        "reference": {
            "software": f"DIPY {dipy_version}",
            "role": "validation oracle only",
        },
        "candidate": {
            "implementation": "FNIT internal NumPy/SciPy OLS-DTI and legacy Descoteaux-2007 basis",
            "source_file": "src/fnit/amico_noddi/kernels.py",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "runtime_imports_dipy": False,
        },
        "scope": {
            "spherical_harmonics": "73 deterministic unit directions, even orders 0 through 12",
            "principal_direction": "two deterministic diffusion tensors, 65-volume two-shell scheme",
            "rotation_basis": "bundled AMICO 500-direction tables",
        },
        "metrics": {
            "spherical_harmonics_max_abs": float(np.max(np.abs(sh_candidate - sh_reference))),
            "spherical_harmonics_mean_abs": float(np.mean(np.abs(sh_candidate - sh_reference))),
            "principal_direction_max_sign_invariant_l2": float(principal_error.max()),
            "principal_direction_mean_sign_invariant_l2": float(principal_error.mean()),
            "rotation_basis_max_abs": float(np.max(np.abs(rotated_candidate - rotated_reference))),
            "rotation_fit_max_abs": float(np.max(np.abs(fit_candidate - fit_reference))),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = compare()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
