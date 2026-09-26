#!/usr/bin/env python3
"""Validate TorchApplyWarp dense fields against an installed FSL applywarp."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import nibabel as nib
import numpy as np
import torch

from fnit.applywarp import TorchApplyWarp
from fnit.applywarp.core import _expand_cubic_coefficients


def save(image, path):
    nib.save(image, str(path))


def measure(call, repeats, synchronize=None):
    call()
    if synchronize is not None:
        synchronize()
    samples = []
    for _ in range(repeats):
        if synchronize is not None:
            synchronize()
        start = time.perf_counter()
        call()
        if synchronize is not None:
            synchronize()
        samples.append(time.perf_counter() - start)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
    }


def run(args):
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    shape = (17, 16, 15)
    affine = np.diag([-2.0, 2.5, 3.0, 1.0])
    x, y, z = np.indices(shape, dtype=np.float32)
    ramp = x + 10 * y + 100 * z + 0.01 * x * y + 0.001 * y * z
    field = np.zeros((*shape, 3), dtype=np.float32)
    field[..., 0] = 0.37 + 0.08 * np.sin(y / 3)
    field[..., 1] = -0.24 + 0.06 * np.cos(z / 2)
    field[..., 2] = 0.19 + 0.03 * np.sin(x / 4)
    moving = work / "coordinate_ramp.nii.gz"
    reference = work / "reference.nii.gz"
    warp = work / "dense_relative_warp.nii.gz"
    save(nib.Nifti1Image(ramp, affine), moving)
    save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), reference)
    save(nib.Nifti1Image(field, affine), warp)
    premat = np.eye(4)
    premat[:3, 3] = (0.18, -0.11, 0.07)
    postmat = np.eye(4)
    postmat[:3, 3] = (-0.13, 0.09, -0.05)
    premat_path = work / "premat.mat"
    postmat_path = work / "postmat.mat"
    np.savetxt(premat_path, premat, fmt="%.12g")
    np.savetxt(postmat_path, postmat, fmt="%.12g")

    records = []
    model = TorchApplyWarp(args.device)
    for interpolation in ("trilinear", "nn"):
        for matrices in (False, True):
            name = f"{interpolation}-{'matrices' if matrices else 'warp-only'}"
            fsl_output = work / f"fsl-{name}.nii.gz"
            torch_output = work / f"torch-{name}.nii.gz"
            command = [
                args.applywarp,
                f"--in={moving}",
                f"--ref={reference}",
                f"--warp={warp}",
                "--rel",
                f"--interp={interpolation}",
                "--datatype=float",
                f"--out={fsl_output}",
            ]
            kwargs = {}
            if matrices:
                command.extend((f"--premat={premat_path}", f"--postmat={postmat_path}"))
                kwargs.update(premat=premat_path, postmat=postmat_path)
            subprocess.run(command, check=True)
            model.run(
                moving,
                reference,
                torch_output,
                warp=warp,
                warp_convention="relative",
                interpolation=interpolation,
                output_dtype="float",
                **kwargs,
            )
            fsl = np.asarray(nib.load(fsl_output).dataobj, dtype=np.float64)
            candidate = np.asarray(nib.load(torch_output).dataobj, dtype=np.float64)
            difference = np.abs(candidate - fsl)
            tolerance = 5e-4 if interpolation == "trilinear" else 0.0
            records.append(
                {
                    "case": name,
                    "maximum_absolute_error": float(difference.max()),
                    "mean_absolute_error": float(difference.mean()),
                    "different_voxels": int(np.count_nonzero(difference)),
                    "tolerance": tolerance,
                    "passed": bool(difference.max() <= tolerance),
                }
            )

    contract_input = work / "contract_input.nii.gz"
    contract_reference = work / "contract_reference.nii.gz"
    fsl_contract = work / "fsl-contract.nii.gz"
    torch_contract = work / "torch-contract.nii.gz"
    contract_data = np.linspace(
        -1.9, 120.9, num=np.prod(shape), dtype=np.float32
    ).reshape(shape)
    contract_input_image = nib.Nifti1Image(contract_data, affine)
    contract_reference_image = nib.Nifti1Image(
        np.zeros(shape, dtype=np.float32), affine
    )
    for image in (contract_input_image, contract_reference_image):
        image.set_qform(affine, 2)
        image.set_sform(affine, 4)
    save(contract_input_image, contract_input)
    save(contract_reference_image, contract_reference)
    subprocess.run(
        [
            args.applywarp,
            f"--in={contract_input}",
            f"--ref={contract_reference}",
            "--datatype=short",
            f"--out={fsl_contract}",
        ],
        check=True,
    )
    model.run(
        contract_input,
        contract_reference,
        torch_contract,
        output_dtype="short",
    )
    fsl_contract_image = nib.load(fsl_contract)
    torch_contract_image = nib.load(torch_contract)
    fsl_contract_data = np.asarray(fsl_contract_image.dataobj)
    torch_contract_data = np.asarray(torch_contract_image.dataobj)
    contract_difference = np.abs(
        torch_contract_data.astype(np.int64) - fsl_contract_data.astype(np.int64)
    )
    header_matches = (
        torch_contract_image.shape == fsl_contract_image.shape
        and torch_contract_image.get_data_dtype() == fsl_contract_image.get_data_dtype()
        and int(torch_contract_image.header["qform_code"])
        == int(fsl_contract_image.header["qform_code"])
        and int(torch_contract_image.header["sform_code"])
        == int(fsl_contract_image.header["sform_code"])
        and np.array_equal(
            torch_contract_image.get_qform(), fsl_contract_image.get_qform()
        )
        and np.array_equal(
            torch_contract_image.get_sform(), fsl_contract_image.get_sform()
        )
        and float(torch_contract_image.header["cal_min"])
        == float(fsl_contract_image.header["cal_min"])
        and float(torch_contract_image.header["cal_max"])
        == float(fsl_contract_image.header["cal_max"])
    )
    records.append(
        {
            "case": "reference-header-and-explicit-short-dtype",
            "maximum_absolute_error": int(contract_difference.max()),
            "mean_absolute_error": float(contract_difference.mean()),
            "different_voxels": int(np.count_nonzero(contract_difference)),
            "tolerance": 0,
            "header_matches": bool(header_matches),
            "passed": bool(contract_difference.max() == 0 and header_matches),
        }
    )

    coefficient = work / "cubic_coefficients.nii.gz"
    residual = work / "cubic_residual.nii.gz"
    with_affine = work / "cubic_with_affine.nii.gz"
    subprocess.run(
        [
            args.fnirtfileutils,
            f"--in={warp}",
            f"--ref={reference}",
            f"--out={coefficient}",
            "--outformat=spline",
            "--knotspace=4",
        ],
        check=True,
    )
    subprocess.run(
        [
            args.fnirtfileutils,
            f"--in={coefficient}",
            f"--ref={reference}",
            f"--out={residual}",
        ],
        check=True,
    )
    subprocess.run(
        [
            args.fnirtfileutils,
            f"--in={coefficient}",
            f"--ref={reference}",
            f"--out={with_affine}",
            "--withaff",
        ],
        check=True,
    )

    coefficient_image = nib.load(coefficient)
    expanded, field_shape, _, embedded_affine = _expand_cubic_coefficients(
        coefficient_image, torch.device(args.device)
    )
    expanded = np.moveaxis(expanded.detach().cpu().numpy(), 0, -1)
    fsl_residual = np.asarray(nib.load(residual).dataobj, dtype=np.float32)
    residual_difference = np.abs(expanded - fsl_residual)
    records.append(
        {
            "case": "cubic-expansion-vs-fnirtfileutils-out",
            "maximum_absolute_error": float(residual_difference.max()),
            "mean_absolute_error": float(residual_difference.mean()),
            "different_voxels": int(np.count_nonzero(residual_difference)),
            "tolerance": 1e-6,
            "passed": bool(residual_difference.max() <= 1e-6),
        }
    )

    voxel_sizes = np.array(
        [
            float(coefficient_image.header["intent_p1"]),
            float(coefficient_image.header["intent_p2"]),
            float(coefficient_image.header["intent_p3"]),
        ]
    )
    voxels = np.indices(field_shape, dtype=np.float64).reshape(3, -1)
    field_mm = voxels * voxel_sizes[:, None]
    affine_part = np.linalg.inv(embedded_affine) - np.eye(4)
    affine_displacement = (
        affine_part[:3, :3] @ field_mm + affine_part[:3, 3:4]
    ).T.reshape(*field_shape, 3)
    decoded_with_affine = expanded + affine_displacement
    fsl_with_affine = np.asarray(nib.load(with_affine).dataobj, dtype=np.float32)
    affine_difference = np.abs(decoded_with_affine - fsl_with_affine)
    records.append(
        {
            "case": "cubic-embedded-affine-vs-fnirtfileutils-withaff",
            "maximum_absolute_error": float(affine_difference.max()),
            "mean_absolute_error": float(affine_difference.mean()),
            "different_voxels": int(np.count_nonzero(affine_difference)),
            "tolerance": 1e-6,
            "passed": bool(affine_difference.max() <= 1e-6),
        }
    )

    for interpolation in ("trilinear", "nn"):
        for matrices in (False, True):
            name = f"cubic-{interpolation}-{'matrices' if matrices else 'warp-only'}"
            fsl_output = work / f"fsl-{name}.nii.gz"
            torch_output = work / f"torch-{name}.nii.gz"
            command = [
                args.applywarp,
                f"--in={moving}",
                f"--ref={reference}",
                f"--warp={coefficient}",
                f"--interp={interpolation}",
                "--datatype=float",
                f"--out={fsl_output}",
            ]
            kwargs = {}
            if matrices:
                command.extend((f"--premat={premat_path}", f"--postmat={postmat_path}"))
                kwargs.update(premat=premat_path, postmat=postmat_path)
            subprocess.run(command, check=True)
            model.run(
                moving,
                reference,
                torch_output,
                warp=coefficient,
                interpolation=interpolation,
                output_dtype="float",
                **kwargs,
            )
            fsl = np.asarray(nib.load(fsl_output).dataobj, dtype=np.float64)
            candidate = np.asarray(nib.load(torch_output).dataobj, dtype=np.float64)
            difference = np.abs(candidate - fsl)
            tolerance = 5e-4 if interpolation == "trilinear" else 0.0
            records.append(
                {
                    "case": name,
                    "maximum_absolute_error": float(difference.max()),
                    "mean_absolute_error": float(difference.mean()),
                    "different_voxels": int(np.count_nonzero(difference)),
                    "tolerance": tolerance,
                    "passed": bool(difference.max() <= tolerance),
                }
            )
    timing_shape = (91, 109, 91)
    timing_affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    tx, ty, tz = np.indices(timing_shape, dtype=np.float32)
    timing_data = np.sin(tx / 11) + np.cos(ty / 13) + tz / 91
    timing_field = np.zeros((*timing_shape, 3), dtype=np.float32)
    timing_field[..., 0] = 0.31 + 0.04 * np.sin(ty / 17)
    timing_field[..., 1] = -0.23 + 0.03 * np.cos(tz / 19)
    timing_field[..., 2] = 0.17 + 0.02 * np.sin(tx / 23)
    timing_input = work / "timing_input.nii.gz"
    timing_reference = work / "timing_reference.nii.gz"
    timing_warp = work / "timing_warp.nii.gz"
    timing_fsl_output = work / "timing_fsl_output.nii.gz"
    timing_cpu_output = work / "timing_torch_cpu_output.nii.gz"
    timing_device_output = work / "timing_torch_device_output.nii.gz"
    save(nib.Nifti1Image(timing_data, timing_affine), timing_input)
    save(
        nib.Nifti1Image(np.zeros(timing_shape, dtype=np.float32), timing_affine),
        timing_reference,
    )
    save(nib.Nifti1Image(timing_field, timing_affine), timing_warp)
    fsl_timing_command = [
        args.applywarp,
        f"--in={timing_input}",
        f"--ref={timing_reference}",
        f"--warp={timing_warp}",
        "--rel",
        "--interp=trilinear",
        "--datatype=float",
        f"--out={timing_fsl_output}",
    ]

    def run_fsl_timing():
        subprocess.run(
            fsl_timing_command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    cpu_model = TorchApplyWarp("cpu")

    def run_torch_cpu_timing():
        cpu_model.run(
            timing_input,
            timing_reference,
            timing_cpu_output,
            warp=timing_warp,
            warp_convention="relative",
            interpolation="trilinear",
            output_dtype="float",
        )

    timings = {
        "fixture": {
            "shape": list(timing_shape),
            "voxel_size_mm": [2.0, 2.0, 2.0],
            "warp": "dense relative",
            "interpolation": "trilinear",
            "scope": "end-to-end process/API call including NIfTI read and write",
            "warmup_excluded": True,
            "repeats": args.timing_repeats,
        },
        "fsl_applywarp_cpu": measure(run_fsl_timing, args.timing_repeats),
        "torch_cpu": measure(run_torch_cpu_timing, args.timing_repeats),
    }
    if torch.device(args.device).type == "cuda":
        device_model = TorchApplyWarp(args.device)

        def run_torch_device_timing():
            device_model.run(
                timing_input,
                timing_reference,
                timing_device_output,
                warp=timing_warp,
                warp_convention="relative",
                interpolation="trilinear",
                output_dtype="float",
            )

        timings["torch_gpu"] = measure(
            run_torch_device_timing, args.timing_repeats, torch.cuda.synchronize
        )

    version_path = Path(os.environ.get("FSLDIR", "")) / "etc/fslversion"
    report = {
        "fsl_version": version_path.read_text().strip() if version_path.is_file() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda"
            else None
        ),
        "device": args.device,
        "precision": {
            "cuda_execution": torch.device(args.device).type == "cuda",
            "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
            "coordinate_and_cubic_dtype": "float64",
            "image_sampling_dtype": "float32",
            "reduced_precision_tensor_dtype": False,
        },
        "scope": (
            "dense relative and cubic coefficient fields; coordinate ramp; "
            "trilinear and nearest; with and without one premat and postmat"
        ),
        "fnirt_coefficients_tested": True,
        "cases": records,
        "timing": timings,
        "passed": all(record["passed"] for record in records),
    }
    rendered = json.dumps(report, indent=2)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--applywarp", default="applywarp")
    parser.add_argument("--fnirtfileutils", default="fnirtfileutils")
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--json-out")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
