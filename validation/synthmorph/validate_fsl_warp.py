"""Real-T1w check of SynthMorph RAS-to-FSL warp conversion and applywarp."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit import SynthMorph, TorchApplyWarp, convert_warp_to_fsl


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    mask = (first != 0) | (second != 0)
    difference = first[mask] - second[mask]
    return {
        "union_voxels": int(mask.sum()),
        "pearson_r": float(np.corrcoef(first[mask], second[mask])[0, 1]),
        "mae": float(np.abs(difference).mean()),
        "p99_abs": float(np.quantile(np.abs(difference), 0.99)),
        "max_abs": float(np.abs(difference).max()),
    }


def comparison_figure(output, synthmorph, torch_result, fsl_result):
    from PIL import Image, ImageDraw, ImageFont

    y = synthmorph.shape[1] // 2
    upper = float(np.quantile(synthmorph[synthmorph > 0], 0.995))
    comparisons = (
        (synthmorph, "SynthMorph direct"),
        (torch_result, "TorchApplyWarp"),
        (fsl_result, "FSL applywarp"),
    )
    differences = (
        (np.abs(torch_result - synthmorph), "Torch vs direct"),
        (np.abs(fsl_result - synthmorph), "FSL vs direct"),
        (np.abs(torch_result - fsl_result), "Torch vs FSL"),
    )
    difference_max = max(
        float(np.quantile(values, 0.995)) for values, _ in differences
    )
    height, width = np.rot90(synthmorph[:, y, :]).shape
    title_height = 34
    figure = Image.new("RGB", (width * 3, (height + title_height) * 2 + 28), "white")
    draw = ImageDraw.Draw(figure)
    font = ImageFont.load_default(size=18)
    for column, (values, title) in enumerate(comparisons):
        pixels = np.rot90(values[:, y, :])
        pixels = np.uint8(np.clip(pixels / upper, 0, 1) * 255)
        figure.paste(Image.fromarray(pixels, mode="L").convert("RGB"),
                     (column * width, title_height))
        draw.text((column * width + 8, 6), title, fill="black", font=font)
    for column, (values, title) in enumerate(differences):
        pixels = np.rot90(values[:, y, :])
        pixels = np.uint8(np.clip(pixels / difference_max, 0, 1) * 255)
        rgb = np.stack((pixels, pixels // 3, np.zeros_like(pixels)), axis=-1)
        figure.paste(Image.fromarray(rgb, mode="RGB"),
                     (column * width, height + 2 * title_height))
        draw.text((column * width + 8, height + title_height + 6),
                  title, fill="black", font=font)
    draw.text((8, (height + title_height) * 2 + 4),
              f"Absolute difference: 0 to {difference_max:.4f} intensity units (99.5th percentile)",
              fill="black", font=ImageFont.load_default(size=14))
    figure.save(output / "synthmorph_fsl_warp_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fsl-applywarp")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = SynthMorph(weights=args.weights, device=args.device, model="joint")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = model(moving=args.moving, fixed=args.fixed)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    registration_seconds = time.perf_counter() - start
    result.moved.save(output / "synthmorph_moved.nii.gz")
    start = time.perf_counter()
    warp = convert_warp_to_fsl(result.transform, moving=args.moving, fixed=args.fixed)
    conversion_seconds = time.perf_counter() - start
    warp.save(output / "synthmorph_fsl_warp.nii.gz")
    registration_peak = (
        float(torch.cuda.max_memory_allocated() / 2**30)
        if args.device.startswith("cuda") else None
    )
    warper = TorchApplyWarp(device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    applied = warper.run(
        input=args.moving,
        reference=args.fixed,
        output=output / "torch_applywarp.nii.gz",
        warp=output / "synthmorph_fsl_warp.nii.gz",
        interpolation="trilinear",
        output_dtype="float",
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    torch_seconds = time.perf_counter() - start
    moved_data = np.asarray(result.moved.dataobj, dtype=np.float32)
    torch_data = np.asarray(applied.image.dataobj, dtype=np.float32)
    report = {
        "data": "OpenNeuro ds000114 defaced T1w, examples/data",
        "moving_sha256": sha256(args.moving),
        "fixed_sha256": sha256(args.fixed),
        "converter_sha256": sha256(
            Path(__file__).resolve().parents[2] / "src/fnit/synthmorph/fsl_warp.py"
        ),
        "device": args.device,
        "output_shape": list(warp.shape[:3]),
        "registration_seconds_loaded_model": registration_seconds,
        "conversion_seconds": conversion_seconds,
        "torch_applywarp_seconds_including_io": torch_seconds,
        "registration_peak_cuda_allocated_gib": registration_peak,
        "applywarp_peak_cuda_allocated_gib": (
            float(torch.cuda.max_memory_allocated() / 2**30)
            if args.device.startswith("cuda") else None
        ),
        "fsl_warp_intent": int(warp.header["intent_code"]),
        "torch_convention": applied.qc["warp_convention"],
        "synthmorph_vs_torch_applywarp": metrics(moved_data, torch_data),
    }
    if args.fsl_applywarp:
        report["fsl_applywarp_binary"] = args.fsl_applywarp
        command = [
            args.fsl_applywarp,
            f"--in={args.moving}",
            f"--ref={args.fixed}",
            f"--warp={output / 'synthmorph_fsl_warp.nii.gz'}",
            "--rel", "--interp=trilinear", "--datatype=float",
            f"--out={output / 'fsl_applywarp.nii.gz'}",
        ]
        start = time.perf_counter()
        subprocess.run(command, check=True)
        report["fsl_applywarp_seconds_including_io"] = time.perf_counter() - start
        fsl_image = nib.load(output / "fsl_applywarp.nii.gz")
        fsl_data = np.asarray(fsl_image.dataobj, dtype=np.float32)
        report["fsl_vs_torch_applywarp"] = metrics(fsl_data, torch_data)
        report["synthmorph_vs_fsl_applywarp"] = metrics(moved_data, fsl_data)
        report["fsl_shape_equal"] = fsl_data.shape == torch_data.shape
        report["fsl_affine_max_abs"] = float(
            np.max(np.abs(fsl_image.affine - applied.image.affine))
        )
        comparison_figure(output, moved_data, torch_data, fsl_data)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
