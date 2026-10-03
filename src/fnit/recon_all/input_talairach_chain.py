"""Connect T1 import, conform, SynthStrip and Talairach affine registration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.synthstrip import SynthStrip

from .input_chain import run_input_chain
from .talairach_synthmorph import register_talairach


def write_voxel_lta_from_ras(source_lta: str | Path, output_lta: str | Path) -> None:
    """Convert the fixed SynthMorph RAS LTA to voxel coordinates.

    Read a type-1 LTA with source/destination volume geometry and write a
    type-0 LTA. Matrix values are float64 to match the previous Surfa
    conversion; the source and destination geometry text is retained.
    """
    lines = Path(source_lta).read_text().splitlines()
    if not lines[0].startswith("type      = 1") or "nxforms   = 1" not in lines:
        raise ValueError("expected a single RAS-to-RAS SynthMorph LTA")
    matrix_row = lines.index("1 4 4") + 1
    matrix = np.asarray([[float(value) for value in row.split()]
                         for row in lines[matrix_row:matrix_row + 4]], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("expected a 4x4 LTA transform")

    def volume_affine(section: str) -> np.ndarray:
        start = lines.index(section) + 1
        geometry = dict(row.split("=", 1) for row in lines[start:start + 8]
                        if "=" in row)
        geometry = {key.strip(): value.split("#", 1)[0].strip()
                    for key, value in geometry.items()}
        dims = np.fromstring(geometry["volume"], sep=" ", dtype=np.float64)
        sizes = np.fromstring(geometry["voxelsize"], sep=" ", dtype=np.float64)
        axes = np.asarray([np.fromstring(geometry[key], sep=" ", dtype=np.float64)
                           for key in ("xras", "yras", "zras")])
        center = np.fromstring(geometry["cras"], sep=" ", dtype=np.float64)
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = axes.T * sizes
        affine[:3, 3] = center - affine[:3, :3] @ (dims / 2)
        return affine

    source = volume_affine("src volume info")
    target = volume_affine("dst volume info")
    voxel = np.linalg.inv(target) @ matrix @ source
    lines[0] = "type      = 0 # LINEAR_VOX_TO_VOX"
    lines[matrix_row:matrix_row + 4] = [
        " ".join(f"{float(value):.15e}" for value in row) for row in voxel
    ]
    output = Path(output_lta)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def save_synthstrip_mgh(source_file: str | Path, stripped_image,
                        output_file: str | Path) -> None:
    """将SynthStrip结果写成源conform网格和dtype的3D MGH/MGZ。

    source_file为conform MGH/MGZ；stripped_image为同网格nibabel图像；
    output_file为输出路径。整数输入要求输出可精确表示，避免截断强度。
    形状、空间、非有限值或整数范围不符时抛ValueError；不改变模型精度。
    """
    source = nib.load(str(source_file))
    values = np.asanyarray(stripped_image.dataobj)
    if not isinstance(source, nib.MGHImage) or len(source.shape) != 3:
        raise ValueError("expected a 3D conformed MGH/MGZ input")
    if values.shape != source.shape or not np.array_equal(stripped_image.affine, source.affine):
        raise ValueError("SynthStrip output must preserve the conformed grid")
    dtype = source.get_data_dtype()
    if not np.isfinite(values).all():
        raise ValueError("SynthStrip output must be finite")
    cast = values.astype(dtype)
    if np.issubdtype(dtype, np.integer) and not np.array_equal(values, cast):
        raise ValueError("SynthStrip output is not exactly representable in the source integer dtype")
    header = source.header.copy()
    header.set_data_dtype(dtype)
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.MGHImage(cast, source.affine, header), str(output))


def run_input_talairach_chain(t1: str | Path, subject_dir: str | Path,
                              weights_dir: str | Path, assets_dir: str | Path,
                              *, device: str = "cpu", threads: int = 4) -> dict:
    """Produce the fixed profile's orig, synthstrip and talairach.xfm files."""
    weights = Path(weights_dir)
    template = Path(assets_dir) / "average/mni305.cor.stripped.mgz"
    for path in (weights / "synthstrip.1.pt",
                 weights / "synthmorph.affine.2.h5", template):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = Path(subject_dir)
    result = run_input_chain(t1, root, device=device)
    strip_file = root / "mri/synthstrip.mgz"
    started = time.perf_counter()
    forwards = []
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        strip = SynthStrip(weights=weights, device=device, threads=threads, configure_precision=False)
        # 经验证的SynthStrip cuDNN FP32例外在模型构造后施加。
        torch.backends.cudnn.allow_tf32 = False
        stripped = strip(result["conformed"], precision_report=forwards)
        save_synthstrip_mgh(source_file=result["conformed"],
                           stripped_image=stripped.image, output_file=strip_file)
    finally:
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
    del strip
    strip_seconds = time.perf_counter() - started
    xfm = root / "mri/transforms/talairach.xfm"
    lta = root / "mri/transforms/synthmorph.mni305/aff.lta"
    started = time.perf_counter()
    child_gpu = None
    if torch.device(device).type == "cuda":
        memory_report = root / "scripts/talairach-child-gpu.json"
        child_env = dict(os.environ)
        child_env.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
        subprocess.run([
            sys.executable, "-m", "fnit.recon_all.talairach_synthmorph",
            str(strip_file), str(template), "--weights", str(weights),
            "--xfm", str(xfm), "--lta", str(lta), "--device", device,
            "--threads", str(threads), "--memory-report", str(memory_report),
        ], check=True, env=child_env)
        child_gpu = json.loads(memory_report.read_text())
    else:
        register_talairach(strip_file, template, weights, xfm, lta,
                           device=device, threads=threads, precision_report=forwards)
    voxel_lta = root / "mri/transforms/talairach.xfm.lta"
    write_voxel_lta_from_ras(source_lta=lta, output_lta=voxel_lta)
    talairach_seconds = time.perf_counter() - started
    return {**result, "synthstrip": str(strip_file), "talairach_xfm": str(xfm),
            "talairach_affine_lta": str(lta),
            "talairach_voxel_lta": str(voxel_lta), "threads": threads,
            "synthstrip_seconds": strip_seconds,
            "talairach_seconds": talairach_seconds,
            "talairach_child_gpu": child_gpu, "actual_forwards": forwards}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("t1", type=Path)
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_input_talairach_chain(
        args.t1, args.subject_dir, args.weights_dir, args.assets_dir,
        device=args.device, threads=args.threads), indent=2))


if __name__ == "__main__":
    main()
