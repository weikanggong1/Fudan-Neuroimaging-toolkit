"""Connect T1 import, conform, SynthStrip and Talairach affine registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import surfa as sf
import torch

from fnit.synthstrip import SynthStrip

from .input_chain import run_input_chain
from .talairach_synthmorph import register_talairach


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
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        strip = SynthStrip(weights=weights, device=device, threads=threads)
        # The SynthStrip constructor enables TF32; exact uint8 masks need FP32 cuDNN.
        torch.backends.cudnn.allow_tf32 = False
        strip(result["conformed"]).image.save(str(strip_file))
    finally:
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
    strip_seconds = time.perf_counter() - started
    xfm = root / "mri/transforms/talairach.xfm"
    lta = root / "mri/transforms/synthmorph.mni305/aff.lta"
    started = time.perf_counter()
    register_talairach(strip_file, template, weights, xfm, lta,
                       device=device, threads=threads)
    voxel_lta = root / "mri/transforms/talairach.xfm.lta"
    sf.load_affine(str(lta)).convert(space="voxel").save(str(voxel_lta))
    talairach_seconds = time.perf_counter() - started
    return {**result, "synthstrip": str(strip_file), "talairach_xfm": str(xfm),
            "talairach_affine_lta": str(lta),
            "talairach_voxel_lta": str(voxel_lta), "threads": threads,
            "synthstrip_seconds": strip_seconds,
            "talairach_seconds": talairach_seconds}


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
