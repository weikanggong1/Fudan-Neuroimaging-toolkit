"""Compare one fixed SynthMorph affine input under CPU and GPU precision settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import surfa as sf
import torch

from fnit.recon_all.sclimbic import _etiv_from_lta
from fnit.synthmorph import SynthMorph


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("moving", "template", "weights_dir", "official_aff",
                 "official_voxel", "prior_aff", "prior_voxel", "output_dir"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    refs = {name: sf.load_affine(str(getattr(args, name))) for name in
            ("official_aff", "official_voxel", "prior_aff", "prior_voxel")}
    official_etiv = _etiv_from_lta(args.official_voxel)
    torch.set_num_threads(args.threads)
    # These flags were left by SynthStrip in the tested recon-all prefix.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    report = {
        "input_sha256": {
            "synthstrip": sha256(args.moving),
            "mni305_template": sha256(args.template),
            "affine_weight": sha256(args.weights_dir / "synthmorph.affine.2.h5"),
        },
        "source_sha256": {
            "pipeline": sha256(Path(__import__("fnit.synthmorph.pipeline", fromlist=["x"]).__file__)),
            "models": sha256(Path(__import__("fnit.synthmorph.models", fromlist=["x"]).__file__)),
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "official_etiv_from_lta_mm3": official_etiv,
        "prior_etiv_from_lta_mm3": _etiv_from_lta(args.prior_voxel),
        "cases": {},
    }

    def run(name: str, model: SynthMorph, matmul_tf32: bool, cudnn_tf32: bool) -> None:
        # SynthMorph.__init__ enables both flags, so set them after construction.
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        if model.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = model(args.moving, args.template)
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        aff_path = args.output_dir / f"{name}.aff.lta"
        voxel_path = args.output_dir / f"{name}.voxel.lta"
        result.transform.save(str(aff_path))
        result.transform.convert(space="voxel").save(str(voxel_path))
        aff, voxel = (sf.load_affine(str(path)) for path in (aff_path, voxel_path))
        etiv = _etiv_from_lta(voxel_path)
        report["cases"][name] = {
            "device": str(model.device),
            "matmul_tf32": matmul_tf32,
            "cudnn_tf32": cudnn_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "seconds": seconds,
            "aff_max_abs_vs_official": float(np.max(np.abs(aff.matrix - refs["official_aff"].matrix))),
            "aff_max_abs_vs_prior": float(np.max(np.abs(aff.matrix - refs["prior_aff"].matrix))),
            "voxel_max_abs_vs_official": float(np.max(np.abs(voxel.matrix - refs["official_voxel"].matrix))),
            "voxel_max_abs_vs_prior": float(np.max(np.abs(voxel.matrix - refs["prior_voxel"].matrix))),
            "etiv_mm3": etiv,
            "etiv_error_vs_official_lta_mm3": etiv - official_etiv,
            "aff_sha256": sha256(aff_path),
            "voxel_sha256": sha256(voxel_path),
            "peak_cuda_allocated_mib": (torch.cuda.max_memory_allocated() / 1048576
                                        if model.device.type == "cuda" else None),
        }
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, seconds, report["cases"][name]["voxel_max_abs_vs_official"],
              report["cases"][name]["etiv_error_vs_official_lta_mm3"], flush=True)

    run("cpu_fp32", SynthMorph(weights=args.weights_dir, device="cpu", model="affine", extent=256), False, False)
    gpu = SynthMorph(weights=args.weights_dir, device=args.device, model="affine", extent=256)
    for name, matmul, cudnn in (
        ("gpu_tf32_default", True, True),
        ("gpu_fp32_both", False, False),
        ("gpu_fp32_matmul_only", False, True),
        ("gpu_fp32_cudnn_only", True, False),
        ("gpu_tf32_default_repeat", True, True),
    ):
        run(name, gpu, matmul, cudnn)


if __name__ == "__main__":
    main()
