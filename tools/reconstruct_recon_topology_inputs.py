"""由冻结 FNIT MRI 前段在新目录重建拓扑阶段的 nofix 与 qsphere。

只读取 FNIT 自产 filled、norm；不接收官方参考，也不续跑或补写整例。
诊断坐标为 surface RAS/mm，真实来源、SHA 和阶段墙钟写入私有回执。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import torch

from fnit.recon_all.extract_main_component_python import extract_main_component
from fnit.recon_all.inflate_standard_run import run_standard_inflate
from fnit.recon_all.pretess_python import pretess_mgh
from fnit.recon_all.smooth_surface_python import smooth_surface
from fnit.recon_all.sphere_quick_python import write_quick_sphere
from fnit.recon_all.tessellate_gpu import tessellate_mgh, write_quad_surface


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*, filled: Path, norm: Path, hemisphere: str, output_directory: Path,
        device: str = "cuda:0") -> dict:
    """重放生产 nofix/qsphere 前缀；所有输出写新目录，失败不标完成。

    filled/norm 是同一自产被试 conformed 网格的三维 MGZ；hemisphere
    为 lh/rh，分别提取标签255/127；device 默认 cuda:0，必须显式
    可用，不自动退回 CPU。输出包含 pretess MGZ、raw quad、nofix、
    smoothwm.nofix、inflated.nofix、qsphere.nofix 与私有 JSON 回执。
    几何头由 pretess MGZ 提供，坐标为 surface RAS/mm；算法失败原样抛错。
    """
    started = time.perf_counter()
    filled, norm, output_directory = Path(filled), Path(norm), Path(output_directory)
    if hemisphere not in ("lh", "rh"):
        raise ValueError("hemisphere must be lh or rh")
    if output_directory.exists():
        raise FileExistsError(output_directory)
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("explicit available CUDA device required for production input replay")
    input_sha = {"filled": _sha(filled), "norm": _sha(norm)}
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output_directory.mkdir(parents=True, mode=0o700)
    surf, mri = output_directory / "surf", output_directory / "mri"
    surf.mkdir()
    mri.mkdir()
    code = 255 if hemisphere == "lh" else 127
    stage_seconds = {}
    paths = {
        "pretess": mri / f"filled-pretess{code}.mgz",
        "raw": surf / f"{hemisphere}.orig.raw.quad",
        "nofix": surf / f"{hemisphere}.orig.nofix",
        "smooth": surf / f"{hemisphere}.smoothwm.nofix",
        "inflated": surf / f"{hemisphere}.inflated.nofix",
        "qsphere": surf / f"{hemisphere}.qsphere.nofix",
    }
    try:
        tick = time.perf_counter()
        pretess_mgh(filled, code, norm, paths["pretess"])
        stage_seconds["pretess"] = time.perf_counter() - tick
        tick = time.perf_counter()
        vertices, quads = tessellate_mgh(paths["pretess"], code, device=device)
        write_quad_surface(paths["raw"], vertices, quads,
                           nib.load(str(paths["pretess"])), Path("../mri") / paths["pretess"].name)
        extract_main_component(paths["raw"], paths["nofix"])
        stage_seconds["tessellate_and_component"] = time.perf_counter() - tick
        tick = time.perf_counter()
        smooth_surface(paths["nofix"], paths["smooth"], device=device)
        run_standard_inflate(input_surface=paths["smooth"],
            inflated_output=paths["inflated"], sulc_output=None,
            save_sulc=False, backend="torch", device=device, profile=False)
        write_quick_sphere(paths["inflated"], paths["qsphere"])
        stage_seconds["smooth_inflate_qsphere"] = time.perf_counter() - tick
        status = "complete"
    except BaseException:
        status = "failed"
        raise
    finally:
        receipt = dict(schema_version=1, status=status, input_sha256=input_sha,
            output_sha256={key: _sha(path) for key, path in paths.items() if path.is_file()},
            stage_seconds=stage_seconds,
            full_wall_seconds_through_before_receipt_write=time.perf_counter() - started,
            device=device, torch_matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            torch_cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
            raw_t1_whole_execution="not_assessed",
            input_reconstruction_exact_match_to_failed_worker="not_assessed")
        (output_directory / "reconstruction.private.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filled", type=Path, required=True)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    result = run(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
