"""从空目录和真实单幅 T1 连续执行当前 recon-all 的体积前段至 filled。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from fnit.recon_all.ants_denoise_python import denoise_volume
from fnit.recon_all.ca_normalize_python import run_ca_normalize
from fnit.recon_all.fill_cutting_plane_python import fill_mgz
from fnit.recon_all.input_talairach_chain import run_input_talairach_chain
from fnit.recon_all.mri_mask_gpu import mask_volume
from fnit.recon_all.n4_itk import correct_volume
from fnit.recon_all.n4_wrapper import make_nu
from fnit.recon_all.native_free import (
    _run_native_em_register, _run_native_wm_edit, _run_native_wm_segment,
    _segment_callosum,
)
from fnit.recon_all.normalization import normalize_t1
from fnit.recon_all.normalization.aseg_pipeline import normalize_t1_aseg
from fnit.recon_all.pretess_python import pretess_mgh
from fnit.recon_all.sclimbic import mri_entowm_seg
from fnit.recon_all.wm_edits_python import fix_ento_wm
from fnit.synthseg_parc import SynthSeg


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_prefix(t1: Path, subject: Path, weights: Path, assets: Path,
               native_bin: Path, *, device: str, threads: int,
               code_commit: str) -> dict:
    """执行与生产调度同序的 MRI 前段；报告失败阶段，输出均在 conform 网格。"""
    if subject.exists() and any(subject.iterdir()):
        raise ValueError("subject directory must be empty")
    subject.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    report = {"code_commit": code_commit, "script_sha256": _sha256(Path(__file__)),
              "host": platform.node(), "cpu": platform.processor(),
              "input_sha256": _sha256(t1),
              "device": device, "threads": threads, "precision":
              {"tf32_default": True, "fp32_exceptions": ["SynthStrip", "SynthSeg",
                                                          "Talairach affine"]},
              "gpu_name": (torch.cuda.get_device_name(device)
                           if torch.device(device).type == "cuda" else None),
              "binaries": {name: _sha256(native_bin / name) for name in
                           ("fnit_n4_itk", "mri_em_register", "mri_segment",
                            "mri_edit_wm_with_aseg")}, "stages": [], "status": "running"}
    report_file = subject / "volume-prefix-run.json"
    started = time.perf_counter()

    def stage(name: str, function, *args, **kwargs):
        tick = time.perf_counter()
        try:
            result = function(*args, **kwargs)
        except Exception as error:
            report.update(status="failed", failed_stage=name, error=repr(error),
                          total_seconds=time.perf_counter() - started)
            report_file.write_text(json.dumps(report, indent=2) + "\n")
            raise
        report["stages"].append({"name": name, "seconds": time.perf_counter() - tick})
        report_file.write_text(json.dumps(report, indent=2) + "\n")
        return result

    first = stage("input_talairach", run_input_talairach_chain,
                  t1, subject, weights, assets, device=device, threads=threads)
    mri, stats = subject / "mri", subject / "stats"
    stats.mkdir(exist_ok=True)
    (subject / "scripts").mkdir(exist_ok=True)
    (mri / "tmp").mkdir(exist_ok=True)
    nu0 = mri / "tmp/nu0.mgz"
    stage("n4", correct_volume, mri / "orig.mgz", nu0,
          binary=native_bin / "fnit_n4_itk")
    stage("nu", make_nu, mri / "orig.mgz", nu0,
          first["talairach_xfm"], mri / "nu.mgz")
    stage("T1_normalize", normalize_t1, mri / "nu.mgz",
          first["talairach_xfm"], mri / "T1.mgz", device=device)
    stage("brainmask", mask_volume, mri / "T1.mgz",
          first["synthstrip"], mri / "brainmask.mgz", device=device)
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    previous_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        result = stage("SynthSeg", lambda: SynthSeg(weights=weights, device=device,
                                                      threads=threads)(mri / "orig.mgz",
                         keep_geometry=True, color_lut=assets / "FreeSurferColorLUT.txt"))
    finally:
        torch.backends.cudnn.allow_tf32 = previous_tf32
    result.segmentation.save(str(mri / "synthseg.rca.mgz"))
    result.write_volumes_csv(mri / "orig.mgz", stats / "synthseg.vol.csv")
    lta = mri / "transforms/talairach.lta"
    gca = assets / "average/RB_all_2020-01-02.gca"
    stage("mri_em_register", _run_native_em_register,
          native_bin / "mri_em_register", mri, gca, assets)
    stage("mri_ca_normalize", run_ca_normalize, mri / "nu.mgz",
          mri / "brainmask.mgz", gca, lta, mri / "norm.mgz", mri / "ctrl_pts.mgz")
    stage("mri_cc", _segment_callosum, mri)
    stage("brain_second_normalize", normalize_t1_aseg,
          mri / "norm.mgz", mri / "aseg.presurf.mgz", mri / "brainmask.mgz",
          mri / "brain.mgz", device="cpu")
    stage("entowm", mri_entowm_seg, mri / "nu.mgz", mri / "entowm.mgz",
          weights, device="cpu", stats_path=stats / "entowm.stats",
          talairach_lta=mri / "transforms/talairach.xfm.lta")
    stage("ants_denoise", denoise_volume, mri / "brain.mgz", mri / "antsdn.brain.mgz")
    stage("mri_segment", _run_native_wm_segment,
          native_bin / "mri_segment", mri, assets)
    stage("mri_edit_wm_with_aseg", _run_native_wm_edit,
          native_bin / "mri_edit_wm_with_aseg", mri, assets)
    stage("wm_pretess", pretess_mgh, mri / "wm.asegedit.mgz", "wm",
          mri / "norm.mgz", mri / "wm.mgz")
    stage("wm_fix_ento", fix_ento_wm, mri / "wm.mgz", mri / "entowm.mgz",
          mri / "wm.mgz", level=3, left_value=255, right_value=255)
    stage("wm_fix_acj", fix_ento_wm, mri / "wm.mgz", mri / "aseg.presurf.mgz",
          mri / "wm.mgz", level=3, left_value=255, right_value=255, acj=True)
    stage("mri_fill", fill_mgz, mri / "wm.mgz", mri / "aseg.presurf.mgz", lta,
          assets / "SubCorticalMassLUT.txt", mri / "filled.mgz",
          subject / "scripts/ponscc.cut.log")
    report.update(status="complete", total_seconds=time.perf_counter() - started)
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("t1", "subject", "weights", "assets", "native_bin"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_prefix(args.t1, args.subject, args.weights, args.assets,
                                args.native_bin, device=args.device,
                                threads=args.threads,
                                code_commit=args.code_commit), indent=2))


if __name__ == "__main__":
    main()
