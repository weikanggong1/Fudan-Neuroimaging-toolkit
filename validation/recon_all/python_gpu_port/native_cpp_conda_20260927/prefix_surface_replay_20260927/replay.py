"""Replay recon-all surface/posterior stages on a completed same-T1 MRI prefix.

This is a validation harness, not the production T1 entry point. It copies the
prior subject's MRI/stats prefix and computes new surface, annotation, parcel
volume and statistics outputs in an empty separate subject directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import nibabel as nib
import numpy as np
import torch

from fnit.recon_all.anatomical_stats_file import write_anatomical_stats
from fnit.recon_all.brain_volume_stats_python import compute_brain_volume_stats
from fnit.recon_all.gcsa_label_python import label_surface
from fnit.recon_all.mris_register_run import run_register_sphere
from fnit.recon_all.native_free import _folding_atlas, _project_parcels, _project_wmparc, _surface_pair
from fnit.recon_all.segstats_aseg_python import write_aseg_stats
from fnit.recon_all.segstats_wmparc_python import write_wmparc_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix_subject", type=Path)
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("assets_dir", type=Path)
    parser.add_argument("native_bin_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    prefix, subject, assets, binaries = (
        path.resolve() for path in (args.prefix_subject, args.subject_dir,
                                    args.assets_dir, args.native_bin_dir))
    if subject.exists() and any(subject.iterdir()):
        raise ValueError("subject_dir must be empty")
    subject.mkdir(parents=True, exist_ok=True)
    shutil.copytree(prefix / "mri", subject / "mri")
    shutil.copytree(prefix / "stats", subject / "stats")
    for name in ("surf", "label", "scripts"):
        (subject / name).mkdir()
    mri, surf, labels, stats = (subject / name for name in ("mri", "surf", "label", "stats"))
    prefix_names = ("orig.mgz", "nu.mgz", "norm.mgz", "brainmask.mgz",
                    "aseg.presurf.mgz", "brain.mgz", "antsdn.brain.mgz",
                    "wm.seg.mgz", "wm.asegedit.mgz", "wm.mgz", "filled.mgz")
    report = {"status": "running", "source_prefix": str(prefix),
              "subject_dir": str(subject), "device": args.device,
              "threads": args.threads, "stages": [],
              "prefix_sha256": {name: hashlib.sha256((mri / name).read_bytes()).hexdigest()
                                for name in prefix_names}}
    report_file = subject / "fnit-prefix-replay-run.json"
    started = time.perf_counter()
    torch.set_num_threads(args.threads)

    def stage(name, fn, *fn_args, **fn_kwargs):
        tick = time.perf_counter()
        try:
            value = fn(*fn_args, **fn_kwargs)
        except Exception as exc:
            report.update(status="failed", failed_stage=name,
                          error=repr(exc), total_seconds=time.perf_counter() - started)
            report_file.write_text(json.dumps(report, indent=2))
            raise
        report["stages"].append({"name": name, "seconds": time.perf_counter() - tick})
        report_file.write_text(json.dumps(report, indent=2))
        return value

    aseg = np.asarray(nib.load(str(mri / "aseg.auto.mgz")).dataobj).astype(np.int16)
    for hemi in ("lh", "rh"):
        result = stage(f"surface_{hemi}", _surface_pair,
                       subject, hemi, mri / "filled.mgz", mri / "norm.mgz", aseg,
                       device=args.device, threads=args.threads,
                       native_topology_binary=binaries / "mris_fix_topology",
                       native_surface_metrics_binary=binaries / "mris_place_surface",
                       native_inflate_binary=binaries / "mris_inflate",
                       native_registration=True, assets=assets)
        report.setdefault("surfaces", {})[hemi] = result
    for hemi in ("lh", "rh"):
        result = stage(f"register_{hemi}", run_register_sphere,
                       surf / f"{hemi}.sphere", surf / f"{hemi}.smoothwm",
                       surf / f"{hemi}.sulc", _folding_atlas(assets, hemi),
                       surf / f"{hemi}.sphere.reg", overlap_device="cpu")
        report.setdefault("registrations", {})[hemi] = result
    for hemi in ("lh", "rh"):
        for atlas, prefix_name in (("aparc", "DKaparc"),
                                   ("aparc.a2009s", "CDaparc"),
                                   ("aparc.DKTatlas", "DKTaparc")):
            atlas_file = assets / "average" / (
                f"{hemi}.{prefix_name}.atlas.acfb40.noaparc.i12.2016-08-02.gcs")
            stage(f"annot_{hemi}_{atlas}", label_surface, subject, hemi,
                  atlas_file, assets / "lib/bem/ic4.tri", assets / "lib/bem/ic7.tri",
                  labels / f"{hemi}.{atlas}.annot", device=args.device)
    stage("project_aparc_volumes", _project_parcels, subject, aseg)
    stage("project_wmparc", _project_wmparc, subject, aseg)
    volumes = stage("brain_volume_stats", compute_brain_volume_stats,
                    subject, assets / "ASegStatsLUT.txt")
    (stats / "brainvol.stats").write_text("".join(
        f"# Measure {name}, {name}, {name}, {value:.6f}, mm^3\n"
        for name, value in volumes.items()))
    stage("aseg_stats", write_aseg_stats, subject, assets / "ASegStatsLUT.txt",
          stats / "aseg.stats")
    stage("wmparc_stats", write_wmparc_stats, subject, assets / "WMParcStatsLUT.txt",
          stats / "wmparc.stats")
    for hemi in ("lh", "rh"):
        for atlas in ("aparc", "aparc.a2009s", "aparc.DKTatlas"):
            stage(f"stats_{hemi}_{atlas}", write_anatomical_stats,
                  subject, hemi, atlas, "white", volumes,
                  stats / f"{hemi}.{atlas}.stats", device=args.device)
    report.update(status="complete", total_seconds=time.perf_counter() - started)
    report_file.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "stages": len(report["stages"]),
                      "total_seconds": report["total_seconds"]}))


if __name__ == "__main__":
    main()
