"""Isolated, CPU-only v3 final-aseg diagnosis on one frozen FreeSurfer subject.

The candidate and official subject directories are read-only. All generated
volumes, native controls, and the JSON report are written under --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import nibabel as nib
import numpy as np

from fnit.recon_all import relabel_hypointensities_python as relabel
from fnit.recon_all import surf2volseg_fix_python as fix
from fnit.recon_all import volmask_python as volmask


INPUTS = (
    "mri/aseg.presurf.mgz", "mri/aseg.mgz",
    "surf/lh.white", "surf/lh.pial", "surf/rh.white", "surf/rh.pial",
    "label/lh.cortex.label", "label/rh.cortex.label",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def values(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def compare(left: Path, right: Path) -> dict:
    a, b = nib.load(str(left)), nib.load(str(right))
    av, bv = np.asarray(a.dataobj), np.asarray(b.dataobj)
    if av.shape != bv.shape:
        raise ValueError(f"shape mismatch: {left}, {right}")
    unequal = av != bv
    first = np.argwhere(unequal)
    return {
        "unequal_voxels": int(np.count_nonzero(unequal)),
        "shape": list(av.shape),
        "first_ijk": first[0].tolist() if len(first) else None,
        "first_values": [float(av[tuple(first[0])]), float(bv[tuple(first[0])])]
        if len(first) else None,
        "dtype": [str(a.get_data_dtype()), str(b.get_data_dtype())],
        "header_exact": bool(a.header.binaryblock == b.header.binaryblock),
        "affine_max_abs": float(np.max(np.abs(a.affine - b.affine))),
    }


def int32_mgh(source: Path, output: Path) -> None:
    image = nib.load(str(source))
    header = image.header.copy()
    header.set_data_dtype(np.int32)
    nib.save(nib.MGHImage(np.asarray(image.dataobj).astype(np.int32),
                          image.affine, header=header), str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--native-bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-native", action="store_true",
                        help="reuse generated Python volumes after native setup failure")
    args = parser.parse_args()
    candidate, official, out = args.candidate, args.official, args.out
    if out.exists() and not args.resume_native:
        raise FileExistsError(out)
    for rel in INPUTS:
        if not (candidate / rel).is_file():
            raise FileNotFoundError(candidate / rel)
    if not args.resume_native:
        out.mkdir(parents=True)
    report = json.loads((out / "report.json").read_text()) if args.resume_native else {
        "candidate": str(candidate), "official": str(official),
        "input_sha256": {rel: sha256(candidate / rel) for rel in INPUTS},
        "official_sha256": {rel: sha256(official / rel) for rel in
                            ("mri/aseg.presurf.mgz", "mri/aseg.presurf.hypos.mgz",
                             "mri/ribbon.mgz", "mri/aseg.mgz")},
        "implementation_sha256": {
            name: sha256(Path(module.__file__)) for name, module in
            (("volmask", volmask), ("relabel", relabel), ("fix", fix))},
        "seconds": {}, "comparisons": {}, "status": "running",
    }
    report_file = out / "report.json"

    def save() -> None:
        report_file.write_text(json.dumps(report, indent=2) + "\n")

    def timed(name: str, function, *positional) -> None:
        start = time.perf_counter()
        function(*positional)
        report["seconds"][name] = time.perf_counter() - start
        save()

    def native(name: str, command: list[str]) -> None:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                   FREESURFER_HOME=str(args.native_bin.parent),
                   SUBJECTS_DIR=str(candidate.parent))
        start = time.perf_counter()
        with (out / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=out, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        report["seconds"][name] = time.perf_counter() - start
        save()

    try:
        cp, op = candidate / "mri", official / "mri"
        report["comparisons"]["presurf_candidate_official"] = compare(
            cp / "aseg.presurf.mgz", op / "aseg.presurf.mgz")
        report["comparisons"]["baseline_aseg_official"] = compare(
            cp / "aseg.mgz", op / "aseg.mgz")
        report["comparisons"]["official_hypos_presurf"] = compare(
            op / "aseg.presurf.hypos.mgz", op / "aseg.presurf.mgz")

        if args.resume_native:
            for name in ("ribbon.mgz", "hypos.python.mgz", "aseg.python.mgz"):
                if not (out / name).is_file():
                    raise FileNotFoundError(out / name)
            if any(sha256(candidate / rel) != digest
                   for rel, digest in report["input_sha256"].items()):
                raise RuntimeError("frozen candidate input changed before resuming")
        else:
            timed("python_volmask", volmask.write_ribbon,
                  cp / "aseg.presurf.mgz", candidate / "surf", out, args.lut)
            timed("python_relabel", relabel.relabel_volume,
                  cp / "aseg.presurf.mgz", candidate / "surf", out / "hypos.python.mgz")
            timed("python_fix", fix.fix_presurf_volume,
                  out / "hypos.python.mgz", out / "ribbon.mgz", candidate / "surf",
                  candidate / "label", out / "aseg.float.python.mgz")
            timed("python_int32_save", int32_mgh,
                  out / "aseg.float.python.mgz", out / "aseg.python.mgz")

        if not args.resume_native or not (out / "hypos.native.mgz").is_file():
            native("native_relabel", [
            str(args.native_bin / "mri_relabel_hypointensities"),
            str(cp / "aseg.presurf.mgz"), str(candidate / "surf"),
            str(out / "hypos.native.mgz")])
        native("native_fix", [
            str(args.native_bin / "mri_surf2volseg"), "--o", str(out / "aseg.native.mgz"),
            "--i", str(out / "hypos.python.mgz"),
            "--fix-presurf-with-ribbon", str(out / "ribbon.mgz"),
            "--threads", "4",
            "--lh-cortex-mask", str(candidate / "label/lh.cortex.label"),
            "--lh-white", str(candidate / "surf/lh.white"),
            "--lh-pial", str(candidate / "surf/lh.pial"),
            "--rh-cortex-mask", str(candidate / "label/rh.cortex.label"),
            "--rh-white", str(candidate / "surf/rh.white"),
            "--rh-pial", str(candidate / "surf/rh.pial")])

        for name, left, right in (
            ("ribbon_candidate_official", out / "ribbon.mgz", op / "ribbon.mgz"),
            ("hypos_candidate_official", out / "hypos.python.mgz", op / "aseg.presurf.hypos.mgz"),
            ("hypos_python_native_same_input", out / "hypos.python.mgz", out / "hypos.native.mgz"),
            ("aseg_python_native_same_input", out / "aseg.python.mgz", out / "aseg.native.mgz"),
            ("aseg_python_official", out / "aseg.python.mgz", op / "aseg.mgz"),
        ):
            report["comparisons"][name] = compare(left, right)

        reference = values(op / "aseg.mgz")
        before = values(cp / "aseg.mgz")
        after = values(out / "aseg.python.mgz")
        report["net_effect"] = {
            "changed_candidate_voxels": int(np.count_nonzero(before != after)),
            "wrong_to_right": int(np.count_nonzero((before != reference) & (after == reference))),
            "right_to_wrong": int(np.count_nonzero((before == reference) & (after != reference))),
            "wrong_to_other_wrong": int(np.count_nonzero(
                (before != reference) & (after != reference) & (before != after))),
        }
        report["source_input_sha256_unchanged"] = all(
            sha256(candidate / rel) == digest
            for rel, digest in report["input_sha256"].items())
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
