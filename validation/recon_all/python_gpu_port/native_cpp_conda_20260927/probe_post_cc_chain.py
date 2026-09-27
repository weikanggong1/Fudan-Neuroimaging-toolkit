"""Replay the existing Python aseg postprocessing ports on an isolated subject.

Run only after the source E2E job has exited successfully. The source and
official subjects are read-only; all outputs go into a fresh scratch copy.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import nibabel as nib
import numpy as np

from fnit.recon_all import relabel_hypointensities_python as relabel
from fnit.recon_all import surf2volseg_fix_python as surf2volseg
from fnit.recon_all import volmask_python as volmask


INPUTS = (
    "mri/aseg.presurf.mgz",
    "surf/lh.white", "surf/lh.pial", "surf/rh.white", "surf/rh.pial",
    "label/lh.cortex.label", "label/rh.cortex.label",
)
OUTPUTS = (
    "mri/ribbon.mgz", "mri/lh.ribbon.mgz", "mri/rh.ribbon.mgz",
    "mri/aseg.presurf.hypos.mgz", "mri/aseg.mgz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".mgz" else data


def _compare(candidate: Path, official: Path) -> dict:
    a, b = nib.load(str(candidate)), nib.load(str(official))
    if a.shape != b.shape:
        return {"shape": [[int(i) for i in a.shape], [int(i) for i in b.shape]],
                "shape_equal": False}
    av, bv = np.asarray(a.dataobj), np.asarray(b.dataobj)
    unequal = av != bv
    first = np.argwhere(unequal)
    result = {
        "shape": [int(i) for i in a.shape],
        "dtype": [str(a.get_data_dtype()), str(b.get_data_dtype())],
        "unequal_voxels": int(np.count_nonzero(unequal)),
        "first_unequal_ijk": first[0].tolist() if len(first) else None,
        "affine_max_abs": float(np.max(np.abs(a.affine - b.affine))),
        "header_exact": a.header.binaryblock == b.header.binaryblock,
    }
    ar, br = _raw(candidate), _raw(official)
    result["decompressed_mgh_exact"] = ar == br
    result["sha256"] = [_sha256(candidate), _sha256(official)]
    return result


def _write_int32_aseg(float_file: Path, output: Path) -> None:
    """Mirror native mri_surf2volseg's MRIalloc(..., MRI_INT) result type."""
    image = nib.load(str(float_file))
    header = image.header.copy()
    header.set_data_dtype(np.int32)
    data = np.asarray(image.dataobj).astype(np.int32)
    nib.save(nib.MGHImage(data, image.affine, header=header), str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--color-lut", type=Path, required=True)
    parser.add_argument("--e2e-status", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()

    if not args.e2e_status.is_file() or args.e2e_status.read_text().strip() != "exit=0":
        raise RuntimeError("candidate E2E exit=0 marker is required before the probe")
    run = args.candidate / "fnit-native-free-run.json"
    if json.loads(run.read_text()).get("status") != "complete":
        raise RuntimeError("candidate run report is not complete")
    for relative in INPUTS:
        if not (args.candidate / relative).is_file():
            raise FileNotFoundError(args.candidate / relative)
    if not args.color_lut.is_file():
        raise FileNotFoundError(args.color_lut)
    if args.scratch.exists():
        raise FileExistsError(f"scratch must be fresh: {args.scratch}")

    args.scratch.mkdir(parents=True)
    report_file = args.scratch / "report.json"
    report = {"status": "running", "candidate": str(args.candidate),
              "official": str(args.official), "color_lut": str(args.color_lut),
              "integration_dependencies": {
                  "source_aseg": "mri_cc -> aseg.auto -> aseg.presurf",
                  "ribbon": "both white/pial surfaces + aseg.presurf + FreeSurferColorLUT",
                  "hypos": "aseg.presurf + both white surfaces",
                  "final_aseg": "hypos + ribbon + both white/pial surfaces + both cortex labels",
                  "missing_entowm": not (args.candidate / "mri/entowm.mgz").is_file(),
                  "cortex_label_limit": "v2 runner uses no-GA label_cortex; official --fix-ga also uses entowm",
                  "runner_handoff": "reload final int32 aseg before project_aparc and later stats",
              },
              "input_sha256": {name: _sha256(args.candidate / name)
                               for name in INPUTS},
              "port_sha256": {name: _sha256(Path(module.__file__)) for name, module in (
                  ("volmask", volmask), ("relabel", relabel),
                  ("surf2volseg_fix", surf2volseg))},
              "seconds": {}, "comparisons": {}}

    def save() -> None:
        report_file.write_text(json.dumps(report, indent=2,
                                          default=lambda value: value.item()
                                          if isinstance(value, np.generic)
                                          else str(value)) + "\n")

    def stage(name: str, function, *positional) -> None:
        start = time.perf_counter()
        function(*positional)
        report["seconds"][name] = time.perf_counter() - start
        save()

    try:
        subject = args.scratch / "subject"
        stage("copy_subject", shutil.copytree, args.candidate, subject)
        for name, digest in report["input_sha256"].items():
            if _sha256(subject / name) != digest:
                raise RuntimeError(f"copied input checksum mismatch: {name}")
        report["comparisons"]["before_aseg"] = _compare(
            subject / "mri/aseg.mgz", args.official / "mri/aseg.mgz")
        shutil.copyfile(subject / "mri/aseg.mgz", subject / "mri/aseg.before_post.mgz")
        mri, surf, label = (subject / name for name in ("mri", "surf", "label"))
        stage("volmask", volmask.write_ribbon, mri / "aseg.presurf.mgz",
              surf, mri, args.color_lut)
        stage("relabel_hypointensities", relabel.relabel_volume,
              mri / "aseg.presurf.mgz", surf, mri / "aseg.presurf.hypos.mgz")
        stage("surf2volseg_fix", surf2volseg.fix_presurf_volume,
              mri / "aseg.presurf.hypos.mgz", mri / "ribbon.mgz",
              surf, label, mri / "aseg.float32_python.mgz")
        stage("aseg_int32_header", _write_int32_aseg,
              mri / "aseg.float32_python.mgz", mri / "aseg.mgz")
        for name in OUTPUTS:
            report["comparisons"][name] = _compare(subject / name,
                                                      args.official / name)
        report["source_input_sha256_unchanged"] = all(
            _sha256(args.candidate / name) == digest
            for name, digest in report["input_sha256"].items())
        report["status"] = "complete"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
