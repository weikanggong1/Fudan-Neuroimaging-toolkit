"""单例 benchmark：按 FNIT 的输入与步骤运行原 SynthStrip 和 FSL 前处理。

这个脚本不属于 FNIT 运行接口。原始影像、命令日志和完整路径只保存到
用户指定的私有结果目录；公开报告仅包含匿名标量、版本和文件哈希。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_check(path, reference=None, *, binary=False, frames=None):
    image = nib.load(str(path))
    array = np.asarray(image.dataobj)
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite output: {path}")
    if reference is not None:
        target = nib.load(str(reference))
        if image.shape[:3] != target.shape[:3] or not np.allclose(
            image.affine, target.affine, atol=1e-4, rtol=0,
        ):
            raise ValueError(f"Output grid differs: {path}")
    if frames is not None and (image.ndim != 4 or image.shape[3] != frames):
        raise ValueError(f"Wrong frame count: {path}")
    if binary and (not np.isin(array, [0, 1]).all() or not array.any()):
        raise ValueError(f"Invalid mask: {path}")
    result = {"shape": list(image.shape), "dtype": str(image.get_data_dtype()),
              "all_finite": True, "sha256": sha256(path)}
    if binary:
        result["voxels"] = int(np.count_nonzero(array))
    if image.ndim == 4:
        result["tr_seconds"] = float(image.header.get_zooms()[3])
        result["time_unit"] = image.header.get_xyzt_units()[1]
    return result


class NativeRun:
    def __init__(self, args):
        self.args = args
        self.output = args.output_dir.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.env = os.environ.copy()
        self.env.update(FSLDIR=str(args.fsl_root), FREESURFER_HOME=str(args.freesurfer_root),
                        FSLOUTPUTTYPE="NIFTI_GZ", OMP_NUM_THREADS=str(args.threads),
                        OPENBLAS_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
        self.env["LD_LIBRARY_PATH"] = str(args.fsl_root / "lib") + os.pathsep + self.env.get("LD_LIBRARY_PATH", "")
        self.env["PATH"] = os.pathsep.join((str(args.fsl_root / "bin"),
                                              str(args.freesurfer_root / "bin"), self.env["PATH"]))
        self.rows = []
        self.commands = []
        self.public = {"schema_version": 1, "stage": args.stage, "steps": self.rows,
                       "fsl_installation_version": (args.fsl_root / "etc/fslversion").read_text().strip(),
                       "threads": args.threads, "input_sha256": {}, "checks": {},
                       "component_packages": {}, "component_sha256": {},
                       "scope": "Original software on one real input; no BET, threshold/dilation, smoothing, GDC/B0 or FIX added.",
                       "privacy": "Anonymous checks and hashes only. Full paths and logs stay in the private output directory."}
        for component in ("fast4", "flirt", "fnirt", "melodic"):
            package_files = sorted((args.fsl_root / "conda-meta").glob(f"fsl-{component}-*.json"))
            if package_files:
                info = json.loads(package_files[-1].read_text())
                self.public["component_packages"][component] = {key: info.get(key) for key in ("name", "version", "build")}

    def command(self, name, argv, expected, *, reference=None, binary=False, frames=None):
        expected = Path(expected)
        if expected.exists():
            raise FileExistsError(f"Use a fresh output directory: {expected}")
        executable = Path(argv[0])
        self.public["component_sha256"][executable.name] = sha256(executable)
        log = self.output / f"{name}.log"
        started = time.perf_counter()
        with log.open("w") as stream:
            process = subprocess.run([str(value) for value in argv], env=self.env, stdout=stream, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - started
        row = {"name": name, "process_wall_seconds": elapsed, "exit_code": process.returncode}
        self.rows.append(row)
        self.commands.append({"name": name, "argv": [str(value) for value in argv]})
        (self.output / f"{self.args.stage}.commands.private.json").write_text(json.dumps(
            {"steps": self.rows, "commands": self.commands}, indent=2) + "\n")
        if process.returncode not in (0, 255) or not expected.is_file():
            raise RuntimeError(f"{name} failed with {process.returncode}; inspect {log}")
        # The installed original programs can return 255 after a complete write.
        # Fresh-file CRC, full array and grid checks are required before acceptance.
        if str(expected).endswith(".nii.gz"):
            subprocess.run(["gzip", "-t", str(expected)], check=True)
        row["output_checks"] = image_check(expected, reference, binary=binary, frames=frames)
        row["validated_abnormal_exit"] = process.returncode == 255
        return expected

    def stats(self, name, *arguments):
        executable = self.args.fsl_root / "bin/fslstats"
        self.public["component_sha256"]["fslstats"] = sha256(executable)
        started = time.perf_counter()
        result = subprocess.run([str(executable), *map(str, arguments)], env=self.env, capture_output=True, text=True)
        seconds = time.perf_counter() - started
        try:
            value = float(result.stdout.strip())
        except ValueError as error:
            raise RuntimeError(f"Invalid {name} result") from error
        if result.returncode not in (0, 255) or not np.isfinite(value) or value <= 0:
            raise RuntimeError(f"{name} failed: exit {result.returncode}")
        self.rows.append({"name": name, "process_wall_seconds": seconds,
                          "exit_code": result.returncode, "value": value})
        return value

    def save_report(self):
        self.public["sum_process_wall_seconds"] = sum(row["process_wall_seconds"] for row in self.rows)
        self.public["timing_boundary"] = "Each original child process includes startup and image read/write. Hashes and integrity checks are outside these timers."
        path = self.args.report_out or self.output / f"{self.args.stage}.public.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.public, indent=2) + "\n")

    def anatomy(self):
        args = self.args
        masks = self.output / "masks"
        anat = self.output / "anat"
        feat = self.output / "feat"
        for directory in (masks, anat, feat):
            directory.mkdir(exist_ok=True)
        for label, path in (("t1w", args.t1w), ("sbref", args.sbref), ("synthstrip_weights", args.synthstrip_weights)):
            self.public["input_sha256"][label] = sha256(path)
        shutil.copyfile(args.sbref, feat / "example_func.nii.gz")
        executable = args.freesurfer_root / "bin/mri_synthstrip"
        self.public["component_sha256"]["mri_synthstrip_source"] = sha256(args.freesurfer_root / "python/scripts/mri_synthstrip")
        for name, source, brain, mask in (
            ("epi_synthstrip", args.sbref, masks / "epi_brain.nii.gz", masks / "epi_mask.nii.gz"),
            ("t1_synthstrip", args.t1w, anat / "T1_brain.nii.gz", anat / "T1_mask.nii.gz"),
        ):
            self.command(name, [executable, "-i", source, "-o", brain, "-m", mask,
                                "--model", args.synthstrip_weights, "-g", "-t", args.threads],
                         brain, reference=source)
            self.public["checks"][name + "_mask"] = image_check(mask, source, binary=True)
        # FAST's built-in K-means initialisation is 15 iterations. Its standard
        # T1 defaults match FNIT FASTConfig: bias=4, fixed=4, FWHM=20, H=.1/R=.3.
        stem = anat / "T1_fast"
        self.command("fast", [args.fsl_root / "bin/fast", "-t", "1", "-n", "3", "-I", "4",
                              "-l", "20", "-H", "0.1", "-R", "0.3", "-o", stem,
                              "-B", "-b", anat / "T1_brain.nii.gz"],
                     anat / "T1_fast_pve_2.nii.gz", reference=args.t1w)
        for label in ("pve_0", "pve_1", "pve_2", "seg", "restore", "bias"):
            self.public["checks"]["fast_" + label] = image_check(anat / f"T1_fast_{label}.nii.gz", args.t1w)
        self.save_report()

    def feat(self):
        args = self.args
        feat = self.output / "feat"
        feat.mkdir(exist_ok=True)
        mc = feat / "mc"
        mc.mkdir(exist_ok=True)
        mask = self.output / "masks/epi_mask.nii.gz"
        image_check(mask, args.sbref, binary=True)
        self.public["input_sha256"] = {label: sha256(path) for label, path in
                                        (("bold", args.bold), ("sbref", args.sbref), ("epi_mask", mask))}
        frames = nib.load(str(args.bold)).shape[3]
        stem = mc / "prefiltered_func_data_mcf"
        corrected = self.command("mcflirt", [args.fsl_root / "bin/mcflirt", "-in", args.bold,
                                              "-out", stem, "-mats", "-plots", "-reffile", args.sbref,
                                              "-rmsrel", "-rmsabs", "-spline_final"],
                                 Path(str(stem) + ".nii.gz"), reference=args.sbref, frames=frames)
        matrices = sorted(Path(str(stem) + ".mat").glob("MAT_*"))
        if len(matrices) != frames or any(np.loadtxt(path).shape != (4, 4) or
                                         not np.isfinite(np.loadtxt(path)).all() for path in matrices):
            raise ValueError("MCFLIRT did not produce all finite 4x4 matrices")
        parameters = np.loadtxt(str(stem) + ".par")
        if parameters.shape != (frames, 6) or not np.isfinite(parameters).all():
            raise ValueError("Invalid MCFLIRT motion parameters")
        executable = args.fsl_root / "bin/fslmaths"
        masked = self.command("mask_bold", [executable, corrected, "-mas", mask,
                                           feat / "masked_func_data.nii.gz"],
                              feat / "masked_func_data.nii.gz", reference=args.sbref, frames=frames)
        median = self.stats("masked_p50", corrected, "-k", mask, "-p", "50")
        factor = 10000.0 / median
        self.public["intensity_factor"] = factor
        scaled = self.command("intensity_scale", [executable, masked, "-mul", repr(factor),
                                                  feat / "scaled_func_data.nii.gz"],
                              feat / "scaled_func_data.nii.gz", reference=args.sbref, frames=frames)
        mean = self.command("temporal_mean", [executable, scaled, "-Tmean", feat / "tempMean.nii.gz"],
                            feat / "tempMean.nii.gz", reference=args.sbref)
        sigma = args.highpass_seconds / (2 * args.tr)
        filtered = self.command("gaussian_highpass", [executable, scaled, "-bptf", repr(sigma), "-1",
                                                       "-add", mean, feat / "filtered_func_data.nii.gz"],
                                feat / "filtered_func_data.nii.gz", reference=args.sbref, frames=frames)
        self.public["highpass_sigma_volumes"] = sigma
        self.public["highpass_cutoff_seconds"] = args.highpass_seconds
        shutil.copyfile(mask, feat / "mask.nii.gz")
        self.command("filtered_mean", [executable, filtered, "-Tmean", feat / "mean_func.nii.gz"],
                     feat / "mean_func.nii.gz", reference=args.sbref)
        self.save_report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("anatomy", "feat"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fsl-root", type=Path, required=True)
    parser.add_argument("--freesurfer-root", type=Path, required=True)
    parser.add_argument("--bold", type=Path)
    parser.add_argument("--sbref", type=Path, required=True)
    parser.add_argument("--t1w", type=Path)
    parser.add_argument("--synthstrip-weights", type=Path)
    parser.add_argument("--tr", type=float, default=0.735)
    parser.add_argument("--highpass-seconds", type=float, default=100.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    if args.stage == "anatomy" and (args.t1w is None or args.synthstrip_weights is None):
        parser.error("anatomy requires --t1w and --synthstrip-weights")
    if args.stage == "feat" and args.bold is None:
        parser.error("feat requires --bold")
    run = NativeRun(args)
    getattr(run, args.stage)()


if __name__ == "__main__":
    main()
