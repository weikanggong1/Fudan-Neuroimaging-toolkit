"""Benchmark the public single-run volume and surface APIs on real BIDS data.

Input images and derivatives remain in the caller's directories. Public reports
contain scalar checks and hashes, never subject identifiers or absolute paths.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.fmri import fMRIVolume_pipeline, fMRISurface_pipeline
from fnit.fmri.bids import locate_bids_inputs
from fnit.fmri.derivatives import sidecar


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(root):
    paths = []
    for component in ("fmri", "msm", "fast", "flirt", "fnirt", "applywarp",
                      "synthstrip", "synthmorph"):
        paths.extend((root / "src/fnit" / component).rglob("*.py"))
    paths.extend((root / "src/fnit/msm/_fastpd_src").glob("*"))
    paths.extend((root / "src/fnit").glob("_*.py"))
    return {p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(set(paths)) if p.is_file()}


def check_volume(path, template=None, mask=None):
    image = nib.load(str(path))
    if image.ndim != 4 or image.get_data_dtype() != np.dtype("float32"):
        raise ValueError("expected float32 4D BOLD")
    if image.header.get_xyzt_units()[1] != "sec":
        raise ValueError("BOLD time unit must be seconds")
    if template is not None and (image.shape[:3] != template.shape[:3] or
            not np.allclose(image.affine, template.affine, atol=1e-4, rtol=0)):
        raise ValueError("BOLD does not match its reference grid")
    # Decode gzip once; repeated proxy slices otherwise repeatedly inflate it.
    values = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite BOLD")
    outside_max = None
    if mask is not None:
        outside_max = float(np.abs(values[~mask]).max())
        if outside_max != 0:
            raise ValueError("nonzero BOLD outside its output brain mask")
    return {"shape": list(image.shape), "dtype": str(image.get_data_dtype()),
            "finite_values": int(values.size), "nonfinite_values": 0,
            "tr_seconds": float(image.header.get_zooms()[3]),
            "time_unit": "sec", "outside_mask_max_abs": outside_max,
            "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("volume", "surface"))
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--derivatives-root", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--mni-template", type=Path)
    parser.add_argument("--mni-brain-mask", type=Path)
    parser.add_argument("--registration-backend", choices=("synthmorph", "fnirt"), default="synthmorph")
    parser.add_argument("--synthstrip-weights", type=Path)
    parser.add_argument("--synthmorph-weights", type=Path)
    parser.add_argument("--recon-all", type=Path)
    parser.add_argument("--hcp-assets-dir", type=Path)
    parser.add_argument("--wb-command", default="wb_command")
    parser.add_argument("--registered-spheres", type=Path, nargs=2,
                        help="Surface control only: replace estimation with two existing registered spheres")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cuda = str(args.device).startswith("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if cuda:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(args.device)
    inputs = locate_bids_inputs(args.bids_root, subject=args.subject)
    if len(inputs.t1w_images) != 1:
        raise ValueError("benchmark requires one unambiguous T1w image")
    input_hashes = {"bold": sha256(inputs.bold), "sbref": sha256(inputs.sbref)
                    if inputs.sbref else None, "t1w": sha256(inputs.t1w_images[0])}
    started = time.perf_counter()
    if args.stage == "volume":
        if args.mni_template is None:
            parser.error("volume requires --mni-template")
        result = fMRIVolume_pipeline(
            args.bids_root, args.derivatives_root, subject=args.subject,
            mni_template=args.mni_template, mni_brain_mask=args.mni_brain_mask,
            registration_backend=args.registration_backend,
            synthstrip_weights=args.synthstrip_weights,
            synthmorph_weights=args.synthmorph_weights,
            regress_wm=True, regress_csf=True, regress_motion=True,
            device=args.device, batch_size=8, random_state=0,
        )
    else:
        if args.recon_all is None or args.hcp_assets_dir is None:
            parser.error("surface requires --recon-all and --hcp-assets-dir")
        result = fMRISurface_pipeline(
            args.bids_root, args.derivatives_root, subject=args.subject,
            recon_all=args.recon_all, hcp_assets_dir=args.hcp_assets_dir,
            wb_command=args.wb_command, device=args.device,
            registered_spheres=args.registered_spheres,
        )
    if cuda:
        torch.cuda.synchronize(args.device)
    api_wall = time.perf_counter() - started
    memory = {"peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if cuda else 0,
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(args.device) if cuda else 0}
    metadata = json.loads(result.metadata.read_text())
    if args.stage == "volume":
        mask_image = nib.load(str(result.mask_mni))
        template = nib.load(str(args.mni_template))
        if mask_image.shape != template.shape or not np.allclose(
                mask_image.affine, template.affine, atol=1e-4, rtol=0):
            raise ValueError("mask grid differs from template")
        mask = np.asarray(mask_image.dataobj) > 0
        if not mask.any():
            raise ValueError("empty MNI mask")
        checks = {"clean_native": check_volume(result.clean_native, nib.load(str(inputs.bold))),
                  "clean_mni": check_volume(result.clean_mni, template, mask),
                  "mask_voxels": int(mask.sum()), "mask_sha256": sha256(result.mask_mni),
                  "bbr_matrix_finite_4x4": bool(np.loadtxt(result.bbr_matrix).shape == (4, 4)
                                               and np.isfinite(np.loadtxt(result.bbr_matrix)).all())}
        pipeline = metadata["FNIT"]["Report"]
        if not pipeline["ica_converged"]:
            raise ValueError("ICA did not converge")
        algorithm = {key: pipeline[key] for key in (
            "registration_backend", "ica_components", "ica_converged",
            "ica_iterations", "aroma_noise_components", "wm_csf_motion_regression")}
        limits = ["No GDC or B0 correction: corresponding raw inputs are unavailable.",
                  "ICA-AROMA replaces UKB FIX, so there is no paired final UKB voxelwise oracle."]
    else:
        cifti = nib.load(str(result.dtseries))
        values = np.asarray(cifti.dataobj, dtype=np.float32)
        if values.shape != (nib.load(str(inputs.bold)).shape[3], 91282) or not np.isfinite(values).all():
            raise ValueError("CIFTI shape or finite-value check failed")
        hemisphere_checks = {}
        for hemi, path in (("L", result.left), ("R", result.right)):
            gifti = nib.load(str(path))
            if len(gifti.darrays) != values.shape[0] or any(
                    array.data.shape != (32492,) or not np.isfinite(array.data).all()
                    for array in gifti.darrays):
                raise ValueError("fsLR32k GIFTI contract failed")
            if not sidecar(path).is_file():
                raise FileNotFoundError(sidecar(path))
            hemisphere_checks[hemi] = {"frames": len(gifti.darrays), "vertices": 32492,
                                      "all_finite": True, "sha256": sha256(path)}
        if not np.isclose(cifti.header.get_axis(0).step, inputs.tr, atol=1e-5):
            raise ValueError("CIFTI TR differs from raw BOLD")
        checks = {"cifti_shape": list(values.shape), "cifti_all_finite": True,
                  "cifti_sha256": sha256(result.dtseries), "hemispheres": hemisphere_checks,
                  "tr_seconds": float(cifti.header.get_axis(0).step),
                  "brain_models": {name: int(model.size) for name, _, model in
                                   cifti.header.get_axis(1).iter_structures()}}
        algorithm = {"registration": metadata["FNIT"]["Registration"],
                     "coverage": metadata["FNIT"]["Coverage"]}
        if args.registered_spheres:
            algorithm["registration"] = "provided registered spheres; MSMSulc estimation skipped"
            input_hashes["registered_spheres"] = [sha256(p) for p in args.registered_spheres]
        limits = ["Surface timing starts from completed volume derivatives and existing recon-all surfaces.",
                  "This API run measures execution and contracts; the controlled official-sphere numerical comparison is separate."]
    if nib.load(str(inputs.bold)).shape[3] != (checks["clean_mni"]["shape"][3]
            if args.stage == "volume" else checks["cifti_shape"][0]):
        raise ValueError("frame count changed")
    report = {"schema_version": 1, "source_revision": args.source_revision,
              "source_sha256": source_hashes(args.source_root), "stage": args.stage,
              "data": {"kind": "one real UK Biobank run", "subjects": 1,
                       "bold_shape": list(nib.load(str(inputs.bold)).shape),
                       "t1w_shape": list(nib.load(str(inputs.t1w_images[0])).shape)},
              "input_sha256": input_hashes, "algorithm": algorithm, "checks": checks,
              "timing_seconds": {"public_api_including_output_save": api_wall,
                                 "pipeline_stages": result.timing_seconds}, "memory": memory,
              "environment": {"host": platform.node(), "python": platform.python_version(),
                              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(args.device) if cuda else None,
                              "cpu_threads": args.threads, "tf32": True,
                              "low_precision_enabled": False},
              "limits": limits + ["Single cold run on a shared H100; not a controlled speed comparison.",
                                  "Hashing and validation occur outside the API timer."],
              "privacy": "Only scalar checks and hashes are published; subject identifiers and image paths are omitted."}
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"stage": args.stage, "api_wall_seconds": api_wall,
                      "memory": memory, "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
