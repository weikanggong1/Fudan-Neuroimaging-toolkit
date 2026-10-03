"""只读核对官方 saved ITK 的方向；不拟合、不修改 MRI 或表面。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


ORACLE = {
    "container_sha256": "8e32238619053c1f9d1739b26f4afd72df809d914f5a5771707bf5da4b1d0f39",
    "software_versions": {"fmriprep": "25.2.4", "smriprep": "0.19.2", "niworkflows": "1.14.4", "nitransforms": "25.1.0"},
    "inspection_scope": "Read-only inspection of installed source in the exact official-reference SIF; this verifier uses nibabel/NumPy rather than running those interfaces.",
    "installed_source_sha256": {
        "smriprep/workflows/surfaces.py": "17784d0bb26db872aaa5b43e7b54ab78b93932b557348b105d6d3f12bc8abe96",
        "smriprep/workflows/outputs.py": "771f6b86738db77e9375e83e8ec93365da82f9abeffd088b4795b66bea3d0428",
        "smriprep/interfaces/surf.py": "b65b68e66cac1d7eb31b4f4b67af03a9ce6b78cb7bf91a4190e7b8dc27b91a03",
        "niworkflows/interfaces/nitransforms.py": "c51afbee61fc7d1439cbd1a2ab28681bf29c2e480c2c5d9406509d4dafa778bb",
        "nitransforms/io/lta.py": "936952a1e4a9d4424ee1d6139ab9edf05fc712c157511bc6a37e9164327786ae",
        "nitransforms/io/itk.py": "627a1e178b6a7c5e9181023d9d7b4bc47909ad28d7eae01db44e443498ac6551",
        "nitransforms/linear.py": "325ef1010529ae876da35c01ef862874e881582ff20c9c7645046fedb8a9cd14",
    },
    "direction_evidence": [
        {"source": "smriprep/workflows/surfaces.py", "lines": [320, 328], "meaning": "RobustRegister source is FreeSurfer T1.mgz; target is the native anatomical image."},
        {"source": "smriprep/workflows/outputs.py", "lines": [668, 702], "meaning": "ConcatenateXFMs converts that LTA, and out_xfm is saved as from-fsnative_to-T1w."},
        {"source": "niworkflows/interfaces/nitransforms.py", "lines": [92, 109], "meaning": "The LTA is loaded into NiTransforms, then written as ITK; the reverse file uses the inverse transform."},
        {"source": "nitransforms/io/lta.py", "lines": [190, 222], "meaning": "LTA loading inverts its RAS matrix to obtain the image-sampling pull transform."},
        {"source": "nitransforms/io/itk.py", "lines": [85, 92, 178, 184], "meaning": "ITK conversion changes LPS/RAS coordinates and applies a nonzero fixed center, without a direction inversion."},
        {"source": "smriprep/interfaces/surf.py", "lines": [202, 211], "meaning": "The saved transform is loaded and applied to scanner-RAS surface coordinates with inverse=True."},
    ],
    "primary_links": [
        "https://github.com/nipreps/smriprep/blob/0.19.2/src/smriprep/workflows/surfaces.py",
        "https://github.com/nipreps/smriprep/blob/0.19.2/src/smriprep/workflows/outputs.py",
        "https://github.com/nipreps/smriprep/blob/0.19.2/src/smriprep/interfaces/surf.py",
        "https://github.com/nipreps/niworkflows/blob/1.14.4/niworkflows/interfaces/nitransforms.py",
        "https://github.com/nipy/nitransforms/blob/25.1.0/nitransforms/io/lta.py",
        "https://github.com/nipy/nitransforms/blob/25.1.0/nitransforms/io/itk.py",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_single_itk_pull(path: Path) -> np.ndarray:
    """读取单个 AffineTransform；float32 文本解析与实测 NiTransforms 25.1.0 一致。"""
    lines = path.read_text().splitlines()
    transform = [line for line in lines if line.startswith("Transform:")]
    parameter_rows = [line for line in lines if line.startswith("Parameters:")]
    center_rows = [line for line in lines if line.startswith("FixedParameters:")]
    if len(transform) != 1 or transform[0].split()[-1] not in (
            "AffineTransform_float_3_3", "AffineTransform_double_3_3"):
        raise ValueError("Expected exactly one 3D ITK affine")
    if len(parameter_rows) != 1 or len(center_rows) != 1:
        raise ValueError("ITK parameters/fixed center are ambiguous")
    parameters = np.asarray(parameter_rows[0].split(":", 1)[1].split(), dtype=np.float32).astype(np.float64)
    center = np.asarray(center_rows[0].split(":", 1)[1].split(), dtype=np.float64)
    if parameters.shape != (12,) or center.shape != (3,) or not np.isfinite(parameters).all() or not np.isfinite(center).all():
        raise ValueError("ITK affine parameters or fixed center are invalid")
    lps_affine = np.eye(4)
    lps_affine[:3, :3] = parameters[:9].reshape(3, 3)
    lps_affine[:3, 3] = parameters[9:] + center - lps_affine[:3, :3] @ center
    lps_to_ras = np.diag([-1., -1., 1., 1.])
    return lps_to_ras @ lps_affine @ lps_to_ras


def verify(attempt: Path, raw_t1w: Path) -> dict:
    own_before = sha256(Path(__file__))
    report_path = attempt / "report.corrected.public.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "complete" or report.get("container_exit_code") != 0:
        raise ValueError("Official fresh reconstruction has not completed")
    if report.get("SIF_sha256") != ORACLE["container_sha256"]:
        raise ValueError("Completed reference used a different container from the inspected source")
    subject = report["subject"]
    prefix = f"sub-{subject}_ses-preop"
    anatomy = attempt / "derivatives" / f"sub-{subject}" / "ses-preop/anat"
    reconstruction = attempt / "derivatives/sourcedata/freesurfer" / prefix
    paths = {
        "reference_completed_report": report_path,
        "raw_T1w": raw_t1w,
        "preproc_T1w": anatomy / f"{prefix}_desc-preproc_T1w.nii.gz",
        "orig001": reconstruction / "mri/orig/001.mgz",
        "orig": reconstruction / "mri/orig.mgz",
        "fs_T1": reconstruction / "mri/T1.mgz",
        "LTA": attempt / f"work/fmriprep_25_2_wf/sub_{subject}_ses_preop_wf/anat_fit_wf/surface_recon_wf/fsnative2t1w_xfm/T1_robustreg.lta",
        "ITK_forward_named": anatomy / f"{prefix}_from-fsnative_to-T1w_mode-image_xfm.txt",
        "ITK_inverse_named": anatomy / f"{prefix}_from-T1w_to-fsnative_mode-image_xfm.txt",
    }
    for hemisphere, abbreviation in (("L", "lh"), ("R", "rh")):
        paths[f"white_{hemisphere}"] = reconstruction / "surf" / f"{abbreviation}.white"
        paths[f"GIFTI_{hemisphere}"] = anatomy / f"{prefix}_hemi-{hemisphere}_white.surf.gii"
    before = {name: sha256(path) for name, path in paths.items()}
    expected_raw = report["input_sha256"].get("t1w", report["input_sha256"].get("T1w"))
    if before["raw_T1w"] != expected_raw:
        raise ValueError("Raw T1w differs from the completed official reference's bound input")
    pull = read_single_itk_pull(paths["ITK_forward_named"])
    forward = np.linalg.inv(pull)
    reverse_named = read_single_itk_pull(paths["ITK_inverse_named"])
    lines = paths["LTA"].read_text().splitlines()
    if not any(line.startswith("type") and line.split("=", 1)[1].split("#", 1)[0].strip() == "1" for line in lines):
        raise ValueError("This proof requires a RAS-to-RAS type-1 LTA")
    index = lines.index("1 4 4")
    lta = np.array([[float(value) for value in row.split()] for row in lines[index + 1:index + 5]])
    original = nib.load(str(paths["orig"]))
    fs_t1 = nib.load(str(paths["fs_T1"]))
    surface_to_scanner = original.affine @ np.linalg.inv(original.header.get_vox2ras_tkr())
    hemispheres = {}
    for hemisphere in ("L", "R"):
        vertices, faces = nib.freesurfer.read_geometry(str(paths[f"white_{hemisphere}"]))
        saved = nib.load(str(paths[f"GIFTI_{hemisphere}"]))
        points = saved.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
        triangles = saved.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
        if len(points) != 1 or len(triangles) != 1 or not np.array_equal(faces, triangles[0].data):
            raise ValueError("Official GIFTI does not preserve the original white face order")
        scanner = nib.affines.apply_affine(surface_to_scanner, vertices)
        difference = nib.affines.apply_affine(forward, scanner) - points[0].data
        wrong_difference = nib.affines.apply_affine(pull, scanner) - points[0].data
        if not np.isfinite(difference).all() or np.abs(difference).max() > 5e-5:
            raise ValueError("Forward matrix does not reproduce the real official saved vertices")
        hemispheres[hemisphere] = {
            "vertices": len(vertices), "face_order_exact": True,
            "forward_candidate_vs_official_gifti_max_abs_mm": float(np.abs(difference).max()),
            "forward_candidate_vs_official_gifti_mean_euclidean_mm": float(np.linalg.norm(difference, axis=1).mean()),
            "wrong_uninverted_pull_vs_official_gifti_max_abs_mm": float(np.abs(wrong_difference).max()),
            "wrong_uninverted_pull_vs_official_gifti_mean_euclidean_mm": float(np.linalg.norm(wrong_difference, axis=1).mean()),
        }
    images = {name: nib.as_closest_canonical(nib.load(str(paths[name])))
              for name in ("raw_T1w", "preproc_T1w", "orig001")}
    raw, preproc, input_image = (images[name] for name in ("raw_T1w", "preproc_T1w", "orig001"))
    if (raw.shape != preproc.shape or raw.shape != input_image.shape
            or not np.allclose(raw.affine, preproc.affine, atol=1e-5, rtol=0)
            or not np.allclose(raw.affine, input_image.affine, atol=1e-5, rtol=0)
            or not np.allclose(fs_t1.affine, original.affine, atol=1e-5, rtol=0)):
        raise ValueError("Native anatomical target and original raw T1w do not share a verified world grid")
    if np.abs(pull @ reverse_named - np.eye(4)).max() > 1e-5 or np.abs(lta - forward).max() > 1e-5:
        raise ValueError("The LTA, forward-named ITK and reverse-named ITK directions disagree")
    after = {name: sha256(path) for name, path in paths.items()}
    own_after = sha256(Path(__file__))
    if before != after or own_before != own_after:
        raise ValueError("Read-only inputs or verifier changed")
    return {
        "status": "passed", "dataset": "OpenNeuro ds001226 v5.0.1 CC0", "subject": subject,
        "scope": "Read-only same-vertex proof of saved image-transform direction, with no fitted registration, MRI/surface modification or whole-pipeline benchmark.",
        "oracle_source_provenance": ORACLE,
        "reference_report_SIF_identity_verified": True,
        "input_sha256_before": before, "input_sha256_after": after,
        "input_sha_guards_equal": True, "verifier_sha256": own_before, "verifier_sha_guard_equal": True,
        "source_T1_and_orig_scanner_affine_max_abs": float(np.abs(fs_t1.affine - original.affine).max()),
        "canonical_raw_preproc_same_shape": raw.shape == preproc.shape,
        "canonical_raw_orig001_same_shape": raw.shape == input_image.shape,
        "canonical_raw_preproc_affine_max_abs": float(np.abs(raw.affine - preproc.affine).max()),
        "canonical_raw_orig001_affine_max_abs": float(np.abs(raw.affine - input_image.affine).max()),
        "orig001_vs_raw_pixel_max_abs": float(np.abs(input_image.get_fdata() - raw.get_fdata()).max()),
        "orig001_vs_preproc_pixel_max_abs": float(np.abs(input_image.get_fdata() - preproc.get_fdata()).max()),
        "itk_named_from_fsnative_to_T1w_semantics": "T1w scanner-RAS to FreeSurfer scanner-RAS pull map after LPS conversion",
        "fnit_required_forward_scanner_ras_matrix": forward.tolist(),
        "itk_bidirectional_inverse_max_abs": float(np.abs(pull @ reverse_named - np.eye(4)).max()),
        "lta_forward_vs_converted_inverse_itk_max_abs": float(np.abs(lta - forward).max()),
        "hemisphere_coordinate_proof": hemispheres,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-attempt", type=Path, required=True)
    parser.add_argument("--raw-t1w", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.reference_attempt, args.raw_t1w)
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(content, end="")
    else:
        with args.output.open("x") as stream:
            stream.write(content)


if __name__ == "__main__":
    main()
