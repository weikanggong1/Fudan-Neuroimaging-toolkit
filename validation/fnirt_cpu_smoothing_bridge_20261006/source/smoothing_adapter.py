"""Private CPU FP32 storage-orientation adapter; never a default backend."""
from __future__ import annotations


def classify_header(header, np):
    """Require two declared, consistent, well-conditioned spatial forms.

    This diagnostic accepts a narrower header contract than NEWIMAGE. It
    rejects unknown, conflicting, or nearly singular forms rather than
    guessing from nibabel's best affine. Determinant values are diagnostic
    NumPy calculations, not a claim of NEWMAT determinant bit equivalence.
    """
    qcode, scode = int(header["qform_code"]), int(header["sform_code"])
    if qcode <= 0 or scode <= 0:
        raise ValueError("diagnostic requires both qform and sform codes")
    matrices = {"qform": header.get_qform(), "sform": header.get_sform()}
    determinants, signs = {}, []
    for name, matrix in matrices.items():
        if not np.isfinite(matrix).all():
            raise ValueError("nonfinite spatial form")
        d64 = float(np.linalg.det(matrix[:3, :3].astype(np.float64)))
        d32 = float(np.linalg.det(matrix[:3, :3].astype(np.float32)))
        if not np.isfinite(d64) or not np.isfinite(d32):
            raise ValueError("nonfinite determinant")
        # Conservative diagnostic guard; it is not FSL's ZERODET threshold.
        if min(abs(d64), abs(d32)) <= 1e-6 or (d64 < 0) != (d32 < 0):
            raise ValueError("ambiguous or nearly singular form")
        determinants[name] = {"float64": d64, "float32": d32}
        signs.append(1 if d64 > 0 else -1)
    if signs[0] != signs[1]:
        raise ValueError("qform/sform orientation conflict")
    zoom = tuple(float(v) for v in header.get_zooms()[:3])
    if len(zoom) != 3 or any(not np.isfinite(v) or v <= 0 for v in zoom):
        raise ValueError("invalid pixdim")
    return {"qform_code": qcode, "sform_code": scode,
            "determinants": determinants, "orientation_sign": signs[0],
            "orientation": "neurological" if signs[0] > 0 else "radiological",
            "flip_x": signs[0] > 0, "pixdim": list(zoom),
            "determinant_arithmetic": "NumPy FP64 and FP32 diagnostics; not NEWMAT bits",
            "guard_scope": "both forms known, consistent, abs(det)>1e-6"}


def smooth_original_storage(volume, fwhm_mm, voxel_sizes, *, flip_x, blur):
    """Flip only CPU FP32/no-grad input storage, call mature blur, restore X.

    No image/reference values, subject identity, or shape-specific arithmetic
    select the implementation. This bounded trial deliberately excludes masks.
    """
    import torch
    if (volume.device.type != "cpu" or volume.dtype != torch.float32
            or volume.requires_grad or volume.ndim != 5):
        raise ValueError("private adapter requires FP32 no-grad 5D CPU input")
    if not isinstance(flip_x, bool):
        raise ValueError("flip_x must be explicit resolved header metadata")
    if fwhm_mm <= 0 or not flip_x:
        return blur(volume, fwhm_mm, voxel_sizes, None, execution="optimized")
    result = blur(volume.flip(2), fwhm_mm, voxel_sizes, None, execution="optimized")
    return result.flip(2)
