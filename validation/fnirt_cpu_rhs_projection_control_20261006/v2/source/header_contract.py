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



def raw_header_bridge(path, header, expected_raw_sha256):
    """Bind stored gzip header before comparing loaded spatial fields.

    Nibabel consumes vox_offset/scl_slope/scl_inter into its data proxy. Its
    loaded binaryblock is consequently a different representation. This
    routine requests only348 uncompressed header bytes, never voxel arrays.
    """
    import gzip
    import hashlib
    with gzip.open(path, "rb") as stream:
        raw = stream.read(348)
    if len(raw) != 348 or hashlib.sha256(raw).hexdigest() != expected_raw_sha256:
        raise ValueError("stored original348 header identity changed")
    loaded = header.binaryblock
    if len(loaded) != 348:
        raise ValueError("declared NIfTI1 loaded header length changed")
    spatial_names = ("dim", "pixdim", "qform_code", "sform_code", "quatern_b", "quatern_c", "quatern_d",
                     "qoffset_x", "qoffset_y", "qoffset_z", "srow_x", "srow_y", "srow_z", "xyzt_units")
    records, changed = [], []
    for name, (dtype, offset) in header.structarr.dtype.fields.items():
        equal = raw[offset:offset + dtype.itemsize] == loaded[offset:offset + dtype.itemsize]
        if not equal:
            changed.append(name)
        if name in spatial_names:
            records.append({"field": name, "offset": offset, "bytes": dtype.itemsize, "bitexact": equal})
    if len(records) != len(spatial_names) or not all(row["bitexact"] for row in records):
        raise ValueError("loaded spatial fields differ from bound stored header")
    if not set(changed).issubset({"vox_offset", "scl_slope", "scl_inter"}):
        raise ValueError("unexpected loaded nonspatial header change")
    return {"stored348_sha256": hashlib.sha256(raw).hexdigest(),
            "loaded_binaryblock_sha256": hashlib.sha256(loaded).hexdigest(),
            "stored348_identity_exact": True, "spatial_fields": records, "spatial_fields_bitexact": True,
            "changed_nonspatial_fields": changed, "header_bytes_requested": 348,
            "pixel_arrays_materialized_by_header_bridge": False}
