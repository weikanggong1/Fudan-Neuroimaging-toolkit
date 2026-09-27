"""Native-free numerical port of FreeSurfer 8.2 ``mri_cc -aseg``.

The fixed source is d932c45. The recon-all dispatcher calls this stage after
`norm.mgz` is generated.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import linalg, ndimage

from .mgh_compat import save_same_dtype_mgh
from .mri_cc_masks_python import callosum_seed, remove_fornix_slice
from .mri_cc_plane_python import search_cutting_plane, unadjusted_voxel_lta


def _resample_nearest(source: np.ndarray, forward: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(forward.astype(np.float64))
    return ndimage.affine_transform(source, inverse[:3, :3], inverse[:3, 3],
                                    output_shape=source.shape, order=0,
                                    mode="constant", cval=0, prefilter=False)


def _component_corrections(aseg: np.ndarray) -> None:
    structure = ndimage.generate_binary_structure(3, 1)
    for label, replacement in ((254, 255), (252, 251)):
        segments, count = ndimage.label(aseg == label, structure=structure)
        if count <= 1:
            continue
        sizes = np.bincount(segments.ravel())
        sizes[0] = 0
        main = int(np.argmax(sizes))
        main_ymax = int(np.argwhere(segments == main)[:, 1].max())
        for index in range(1, count + 1):
            if index == main:
                continue
            coords = np.argwhere(segments == index)
            if len(coords) and int(coords[:, 1].min()) > main_ymax:
                aseg[segments == index] = replacement


def segment_callosum(aseg: np.ndarray, norm: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Return labelled ``aseg.auto``, adjusted ``cc_up`` voxel LTA, diagnostics."""
    if aseg.shape != norm.shape or aseg.ndim != 3:
        raise ValueError("aseg and norm must have the same 3D shape")
    if np.any((aseg >= 251) & (aseg <= 255)):
        raise ValueError("input aseg already contains corpus callosum labels")
    plane = search_cutting_plane(aseg)
    forward = unadjusted_voxel_lta(plane)
    cleaned = np.asarray(aseg, dtype=np.int32).copy()
    cleaned[norm < 40] = 0
    aseg_x = _resample_nearest(cleaned, forward)
    seed, best, _ = callosum_seed(aseg_x)
    selected = np.zeros(aseg.shape, dtype=np.uint8)
    for x in range(best - 2, best + 3):
        edited, _, _ = remove_fornix_slice(seed[x].T)
        selected[x] = (edited.T > 0).astype(np.uint8) * 100
    cc = ndimage.minimum_filter(ndimage.maximum_filter(selected, size=3, mode="nearest"),
                                size=3, mode="nearest")
    coords = np.argwhere(cc > 1)
    coords = coords[np.lexsort((coords[:, 0], coords[:, 1], coords[:, 2]))]
    mean = coords.mean(axis=0)
    cov = np.zeros((3, 3), dtype=np.float32)
    for point in coords:
        vector = np.asarray((point - mean) * 100, dtype=np.float32)
        cov += np.outer(vector, vector)
    cov *= np.float32(1.0 / (len(coords) * 100))
    _, eigenvectors = linalg.eigh(cov, driver="evr")
    direction = eigenvectors[:, np.argmax(np.abs(eigenvectors[2]))].astype(np.float64)
    if direction[2] < 0:
        direction = -direction
    central = np.argwhere(cc[best] > 0)
    positions = np.column_stack((np.full(len(central), best), central))
    central_zf = (positions - mean) @ direction
    zf_low, zf_high = float(central_zf.min()), float(central_zf.max())
    zf = (coords - mean) @ direction
    divisions = np.floor((zf - zf_low) / ((zf_high - zf_low + 1) / 5)).astype(int)
    divisions = np.clip(divisions, 0, 4)
    values = np.full(len(coords), 100, dtype=np.uint8)
    valid = zf >= zf_low - 10
    values[valid] = (divisions[valid] + 1) * 20 + 10
    labelled = np.zeros(aseg.shape, dtype=np.uint8)
    labelled[coords[:, 0], coords[:, 1], coords[:, 2]] = values
    # The second transform in main() receives the inverse LTA, so output
    # coordinates sample the callosum image through this forward matrix.
    pasted = ndimage.affine_transform(labelled, forward[:3, :3], forward[:3, 3],
                                       output_shape=aseg.shape, order=0,
                                       mode="constant", cval=0, prefilter=False)
    output = np.asarray(aseg).copy()
    hit = pasted > 0
    compartment = np.clip((pasted[hit].astype(int) - 10) // 20 - 1, 0, 4)
    output[hit] = 251 + compartment
    for x in range(1, output.shape[0] - 1):
        wm = (output[x] == 2) | (output[x] == 41)
        left = (output[x + 1] >= 251) & (output[x + 1] <= 255)
        right = (output[x - 1] >= 251) & (output[x - 1] <= 255)
        holes = wm & left & right
        output[x][holes] = output[x + 1][holes]
    _component_corrections(output)
    # LTAwrite uses a copy shifted so the selected sagittal slice is x=128.
    saved_lta = forward.copy()
    saved_lta[0, 3] += np.float32(128 - best)
    info: dict[str, float | int] = {
        "best_slice": best,
        "plane_score": plane[5],
        "plane_y_degrees": float(np.degrees(plane[3])),
        "plane_z_degrees": float(np.degrees(plane[4])),
        "changed_voxels": int(np.count_nonzero(output != aseg)),
    }
    return output, saved_lta, info


def write_cc_lta(path: str | Path, matrix: np.ndarray, norm_file: str | Path) -> None:
    """Write a voxel-to-voxel LTA with FreeSurfer volume geometry."""
    norm_file = Path(norm_file).absolute()
    image = nib.load(str(norm_file))
    data = np.asarray(image.dataobj)
    bbox = np.argwhere(data >= 70)
    lo, hi = bbox.min(axis=0), bbox.max(axis=0)
    mean = lo + (hi - lo + 1) // 2
    header = image.header
    dims = header["dims"][:3]
    size = header["delta"]
    mdc = header["Mdc"]
    center = header["Pxyz_c"]
    geometry = ["valid = 1  # volume info valid", f"filename = {norm_file}",
                "volume = " + " ".join(str(int(v)) for v in dims),
                "voxelsize = " + " ".join(f"{float(v):.15e}" for v in size)]
    for key, row in zip(("xras", "yras", "zras"), mdc):
        geometry.append(f"{key}   = " + " ".join(f"{float(v):.15e}" for v in row))
    geometry.append("cras   = " + " ".join(f"{float(v):.15e}" for v in center))
    path = Path(path)
    lines = [f"# transform file {path.name}", "type      = 0 # LINEAR_VOX_TO_VOX",
             "nxforms   = 1", "mean      = " + " ".join(f"{float(v):.4f}" for v in mean),
             "sigma     = 10000.0000", "1 4 4"]
    lines.extend(" ".join(f"{float(v):.15e}" for v in row) for row in matrix)
    lines.extend(["src volume info", *geometry, "dst volume info", *geometry])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_mri_cc(aseg_file: str | Path, norm_file: str | Path,
               output_file: str | Path, lta_file: str | Path) -> dict[str, float | int]:
    """File-oriented Python stage matching the recon-all ``mri_cc`` call."""
    aseg = np.asarray(nib.load(str(aseg_file)).dataobj)
    norm = np.asarray(nib.load(str(norm_file)).dataobj)
    output, matrix, info = segment_callosum(aseg, norm)
    save_same_dtype_mgh(aseg_file, output_file, output)
    write_cc_lta(lta_file, matrix, norm_file)
    return info
