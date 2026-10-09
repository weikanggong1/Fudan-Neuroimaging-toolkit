"""FreeSurfer 8.2.0 aseg-guided mri_fill, including its Talairach CC cut."""

from __future__ import annotations

import argparse
import gzip
import struct
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from .fill_aseg_python import _fill_preclassified, _replace_cc_with_wm


_WM_LABELS = (2, 41, 187, 186, 251, 252, 253, 254, 255, 28, 60, 7, 46)
_EDIT_MARKERS = (200, 210, 220, 230, 240)


def read_vox_to_tal_lta(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the VOX_TO_VOX matrix and destination CRAS used by mri_fill."""
    lines = Path(path).read_text().splitlines()
    if not any(line.strip().startswith("type") and line.split("=")[1].split()[0] == "0"
               for line in lines):
        raise ValueError("mri_fill port currently requires a type 0 VOX_TO_VOX LTA")
    start = next(i for i, line in enumerate(lines) if line.strip() == "1 4 4")
    matrix = np.array([[float(value) for value in line.split()]
                       for line in lines[start + 1:start + 5]], dtype=np.float64)
    dst = lines.index("dst volume info")
    cras = np.array([float(v) for v in next(line for line in lines[dst + 1:]
                                         if line.startswith("cras")).split("=")[1].split()])
    if matrix.shape != (4, 4) or cras.shape != (3,):
        raise ValueError("invalid Talairach LTA geometry")
    return matrix, cras


def compute_cc_cut_mask(wm: np.ndarray, classified_aseg: np.ndarray,
                        vox_to_tal: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Find CC midline and map the native one-voxel Talairach cut into WM space."""
    if wm.shape != classified_aseg.shape or wm.ndim != 3 or wm.dtype != np.uint8:
        raise ValueError("wm and classified_aseg must have equal 3D shapes; wm must be uint8")
    inverse = np.linalg.inv(vox_to_tal)
    clean_wm = wm.copy()
    clean_wm[np.isin(clean_wm, _EDIT_MARKERS)] = 0
    tal_wm = ndi.affine_transform(clean_wm, inverse[:3, :3], offset=inverse[:3, 3],
                                  output_shape=wm.shape, order=1, mode="constant",
                                  output=np.float64) >= 53.5
    tal_seg = ndi.affine_transform(classified_aseg, inverse[:3, :3],
                                   offset=inverse[:3, 3], output_shape=wm.shape,
                                   order=0, mode="constant", cval=0)
    wm_label = np.isin(tal_seg, _WM_LABELS)
    left_label = np.concatenate((tal_seg[:1], tal_seg[:-1]), axis=0)
    right_label = np.concatenate((tal_seg[1:], tal_seg[-1:]), axis=0)
    left_wm = np.concatenate((wm_label[:1], wm_label[:-1]), axis=0)
    right_wm = np.concatenate((wm_label[1:], wm_label[-1:]), axis=0)
    midline = left_wm & right_wm & ((left_label != tal_seg) | (right_label != tal_seg))

    # The seed finder also requires the center label to be WM, whereas the
    # subsequent non-midline eraser checks only the two neighboring labels.
    eligible = wm_label & midline
    votes = ndi.convolve1d((eligible & tal_wm).astype(np.int16), np.ones(11, np.int16),
                            axis=2, mode="nearest")
    votes = ndi.convolve1d(votes, np.ones(11, np.int16), axis=1, mode="nearest")
    votes[~eligible] = 0
    if not votes.max():
        raise ValueError("no aseg-guided CC seed found")
    x, y, z = (int(v) for v in np.unravel_index(int(votes.argmax()), votes.shape))
    plane = tal_wm[x].T
    labels, _ = ndi.label(plane, structure=ndi.generate_binary_structure(2, 1))
    if labels[z, y] == 0:
        raise ValueError("CC seed is not in the Talairach WM plane")
    component = labels == labels[z, y]
    first_y = np.where(component, np.arange(wm.shape[1])[None, :], wm.shape[1]).min(axis=1)
    cut = np.arange(wm.shape[1])[None, :] >= first_y[:, None]
    tal_cut = np.zeros(wm.shape, dtype=np.uint8)
    tal_cut[x] = np.where(cut.T & midline[x], 200, 0).astype(np.uint8)
    # MRIfromTalairachEx uses trilinear sampling into UCHAR, then MRIbinarize(1).
    source_cut = ndi.affine_transform(tal_cut, vox_to_tal[:3, :3],
                                      offset=vox_to_tal[:3, 3], output_shape=wm.shape,
                                      order=1, mode="constant", cval=0, output=np.uint8)
    return source_cut >= 1, (x, y, z)


def _colortable_tag(path: str | Path) -> bytes:
    """Encode the source's TAG_OLD_COLORTABLE version 2 payload."""
    entries: dict[int, tuple[str, int, int, int, int]] = {}
    for line in Path(path).read_text().splitlines():
        tokens = line.split()
        if len(tokens) >= 6 and tokens[0].isdigit():
            index = int(tokens[0])
            entries[index] = (tokens[1], *(int(v) for v in tokens[2:6]))
    if not entries:
        raise ValueError("empty FreeSurfer color table")
    name = str(path).encode() + b"\0"
    tag = bytearray(struct.pack(">iiii", 1, -2, max(entries) + 1, len(name)))
    tag.extend(name)
    tag.extend(struct.pack(">i", len(entries)))
    for index, (label, r, g, b, transparency) in sorted(entries.items()):
        encoded = label.encode() + b"\0"
        tag.extend(struct.pack(">ii", index, len(encoded)))
        tag.extend(encoded)
        tag.extend(struct.pack(">iiii", r, g, b, transparency))
    return bytes(tag)


def save_filled_mgz(wm_file: str | Path, output_file: str | Path,
                    filled: np.ndarray, colortable_file: str | Path,
                    provenance: str | None = None) -> None:
    """Keep WM MGH geometry and tags, and embed the FreeSurfer color table."""
    source = Path(wm_file)
    image = nib.load(str(source))
    if (not isinstance(image, nib.MGHImage) or image.shape != filled.shape
            or image.get_data_dtype().newbyteorder("=") != np.dtype(np.uint8)
            or filled.dtype != np.uint8):
        raise ValueError("filled and input MGH must be matching uint8 volumes")
    raw = gzip.decompress(source.read_bytes()) if source.suffix == ".mgz" else source.read_bytes()
    end = 284 + filled.size
    tail = raw[end:]
    # The first 20 bytes are MGH scan parameters. All tags in the WM input
    # have 64-bit lengths; its first command-line tag is where mri_fill writes CTAB.
    cursor = 20
    prefix = bytearray(tail[:20])
    while cursor + 12 <= len(tail):
        tag, length = struct.unpack_from(">iq", tail, cursor)
        if tag == 1:
            raise ValueError("input WM already has an embedded color table")
        if tag == 3:
            break
        end_tag = cursor + 12 + length
        if end_tag > len(tail):
            raise ValueError("invalid source MGH tag length")
        if tag == 41:
            # MRIcopyHeader does not copy pedir, so mri_fill writes UNKNOWN
            # without a trailing NUL, regardless of the input pedir tag.
            prefix.extend(struct.pack(">iq", 41, 7) + b"UNKNOWN")
        else:
            prefix.extend(tail[cursor:end_tag])
        cursor = end_tag
    if cursor > len(tail):
        raise ValueError("invalid source MGH tag length")
    result = raw[:284] + filled.tobytes(order="F") + bytes(prefix) + _colortable_tag(colortable_file) + tail[cursor:]
    if provenance is not None:
        record = provenance.encode() + b"\0"
        result += struct.pack(">iq", 3, len(record)) + record
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(result, mtime=0) if output.suffix == ".mgz" else result)


def fill_mgz(wm_file: str | Path, aseg_file: str | Path, lta_file: str | Path,
             colortable_file: str | Path, output_file: str | Path,
             cut_log_file: str | Path | None = None, *,
             cc_boundary_backend: str = "python", device: str = "cpu",
             cc_marching_backend: str = "python",
             cc_profiles: list[dict] | None = None) -> tuple[int, int, int]:
    """nibabel读取同网格WM/aseg，复用完整aseg引导fill并写出MGZ。

    wm_file为3D uint8；aseg_file为同mm affine的整数语义标签；lta_file
    为type0 VOX_TO_VOX；colortable_file为已声明LUT；output_file保持WM
    网格/头并写0/127/255。cut_log_file默认None，可写CC体素/Talairach
    mm坐标；返回CC种子(x,y,z)体素。cc_boundary_backend默认python，
    torch只替换外部距离边界初始化；device默认cpu；cc_profiles默认
    None，指定list追加两侧分段秒。cc_marching_backend默认python，
    numba编译同一有序堆传播，首次JIT需时间。其余算法/有序扫描不变，
    不调用原生。
    参数/几何错误抛ValueError，读取/写出/CUDA异常传播，不静默回退。
    """
    wm_image = nib.load(str(wm_file))
    aseg_image = nib.load(str(aseg_file))
    wm = np.asarray(wm_image.dataobj)
    aseg = np.asarray(aseg_image.dataobj)
    if wm.shape != aseg.shape or not np.allclose(wm_image.affine, aseg_image.affine):
        raise ValueError("wm and aseg must have matching voxel geometry")
    matrix, tal_cras = read_vox_to_tal_lta(lta_file)
    classified = _replace_cc_with_wm(aseg, boundary_backend=cc_boundary_backend,
                                    device=device, marching_backend=cc_marching_backend,
                                    profiles=cc_profiles)
    cut, seed = compute_cc_cut_mask(wm, classified, matrix)
    filled = _fill_preclassified(wm, classified, float(wm_image.header["delta"][0]), cut)
    provenance = ("fnit.recon_all.fill_cutting_plane_python "
                  f"{wm_file} {aseg_file} {lta_file} {colortable_file} {output_file}")
    save_filled_mgz(wm_file, output_file, filled, colortable_file, provenance)
    if cut_log_file is not None:
        tal_affine = wm_image.affine.copy()
        center = np.array(wm.shape, dtype=np.float64) / 2
        tal_affine[:3, 3] += tal_cras - (wm_image.affine[:3, :3] @ center + wm_image.affine[:3, 3])
        tx, ty, tz = (tal_affine @ np.array([*seed, 1.0]))[:3]
        log = ("# This file contains the column, row, slice (CRS) and talirach XYZ (TAL)\n"
               "# of the cutting planes for the corpus callosum (CC) and pons\n"
               "# as generated by mri_fill. \n"
               f"CC-CRS {seed[0]} {seed[1]} {seed[2]}\n"
               f"CC-TAL {tx:5.1f} {ty:5.1f} {tz:5.1f}\n")
        Path(cut_log_file).write_text(log)
    return seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wm")
    parser.add_argument("aseg")
    parser.add_argument("lta")
    parser.add_argument("colortable")
    parser.add_argument("output")
    parser.add_argument("--cut-log")
    parser.add_argument("--cc-boundary-backend", choices=("python", "torch"), default="python")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cc-marching-backend", choices=("python", "numba"), default="python")
    args = parser.parse_args()
    fill_mgz(wm_file=args.wm, aseg_file=args.aseg, lta_file=args.lta,
             colortable_file=args.colortable, output_file=args.output, cut_log_file=args.cut_log,
             cc_boundary_backend=args.cc_boundary_backend, device=args.device,
             cc_marching_backend=args.cc_marching_backend)


if __name__ == "__main__":
    main()
