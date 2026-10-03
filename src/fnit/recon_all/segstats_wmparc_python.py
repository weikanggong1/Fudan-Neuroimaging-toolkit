"""The fixed recon-all ``mri_segstats`` call for ``wmparc.stats``.

The partial-volume calculation follows FreeSurfer 8.2.0's
``MRIvoxelsInLabelWithPartialVolumeEffects``.  Each border voxel is evaluated
once for all labels; updates for each label retain source voxel scan order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from numba import njit

from .anatomical_stats_global import read_brain_volume_stats
from .estimated_tiv import estimate_tiv


@njit(cache=True)
def _partial_volume(seg: np.ndarray, intensity: np.ndarray,
                    selected: np.ndarray, voxel_volume: np.float32) -> np.ndarray:
    width, height, depth = seg.shape
    volumes = np.zeros(len(selected), dtype=np.float32)
    near_ids = np.empty(27, dtype=np.int32)
    large_count = np.empty(27, dtype=np.int32)
    large_sum = np.empty(27, dtype=np.float32)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                own = int(seg[x, y, z])
                border = False
                interesting = selected[own]
                for axis in range(3):
                    for step in (-1, 1):
                        xx, yy, zz = x, y, z
                        if axis == 0:
                            xx += step
                        elif axis == 1:
                            yy += step
                        else:
                            zz += step
                        if xx < 0 or xx >= width or yy < 0 or yy >= height or zz < 0 or zz >= depth:
                            continue
                        label = int(seg[xx, yy, zz])
                        if label != own:
                            border = True
                            if selected[label]:
                                interesting = True
                if not border:
                    if selected[own]:
                        volumes[own] += voxel_volume
                    continue
                if not interesting:
                    continue

                # MRIcomputeLabelNbhd(..., 1) uses all 27 voxels, with
                # FreeSurfer's clamped index arrays at the volume boundary.
                nnear = 0
                for dx in range(-1, 2):
                    xx = min(max(x + dx, 0), width - 1)
                    for dy in range(-1, 2):
                        yy = min(max(y + dy, 0), height - 1)
                        for dz in range(-1, 2):
                            zz = min(max(z + dz, 0), depth - 1)
                            label = int(seg[xx, yy, zz])
                            index = 0
                            while index < nnear and near_ids[index] != label:
                                index += 1
                            if index == nnear:
                                near_ids[index] = label
                                nnear += 1

                for k in range(1, nnear):
                    label = near_ids[k]
                    j = k - 1
                    while j >= 0 and near_ids[j] > label:
                        near_ids[j + 1] = near_ids[j]
                        j -= 1
                    near_ids[j + 1] = label

                for k in range(nnear):
                    large_count[k] = 0
                    large_sum[k] = np.float32(0)
                for dx in range(-7, 8):
                    xx = min(max(x + dx, 0), width - 1)
                    for dy in range(-7, 8):
                        yy = min(max(y + dy, 0), height - 1)
                        for dz in range(-7, 8):
                            zz = min(max(z + dz, 0), depth - 1)
                            label = int(seg[xx, yy, zz])
                            for k in range(nnear):
                                if near_ids[k] == label:
                                    large_count[k] += 1
                                    large_sum[k] += np.float32(intensity[xx, yy, zz])
                                    break

                current = np.float32(intensity[x, y, z])
                mean_own = np.float32(0)
                for k in range(nnear):
                    if near_ids[k] == own:
                        mean_own = large_sum[k] / large_count[k]
                        break
                neighbor = -1
                max_count = 0
                mean_neighbor = np.float32(0)
                # Source scans IDs ascending and retains the first count tie.
                for k in range(nnear):
                    if near_ids[k] == own:
                        continue
                    mean = large_sum[k] / large_count[k]
                    if (large_count[k] > max_count and
                            np.float32(mean - current) * np.float32(mean_own - current) < 0):
                        neighbor = near_ids[k]
                        max_count = large_count[k]
                        mean_neighbor = mean

                if neighbor < 0:
                    if selected[own]:
                        volumes[own] += voxel_volume
                    continue
                fraction = np.float32((current - mean_neighbor) / (mean_own - mean_neighbor))
                if fraction > 1:
                    fraction = np.float32(1)
                if fraction < 0:
                    continue
                if selected[own]:
                    volumes[own] += voxel_volume * fraction
                if selected[neighbor]:
                    # MRIsegBorder only admits an out-of-label border voxel
                    # when this chosen label is a face neighbor.
                    direct = False
                    if x > 0 and seg[x - 1, y, z] == neighbor:
                        direct = True
                    if x + 1 < width and seg[x + 1, y, z] == neighbor:
                        direct = True
                    if y > 0 and seg[x, y - 1, z] == neighbor:
                        direct = True
                    if y + 1 < height and seg[x, y + 1, z] == neighbor:
                        direct = True
                    if z > 0 and seg[x, y, z - 1] == neighbor:
                        direct = True
                    if z + 1 < depth and seg[x, y, z + 1] == neighbor:
                        direct = True
                    if direct:
                        volumes[neighbor] += voxel_volume * np.float32(1 - fraction)
    return volumes


def _lut(path: Path) -> dict[int, str]:
    names = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            names[int(parts[0])] = parts[1]
    return names


def _statistics(seg: np.ndarray, intensity: np.ndarray, ids: list[int],
                voxel_volume: float) -> list[tuple[int, int, float, float, float, float, float]]:
    maximum = int(seg.max())
    selected = np.zeros(maximum + 1, dtype=np.bool_)
    selected[ids] = True
    volumes = _partial_volume(seg, intensity, selected, np.float32(voxel_volume))
    seg_flat = seg.ravel(order="F")
    values = intensity.ravel(order="F")
    counts = np.bincount(seg_flat, minlength=maximum + 1)
    sums = np.bincount(seg_flat, weights=values, minlength=maximum + 1)
    squared = np.bincount(seg_flat, weights=values.astype(np.float64) ** 2,
                          minlength=maximum + 1)
    rows = []
    for label in ids:
        count = int(counts[label])
        if not count:
            continue
        own_values = values[seg_flat == label]
        mean = np.float32(sums[label] / count)
        # MRIsegStats retains a float mean, then calculates sample variance.
        # A single voxel has the existing zero standard deviation; avoid
        # evaluating its unused division by zero before selecting that value.
        if count > 1:
            first_term = float(np.float32(np.float32(count * mean) * mean))
            variance = (first_term -
                        2 * float(mean) * sums[label] + squared[label]) / (count - 1)
            std = np.float32(np.sqrt(max(variance, 0)))
        else:
            std = np.float32(0)
        lower = float(own_values.min())
        upper = float(own_values.max())
        rows.append((label, count, float(volumes[label]), float(mean),
                     float(std), lower, upper))
    return rows


def write_wmparc_stats(subject: str | Path, wm_lut: str | Path,
                       output: str | Path, *, brainvol_stats: str | Path | None = None,
                       stiv: str | Path | None = None) -> Path:
    """Write numeric-equivalent ``wmparc.stats`` from the fixed recon-all inputs."""
    subject = Path(subject)
    wm_lut = Path(wm_lut)
    output = Path(output)
    image = nib.load(str(subject / "mri" / "wmparc.mgz"))
    seg = np.asarray(image.dataobj).astype(np.int32, copy=False)
    intensity = np.asarray(nib.load(str(subject / "mri" / "norm.mgz")).dataobj)
    if seg.shape != intensity.shape or seg.min() < 0:
        raise ValueError("Expected matching nonnegative wmparc and norm volumes")
    names = _lut(wm_lut)
    ids = [label for label in sorted(names) if label and label <= seg.max()]
    rows = _statistics(seg, intensity, ids, float(np.prod(image.header.get_zooms()[:3])))
    brainvol_stats = (Path(brainvol_stats) if brainvol_stats is not None else
                      subject / "stats" / "brainvol.stats")
    stiv = Path(stiv) if stiv is not None else subject / "stats" / "synthseg.tiv.dat"
    global_stats = read_brain_volume_stats(brainvol_stats)
    etiv = estimate_tiv(subject / "mri" / "transforms" / "talairach.xfm")
    lines = ["# Title Segmentation Statistics ", "#",
             "# generating_program fnit", "# anatomy_type volume",
             f"# subjectname {subject.name}",
             "# BrainVolStatsFixed-NotNeeded because voxelvolume=1mm3"]
    for group, key, description in (
        ("VentricleChoroidVol", "VentricleChoroidVol", "Volume of ventricles and choroid plexus"),
        ("lhCerebralWhiteMatter", "lhCerebralWhiteMatterVol", "Left hemisphere cerebral white matter volume"),
        ("rhCerebralWhiteMatter", "rhCerebralWhiteMatterVol", "Right hemisphere cerebral white matter volume"),
        ("CerebralWhiteMatter", "CerebralWhiteMatterVol", "Total cerebral white matter volume"),
        ("Mask", "MaskVol", "Mask Volume"),
    ):
        lines.append(f"# Measure {group}, {key}, {description}, {global_stats[key]:.6f}, mm^3")
    lines.extend((
        "# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, "
        f"{etiv:.6f}, mm^3",
        "# Measure SegmentedTotalIntraCranialVol, sTIV, Segmented Total Intracranial Volume, "
        f"{float(stiv.read_text().split()[0]):.6f}, mm^3",
        f"# SegVolFile {subject / 'mri' / 'wmparc.mgz'}",
        f"# ColorTable {wm_lut}",
        f"# InVolFile {subject / 'mri' / 'norm.mgz'}",
        f"# PVVolFile {subject / 'mri' / 'norm.mgz'}",
        "# ExcludeSegId 0", "# Only reporting non-empty segmentations",
        f"# VoxelVolume_mm3 {np.prod(image.header.get_zooms()[:3]):g}",
    ))
    columns = (("Index", "Index", "NA"), ("SegId", "Segmentation Id", "NA"),
               ("NVoxels", "Number of Voxels", "unitless"),
               ("Volume_mm3", "Volume", "mm^3"),
               ("StructName", "Structure Name", "NA"))
    columns += tuple((f"norm{suffix}", f"Intensity norm{suffix}", "MR")
                     for suffix in ("Mean", "StdDev", "Min", "Max", "Range"))
    for index, (header, field, units) in enumerate(columns, 1):
        lines.extend((f"# TableCol {index:2d} ColHeader {header}",
                      f"# TableCol {index:2d} FieldName {field}",
                      f"# TableCol {index:2d} Units     {units}"))
    lines.extend((f"# NRows {len(rows)}", "# NTableCols 10",
                  "# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange"))
    for index, (label, count, volume, mean, std, lower, upper) in enumerate(rows, 1):
        lines.append(f"{index:3d} {label:3d}  {count:8d} {volume:10.1f}  "
                     f"{names[label]:<30s} {mean:10.4f} {std:10.4f} "
                     f"{lower:10.4f} {upper:10.4f} {upper-lower:10.4f} ")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=Path)
    parser.add_argument("wm_lut", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_wmparc_stats(args.subject, args.wm_lut, args.output)


if __name__ == "__main__":
    main()
