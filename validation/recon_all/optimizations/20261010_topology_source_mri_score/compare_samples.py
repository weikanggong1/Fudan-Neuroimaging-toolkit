"""完整 first-defect 22点 MRI 采样记录的类型边界诊断。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import nibabel as nib
import numpy as np
from numba import njit
from fnit.recon_all.topology_edge_mri_score import _sample as sample_ras
from fnit.recon_all.place_surface_border import _sample as sample_voxel


@njit(cache=True, fastmath=False)
def replay(volume, matrix, values):
    own_voxels = np.empty((len(values), 3), np.float64)
    ras_values = np.empty(len(values), np.float64)
    source_voxel_values = np.empty(len(values), np.float64)
    for row in range(len(values)):
        point = np.empty(4, np.float32)
        for axis in range(3):
            point[axis] = np.float32(values[row, axis])
        point[3] = 1
        for axis in range(3):
            value = np.float64(0)
            for column in range(4):
                value = value + np.float64(matrix[axis, column]) * np.float64(point[column])
            own_voxels[row, axis] = np.float64(np.float32(value))
        ras_values[row] = sample_ras(volume, matrix, values[row, 0], values[row, 1], values[row, 2])
        source_voxel_values[row] = sample_voxel(volume, values[row, 3], values[row, 4], values[row, 5])
    return own_voxels, ras_values, source_voxel_values


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def errors(a, b):
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(different_elements=int(np.count_nonzero(difference)), max=float(difference.max(initial=0)),
                p99=float(np.percentile(difference, 99)) if difference.size else 0,
                mean=float(difference.mean()) if difference.size else 0)


def compare(*, sample_capture, native_preparation, intensity_volume, output_report,
            active_topology_matrix=None):
    started = time.perf_counter()
    records = np.fromfile(sample_capture, dtype=np.dtype([("endpoints", "<i4", (2,)), ("values", "<f8", (7,))]))
    with open(native_preparation, "rb") as stream:
        header = np.fromfile(stream, "<i4", 5)
        matrix = np.fromfile(stream, "<f4", 16).reshape(4, 4)
    if active_topology_matrix is not None:
        entries = np.fromfile(active_topology_matrix, "<f4")
        if entries.size != 16 or not np.isfinite(entries).all():
            raise ValueError("actual topology matrix requires exactly 16 finite float32 entries")
        matrix = entries.reshape(4, 4)
    volume = np.ascontiguousarray(np.asarray(nib.load(str(intensity_volume)).dataobj))
    if tuple(header[2:]) != volume.shape or volume.dtype != np.uint8 or len(records) % 22:
        raise ValueError("complete 22-point records and identical uint8 MRI required")
    values = np.ascontiguousarray(records["values"])
    voxels, ras, from_voxels = replay(volume, matrix, values)
    different = np.flatnonzero(ras != values[:, 6])
    first = int(different[0]) if len(different) else None
    report = dict(status="complete_source_sample_replay", production_changed=False,
                  scope="entire first patch before pruning; all pairs/endpoints/interior samples",
                  number_sample_records=len(records), number_edge_records=len(records)//22,
                  source_endpoints_first=records["endpoints"][0].tolist() if len(records) else [],
                  own_ras_voxels_vs_source=errors(voxels, values[:, 3:6]),
                  own_ras_sampler_vs_source=errors(ras, values[:, 6]),
                  existing_voxel_sampler_vs_source=errors(from_voxels, values[:, 6]),
                  first_ras_difference=dict(index=first, endpoints=records["endpoints"][first].tolist(),
                      source_record=values[first].tolist(), own_voxel=voxels[first].tolist(),
                      own_ras_value=float(ras[first]), own_voxel_value=float(from_voxels[first])) if first is not None else None,
                  artifact_sha256={name:sha(path) for name,path in dict(sample_capture=sample_capture,
                      native_preparation=native_preparation, intensity_volume=intensity_volume).items()},
                  matrix_scope="actual MATRIX_ALLOCATION" if active_topology_matrix is not None else "fresh MRIS_makeRAS2VoxelMap; diagnostic inactive branch",
                  active_matrix_sha256=sha(active_topology_matrix) if active_topology_matrix is not None else None,
                  actual_matrix=matrix.tolist(),
                  script_sha256=sha(__file__), elapsed_seconds=time.perf_counter()-started)
    output_report = Path(output_report)
    if output_report.exists():
        raise FileExistsError("new report only")
    output_report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["sample-capture", "native-preparation", "intensity-volume", "output-report"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--active-topology-matrix", type=Path)
    compare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
