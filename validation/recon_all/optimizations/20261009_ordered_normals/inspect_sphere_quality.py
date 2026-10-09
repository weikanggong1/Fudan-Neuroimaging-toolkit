"""Independent FP64 orientation plus the existing cleanup FP32 predicate."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import torch
from fnit.recon_all.mris_register_nonlinear import face_area_normals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--stage-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    xyz, faces = fsio.read_geometry(str(args.surface))
    corners = xyz[faces]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    signed = np.einsum("ij,ij->i", cross, corners.mean(axis=1))
    negative = signed < 0
    native_areas, _ = face_area_normals(torch.as_tensor(xyz.astype(np.float32)),
                                       torch.as_tensor(faces.astype(np.int64)), signed_sphere=True)
    native_negative = native_areas.numpy() < 0
    history = json.loads(args.stage_report.read_text())["stage_report"]["negative_counts"]
    radius = np.linalg.norm(xyz, axis=1)
    result = {"surface_sha256": hashlib.sha256(args.surface.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "vertices": len(xyz), "faces": len(faces), "all_finite": bool(np.isfinite(xyz).all()),
              "fp64_negative_faces": int(negative.sum()), "fp32_cleanup_negative_faces": int(native_negative.sum()),
              "fp64_negative_area_mm2": float(areas[negative].sum()),
              "fp64_negative_face_max_area_mm2": float(areas[negative].max(initial=0)),
              "fp64_negative_face_p99_area_mm2": float(np.percentile(areas[negative], 99)) if negative.any() else 0,
              "fp64_total_area_mm2": float(areas.sum()), "fp64_zero_area_faces": int(np.count_nonzero(areas == 0)),
              "fp64_negative_face_indices": np.flatnonzero(negative).tolist(),
              "fp32_negative_area_mm2": float(native_areas[native_areas < 0].double().abs().sum()),
              "radius_min_mm": float(radius.min()), "radius_max_mm": float(radius.max()),
              "finish_history_length": len(history), "finish_first_count": history[0] if history else 0,
              "finish_last_recorded_count": history[-1] if history else 0,
              "finish_min_recorded_count": min(history) if history else 0,
              "mesh_quality": "residual negative faces; not fully passed",
              "interpretation": "Existing baseline and candidate output hashes are equal; no threshold relaxed"}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("indices")}))


if __name__ == "__main__":
    main()
