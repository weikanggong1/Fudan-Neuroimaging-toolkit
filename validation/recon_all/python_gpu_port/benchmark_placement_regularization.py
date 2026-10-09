"""真实固定 pial 输入的 CPU/PyTorch 正则梯度配对测试；不等于完整 pial 或整例。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=32768)
    args = parser.parse_args()
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.candidate_directory))
    from fnit.recon_all.place_surface_regularization_torch import PlacementRegularizationTorch
    from fnit.recon_all.place_surface_border import compute_border_values_first_pass
    from fnit.recon_all.place_surface_curvature import quadratic_curvature, tangent_basis, two_ring_neighbors
    from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
    from fnit.recon_all.place_surface_gradient_average import average_signed_gradients
    from fnit.recon_all.place_surface_intensity import intensity_gradient
    from fnit.recon_all.place_surface_normals import initial_vertex_normals
    from fnit.recon_all.place_surface_repulsion import original_vertex_normals, surface_repulsion_gradient, vertex_buckets
    from fnit.recon_all.place_surface_rip import rip_outside_label
    from fnit.recon_all.place_surface_smoothing import average_marked_values, _ordered_neighbors
    from fnit.recon_all.place_surface_spring import spring_gradient
    from fnit.recon_all.place_surface_volume import prepare_placement_volume

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    report = {"scope": "frozen_same_input_pial_regularization_only", "hostname": platform.node(),
              "torch": torch.__version__, "numpy": np.__version__, "threads": args.threads,
              "device": str(device), "gpu": torch.cuda.get_device_name(device),
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "tf32_cudnn": torch.backends.cudnn.allow_tf32, "dtype": "float32/explicit float64 sums",
              "autocast": False, "chunk_size": args.chunk_size, "order": ["cpu", "torch", "torch", "cpu"],
              "kernel_tolerance_declared": {"max_absolute": 1e-5},
              "admission": "not_assessed_until_complete_pial_regression", "hemispheres": {}}
    report["source_sha256"] = {}
    for module in (PlacementRegularizationTorch, quadratic_curvature, average_signed_gradients, spring_gradient):
        import inspect
        source = inspect.getfile(module)
        report["source_sha256"][Path(source).name] = sha256(source)
    for hemi in ("lh", "rh"):
        started = time.perf_counter()
        white = args.subject / f"surf/{hemi}.white"
        xyz, faces, metadata = fs.read_geometry(str(white), read_metadata=True)
        xyz = xyz.astype(np.float32)
        label = args.subject / f"label/{hemi}.cortex+hipamyg.label"
        ripped = np.asarray(rip_outside_label(len(xyz), fs.read_label(str(label))), dtype=np.bool_)
        brain_path, wm_path, aseg_path = (args.subject / f"mri/{name}.mgz" for name in
                                        ("brain.finalsurfs", "wm", "aseg.presurf"))
        stats_path = args.subject / f"surf/autodet.gw.stats.{hemi}.dat"
        brain = nib.load(str(brain_path))
        wm, aseg = (np.asarray(nib.load(str(path)).dataobj) for path in (wm_path, aseg_path))
        stats = dict(line.split()[:2] for line in stats_path.read_text().splitlines() if len(line.split()) >= 2)
        volume, bright = prepare_placement_volume(np.asarray(brain.dataobj), wm, surface="pial", mid_gray=float(stats["MID_GRAY"]))
        placement = volume.copy()
        placement[bright == 130] = 0
        affine = surface_ras_to_voxel(brain.header, metadata)
        normals = initial_vertex_normals(xyz, faces)
        thresholds = np.array([float(stats[f"pial_{name}"]) for name in
                               ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")])
        border = compute_border_values_first_pass(volume, aseg, xyz, normals, xyz, ripped,
                    np.full(len(xyz), -1.0, dtype=np.float32), affine, thresholds, hemisphere=hemi, surface="pial")
        values = average_marked_values(border[0], border[4], ripped, faces, 5)
        neighbors, valid, _ = _ordered_neighbors(faces, len(xyz))
        ordered = neighbors, valid
        offsets, candidates = two_ring_neighbors(faces, len(xyz), ordered_neighbors=ordered)
        intensity = intensity_gradient(placement, xyz, normals, ripped, values, border[5], affine,
                                       brain.header.get_zooms()[:3], weight=.2, sigma_global=2.)
        rep_offsets, rep_candidates = vertex_buckets(xyz, xyz, ripped)
        repulsion = surface_repulsion_gradient(xyz, normals, xyz,
                    original_vertex_normals(xyz, faces), ripped,
                    rep_offsets, rep_candidates, weight=5., cropped=np.zeros(len(xyz), dtype=np.int32))
        gradient = np.float32(intensity + repulsion)
        prepare_seconds = time.perf_counter() - started
        def cpu():
            averaged = average_signed_gradients(gradient, faces, ripped, 16, ordered_neighbors=ordered)
            normal = spring_gradient(xyz, normals, faces, ripped, weight=.3, direction="normal", ordered_neighbors=ordered)
            scalar = quadratic_curvature(xyz, normals, tangent_basis(normals), ripped, offsets, candidates)
            tangent = spring_gradient(xyz, normals, faces, ripped, weight=.3, direction="tangent", ordered_neighbors=ordered)
            return np.float32(np.float32(np.float32(averaged + normal) + np.float32(scalar[:, None] * normals)) + tangent)
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        context = PlacementRegularizationTorch(neighbors=neighbors, valid=valid, offsets=offsets,
                    candidates=candidates, ripped=ripped, device=str(device), chunk_size=args.chunk_size)
        torch.cuda.synchronize(device)
        context_seconds = time.perf_counter() - tick
        def gpu():
            return context.regularize(vertices=xyz, normals=normals, gradient=gradient,
                                      iterations=16, spring_weight=.3)
        # 首次JIT/设备加载单列；配对计时包含动态坐标上传、GPU运算和完整梯度回传。
        tick = time.perf_counter(); expected = cpu(); cpu_cold = time.perf_counter()-tick
        torch.cuda.synchronize(device)
        tick = time.perf_counter(); actual = gpu(); torch.cuda.synchronize(device); gpu_cold = time.perf_counter()-tick
        rows = []
        torch.cuda.reset_peak_memory_stats(device)
        for name in report["order"]:
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            result = cpu() if name == "cpu" else gpu()
            torch.cuda.synchronize(device)
            rows.append({"backend": name, "seconds": time.perf_counter()-tick,
                         "sha256": hashlib.sha256(result.tobytes()).hexdigest()})
        delta = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
        norm = np.linalg.norm(actual.astype(np.float64)-expected.astype(np.float64), axis=1)
        cpu_time = statistics.median(row["seconds"] for row in rows if row["backend"] == "cpu")
        gpu_time = statistics.median(row["seconds"] for row in rows if row["backend"] == "torch")
        report["hemispheres"][hemi] = {"vertices": len(xyz), "faces": len(faces),
            "input_sha256": {path.name: sha256(path) for path in (white, label, brain_path, wm_path, aseg_path, stats_path)},
            "prepare_seconds": prepare_seconds, "context_seconds": context_seconds,
            "cpu_cold_seconds": cpu_cold, "gpu_cold_seconds": gpu_cold, "paired": rows,
            "cpu_median_seconds": cpu_time, "gpu_median_seconds": gpu_time, "speed_ratio_cpu_over_gpu": cpu_time/gpu_time,
            "different_elements": int(np.count_nonzero(delta)), "max_absolute": float(delta.max()),
            "p99_absolute": float(np.percentile(delta,99)), "max_vertex_vector_error": float(norm.max()),
            "finite": bool(np.isfinite(actual).all()), "kernel_tolerance_pass": bool(delta.max() <= 1e-5),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        print(json.dumps({hemi: report["hemispheres"][hemi]}, ensure_ascii=False), flush=True)
        del context
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
