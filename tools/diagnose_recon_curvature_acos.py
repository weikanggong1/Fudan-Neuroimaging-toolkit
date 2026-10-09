"""冻结真实几何下隔离Torch CPU acos与系统acosf；不改变生产实现或验收门。"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

import fnit.recon_all.discrete_curvature_torch as module
from fnit.recon_all.discrete_curvature_torch import DiscreteCurvatureTopology, _dot, _length


def comparison(first, second):
    # nibabel morph可能保留大端dtype；位图比较先统一字节序，数值不改。
    first, second = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    delta = np.abs(first.astype(float) - second.astype(float))
    return {"different_bits": int(np.count_nonzero(first.view(np.uint32) != second.view(np.uint32))),
            "max_abs": float(delta.max(initial=0)), "p99_abs": float(np.percentile(delta, 99))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    lib = ctypes.CDLL("libm.so.6")
    acosf = lib.acosf
    acosf.argtypes, acosf.restype = [ctypes.c_float], ctypes.c_float
    original = json.loads(args.benchmark.read_text())
    report = {"host": platform.node(), "scope": "CPU-only fixed-geometry libm acosf isolation, not CUDA regression",
              "benchmark_sha256": sha(args.benchmark), "script_sha256": sha(__file__),
              "module_sha256": sha(module.__file__), "threads": 4, "cases": []}
    for index, case in enumerate(original["cases"]):
        subject = Path(case["subject"])
        if not subject.is_absolute():
            subject = args.source_root / subject
        vertices_np, faces = fsio.read_geometry(str(subject / "surf" / f"{case['hemi']}.smoothwm"))
        vertices = torch.tensor(vertices_np, dtype=torch.float32)
        context = DiscreteCurvatureTopology(faces=faces, nvertices=len(vertices), device="cpu")
        result = context.evaluate(vertices=vertices)
        normal = result["face_normal"]
        first, second = normal[context.face_ids], normal[context.next_faces]
        ratio = (_dot(first, second) / (_length(first) * _length(second))).clamp(-1, 1)
        lib_angle = np.fromiter((acosf(float(value)) for value in ratio.numpy().ravel()), np.float32,
                               count=ratio.numel()).reshape(ratio.shape)
        lib_angle = torch.from_numpy(lib_angle)
        outer, inner = vertices[context.outer_vertices], vertices[context.inner_vertices]
        positive = _length((outer + first) - (inner + second))
        negative = _length((outer - first) - (inner - second))
        signed = lib_angle * torch.where(positive < negative, 1, -1)
        edge_length = _length(vertices[context.edge_vertices[..., 1]] - vertices[context.edge_vertices[..., 0]])
        area_sum = torch.zeros(len(vertices), dtype=torch.float32)
        normal_sum = torch.zeros_like(area_sum)
        for slot in range(context.width):
            active = context.valid[:, slot]
            area_sum = area_sum + torch.where(active, result["face_area"][context.face_ids[:, slot]], 0)
            normal_sum = normal_sum + torch.where(active, signed[:, slot] * edge_length[:, slot], 0)
        lib_h = (.75 / area_sum.double() * normal_sum.double()).float().numpy()
        frozen = args.benchmark.parent / f"case_{index // 2}_{case['hemi']}" / "reference_subject"
        reference = fsio.read_morph_data(str(frozen / "surf" / f"{case['hemi']}.smoothwm.H.crv"))
        report["cases"].append({"hemi": case["hemi"], "surface_sha256": case["surface_sha256"],
            "torch_cpu_H_vs_reference": comparison(result["H"].numpy(), reference),
            "libm_H_vs_reference": comparison(lib_h, reference),
            "torch_cpu_angle_vs_libm": comparison(torch.acos(ratio).numpy(), lib_angle.numpy())})
    report["status"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
