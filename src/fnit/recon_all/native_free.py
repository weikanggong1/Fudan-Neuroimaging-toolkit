"""Experimental T1-to-morphometry reconstruction without FreeSurfer programs.

The surface repair, placement, and registration steps are approximations. The
run report records them explicitly; this is not a numerically equivalent
replacement for FreeSurfer recon-all.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import torch


def _save_like(source: Path, output: Path, values: np.ndarray) -> None:
    image = nib.load(str(source))
    array = np.ascontiguousarray(values)
    header = image.header.copy()
    header.set_data_dtype(array.dtype)
    nib.save(nib.MGHImage(array, image.affine, header), str(output))


def _replace_vertices(source: Path, output: Path, vertices: np.ndarray) -> None:
    """Retain the triangle order and volume-geometry tag of a source surface."""
    raw = source.read_bytes()
    if raw[:3] != b"\xff\xff\xfe":
        raise ValueError(f"expected a triangular surface: {source}")
    first = raw.index(b"\n\n", 3) + 2
    count = int.from_bytes(raw[first:first + 4], "big")
    if len(vertices) != count:
        raise ValueError("surface vertex count changed")
    start = first + 8
    output.write_bytes(raw[:start] + np.asarray(vertices, dtype=">f4").tobytes()
                       + raw[start + 12 * count:])


def _largest_filled(mask: np.ndarray) -> np.ndarray:
    components, count = ndi.label(mask)
    if count == 0:
        raise ValueError("hemisphere has no white matter")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return ndi.binary_fill_holes(components == np.argmax(sizes))


def _pial_from_cortex(white: np.ndarray, faces: np.ndarray,
                      image: nib.MGHImage, segmentation: np.ndarray,
                      hemi: str) -> tuple[np.ndarray, np.ndarray]:
    from .inflate_python import vertex_normals

    normal = vertex_normals(np.asarray(white, np.float32), np.asarray(faces, np.int32))
    center = white.mean(axis=0)
    if np.median(np.sum(normal * (white - center), axis=1)) < 0:
        normal = -normal
    wm_id, cortex_id = ((2, 3) if hemi == "lh" else (41, 42))
    allowed = (segmentation == wm_id) | (segmentation == cortex_id)
    inverse = np.linalg.inv(image.header.get_vox2ras_tkr())
    distance = np.full(len(white), np.float32(6), dtype=np.float32)
    active = np.ones(len(white), bool)
    entered_cortex = np.zeros(len(white), bool)
    for step in range(1, 13):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        points = white[indices] + np.float32(step * .5) * normal[indices]
        voxels = np.rint(points @ inverse[:3, :3].T + inverse[:3, 3]).astype(np.int32)
        in_bounds = np.all((voxels >= 0) & (voxels < np.asarray(allowed.shape)), axis=1)
        voxels = np.clip(voxels, 0, np.asarray(allowed.shape) - 1)
        labels = segmentation[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
        entered_cortex[indices] |= in_bounds & (labels == cortex_id)
        finished = (~in_bounds | ~allowed[voxels[:, 0], voxels[:, 1], voxels[:, 2]]) & entered_cortex[indices]
        distance[indices[finished]] = np.float32(step * .5)
        active[indices[finished]] = False
    distance = np.clip(distance, 1, 6)
    return (white + normal * distance[:, None]).astype(np.float32), distance


def _surface_pair(subject: Path, hemi: str, filled: Path, norm: Path,
                  aseg: np.ndarray, *, device: str) -> dict:
    from .extract_main_component_python import extract_main_component
    from .pretess_python import pretess_mgh
    from .smooth_surface_python import smooth_surface
    from .sphere_python import project_radially
    from .tessellate_gpu import tessellate_mgh, write_quad_surface
    from .surface_area_gpu import area_map, mid_area_map
    from .surface_curvature_gpu import curvature_map
    from .surface_roi_gpu import vertex_volume_map

    surf, mri, labels = (subject / name for name in ("surf", "mri", "label"))
    code = 255 if hemi == "lh" else 127
    pretess = mri / f"filled-pretess{code}.mgz"
    pretess_mgh(filled, code, norm, pretess)
    raw = surf / f"{hemi}.orig.raw.quad"
    vertices, quads = tessellate_mgh(pretess, code, device=device)
    write_quad_surface(raw, vertices, quads, nib.load(str(pretess)),
                       Path("../mri") / pretess.name)
    orig = surf / f"{hemi}.orig.nofix"
    components = extract_main_component(raw, orig)
    shutil.copyfile(orig, surf / f"{hemi}.orig")
    smoothwm = surf / f"{hemi}.smoothwm"
    smooth_surface(orig, smoothwm, device=device)
    shutil.copyfile(smoothwm, surf / f"{hemi}.white.preaparc")
    shutil.copyfile(smoothwm, surf / f"{hemi}.white")
    shutil.copyfile(smoothwm, surf / f"{hemi}.smoothwm.nofix")

    white, faces = fs.read_geometry(str(smoothwm))
    image = nib.load(str(mri / "aseg.mgz"))
    pial, thickness = _pial_from_cortex(white, faces, image, aseg, hemi)
    _replace_vertices(smoothwm, surf / f"{hemi}.pial", pial)
    shutil.copyfile(surf / f"{hemi}.pial", surf / f"{hemi}.pial.T1")

    inflated = surf / f"{hemi}.inflated"
    smooth_surface(smoothwm, inflated, iterations=35, device=device)
    inflated_xyz, _ = fs.read_geometry(str(inflated))
    sphere_xyz = project_radially(inflated_xyz)
    _replace_vertices(inflated, surf / f"{hemi}.sphere", sphere_xyz)
    shutil.copyfile(surf / f"{hemi}.sphere", surf / f"{hemi}.sphere.reg")
    fs.write_morph_data(str(surf / f"{hemi}.sulc"),
                        np.linalg.norm(inflated_xyz - white, axis=1).astype(np.float32))
    fs.write_morph_data(str(surf / f"{hemi}.thickness"), thickness)
    area_map(surf / f"{hemi}.white", surf / f"{hemi}.area", device=device)
    area_map(surf / f"{hemi}.pial", surf / f"{hemi}.area.pial", device=device)
    mid_area_map(surf / f"{hemi}.area", surf / f"{hemi}.area.pial",
                 surf / f"{hemi}.area.mid", device=device)
    from .label_cortex_python import label_cortex
    cortex = labels / f"{hemi}.cortex.label"
    cortex_vertices = label_cortex(surf / f"{hemi}.white", mri / "aseg.mgz", cortex)
    vertex_volume_map(surf / f"{hemi}.white", surf / f"{hemi}.pial",
                      cortex, surf / f"{hemi}.volume", device=device)
    curvature_map(surf / f"{hemi}.white", surf / f"{hemi}.curv", device=device)
    curvature_map(surf / f"{hemi}.pial", surf / f"{hemi}.curv.pial", device=device)
    from .surface_roi_curvature_gpu import principal_curvatures
    for surface, prefix in (("white", "white.preaparc"),
                            ("inflated", "inflated")):
        xyz, mesh = fs.read_geometry(str(surf / f"{hemi}.{surface}"))
        k1, k2 = principal_curvatures(xyz, mesh, device=device)
        fs.write_morph_data(str(surf / f"{hemi}.{prefix}.H"),
                            ((k1 + k2) / 2).astype(np.float32))
        fs.write_morph_data(str(surf / f"{hemi}.{prefix}.K"),
                            (k1 * k2).astype(np.float32))
        if surface == "white":
            for name, values in (("H", (k1 + k2) / 2), ("K", k1 * k2),
                                 ("K1", k1), ("K2", k2)):
                fs.write_morph_data(str(surf / f"{hemi}.smoothwm.{name}.crv"),
                                    np.asarray(values, np.float32))
    return {"hemisphere": hemi, "vertices": len(white), "faces": len(faces),
            "raw_components": components, "cortex_vertices": len(cortex_vertices),
            "mean_thickness_mm": float(np.mean(thickness)),
            "approximations": ["filled-from-SynthSeg-WM", "topology-unrepaired",
                               "white-from-smoothed-tessellation", "pial-normal-ray",
                               "inflated-Laplacian", "sphere-radial",
                               "sphere.reg-unregistered"]}


def _project_parcels(subject: Path, aseg: np.ndarray) -> None:
    """Assign cortical voxels to the closest same-hemisphere white vertex."""
    mri, surf, label = (subject / name for name in ("mri", "surf", "label"))
    image = nib.load(str(mri / "aseg.mgz"))
    transform = image.header.get_vox2ras_tkr()
    for atlas, output in (("aparc", "aparc+aseg.mgz"),
                          ("aparc.a2009s", "aparc.a2009s+aseg.mgz"),
                          ("aparc.DKTatlas", "aparc.DKTatlas+aseg.mgz")):
        projected = aseg.astype(np.int32, copy=True)
        for hemi, cortical_id, offset in (("lh", 3, 1000), ("rh", 42, 2000)):
            vertices, _ = fs.read_geometry(str(surf / f"{hemi}.white"))
            ids, _, _ = fs.read_annot(str(label / f"{hemi}.{atlas}.annot"))
            tree = cKDTree(vertices)
            positions = np.argwhere(aseg == cortical_id)
            for block in np.array_split(positions, max(1, (len(positions) + 99999) // 100000)):
                if not len(block):
                    continue
                points = block @ transform[:3, :3].T + transform[:3, 3]
                nearest = tree.query(points, workers=4)[1]
                projected[block[:, 0], block[:, 1], block[:, 2]] = offset + np.maximum(ids[nearest], 0)
        _save_like(mri / "aseg.mgz", mri / output, projected)


def _project_wmparc(subject: Path, aseg: np.ndarray) -> None:
    """Assign white matter voxels to nearby aparc regions."""
    mri, surf, label = (subject / name for name in ("mri", "surf", "label"))
    image = nib.load(str(mri / "aseg.mgz"))
    transform = image.header.get_vox2ras_tkr()
    projected = aseg.astype(np.int32, copy=True)
    for hemi, wm_id, offset in (("lh", 2, 3000), ("rh", 41, 4000)):
        vertices, _ = fs.read_geometry(str(surf / f"{hemi}.white"))
        ids, _, _ = fs.read_annot(str(label / f"{hemi}.aparc.annot"))
        tree = cKDTree(vertices)
        positions = np.argwhere(aseg == wm_id)
        for block in np.array_split(positions, max(1, (len(positions) + 99999) // 100000)):
            if not len(block):
                continue
            points = block @ transform[:3, :3].T + transform[:3, 3]
            nearest = tree.query(points, workers=4)[1]
            projected[block[:, 0], block[:, 1], block[:, 2]] = offset + np.maximum(ids[nearest], 0)
    _save_like(mri / "aseg.mgz", mri / "wmparc.mgz", projected)


def run_recon_all_python(t1: str | Path, subject_dir: str | Path,
                         weights_dir: str | Path, assets_dir: str | Path,
                         *, device: str = "cuda:0", threads: int = 4,
                         n4_python: str | Path | None = None) -> dict:
    """Run an explicitly approximate, FreeSurfer-executable-free fixed profile."""
    from fnit.synthseg_parc import SynthSeg
    from .brain_volume_stats_python import compute_brain_volume_stats
    from .ca_normalize_python import run_ca_normalize
    from .gcsa_label_python import label_surface
    from .input_talairach_chain import run_input_talairach_chain
    from .mri_em_register_python import register_t1
    from .mri_mask_gpu import mask_volume
    from .n4_wrapper import make_nu
    from .normalization import normalize_t1
    from .segstats_aseg_python import write_aseg_stats
    from .segstats_wmparc_python import write_wmparc_stats
    from .anatomical_stats_file import write_anatomical_stats

    t1, subject = Path(t1).resolve(), Path(subject_dir).resolve()
    weights, assets = Path(weights_dir).resolve(), Path(assets_dir).resolve()
    if not t1.is_file() or not weights.is_dir() or not assets.is_dir():
        raise FileNotFoundError("T1, weights, and assets must exist")
    if subject.exists() and any(subject.iterdir()):
        raise ValueError("subject_dir must be empty")
    torch.set_num_threads(threads)
    started = time.perf_counter()
    report: dict = {"profile": "experimental-native-free-core-v1", "input": str(t1),
                    "subject_dir": str(subject), "device": device,
                    "n4_python": str(n4_python or sys.executable), "threads": threads,
                    "stages": [], "status": "running"}

    def stage(name, function, *args, **kwargs):
        tick = time.perf_counter()
        try:
            value = function(*args, **kwargs)
        except Exception as error:
            report.update(status="failed", failed_stage=name, error=repr(error),
                          total_seconds=time.perf_counter() - started)
            (subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))
            raise
        report["stages"].append({"name": name, "seconds": time.perf_counter() - tick})
        (subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))
        return value

    initial = stage("input_talairach", run_input_talairach_chain,
                    t1, subject, weights, assets, device=device, threads=threads)
    mri, surf, labels, stats = (subject / name for name in ("mri", "surf", "label", "stats"))
    for folder in (surf, labels, stats, mri / "tmp"):
        folder.mkdir(parents=True, exist_ok=True)
    nu0 = mri / "tmp/nu0.mgz"
    if n4_python is None:
        from .n4_sitk import correct_volume
        stage("n4", correct_volume, mri / "orig.mgz", nu0)
    else:
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        stage("n4", subprocess.run,
              [str(n4_python), "-m", "fnit.recon_all.n4_sitk", "--i",
               str(mri / "orig.mgz"), "--o", str(nu0)],
              env=env, check=True)
    stage("nu", make_nu, mri / "orig.mgz", nu0,
          initial["talairach_xfm"], mri / "nu.mgz")
    stage("T1_normalize", normalize_t1, mri / "nu.mgz",
          initial["talairach_xfm"], mri / "T1.mgz", device=device)
    stage("brainmask", mask_volume, mri / "T1.mgz",
          initial["synthstrip"], mri / "brainmask.mgz", device=device)

    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        result = stage("SynthSeg", lambda: SynthSeg(weights=weights, device=device,
                                                      threads=threads)(mri / "orig.mgz",
                         keep_geometry=True, color_lut=assets / "FreeSurferColorLUT.txt"))
    finally:
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
    stiv_mm3 = result.total_intracranial_mm3
    result.segmentation.save(str(mri / "synthseg.rca.mgz"))
    result.write_volumes_csv(mri / "orig.mgz", stats / "synthseg.vol.csv")
    seg_image = nib.load(str(mri / "synthseg.rca.mgz"))
    aseg = np.asarray(seg_image.dataobj).astype(np.int16)
    for name in ("aseg.auto.mgz", "aseg.presurf.mgz", "aseg.mgz"):
        shutil.copyfile(mri / "synthseg.rca.mgz", mri / name)

    lta = mri / "transforms/talairach.lta"
    stage("mri_em_register", register_t1, mri / "nu.mgz",
          assets / "average/RB_all_2020-01-02.gca", mri / "brainmask.mgz", lta)
    stage("mri_ca_normalize", run_ca_normalize, mri / "nu.mgz",
          mri / "brainmask.mgz", assets / "average/RB_all_2020-01-02.gca",
          lta, mri / "norm.mgz", mri / "ctrl_pts.mgz")

    wm = np.where(np.isin(aseg, (2, 41, 77, 78, 79)), 255, 0).astype(np.uint8)
    _save_like(mri / "T1.mgz", mri / "wm.mgz", wm)
    filled = np.zeros(aseg.shape, np.uint8)
    for code, label_id in ((255, 2), (127, 41)):
        filled[_largest_filled(aseg == label_id)] = code
    _save_like(mri / "T1.mgz", mri / "filled.mgz", filled)
    for hemi in ("lh", "rh"):
        result = stage(f"surface_{hemi}", _surface_pair, subject, hemi,
                       mri / "filled.mgz", mri / "norm.mgz", aseg, device=device)
        report.setdefault("surfaces", {})[hemi] = result

    for hemi in ("lh", "rh"):
        for atlas, prefix in (("aparc", "DKaparc"),
                              ("aparc.a2009s", "CDaparc"),
                              ("aparc.DKTatlas", "DKTaparc")):
            atlas_file = assets / "average" / (
                f"{hemi}.{prefix}.atlas.acfb40.noaparc.i12.2016-08-02.gcs")
            stage(f"annot_{hemi}_{atlas}", label_surface, subject, hemi,
                  atlas_file, assets / "lib/bem/ic4.tri",
                  assets / "lib/bem/ic7.tri",
                  labels / f"{hemi}.{atlas}.annot", device=device)
    stage("project_aparc_volumes", _project_parcels, subject, aseg)
    stage("project_wmparc", _project_wmparc, subject, aseg)

    volumes = stage("brain_volume_stats", compute_brain_volume_stats,
                    subject, assets / "ASegStatsLUT.txt")
    (stats / "brainvol.stats").write_text("".join(
        f"# Measure {name}, {name}, {name}, {value:.6f}, mm^3\n"
        for name, value in volumes.items()))
    (stats / "synthseg.tiv.dat").write_text(f"{stiv_mm3:.6f}\n")
    stage("aseg_stats", write_aseg_stats, subject, assets / "ASegStatsLUT.txt",
          stats / "aseg.stats")
    stage("wmparc_stats", write_wmparc_stats, subject, assets / "WMParcStatsLUT.txt",
          stats / "wmparc.stats")
    for hemi in ("lh", "rh"):
        for atlas in ("aparc", "aparc.a2009s", "aparc.DKTatlas"):
            stage(f"stats_{hemi}_{atlas}", write_anatomical_stats,
                  subject, hemi, atlas, "white", volumes,
                  stats / f"{hemi}.{atlas}.stats", device=device)
    report.update(status="complete", total_seconds=time.perf_counter() - started)
    (subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("t1", type=Path)
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n4-python", type=Path)
    args = parser.parse_args(argv)
    report = run_recon_all_python(args.t1, args.subject_dir, args.weights_dir,
                                  args.assets_dir, device=args.device,
                                  threads=args.threads, n4_python=args.n4_python)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
