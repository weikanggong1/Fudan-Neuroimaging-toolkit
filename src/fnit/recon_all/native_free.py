"""T1-to-morphometry reconstruction with Conda WM and optional surface stages.

Surface placement and registration are still approximate; the run report
records each selected implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np
import torch


def _native_binary(native_bin_dir: str | Path, name: str) -> tuple[Path, str]:
    binary = Path(native_bin_dir).resolve() / name
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"executable {name} not found: {binary}")
    digest = hashlib.sha256()
    with binary.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return binary, digest.hexdigest()


def _native_em_register_binary(native_bin_dir: str | Path) -> tuple[Path, str]:
    return _native_binary(native_bin_dir, "mri_em_register")


def _native_topology_binary(native_bin_dir: str | Path) -> tuple[Path, str]:
    return _native_binary(native_bin_dir, "mris_fix_topology_fnit")


def _native_surface_metrics_binary(native_bin_dir: str | Path) -> tuple[Path, str]:
    return _native_binary(native_bin_dir, "mris_place_surface")


def _native_inflate_binary(native_bin_dir: str | Path) -> tuple[Path, str]:
    return _native_binary(native_bin_dir, "mris_inflate")


def _run_native_wm_segment(binary: Path, mri: Path, assets: Path) -> None:
    env = dict(os.environ, FREESURFER_HOME=str(assets))
    subprocess.run([str(binary), "-wsizemm", "13", "-mprage",
                    "antsdn.brain.mgz", "wm.seg.mgz"],
                   cwd=mri, env=env, check=True)


def _run_native_wm_edit(binary: Path, mri: Path, assets: Path) -> None:
    env = dict(os.environ, FREESURFER_HOME=str(assets))
    subprocess.run([str(binary), "-keep-in", "-fix-ento-wm", "entowm.mgz",
                    "3", "255", "255", "-fix-acj", "aseg.presurf.mgz",
                    "255", "255", "-fill-seg-wm", "-fix-scm-ha", "1",
                    "wm.seg.mgz", "brain.mgz", "aseg.presurf.mgz",
                    "wm.asegedit.mgz"], cwd=mri, env=env, check=True)


def _folding_atlas(assets: Path, hemi: str) -> Path:
    atlas = assets / "average" / (
        f"{hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif")
    if not atlas.is_file():
        raise FileNotFoundError(f"folding atlas not found: {atlas}")
    return atlas


def _run_native_em_register(binary: Path, mri: Path, atlas: Path,
                            assets: Path) -> None:
    env = dict(os.environ, FREESURFER_HOME=str(assets))
    subprocess.run([str(binary), "-uns", "3", "-mask", "brainmask.mgz",
                    "nu.mgz", str(atlas), "transforms/talairach.lta"],
                   cwd=mri, env=env, check=True)
    if not (mri / "transforms/talairach.lta").is_file():
        raise FileNotFoundError(mri / "transforms/talairach.lta")


def _run_native_topology(binary: Path, subject: Path, hemi: str,
                         assets: Path) -> None:
    from .topology_conda_ga import run_topology_ga_conda

    result = run_topology_ga_conda(subject, hemi, binary, assets)
    output = Path(result["output"])
    if not output.is_file():
        raise FileNotFoundError(f"topology GA produced no surface: {output}")


def _run_native_sphere_step(binary: Path, subject: Path, assets: Path,
                            arguments: list[str], outputs: tuple[Path, ...]) -> float:
    scripts = subject / "scripts"
    scripts.mkdir(exist_ok=True)
    env = dict(os.environ, SUBJECTS_DIR=str(subject.parent),
               FREESURFER_HOME=str(assets))
    tick = time.perf_counter()
    subprocess.run([str(binary), *arguments], cwd=scripts, env=env, check=True)
    for output in outputs:
        if not output.is_file():
            raise FileNotFoundError(f"{binary.name} produced no output: {output}")
    return time.perf_counter() - tick


def _prepare_native_topology(binary: Path, subject: Path, hemi: str,
                             assets: Path, device: str,
                             native_inflate_binary: Path | None = None
                             ) -> tuple[float, float, dict[str, float], float, float]:
    from .mris_remesh_python import remesh_surface
    from .mris_remove_intersection_python import remove_intersection_surface
    from .smooth_surface_python import smooth_surface
    from .sphere_quick_python import write_quick_sphere

    surf = subject / "surf"
    tick = time.perf_counter()
    smooth_nofix = surf / f"{hemi}.smoothwm.nofix"
    inflated_nofix = surf / f"{hemi}.inflated.nofix"
    qsphere_nofix = surf / f"{hemi}.qsphere.nofix"
    smooth_surface(surf / f"{hemi}.orig.nofix", smooth_nofix, device=device)
    python_seconds = time.perf_counter() - tick
    sphere_seconds = {}
    if native_inflate_binary is not None:
        sphere_seconds["inflate_nofix"] = _run_native_sphere_step(
            native_inflate_binary, subject, assets,
            ["-no-save-sulc", str(smooth_nofix), str(inflated_nofix)],
            (inflated_nofix,))
    else:
        from .inflate_python import inflate_surface
        tick = time.perf_counter()
        inflate_surface(smooth_nofix, inflated_nofix)
        python_seconds += time.perf_counter() - tick
    tick = time.perf_counter()
    write_quick_sphere(inflated_nofix, qsphere_nofix)
    quick_seconds = time.perf_counter() - tick
    if native_inflate_binary is not None:
        sphere_seconds["qsphere_nofix_python"] = quick_seconds
    else:
        python_seconds += quick_seconds
    native_started = time.perf_counter()
    _run_native_topology(binary, subject, hemi, assets)
    native_seconds = time.perf_counter() - native_started
    tick = time.perf_counter()
    remesh_surface(surf / f"{hemi}.orig.premesh", surf / f"{hemi}.orig", iterations=3)
    remesh_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    orig = surf / f"{hemi}.orig"
    remove_intersection_surface(orig, orig)
    intersection_seconds = time.perf_counter() - tick
    return (python_seconds + remesh_seconds + intersection_seconds, native_seconds,
            sphere_seconds, remesh_seconds, intersection_seconds)


def _run_accurate_sphere_pair(inflate_binary: Path, subject: Path,
                              hemi: str, assets: Path) -> tuple[dict, dict]:
    from .sphere_standard_run import run_standard_sphere

    surf = subject / "surf"
    inflated, sulc = surf / f"{hemi}.inflated", surf / f"{hemi}.sulc"
    inflate_seconds = _run_native_sphere_step(
        inflate_binary, subject, assets,
        [str(surf / f"{hemi}.smoothwm"), str(inflated)], (inflated, sulc))
    sphere_report = run_standard_sphere(
        inflated, surf / f"{hemi}.smoothwm", surf / f"{hemi}.sphere",
        finish_device="cpu")
    return ({"inflate": inflate_seconds,
             "sphere": sphere_report["total_seconds_including_io"]}, sphere_report)


def _run_native_surface_metrics(binary: Path, subject: Path, hemi: str,
                                assets: Path) -> dict[str, float]:
    surf = subject / "surf"
    white, pial = surf / f"{hemi}.white", surf / f"{hemi}.pial"
    commands = (
        ("thickness", ["--thickness", str(white), str(pial), "20", "5",
                       str(surf / f"{hemi}.thickness")]),
        ("area", ["--area-map", str(white), str(surf / f"{hemi}.area")]),
        ("area.pial", ["--area-map", str(pial), str(surf / f"{hemi}.area.pial")]),
        ("curv", ["--curv-map", str(white), "2", "10",
                  str(surf / f"{hemi}.curv")]),
        ("curv.pial", ["--curv-map", str(pial), "2", "10",
                       str(surf / f"{hemi}.curv.pial")]),
    )
    scripts = subject / "scripts"
    scripts.mkdir(exist_ok=True)
    env = dict(os.environ, SUBJECTS_DIR=str(subject.parent),
               FREESURFER_HOME=str(assets))
    timings = {}
    for name, args in commands:
        tick = time.perf_counter()
        subprocess.run([str(binary), *args], cwd=scripts, env=env, check=True)
        output = surf / f"{hemi}.{name}"
        if not output.is_file():
            raise FileNotFoundError(f"mris_place_surface produced no morph: {output}")
        timings[name] = time.perf_counter() - tick
    return timings


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


def _run_white_mri_chain(subject: Path, weights: Path, assets: Path,
                         threads: int, stage) -> None:
    from .finalsurfs_python import run_finalsurfs
    from .mni_aux_chain import run_mni_aux_chain

    stage("mni_aux", run_mni_aux_chain, subject, weights, assets,
          device="cpu", threads=threads)
    stage("brain_finalsurfs", run_finalsurfs, subject, device="cpu")


def _place_preaparc_and_smooth(subject: Path, hemi: str, binary: Path,
                               assets: Path, threads: int) -> dict:
    from .smooth_surface_python import smooth_surface
    from .white_preaparc_conda import run_white_preaparc

    report = run_white_preaparc(subject, hemi, binary, assets, threads=threads)
    tick = time.perf_counter()
    surf = subject / "surf"
    smooth_surface(surf / f"{hemi}.white.preaparc", surf / f"{hemi}.smoothwm",
                   iterations=3, device="cpu")
    report["smoothwm_seconds"] = time.perf_counter() - tick
    return report


def _write_principal_curvature_maps(surf: Path, hemi: str,
                                    device: str, placed_preaparc: bool) -> None:
    from .surface_roi_curvature_gpu import principal_curvatures

    preaparc_source = "white.preaparc" if placed_preaparc else "white"
    smoothwm_curv = None
    for surface, prefix in ((preaparc_source, "white.preaparc"),
                            ("inflated", "inflated")):
        xyz, mesh = fs.read_geometry(str(surf / f"{hemi}.{surface}"))
        k1, k2 = principal_curvatures(xyz, mesh, device=device)
        fs.write_morph_data(str(surf / f"{hemi}.{prefix}.H"),
                            ((k1 + k2) / 2).astype(np.float32))
        fs.write_morph_data(str(surf / f"{hemi}.{prefix}.K"),
                            (k1 * k2).astype(np.float32))
        if surface == "white":
            smoothwm_curv = (k1, k2)
    if smoothwm_curv is None:
        xyz, mesh = fs.read_geometry(str(surf / f"{hemi}.smoothwm"))
        smoothwm_curv = principal_curvatures(xyz, mesh, device=device)
    k1, k2 = smoothwm_curv
    for name, values in (("H", (k1 + k2) / 2), ("K", k1 * k2),
                         ("K1", k1), ("K2", k2)):
        fs.write_morph_data(str(surf / f"{hemi}.smoothwm.{name}.crv"),
                            np.asarray(values, np.float32))


def _surface_pair(subject: Path, hemi: str, filled: Path, norm: Path,
                  aseg: np.ndarray, *, device: str, threads: int = 4,
                  native_topology_binary: Path | None = None,
                  native_surface_metrics_binary: Path | None = None,
                  native_inflate_binary: Path | None = None,
                  native_registration: bool = False,
                  native_white_preaparc_binary: Path | None = None,
                  assets: Path | None = None) -> dict:
    from .extract_main_component_python import extract_main_component
    from .pretess_python import pretess_mgh
    from .smooth_surface_python import smooth_surface
    from .sphere_python import project_radially
    from .tessellate_gpu import tessellate_mgh, write_quad_surface
    from .surface_area_gpu import area_map, mid_area_map
    from .surface_curvature_gpu import curvature_map
    from .surface_roi_gpu import vertex_volume_map

    if native_inflate_binary is not None and native_topology_binary is None:
        raise ValueError("native_sphere requires native_topology")
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
    smoothwm = surf / f"{hemi}.smoothwm"
    topology_python_seconds = topology_native_seconds = topology_remesh_seconds = topology_intersection_seconds = None
    native_sphere_seconds = None
    if native_topology_binary is not None:
        if assets is None:
            raise ValueError("assets are required for native topology repair")
        topology_python_seconds, topology_native_seconds, pre_sphere_seconds, topology_remesh_seconds, topology_intersection_seconds = (
            _prepare_native_topology(native_topology_binary, subject, hemi,
                                     assets, device, native_inflate_binary))
        if native_inflate_binary is not None:
            native_sphere_seconds = pre_sphere_seconds
    else:
        shutil.copyfile(orig, surf / f"{hemi}.orig")
    white_preaparc_report = None
    if native_white_preaparc_binary is not None:
        if assets is None:
            raise ValueError("assets are required for native white pre-aparc")
        white_preaparc_report = _place_preaparc_and_smooth(
            subject, hemi, native_white_preaparc_binary, assets, threads)
    else:
        smooth_surface(surf / f"{hemi}.orig", smoothwm, device=device)
        shutil.copyfile(smoothwm, surf / f"{hemi}.white.preaparc")
    if native_white_preaparc_binary is not None:
        from .label_cortex_fix_ga_python import label_cortex_fix_ga
        from .label_cortex_python import label_cortex

        base, ga = label_cortex_fix_ga(
            surf / f"{hemi}.white.preaparc", mri / "aseg.presurf.mgz",
            mri / "entowm.mgz", hemi, labels / f"{hemi}.cortex.label")
        label_cortex(surf / f"{hemi}.white.preaparc",
                     mri / "aseg.presurf.mgz",
                     labels / f"{hemi}.cortex+hipamyg.label",
                     keep_hip_amyg=True)
        cortex_count = len(base) + len(ga)
    else:
        shutil.copyfile(smoothwm, surf / f"{hemi}.white")
    if native_topology_binary is None:
        shutil.copyfile(smoothwm, surf / f"{hemi}.smoothwm.nofix")

    white, faces = fs.read_geometry(str(smoothwm))
    if native_white_preaparc_binary is None:
        image = nib.load(str(mri / "aseg.mgz"))
        pial, thickness = _pial_from_cortex(white, faces, image, aseg, hemi)
        _replace_vertices(smoothwm, surf / f"{hemi}.pial", pial)
        shutil.copyfile(surf / f"{hemi}.pial", surf / f"{hemi}.pial.T1")

    standard_sphere_report = None
    if native_inflate_binary is not None:
        if assets is None:
            raise ValueError("assets are required for native sphere generation")
        timings, standard_sphere_report = _run_accurate_sphere_pair(
            native_inflate_binary, subject, hemi, assets)
        native_sphere_seconds.update(timings)
    else:
        inflated = surf / f"{hemi}.inflated"
        smooth_surface(smoothwm, inflated, iterations=35, device=device)
        inflated_xyz, _ = fs.read_geometry(str(inflated))
        sphere_xyz = project_radially(inflated_xyz)
        _replace_vertices(inflated, surf / f"{hemi}.sphere", sphere_xyz)
        fs.write_morph_data(str(surf / f"{hemi}.sulc"),
                            np.linalg.norm(inflated_xyz - white, axis=1).astype(np.float32))
    if not native_registration:
        shutil.copyfile(surf / f"{hemi}.sphere", surf / f"{hemi}.sphere.reg")
    if native_white_preaparc_binary is not None:
        _write_principal_curvature_maps(surf, hemi, device, placed_preaparc=True)
        return {"hemisphere": hemi, "vertices": len(white), "faces": len(faces),
                "raw_components": components, "cortex_vertices": cortex_count,
                "topology_python_seconds": topology_python_seconds,
                "topology_native_seconds": topology_native_seconds,
                "topology_remesh_seconds": topology_remesh_seconds,
                "topology_intersection_seconds": topology_intersection_seconds,
                "native_sphere_seconds": native_sphere_seconds,
                "standard_sphere_report": standard_sphere_report,
                "white_preaparc_report": white_preaparc_report,
                "placement_pending": True}
    native_metric_seconds = None
    if native_surface_metrics_binary is not None:
        if assets is None:
            raise ValueError("assets are required for native surface metrics")
        native_metric_seconds = _run_native_surface_metrics(
            native_surface_metrics_binary, subject, hemi, assets)
    else:
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
    if native_surface_metrics_binary is None:
        curvature_map(surf / f"{hemi}.white", surf / f"{hemi}.curv", device=device)
        curvature_map(surf / f"{hemi}.pial", surf / f"{hemi}.curv.pial", device=device)
    _write_principal_curvature_maps(
        surf, hemi, device, placed_preaparc=white_preaparc_report is not None)
    approximations = []
    if native_topology_binary is None:
        approximations.append("topology-unrepaired")
    if native_surface_metrics_binary is not None:
        approximations.append("native-metrics-on-approximate-white-pial")
    registration_approximation = (
        "sphere.reg-unregistered" if not native_registration
        else "sphere.reg-python-on-approximate-upstream" if native_inflate_binary is not None
        else "sphere.reg-python-on-approximate-sphere-sulc")
    sphere_approximations = ([] if native_inflate_binary is not None else
                             ["inflated-Laplacian", "sphere-radial"])
    return {"hemisphere": hemi, "vertices": len(white), "faces": len(faces),
            "raw_components": components, "cortex_vertices": len(cortex_vertices),
            "mean_thickness_mm": float(np.mean(fs.read_morph_data(
                str(surf / f"{hemi}.thickness")))),
            "topology_python_seconds": topology_python_seconds,
            "topology_native_seconds": topology_native_seconds,
            "topology_remesh_seconds": topology_remesh_seconds,
            "topology_intersection_seconds": topology_intersection_seconds,
            "native_metric_seconds": native_metric_seconds,
            "native_sphere_seconds": native_sphere_seconds,
            "standard_sphere_report": standard_sphere_report,
            "white_preaparc_report": white_preaparc_report,
            "approximations": approximations + [
                ("white-from-preaparc-smooth-only" if white_preaparc_report
                 else "white-from-smoothed-tessellation"), "pial-normal-ray",
                *sphere_approximations, registration_approximation]}


def _finish_cortical_surface(subject: Path, hemi: str, binary: Path,
                             assets: Path, *, device: str, threads: int,
                             metrics_binary: Path | None = None) -> dict:
    """Place final white/pial after sphere registration and annotation."""
    from .final_white_conda import run_final_white
    from .place_pial_python import place_pial_t1
    from .surface_area_gpu import area_map, mid_area_map
    from .surface_curvature_gpu import curvature_map
    from .surface_roi_gpu import vertex_volume_map
    from .surface_thickness_gpu import thickness_map

    surf, labels = subject / "surf", subject / "label"
    white_report = run_final_white(subject, hemi, binary, assets, threads=threads)
    pial_report = place_pial_t1(subject, hemi)
    shutil.copyfile(surf / f"{hemi}.pial.T1", surf / f"{hemi}.pial")
    metric_seconds = None
    if metrics_binary is not None:
        metric_seconds = _run_native_surface_metrics(metrics_binary, subject, hemi, assets)
    else:
        thickness_map(surf / f"{hemi}.white", surf / f"{hemi}.pial",
                      surf / f"{hemi}.thickness", device=device)
        area_map(surf / f"{hemi}.white", surf / f"{hemi}.area", device=device)
        area_map(surf / f"{hemi}.pial", surf / f"{hemi}.area.pial", device=device)
        curvature_map(surf / f"{hemi}.white", surf / f"{hemi}.curv", device=device)
        curvature_map(surf / f"{hemi}.pial", surf / f"{hemi}.curv.pial", device=device)
    mid_area_map(surf / f"{hemi}.area", surf / f"{hemi}.area.pial",
                 surf / f"{hemi}.area.mid", device=device)
    vertex_volume_map(surf / f"{hemi}.white", surf / f"{hemi}.pial",
                      labels / f"{hemi}.cortex.label",
                      surf / f"{hemi}.volume", device=device)
    return {"final_white_report": white_report, "pial_report": pial_report,
            "native_metric_seconds": metric_seconds,
            "mean_thickness_mm": float(np.mean(fs.read_morph_data(
                str(surf / f"{hemi}.thickness")))),
            "placement_pending": False,
            "approximations": ["final-white-conda-cpp", "pial-python",
                               "sphere-and-upstream-parity-unverified"]}


def _project_parcels(subject: Path) -> None:
    """Run the validated bilateral cortex-volume mapping for three atlases."""
    from .surf2volseg_cortex_python import label_cortex_volume

    mri, surf, labels = (subject / name for name in ("mri", "surf", "label"))
    for atlas, output in (("aparc", "aparc+aseg.mgz"),
                          ("aparc.a2009s", "aparc.a2009s+aseg.mgz"),
                          ("aparc.DKTatlas", "aparc.DKTatlas+aseg.mgz")):
        label_cortex_volume(mri / "aseg.mgz", surf, labels, mri / output,
                            atlas=atlas)


def _project_wmparc(subject: Path) -> None:
    """Run the validated white-matter volume mapping from aparc+aseg."""
    from .surf2volseg_wm_python import label_wm_volume

    mri, surf, labels = (subject / name for name in ("mri", "surf", "label"))
    label_wm_volume(mri / "aparc+aseg.mgz", surf, labels, mri / "wmparc.mgz")


def _segment_callosum(mri: Path) -> dict[str, float | int]:
    from .mri_cc_python import run_mri_cc

    source = mri / "aseg.auto_noCCseg.mgz"
    output = mri / "aseg.auto.mgz"
    shutil.copyfile(mri / "synthseg.rca.mgz", source)
    info = run_mri_cc(source, mri / "norm.mgz", output,
                      mri / "transforms/cc_up.lta")
    for name in ("aseg.presurf.mgz", "aseg.mgz"):
        shutil.copyfile(output, mri / name)
    return info


def run_recon_all_python(t1: str | Path, subject_dir: str | Path,
                         weights_dir: str | Path, assets_dir: str | Path,
                         *, device: str = "cuda:0", threads: int = 4,
                         native_bin_dir: str | Path | None = None,
                         native_topology: bool = False,
                         native_surface_metrics: bool = False,
                         native_registration: bool = False,
                         native_sphere: bool = False,
                         native_white_preaparc: bool = False) -> dict:
    """Reconstruct one T1 into FreeSurfer-style mri/surf/label/stats folders.

    Return the same run-report dict written to fnit-native-free-run.json:
    stage timings, per-hemisphere surfaces, implementation provenance and
    success/failure status. The subject directory must be empty on entry.
    """
    from fnit.synthseg_parc import SynthSeg
    from .brain_volume_stats_python import compute_brain_volume_stats
    from .ca_normalize_python import run_ca_normalize
    from .gcsa_label_python import label_surface
    from .input_talairach_chain import run_input_talairach_chain
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
    if native_bin_dir is None:
        raise ValueError("native stages require native_bin_dir")
    if native_registration and not native_topology:
        raise ValueError("native_registration requires native_topology")
    if native_sphere and not native_topology:
        raise ValueError("native_sphere requires native_topology")
    if native_white_preaparc and not native_topology:
        raise ValueError("native_white_preaparc requires native_topology")
    if native_white_preaparc:
        from .mni_aux_chain import validate_mni_aux_assets

        validate_mni_aux_assets(weights, assets)
    native_em = _native_em_register_binary(native_bin_dir)
    n4_binary = _native_binary(native_bin_dir, "fnit_n4_itk")
    topology_binary = (_native_topology_binary(native_bin_dir)
                       if native_topology else None)
    metrics_binary = (_native_surface_metrics_binary(native_bin_dir)
                      if native_surface_metrics else None)
    white_binary = ((metrics_binary or _native_surface_metrics_binary(native_bin_dir))
                    if native_white_preaparc else None)
    inflate_binary = (_native_inflate_binary(native_bin_dir)
                      if native_sphere else None)
    wm_segment_binary = _native_binary(native_bin_dir, "mri_segment")
    wm_edit_binary = _native_binary(native_bin_dir, "mri_edit_wm_with_aseg")
    registration_atlases = ({hemi: _folding_atlas(assets, hemi)
                            for hemi in ("lh", "rh")} if native_registration else {})
    torch.set_num_threads(threads)
    started = time.perf_counter()
    profile = "conda-wmchain-core-v5"
    report: dict = {"profile": profile, "input": str(t1),
                    "subject_dir": str(subject), "device": device,
                    "n4_binary": {"binary": str(n4_binary[0]),
                                  "sha256": n4_binary[1]}, "threads": threads,
                    "stages": [], "status": "running"}
    report["gca_registration"] = {"implementation": "native-c++",
                                  "binary": str(native_em[0]), "sha256": native_em[1]}
    report["topology_repair"] = (
        {"implementation": "native-c++", "binary": str(topology_binary[0]),
         "sha256": topology_binary[1],
         "upstream": "intensity-derived WM and aseg-guided fill"}
        if topology_binary else {"implementation": "unrepaired approximation"})
    report["white_matter_chain"] = {
        "implementation": "Python + Conda C++",
        "mri_segment_sha256": wm_segment_binary[1],
        "mri_edit_wm_with_aseg_sha256": wm_edit_binary[1]}
    report["white_preaparc"] = (
        {"implementation": "Python MNI/aux/finalsurfs + Conda C++ placement",
         "binary": str(white_binary[0]), "sha256": white_binary[1],
         "final_smoothwm": "Python 3 passes on CPU",
         "final_white_pial": "Conda C++ final white + Python pial after annotation"}
        if white_binary else {"implementation": "smoothed orig copy"})
    report["surface_metrics"] = (
        {"implementation": "native-c++", "binary": str(metrics_binary[0]),
         "sha256": metrics_binary[1],
         "upstream": ("placed final white/pial" if native_white_preaparc
                      else "current approximate white/pial surfaces")}
        if metrics_binary else {"implementation": "python"})
    report["sphere_generation"] = (
        {"implementation": "Conda mris_inflate + Python quick/standard sphere",
         "mris_inflate": {"binary": str(inflate_binary[0]), "sha256": inflate_binary[1]},
         "finish_device": "cpu",
         "upstream": "native topology repair on WM-chain filled"}
        if native_sphere else {"implementation": "python radial approximation"})
    report["sphere_registration"] = (
        {"implementation": "Python/Numba", "overlap_device": "cpu",
         "upstream": ("accurate sphere/sulc on approximate earlier geometry" if native_sphere
                      else "radial approximate sphere and non-native sulc/smoothwm"),
         "hemisphere_seconds": {}, "reports": {}}
        if native_registration else {"implementation": "unregistered copy"})

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
    for folder in (surf, labels, stats, mri / "tmp", subject / "scripts"):
        folder.mkdir(parents=True, exist_ok=True)
    nu0 = mri / "tmp/nu0.mgz"
    from .n4_itk import correct_volume
    stage("n4", correct_volume, mri / "orig.mgz", nu0, binary=n4_binary[0])
    stage("nu", make_nu, mri / "orig.mgz", nu0,
          initial["talairach_xfm"], mri / "nu.mgz")
    stage("T1_normalize", normalize_t1, mri / "nu.mgz",
          initial["talairach_xfm"], mri / "T1.mgz", device=device)
    stage("brainmask", mask_volume, mri / "T1.mgz",
          initial["synthstrip"], mri / "brainmask.mgz", device=device)
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        result = stage("SynthSeg", lambda: SynthSeg(weights=weights, device=device,
                                                      threads=threads)(mri / "orig.mgz",
                         keep_geometry=True, color_lut=assets / "FreeSurferColorLUT.txt"))
    finally:
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    stiv_mm3 = result.total_intracranial_mm3
    result.segmentation.save(str(mri / "synthseg.rca.mgz"))
    result.write_volumes_csv(mri / "orig.mgz", stats / "synthseg.vol.csv")
    lta = mri / "transforms/talairach.lta"
    gca = assets / "average/RB_all_2020-01-02.gca"
    stage("mri_em_register", _run_native_em_register, native_em[0], mri, gca, assets)
    stage("mri_ca_normalize", run_ca_normalize, mri / "nu.mgz",
          mri / "brainmask.mgz", gca,
          lta, mri / "norm.mgz", mri / "ctrl_pts.mgz")
    report["corpus_callosum"] = stage("mri_cc", _segment_callosum, mri)
    aseg = np.asarray(nib.load(str(mri / "aseg.auto.mgz")).dataobj).astype(np.int16)

    from .ants_denoise_python import denoise_volume
    from .fill_cutting_plane_python import fill_mgz
    from .normalization.aseg_pipeline import normalize_t1_aseg
    from .pretess_python import pretess_mgh
    from .sclimbic import mri_entowm_seg
    from .wm_edits_python import fix_ento_wm

    stage("brain_second_normalize", normalize_t1_aseg,
          mri / "norm.mgz", mri / "aseg.presurf.mgz",
          mri / "brainmask.mgz", mri / "brain.mgz", device="cpu")
    stage("entowm", mri_entowm_seg, mri / "nu.mgz", mri / "entowm.mgz",
          weights, device="cpu")
    stage("ants_denoise", denoise_volume, mri / "brain.mgz",
          mri / "antsdn.brain.mgz")
    stage("mri_segment", _run_native_wm_segment,
          wm_segment_binary[0], mri, assets)
    stage("mri_edit_wm_with_aseg", _run_native_wm_edit,
          wm_edit_binary[0], mri, assets)
    stage("wm_pretess", pretess_mgh, mri / "wm.asegedit.mgz", "wm",
          mri / "norm.mgz", mri / "wm.mgz")
    stage("wm_fix_ento", fix_ento_wm, mri / "wm.mgz",
          mri / "entowm.mgz", mri / "wm.mgz", level=3,
          left_value=255, right_value=255)
    stage("wm_fix_acj", fix_ento_wm, mri / "wm.mgz",
          mri / "aseg.presurf.mgz", mri / "wm.mgz", level=3,
          left_value=255, right_value=255, acj=True)
    stage("mri_fill", fill_mgz, mri / "wm.mgz",
          mri / "aseg.presurf.mgz", lta,
          assets / "SubCorticalMassLUT.txt", mri / "filled.mgz",
          subject / "scripts/ponscc.cut.log")
    if native_white_preaparc:
        _run_white_mri_chain(subject, weights, assets, threads, stage)
    for hemi in ("lh", "rh"):
        result = stage(f"surface_{hemi}", _surface_pair, subject, hemi,
                       mri / "filled.mgz", mri / "norm.mgz", aseg,
                       device=device, threads=threads,
                       native_topology_binary=topology_binary[0] if topology_binary else None,
                       native_surface_metrics_binary=metrics_binary[0] if metrics_binary else None,
                       native_inflate_binary=inflate_binary[0] if inflate_binary else None,
                       native_registration=native_registration,
                       native_white_preaparc_binary=white_binary[0] if white_binary else None,
                       assets=assets if topology_binary or metrics_binary or white_binary else None)
        report.setdefault("surfaces", {})[hemi] = result

    if native_registration:
        from .mris_register_run import run_register_sphere
        for hemi in ("lh", "rh"):
            surf = subject / "surf"
            result = stage(f"register_{hemi}", run_register_sphere,
                           surf / f"{hemi}.sphere", surf / f"{hemi}.smoothwm",
                           surf / f"{hemi}.sulc", registration_atlases[hemi],
                           surf / f"{hemi}.sphere.reg", overlap_device="cpu")
            report["sphere_registration"]["hemisphere_seconds"][hemi] = result[
                "total_seconds_including_io"]
            report["sphere_registration"]["reports"][hemi] = result

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
    if native_white_preaparc:
        for hemi in ("lh", "rh"):
            result = stage(f"finish_surface_{hemi}", _finish_cortical_surface,
                           subject, hemi, white_binary[0], assets,
                           device=device, threads=threads,
                           metrics_binary=metrics_binary[0] if metrics_binary else None)
            report["surfaces"][hemi].update(result)
    stage("project_aparc_volumes", _project_parcels, subject)
    stage("project_wmparc", _project_wmparc, subject)

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
    parser.add_argument("--native-bin-dir", type=Path)
    parser.add_argument("--native-topology", action="store_true")
    parser.add_argument("--native-surface-metrics", action="store_true")
    parser.add_argument("--native-registration", action="store_true")
    parser.add_argument("--native-sphere", action="store_true")
    parser.add_argument("--native-white-preaparc", action="store_true")
    args = parser.parse_args(argv)
    report = run_recon_all_python(args.t1, args.subject_dir, args.weights_dir,
                                  args.assets_dir, device=args.device,
                                  threads=args.threads,
                                  native_bin_dir=args.native_bin_dir,
                                  native_topology=args.native_topology,
                                  native_surface_metrics=args.native_surface_metrics,
                                  native_registration=args.native_registration,
                                  native_sphere=args.native_sphere,
                                  native_white_preaparc=args.native_white_preaparc)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
