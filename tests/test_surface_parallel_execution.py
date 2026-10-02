"""Thread, I/O and publication contracts; these controls are not benchmarks."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import shutil
import subprocess
import threading

import nibabel as nib
import numpy as np
import pytest

from fnit._hemisphere_parallel import (
    hemisphere_items, map_hemispheres, resolve_cpu_threads, workbench_environment,
)
from fnit.fmri import surface_fmriprep as projection
from fnit.fmri import surface_prepare as preparation
from test_fmri_surface_preparation import reconstruction
from test_fmri_surface_contracts import cifti_inputs, official_assets, _hemisphere
from test_fmri_surface_public_contracts import public_case


@pytest.mark.parametrize("budget, expected", [
    (1, (("L", 1), ("R", 1))), (2, (("L", 1), ("R", 1))),
    (3, (("L", 2), ("R", 1))), (8, (("L", 4), ("R", 4))),
])
def test_total_cpu_budget_and_serial_path(budget, expected):
    assert hemisphere_items(cpu_threads=budget) == expected
    assert hemisphere_items(parallel=False, cpu_threads=budget) == (("L", budget), ("R", budget))


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "8"])
def test_invalid_explicit_cpu_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="positive integer"):
        resolve_cpu_threads(budget)


def test_budget_and_child_environment_leave_parent_state_unchanged(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "12")
    before = dict(os.environ)
    assert resolve_cpu_threads() == 8
    child = workbench_environment(4)
    assert child["OMP_NUM_THREADS"] == child["MKL_NUM_THREADS"] == "4"
    assert dict(os.environ) == before
    monkeypatch.delenv("OMP_NUM_THREADS")
    import torch
    monkeypatch.setattr(torch, "get_num_threads", lambda: 6)
    assert resolve_cpu_threads() == 6


def test_parallel_workers_overlap_and_results_stay_left_right():
    barrier = threading.Barrier(2)
    worker_ids = set()

    def work(hemi, threads):
        worker_ids.add(threading.get_ident())
        barrier.wait(timeout=5)
        return hemi, threads

    assert map_hemispheres(work, cpu_threads=8) == (("L", 4), ("R", 4))
    assert len(worker_ids) == 2


@pytest.mark.parametrize("parallel, budget", [(False, 8), (True, 1)])
def test_serial_controls_use_one_thread_in_fixed_order(parallel, budget):
    seen = []
    caller = threading.get_ident()

    def work(hemi, threads):
        seen.append((hemi, threads, threading.get_ident()))
        return hemi

    assert map_hemispheres(work, parallel=parallel, cpu_threads=budget) == ("L", "R")
    assert seen == [("L", budget, caller), ("R", budget, caller)]


def test_failed_worker_waits_for_other_before_staging_cleanup(tmp_path):
    started = threading.Event()
    failed = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    staged_paths = []

    def task():
        with TemporaryDirectory(dir=tmp_path) as temporary:
            staging = Path(temporary)
            staged_paths.append(staging)

            def work(hemi, threads):
                if hemi == "L":
                    assert started.wait(5)
                    failed.set()
                    raise RuntimeError("left failed")
                started.set()
                assert release.wait(5)
                assert staging.is_dir()
                (staging / "right.result").write_text("finished")
                completed.set()

            map_hemispheres(work, cpu_threads=8)

    with ThreadPoolExecutor(max_workers=1) as controller:
        future = controller.submit(task)
        try:
            assert failed.wait(5)
            assert not future.done()
            assert staged_paths[0].is_dir()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="left failed"):
            future.result(timeout=5)
    assert completed.is_set()
    assert not staged_paths[0].exists()


@pytest.mark.parametrize("affine", [
    np.diag([2., 3., 4., 1.]),
    np.array([[0., 0., -2., 10.], [3., 0., 0., 20.], [0., -4., 0., 30.], [0., 0., 0., 1.]]),
])
def test_las_grid_matches_reorientation_without_reading_image_data(affine):
    values = np.arange(3 * 4 * 5 * 2, dtype=np.float32).reshape(3, 4, 5, 2)
    image = nib.Nifti1Image(values, affine)
    expected = projection._las(image)

    class UnreadableData:
        shape = image.shape
        ndim = image.ndim

        def __array__(self, *args, **kwargs):
            raise AssertionError("Grid validation decoded the 4D BOLD")

    image._dataobj = UnreadableData()
    actual = projection._las_grid(image)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.affine, expected.affine)
    assert actual.header is image.header


def test_parallel_geometry_preserves_serial_vertices_and_files(reconstruction, tmp_path):
    serial = preparation.prepare_t1w_surface_geometry(
        reconstruction, tmp_path / "serial", parallel=False, cpu_threads=8)
    parallel = preparation.prepare_t1w_surface_geometry(
        reconstruction, tmp_path / "parallel", parallel=True, cpu_threads=8)
    for left, right in ((serial.left, parallel.left), (serial.right, parallel.right)):
        for name in ("white", "pial", "midthickness"):
            first, second = getattr(left, name), getattr(right, name)
            assert first.read_bytes() == second.read_bytes()
            np.testing.assert_array_equal(nib.load(first).darrays[0].data,
                                          nib.load(second).darrays[0].data)


def test_workbench_preparation_is_parallel_with_split_subprocess_budget(
    reconstruction, tmp_path, monkeypatch,
):
    barrier = threading.Barrier(2)
    thread_settings = []
    monkeypatch.setattr(shutil, "which", lambda _: "/mock/wb_command")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")

    def workbench(arguments, **kwargs):
        if arguments[1] == "-surface-sphere-project-unproject":
            barrier.wait(timeout=5)
        thread_settings.append(kwargs["env"]["OMP_NUM_THREADS"])
        Path(arguments[-1]).write_bytes(b"controlled workbench output")

    monkeypatch.setattr(subprocess, "run", workbench)
    output = tmp_path / "prepared"
    preparation.prepare_fmriprep_surface_inputs(
        reconstruction, tmp_path / "assets", output, cpu_threads=8)
    assert len(thread_settings) == 6 and set(thread_settings) == {"4"}
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert len([path for path in output.rglob("*") if path.is_file()]) == 16


def test_parallel_projection_keeps_cifti_values_axes_and_metadata(
    cifti_inputs, tmp_path, monkeypatch,
):
    bold, left_metric, right_metric, left_roi, right_roi, dseg, _ = cifti_inputs
    hemis = (_hemisphere(tmp_path, "L", left_roi), _hemisphere(tmp_path, "R", right_roi))
    monkeypatch.setattr(shutil, "which", lambda _: "/mock/wb_command")
    overlap = threading.Barrier(2)
    budgets = []
    use_parallel = False

    def workbench(arguments, **kwargs):
        command = arguments[1]
        index = {"-volume-to-surface-mapping": 4, "-metric-dilate": 5,
                 "-metric-mask": 4, "-metric-resample": 6}[command]
        destination = Path(arguments[index])
        if use_parallel and command == "-volume-to-surface-mapping":
            overlap.wait(timeout=5)
        budgets.append(kwargs["env"]["OMP_NUM_THREADS"])
        shutil.copyfile(left_metric if destination.name.startswith("L.") else right_metric, destination)

    monkeypatch.setattr(subprocess, "run", workbench)
    arguments = (bold, bold, *hemis, left_roi, right_roi, dseg)
    serial = projection.run_fmriprep_surface_projection(
        *arguments, tmp_path / "serial", tr_seconds=0.8, parallel=False, cpu_threads=8)
    assert set(budgets) == {"8"}
    budgets.clear()
    use_parallel = True
    parallel = projection.run_fmriprep_surface_projection(
        *arguments, tmp_path / "parallel", tr_seconds=0.8, parallel=True, cpu_threads=8)
    assert set(budgets) == {"4"}
    first = nib.load(serial.dtseries, mmap=False, keep_file_open=False)
    second = nib.load(parallel.dtseries, mmap=False, keep_file_open=False)
    np.testing.assert_array_equal(np.asarray(first.dataobj), np.asarray(second.dataobj))
    assert first.header.get_axis(0) == second.header.get_axis(0)
    assert first.header.get_axis(1) == second.header.get_axis(1)
    assert first.header.matrix.metadata == second.header.matrix.metadata
    assert serial.left_metric.read_bytes() == parallel.left_metric.read_bytes()
    assert serial.right_metric.read_bytes() == parallel.right_metric.read_bytes()


def test_surface_cli_forwards_parallel_and_total_cpu_budget(monkeypatch):
    from fnit.fmri import cli
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr(cli, "fMRISurface_pipeline", lambda **kwargs:
                        calls.append(kwargs) or SimpleNamespace(dtseries="output.dtseries.nii"))
    common = ["surface", "--bids-root", "/bids", "--derivatives-root", "/out",
              "--subject", "example", "--recon-all", "/recon", "--surface-assets-dir", "/assets"]
    cli.main([*common, "--threads", "8"])
    cli.main([*common, "--threads", "3", "--serial-hemispheres"])
    assert [(call["parallel"], call["cpu_threads"]) for call in calls] == [(True, 8), (False, 3)]


def test_public_pipeline_forwards_one_shared_execution_policy(public_case, monkeypatch):
    import json
    from fnit.fmri import surface_pipeline as pipeline
    from fnit.fmri.derivatives import sidecar

    seen = {}
    for name in ("prepare_fmriprep_surface_inputs", "prepare_msmsulc_inputs", "run_msmsulc"):
        original = getattr(pipeline, name)

        def observed(*args, _name=name, _original=original, **kwargs):
            seen[_name] = kwargs
            return _original(*args, **kwargs)

        monkeypatch.setattr(pipeline, name, observed)
    result = pipeline.fMRISurface_pipeline(**public_case.arguments, parallel=False, cpu_threads=8)
    for options in (*seen.values(), public_case.calls[0]):
        assert options["parallel"] is False and options["cpu_threads"] == 8
    metadata = json.loads(sidecar(result.dtseries).read_text())
    assert metadata["FNIT"]["HemisphereExecution"] == {
        "Parallel": False, "CPUThreads": 8, "CPUThreadsPerHemisphere": {"L": 8, "R": 8},
    }
