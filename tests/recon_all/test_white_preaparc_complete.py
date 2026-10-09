"""完整 preaparc 的轮间状态与失败契约；模拟输入不充当真实 benchmark。"""

import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np
import pytest

from fnit.recon_all import place_white_preaparc_python as stage


@pytest.fixture
def white_inputs(tmp_path, monkeypatch):
    for folder in ("surf", "mri"):
        (tmp_path / folder).mkdir()
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    fs.write_geometry(str(tmp_path / "surf/lh.orig"), xyz, faces, volume_info={
        "head": np.array([20]), "valid": "1", "filename": "contract.mgz",
        "volume": np.array([4, 4, 4]), "voxelsize": np.ones(3),
        "xras": np.array([1., 0, 0]), "yras": np.array([0., 1, 0]),
        "zras": np.array([0., 0, 1]), "cras": np.zeros(3),
    })
    for name in ("brain.finalsurfs", "wm", "aseg.presurf"):
        nib.save(nib.MGHImage(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)),
                 str(tmp_path / f"mri/{name}.mgz"))
    (tmp_path / "surf/autodet.gw.stats.lh.dat").write_text(
        "MID_GRAY 50\n" + "".join(f"white_{name} 50\n" for name in
        ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")))
    observed = {"rip": [], "border": [], "average": [], "cleanup": [], "context": [], "gpu_average": []}
    monkeypatch.setattr(stage, "average_vertex_positions", lambda vertices, *a: vertices.copy())
    monkeypatch.setattr(stage, "prepare_placement_volume", lambda image, *a, **k: (image, None))

    def rip(vertices, *args, ripped=None, values=None, **kwargs):
        flags = np.zeros(3, dtype=np.int32) if ripped is None else ripped.copy()
        # 第一轮两次初始化；随后每轮扩展冻结集合，检验 GPU 上下文须更新。
        if len(observed["rip"]) >= 2:
            flags[len(observed["rip"]) - 2] = 1
        observed["rip"].append(flags.copy())
        return flags, np.zeros(3, dtype=np.float32) if values is None else values.copy()

    def border(volume, seg, current, normals, original, ripped, values, *args, **kwargs):
        observed["border"].append((kwargs["sigma"], kwargs["surface"], original.copy()))
        target = np.full(3, len(observed["border"]), dtype=np.float32)
        return target, None, None, None, np.ones(3, dtype=np.bool_), np.ones(3)

    def average(values, faces, ripped, iterations, **kwargs):
        observed["average"].append(iterations)
        return values

    def cleanup(vertices, faces, ripped):
        observed["cleanup"].append(ripped.copy())
        return vertices.copy(), {"intersecting_faces_after": 0}

    monkeypatch.setattr(stage, "rip_white_preaparc_pass", rip)
    monkeypatch.setattr(stage, "compute_border_values_first_pass", border)
    monkeypatch.setattr(stage, "average_marked_values", lambda values, *a: values)
    monkeypatch.setattr(stage, "intensity_error", lambda volume, current, values, *a: (float(values[0]), 1., None))
    monkeypatch.setattr(stage, "intensity_gradient", lambda *a, **k: np.ones((3, 3), dtype=np.float32))
    monkeypatch.setattr(stage, "average_signed_gradients", average)
    monkeypatch.setattr(stage, "spring_gradient", lambda *a, **k: np.zeros((3, 3), dtype=np.float32))
    monkeypatch.setattr(stage, "quadratic_curvature", lambda *a, **k: np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(stage, "self_repulsion_gradient", lambda *a, **k: np.zeros((3, 3), dtype=np.float32))
    monkeypatch.setattr(stage, "self_repulsion_energy", lambda *a, **k: 0.)
    monkeypatch.setattr(stage, "mean_vertex_spacing", lambda *a: 1.)
    monkeypatch.setattr(stage, "vertex_buckets_current", lambda *a, **k: (None, None))
    monkeypatch.setattr(stage, "tangential_spring_energy", lambda *a, **k: 0.)
    monkeypatch.setattr(stage, "surface_total_area", lambda *a: 1.)
    monkeypatch.setattr(stage, "unconstrained_step_with_offsets",
                        lambda current, *a, **k: (current + np.float32(.1), np.ones_like(current) * .1))
    monkeypatch.setattr(stage, "asynchronous_first_step", lambda current, faces, proposal, *a, **k: (proposal, None))
    monkeypatch.setattr(stage, "repair_intersections", cleanup)
    return tmp_path, xyz, faces, observed


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_terminal_rejection_restores_coordinates_and_refreshes_pass_state(white_inputs, monkeypatch, backend):
    subject, xyz, faces, observed = white_inputs
    if backend == "torch":
        from fnit.recon_all import place_surface_regularization_torch as regularization

        class Context:
            def __init__(self, *, ripped, **kwargs):
                observed["context"].append(ripped.copy())

            def regularize(self, *, gradient, iterations, **kwargs):
                observed["gpu_average"].append(iterations)
                return gradient

        monkeypatch.setattr(regularization, "PlacementRegularizationTorch", Context)
    monkeypatch.setattr(stage, "pial_step_decision", lambda ls, lr, s, r, dt, red:
                        (dt * .5, red + 1, True, True, red + 1 > 2))
    trace = []
    volume_output = subject / "diagnostic/placement.mgz"
    report = stage.place_white_preaparc(
        subject_dir=subject, hemi="lh", output=subject / "diagnostic/lh.white.preaparc",
        max_steps=4, output_volume=volume_output, regularization_backend=backend,
        device="cpu" if backend == "torch" else None,
        trace_callback=lambda *args: trace.append(args),
    )
    assert report["complete_four_passes"] is True
    assert report["pass_ends"] == [1, 2, 3, 4]
    assert [item[0] for item in observed["border"]] == [2., 1., .5, .25]
    assert all(item[1] == "white" for item in observed["border"])
    assert len(observed["rip"]) == 5
    np.testing.assert_array_equal([row["initial_sse"] for row in report["passes"]],
                                  np.float32(.2) * np.arange(1, 5))
    assert all(row["trials"] == 3 and row["trial_trace"][-1]["rejected"] for row in report["per_step"])
    assert all(row["trial_trace"][0]["dt_used"] == .5 for row in report["per_step"])
    assert [row[1] for row in trace] == [0, 1, 2, 3]
    trace[0][3]["trial_trace"].clear()
    assert len(report["per_step"][0]["trial_trace"]) == 3
    actual, actual_faces, metadata = fs.read_geometry(report["output"], read_metadata=True)
    np.testing.assert_array_equal(actual, xyz)
    np.testing.assert_array_equal(actual_faces, faces)
    assert metadata["filename"] == "contract.mgz"
    assert len(observed["cleanup"]) == 2
    assert not observed["cleanup"][0].any()
    np.testing.assert_array_equal(observed["cleanup"][-1], observed["rip"][-1])
    if backend == "torch":
        assert observed["gpu_average"] == [4, 2, 1, 0]
        assert len(observed["context"]) == 4
        for actual_mask, expected in zip(observed["context"], [observed["rip"][1], *observed["rip"][2:]]):
            np.testing.assert_array_equal(actual_mask, expected)
    else:
        assert observed["average"] == [4, 2, 1, 0]
    actual_volume = nib.load(str(volume_output))
    assert actual_volume.get_data_dtype() == np.dtype("uint8")
    np.testing.assert_array_equal(actual_volume.affine, nib.load(str(subject / "mri/brain.finalsurfs.mgz")).affine)
    assert sum(report["stage_seconds"].values()) == pytest.approx(report["seconds"])
    assert sum(report["prepare_components"].values()) == pytest.approx(report["stage_seconds"]["prepare"])


def test_each_pass_obeys_native_100_iteration_limit(white_inputs, monkeypatch):
    subject, xyz, faces, observed = white_inputs
    monkeypatch.setattr(stage, "pial_step_decision", lambda ls, lr, s, r, dt, red:
                        (dt, red, False, False, False))
    report = stage.place_white_preaparc(subject_dir=subject, hemi="lh",
                                       output=subject / "diagnostic/white", max_steps=400)
    assert report["pass_ends"] == [100, 200, 300, 400]
    assert all(row["reason"] == "iteration_limit" for row in report["passes"])
    assert observed["average"] == [4] * 100 + [2] * 100 + [1] * 100 + [0] * 100


def test_incomplete_four_passes_and_residual_intersections_do_not_write(white_inputs, monkeypatch):
    subject, xyz, faces, observed = white_inputs
    output = subject / "diagnostic/white"
    monkeypatch.setattr(stage, "pial_step_decision", lambda ls, lr, s, r, dt, red:
                        (dt * .5, red + 1, True, True, red + 1 > 2))
    with pytest.raises(RuntimeError, match="did not complete four passes"):
        stage.place_white_preaparc(subject_dir=subject, hemi="lh", output=output, max_steps=3)
    assert not output.exists()
    observed["rip"].clear()
    monkeypatch.setattr(stage, "repair_intersections", lambda vertices, *a:
                        (vertices, {"intersecting_faces_after": 2}))
    with pytest.raises(RuntimeError, match="white cleanup") as error:
        stage.place_white_preaparc(subject_dir=subject, hemi="lh", output=output)
    assert error.value.intersection_cleanup["initial"]["intersecting_faces_after"] == 2
    assert error.value.intersection_cleanup["final"]["intersecting_faces_after"] == 2
    assert error.value.partial_stage["steps"] == 4
    assert len(error.value.partial_stage["passes"]) == 4
    assert error.value.intersection_coordinates.shape == (3, 3)
    assert error.value.intersection_faces.shape == (1, 3)
    assert not output.exists()


@pytest.mark.parametrize("backend", ["source_numba", "source_torch"])
def test_same_explicit_marker_is_used_for_initial_and_final_cleanup(white_inputs, monkeypatch, backend):
    subject, xyz, faces, observed = white_inputs
    calls = []

    def cleanup(vertices, faces, ripped, **kwargs):
        calls.append(kwargs)
        return vertices.copy(), {"intersecting_faces_after": 0}

    monkeypatch.setattr(stage, "repair_intersections", cleanup)
    monkeypatch.setattr(stage, "pial_step_decision", lambda ls, lr, s, r, dt, red:
                        (dt * .5, red + 1, True, True, red + 1 > 2))
    result = stage.place_white_preaparc(subject_dir=subject, hemi="lh", output=subject / "diagnostic/white",
        cleanup_marking_backend=backend, device="cpu" if backend == "source_torch" else None)
    assert len(calls) == 2
    assert all(call["marking_backend"] == backend for call in calls)
    assert result["cleanup_marking_backend"] == backend


def test_initial_residual_continues_placement_but_final_mesh_must_be_clear(white_inputs, monkeypatch):
    subject, xyz, faces, observed = white_inputs
    output = subject / "diagnostic/white"
    cleanups = iter(({"intersecting_faces_after": 2}, {"intersecting_faces_after": 0}))
    monkeypatch.setattr(stage, "repair_intersections", lambda vertices, *a: (vertices, next(cleanups)))
    monkeypatch.setattr(stage, "pial_step_decision", lambda ls, lr, s, r, dt, red:
                        (dt * .5, red + 1, True, True, red + 1 > 2))
    result = stage.place_white_preaparc(subject_dir=subject, hemi="lh", output=output)
    assert result["complete_four_passes"]
    assert len(result["passes"]) == 4
    assert result["initial_cleanup"]["intersecting_faces_after"] == 2
    assert result["cleanup"]["intersecting_faces_after"] == 0
    assert output.exists()


def test_prefix_preserves_original_rejected_stop_failure(white_inputs, monkeypatch):
    subject, xyz, faces, observed = white_inputs
    monkeypatch.setattr(stage, "pial_step_decision", lambda ls, lr, s, r, dt, red:
                        (dt * .5, red + 1, True, True, red + 1 > 2))
    with pytest.raises(RuntimeError, match="white prefix rejected"):
        stage.place_white_preaparc_prefix(subject_dir=subject, hemi="lh",
                                          output=subject / "diagnostic/prefix")
    assert not observed["cleanup"]


def test_outputs_cannot_overwrite_inputs_or_each_other(tmp_path):
    with pytest.raises(ValueError, match="overwrite"):
        stage.place_white_preaparc(subject_dir=tmp_path, hemi="lh", output=tmp_path / "surf/lh.orig")
    with pytest.raises(ValueError, match="overwrite"):
        stage.place_white_preaparc(subject_dir=tmp_path, hemi="lh", output=tmp_path / "new",
                                   output_volume=tmp_path / "mri/brain.finalsurfs.mgz")
    with pytest.raises(ValueError, match="overwrite"):
        stage.place_white_preaparc(subject_dir=tmp_path, hemi="lh", output=tmp_path / "new",
                                   output_volume=tmp_path / "new")


def test_different_mri_grid_fails_before_optimization(white_inputs):
    subject, xyz, faces, observed = white_inputs
    nib.save(nib.MGHImage(np.ones((3, 4, 4), dtype=np.uint8), np.eye(4)), str(subject / "mri/wm.mgz"))
    with pytest.raises(ValueError, match="MRI grids differ"):
        stage.place_white_preaparc(subject_dir=subject, hemi="lh", output=subject / "diagnostic/white")
    assert not observed["border"]


@pytest.mark.parametrize("implementation", ["torch", "triton"])
def test_gpu_sampler_caches_mri_once_and_receives_fresh_white_pass_state(white_inputs, monkeypatch, implementation):
    subject, xyz, faces, observed = white_inputs
    from fnit.recon_all import place_surface_sampling as sampling
    created, calls = [], []

    class Sampler:
        def __init__(self, volume, affine, *, device, implementation):
            created.append((volume.copy(), affine.copy(), device, implementation))

        def gradient(self, current, normals, ripped, values, sigmas, sizes, *, weight, sigma_global):
            calls.append((sigma_global, values.copy(), ripped.copy(), weight))
            return np.ones_like(current)

    monkeypatch.setattr(sampling, "PlacementSampling", Sampler)
    monkeypatch.setattr(stage, "intensity_gradient", lambda *a, **k: pytest.fail("CPU intensity was called"))
    monkeypatch.setattr(stage, "pial_step_decision", lambda ls, lr, s, r, dt, red:
                        (dt * .5, red + 1, True, True, red + 1 > 2))
    report = stage.place_white_preaparc(subject_dir=subject, hemi="lh", output=subject / "diagnostic/white",
                                       max_steps=4, sampling_backend=implementation, device="cuda:0")
    assert len(created) == 1
    assert created[0][2:] == ("cuda:0", implementation)
    assert created[0][0].dtype == np.uint8
    assert [row[0] for row in calls] == [2., 1., .5, .25]
    for index, row in enumerate(calls):
        np.testing.assert_array_equal(row[1], np.full(3, index + 1))
        np.testing.assert_array_equal(row[2], observed["rip"][index + 1])
        assert row[3] == .2
    assert report["sampling_backend"] == implementation


def test_collision_profile_is_separate_from_trial_decisions(white_inputs, monkeypatch):
    subject, xyz, faces, observed = white_inputs
    monkeypatch.setattr(stage, "pial_step_decision", lambda ls, lr, s, r, dt, red:
                        (dt, red, False, False, True))

    def collision(current, faces, proposal, *args, candidate_diagnostics=None, **kwargs):
        candidate_diagnostics.update(candidate_pairs=23, candidate_build_seconds=.001,
                                     ordered_acceptance_seconds=.002)
        return proposal, None

    monkeypatch.setattr(stage, "asynchronous_first_step", collision)
    result = stage.place_white_preaparc(subject_dir=subject, hemi="lh",
        output=subject / "diagnostic/white", max_steps=4, collision_profile=True)
    assert len(result["collision_details"]) == 4
    assert all(row["candidate_pairs"] == 23 for row in result["collision_details"])
    assert all("candidate_build_seconds" not in row for row in result["per_step"])


def test_nondefault_grid_is_not_silently_ignored(white_inputs):
    subject, *_ = white_inputs
    with pytest.raises(ValueError, match="nondefault candidate grid"):
        stage.place_white_preaparc(subject_dir=subject, hemi="lh",
            output=subject / "diagnostic/white", candidate_grid_cells_per_axis=3)


def test_compiled_retained_requires_snapshot_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError,match="compiled retained MHT requires snapshot"):
        stage.place_white_preaparc(subject_dir=tmp_path,hemi="lh",output=tmp_path/"white",
                                  retained_mht_backend="compiled")


def test_complete_white_forwards_explicit_retained_policy(white_inputs,monkeypatch):
    subject,xyz,faces,observed=white_inputs
    observed_retained=[]

    def collision(current,faces,proposal,*args,**kwargs):
        observed_retained.append(kwargs["retained_mht_backend"])
        return proposal,None

    monkeypatch.setattr(stage,"asynchronous_first_step",collision)
    monkeypatch.setattr(stage,"pial_step_decision",lambda ls,lr,s,r,dt,red:
                        (dt*.5,red+1,True,True,red+1>2))
    result=stage.place_white_preaparc(subject_dir=subject,hemi="lh",output=subject/"white",
        candidate_backend="snapshot",retained_mht_backend="compiled")
    assert result["retained_mht_backend"]=="compiled"
    assert observed_retained==["compiled"]*12
