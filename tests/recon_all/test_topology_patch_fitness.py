"""评分阶段接线契约：同版输入、自产距离、原调用顺序与状态保护。"""
from copy import deepcopy

import numpy as np
import pytest

from fnit.recon_all import topology_patch_fitness as stage


def arguments():
    xyz = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0]], np.float32)
    faces = np.array([[0, 1, 2]], np.int32)
    normals = np.tile(np.array([0, 0, 1], np.float32), (3, 1))
    face_normals = normals[:1].copy()
    patch = np.arange(3, dtype=np.int32)
    areas = np.zeros(3, np.float32)
    origin = np.array([-1, -1, -2], np.float32)
    scale = np.ones(3, np.float32)
    shape = (7, 7, 7)
    hist = dict(bins=np.array([-1, 0, 1], np.float32),
                counts=np.array([.25, .5, .25], np.float32), bin_size=np.float32(1))
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = 3
    mri = dict(original_vertices=xyz, faces=faces, vertex_normals=normals,
        face_normals=face_normals, vertex_areas_mm2=areas, face_areas_mm2=np.zeros(1, np.float32),
        patch_vertices=patch, patch_faces=np.array([0], np.int32),
        volume=np.full((9, 9, 9), 100, np.uint8), ras_to_voxel=matrix,
        gray_white_pdf=np.full((256, 256, 1), np.float32(1 / 65536), np.float32))
    normal = dict(original_vertices=xyz.copy(), vertex_normals=normals.copy(),
        vertex_areas_mm2=areas.copy(), patch_vertices=patch.copy(),
        neighbor_offsets=np.array([0, 2, 4, 6], np.int32),
        ordered_neighbors=np.array([1, 2, 0, 2, 0, 1], np.int32), normal_dot_histogram=hist)
    curvature = dict(principal_curvature_k1_mm_inv=np.zeros(3, np.float32),
        principal_curvature_k2_mm_inv=np.zeros(3, np.float32), vertex_areas_mm2=areas.copy(),
        patch_vertices=patch.copy(), k1_histogram=deepcopy(hist), k2_histogram=deepcopy(hist),
        joint_curvature_pdf=np.full((3, 3), np.float32(1 / 9), np.float32),
        joint_origin_mm_inv=np.array([-1, -1], np.float32),
        joint_spacing_mm_inv=np.ones(2, np.float32))
    distance = dict(original_vertices=xyz.copy(), faces=faces.copy(), face_normals=face_normals.copy(),
        incident_offsets=np.array([0, 1, 2, 3], np.int32), incident_faces=np.array([0, 0, 0], np.int32),
        incident_corners=np.array([0, 1, 2], np.int32), volume_shape=shape,
        volume_origin_mm=origin.copy(), volume_index_scale_per_mm=scale.copy())
    volume = dict(defect_intensity_volume=np.full(shape, 100, np.uint8),
        white_log_probability=np.full(shape, -.25, np.float32),
        gray_log_probability=np.full(shape, -.5, np.float32),
        initial_white_mean_log_probability=np.float32(-.25),
        initial_gray_mean_log_probability=np.float32(-.5),
        initial_white_mean_intensity=np.float32(100), initial_gray_mean_intensity=np.float32(100),
        volume_origin_mm=origin.copy(), volume_index_scale_per_mm=scale.copy(),
        vertex_sampling_positions_mm=xyz.copy(), initial_vertex_curvbak=np.ones(3, np.float32),
        translated_defect_vertices=patch.copy(), discarded_defect_vertices=np.zeros(3, bool),
        distance_unset_sentinel_mm=np.float32(1000))
    return dict(mri_inputs=mri, normal_inputs=normal, curvature_inputs=curvature,
                distance_inputs=distance, volume_inputs=volume, valid_faces=True)


def mock_components(monkeypatch):
    calls = []
    generated = np.full((7, 7, 7), .25, np.float32)

    def mri(**kwargs):
        calls.append("mri")
        return dict(face_ll=-1., vertex_ll=-2., area_weighted_face_ll=np.nan)

    def distance(**kwargs):
        calls.append("signed_distance")
        return dict(distance_mm=generated)

    def volume(**kwargs):
        calls.append("unmri")
        assert kwargs["signed_distance_mm"] is generated
        feedback = kwargs["initial_vertex_curvbak"].copy() * np.float32(.25)
        return dict(unmri_ll=-3., vertex_curvbak=feedback)

    def curvature(**kwargs):
        calls.append("curvature")
        return dict(qcurv_ll=-4., qcll=np.nan)

    def normal(**kwargs):
        calls.append("normal")
        return dict(curv_ll=-5., cll=np.nan)

    for name, function in [("patch_mri_likelihood", mri), ("defect_signed_distance", distance),
            ("defect_volume_likelihood", volume), ("patch_curvature_likelihood", curvature),
            ("patch_normal_dot_likelihood", normal)]:
        monkeypatch.setattr(stage, name, function)
    return calls, generated


def test_source_order_own_distance_and_unmodified_feedback(monkeypatch):
    args = arguments()
    original = args["volume_inputs"]["initial_vertex_curvbak"].copy()
    calls, generated = mock_components(monkeypatch)
    result = stage.evaluate_prepared_patch_fitness(**args)
    assert calls == result["executed_components"] == ["mri", "signed_distance", "unmri", "curvature", "normal"]
    assert result["fitness"] == -42.
    assert result["distance"]["distance_mm"] is generated
    np.testing.assert_array_equal(original, args["volume_inputs"]["initial_vertex_curvbak"])
    np.testing.assert_array_equal(result["vertex_curvbak"], original * .25)
    assert np.isnan(result["components"]["normal"]["cll"])


@pytest.mark.parametrize("key", ["reference_scores", "reference_results", "reference_curvbak", "reference_anything", "signed_distance_mm"])
def test_reference_output_or_caller_distance_rejected_before_computation(monkeypatch, key):
    args = arguments(); args["volume_inputs"][key] = np.zeros(1)
    calls, _ = mock_components(monkeypatch)
    with pytest.raises(ValueError, match="not stage inputs"):
        stage.evaluate_prepared_patch_fitness(**args)
    assert calls == []


@pytest.mark.parametrize("mapping,field", [
    ("normal_inputs", "original_vertices"), ("normal_inputs", "vertex_normals"),
    ("normal_inputs", "vertex_areas_mm2"), ("normal_inputs", "patch_vertices"),
    ("curvature_inputs", "vertex_areas_mm2"), ("curvature_inputs", "patch_vertices"),
    ("distance_inputs", "faces"), ("distance_inputs", "face_normals"),
    ("distance_inputs", "volume_origin_mm"), ("distance_inputs", "volume_index_scale_per_mm")])
def test_candidate_versions_must_match_before_any_component(monkeypatch, mapping, field):
    args = arguments()
    args[mapping][field] = args[mapping][field].copy()
    args[mapping][field].flat[0] += 1
    calls, _ = mock_components(monkeypatch)
    with pytest.raises(ValueError, match="same candidate version"):
        stage.evaluate_prepared_patch_fitness(**args)
    assert calls == []


@pytest.mark.parametrize("shape", [(5, 7, 7), (7, 5, 7), (7, 7, 5)])
def test_source_small_volume_skips_only_unmri(monkeypatch, shape):
    args = arguments()
    args["distance_inputs"]["volume_shape"] = shape
    for field in ["defect_intensity_volume", "white_log_probability", "gray_log_probability"]:
        args["volume_inputs"][field] = np.zeros(shape, args["volume_inputs"][field].dtype)
    calls, _ = mock_components(monkeypatch)
    result = stage.evaluate_prepared_patch_fitness(**args)
    assert calls == ["mri", "curvature", "normal"]
    assert result["fitness"] == -12.
    assert result["distance"] is None and result["vertex_curvbak"] is None
    assert result["composition"]["tp_fields"]["unmri_ll"] == 0.


def test_weights_keep_source_epsilon_and_negative_values(monkeypatch):
    args = arguments()
    calls, _ = mock_components(monkeypatch)
    epsilon = float(np.finfo(np.float32).eps)
    result = stage.evaluate_prepared_patch_fitness(**args, mri_weight=epsilon / 2,
        volume_unmri_weight=0, normal_dot_weight=-2, quadratic_curvature_weight=epsilon)
    assert calls == ["curvature", "normal"]
    assert result["fitness"] == 10 - 4 * epsilon
    assert result["composition"]["tp_fields"]["face_ll"] == 0.


def test_face_penalty_is_source_scoring_and_not_mesh_qc(monkeypatch):
    args = arguments(); args["valid_faces"] = False
    mock_components(monkeypatch)
    result = stage.evaluate_prepared_patch_fitness(**args)
    assert result["fitness"] == -10_000_042.
    assert len(result["executed_components"]) == 5


@pytest.mark.parametrize("field", ["initial_vertex_curvbak"])
def test_full_vertex_state_order_required(monkeypatch, field):
    args = arguments(); args["volume_inputs"][field] = np.ones(2, np.float32)
    calls, _ = mock_components(monkeypatch)
    with pytest.raises(ValueError, match="full same-order"):
        stage.evaluate_prepared_patch_fitness(**args)
    assert calls == []


def test_missing_field_and_wrong_mapping_fail_explicitly():
    args = arguments(); del args["distance_inputs"]["faces"]
    with pytest.raises(KeyError, match="faces"):
        stage.evaluate_prepared_patch_fitness(**args)
    args = arguments(); args["normal_inputs"] = None
    with pytest.raises(ValueError, match="mappings"):
        stage.evaluate_prepared_patch_fitness(**args)


def test_complete_api_calls_existing_kernels_and_preserves_nan_diagnostics():
    args = arguments()
    result = stage.evaluate_prepared_patch_fitness(**args)
    assert np.isfinite(result["fitness"])
    assert result["distance"]["distance_mm"].dtype == np.float32
    assert result["vertex_curvbak"].dtype == np.float32
    assert result["executed_components"] == ["mri", "signed_distance", "unmri", "curvature", "normal"]
    for component, field in [("normal", "cll"), ("curvature", "qcll")]:
        assert not result["components"][component][field + "_defined"]
        assert np.isnan(result["components"][component][field])
    np.testing.assert_array_equal(args["volume_inputs"]["initial_vertex_curvbak"], np.ones(3, np.float32))


def test_explicit_distance_hook_keeps_order_and_skip(monkeypatch):
    args=arguments();calls,generated=mock_components(monkeypatch)
    def explicit(**kwargs):
        calls.append('signed_distance')
        return dict(distance_mm=generated,actual_backend='gpu_ordered_voxels')
    result=stage.evaluate_prepared_patch_fitness(**args,distance_evaluator=explicit)
    assert calls==['mri','signed_distance','unmri','curvature','normal']
    assert result['fitness']==-42.
    assert result['distance']['actual_backend']=='gpu_ordered_voxels'
    calls.clear()
    result=stage.evaluate_prepared_patch_fitness(**args,distance_evaluator=explicit,volume_unmri_weight=0.)
    assert calls==['mri','curvature','normal']
    assert result['distance'] is None


def test_distance_hook_failure_propagates(monkeypatch):
    args=arguments();calls,_=mock_components(monkeypatch)
    def failing(**kwargs):raise RuntimeError('explicit CUDA failure')
    with pytest.raises(RuntimeError,match='explicit CUDA failure'):
        stage.evaluate_prepared_patch_fitness(**args,distance_evaluator=failing)
    assert calls==['mri']
