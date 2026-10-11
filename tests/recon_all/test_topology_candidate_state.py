"""DVS字段/别名/容量及恢复失败事务；不代替真实候选回归。"""
from dataclasses import replace
import copy
import numpy as np
import pytest

from fnit.recon_all.topology_candidate_state import record_candidate_state, restore_candidate_state


def state():
    return dict(
        faces=np.array([[0, 1, 2], [0, 2, 3], [0, 1, 3]], np.int32), nfaces=2,
        original_vertices=np.arange(12, dtype=np.float32).reshape(4, 3),
        vertex_normals=np.tile(np.array([0, 0, 1], np.float32), (4, 1)),
        vertex_rip=np.zeros(4, np.bool_),
        vertex_topology=np.array([[1, 2, 2, 2, 3, 3, 2], [1, 1, 2, 2, 2, 2, 1],
                                  [1, 1, 3, 3, 3, 3, 2], [1, 1, 2, 2, 2, 2, 1]], np.int32),
        neighbor_storage=[[1, 2, 3, 999], [0, 2], [0, 1, 3], [0, 2]],
        incident_faces=[[0, 1, 999], [0], [0, 1], [1]],
        incident_corners=[[0, 0, 999], [1], [2, 1], [2]],
        curvbak=np.zeros(4, np.float32), rp_statistics={"fitness": []})


def record(s, **kwargs):
    params = dict(surface_state=s, defect_vertices=[0], convex_hull_vertices=[2],
                  border_vertices=[1], vertex_translation=[0, 1, 2, 3])
    params.update(kwargs)
    return record_candidate_state(**params)


def mutate(s):
    m = copy.deepcopy(s)
    m["original_vertices"] += 10
    m["vertex_normals"] += 2
    m["vertex_topology"][:, :6] = [1, 1, 0, 0, 0, 0]
    m["vertex_topology"][:, 6] = 0
    m["neighbor_storage"] = [[3, 1], [2], [0, 1], [2]]
    m["incident_faces"] = [[], [], [], []]
    m["incident_corners"] = [[], [], [], []]
    m["nfaces"] = 3
    m["curvbak"][:] = 17
    m["rp_statistics"]["fitness"].append(-91.)
    return m


def test_full_restore_only_saved_fields_and_preserves_statistics():
    baseline = state(); snapshot = record(baseline); candidate = mutate(baseline)
    candidate["faces"][0] = [0, 2, 1]  # DVS does not back up actual face contents.
    result, report = restore_candidate_state(surface_state=candidate, snapshot=snapshot)
    for key in ("original_vertices", "vertex_normals", "vertex_topology"):
        np.testing.assert_array_equal(result[key][[0, 2]], baseline[key][[0, 2]])
        np.testing.assert_array_equal(result[key][[1, 3]], candidate[key][[1, 3]])
    assert result["nfaces"] == 2 and candidate["nfaces"] == 3
    assert result["faces"] is candidate["faces"]
    assert result["curvbak"] is candidate["curvbak"]
    assert result["rp_statistics"] is candidate["rp_statistics"]
    assert result["neighbor_storage"][0].tolist() == [1, 2, 3]
    assert report["vsize_exceeds_vtotal"] == 1
    assert report["restored_records"] == 2
    assert not np.shares_memory(result["original_vertices"], candidate["original_vertices"])


def test_face_only_restore_does_not_restore_geometry_or_neighbor_counts():
    baseline = state(); snapshot = record(baseline); candidate = mutate(baseline)
    result, report = restore_candidate_state(surface_state=candidate, snapshot=snapshot, mode="faces")
    assert result["original_vertices"] is candidate["original_vertices"]
    assert result["neighbor_storage"] is candidate["neighbor_storage"]
    np.testing.assert_array_equal(result["vertex_topology"][:, :6], candidate["vertex_topology"][:, :6])
    assert result["vertex_topology"][0, 6] == 2
    assert result["incident_faces"][2].tolist() == [0, 1]
    assert result["incident_corners"][2].tolist() == [2, 1]
    assert result["nfaces"] == 2 and report["mode"] == "faces"


def test_snapshot_has_no_aliases_and_restores_repeatably():
    baseline = state(); snapshot = record(baseline)
    baseline["original_vertices"][0] = 90
    baseline["neighbor_storage"][0][0] = 3
    assert snapshot.original_vertices[0].tolist() == [0, 1, 2]
    assert snapshot.neighbor_storage[0].tolist() == [1, 2, 3]
    with pytest.raises(ValueError): snapshot.original_vertices[0, 0] = 1
    candidate = mutate(baseline)
    first, _ = restore_candidate_state(surface_state=candidate, snapshot=snapshot)
    first["original_vertices"][0] = -90
    second, _ = restore_candidate_state(surface_state=candidate, snapshot=snapshot)
    assert second["original_vertices"][0].tolist() == [0, 1, 2]
    assert candidate["original_vertices"][0].tolist() == [100, 100, 100]


def test_ripped_and_negative_mapping_are_skipped_even_if_current_rip_changed():
    baseline = state(); baseline["vertex_rip"][0] = True
    snapshot = record(baseline, defect_vertices=[0, 1], convex_hull_vertices=[2],
                      vertex_translation=[0, -3, 2, 3])
    assert snapshot.vertex_numbers.tolist() == [0, -3, 2]
    assert snapshot.restore_vertex.tolist() == [False, False, True]
    candidate = mutate(baseline); candidate["vertex_rip"][:] = False
    result, report = restore_candidate_state(surface_state=candidate, snapshot=snapshot)
    np.testing.assert_array_equal(result["original_vertices"][0], candidate["original_vertices"][0])
    np.testing.assert_array_equal(result["original_vertices"][2], baseline["original_vertices"][2])
    assert result["vertex_rip"] is candidate["vertex_rip"]
    assert report["skipped_records"] == 2


def test_compile_profile_border_is_explicit_and_keeps_original_order():
    snapshot = record(state(), defect_vertices=[2, 0], profile="border")
    assert snapshot.vertex_numbers.tolist() == [2, 0, 1]


def test_duplicate_source_records_are_not_sorted_or_deduplicated():
    snapshot = record(state(), defect_vertices=[2, 0, 2], convex_hull_vertices=[])
    assert snapshot.vertex_numbers.tolist() == [2, 0, 2]
    result, report = restore_candidate_state(surface_state=mutate(state()), snapshot=snapshot)
    assert report["restored_records"] == 3
    np.testing.assert_array_equal(result["original_vertices"][2], state()["original_vertices"][2])


def test_zero_valid_lengths_retain_current_storage_buffers():
    baseline = state(); baseline["vertex_topology"][0] = [0, 0, 0, 0, 0, 0, 0]
    baseline["neighbor_storage"][0] = []; baseline["incident_faces"][0] = []
    baseline["incident_corners"][0] = []
    snapshot = record(baseline, convex_hull_vertices=[])
    candidate = mutate(baseline)
    result, _ = restore_candidate_state(surface_state=candidate, snapshot=snapshot)
    assert result["neighbor_storage"][0] is candidate["neighbor_storage"][0]
    assert result["incident_faces"][0] is candidate["incident_faces"][0]
    assert result["vertex_topology"][0].tolist() == [0, 0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("field,value", [
    ("neighbor_storage", [[1, 2], [0, 2], [0, 1, 3], [0, 2]]),
    ("incident_faces", [[0], [0], [0, 1], [1]]),
    ("incident_corners", [[0, 3], [1], [2, 1], [2]]),
])
def test_unsafe_saved_storage_fails_without_mutation(field, value):
    baseline = state(); baseline[field] = value
    before = copy.deepcopy(baseline)
    with pytest.raises(ValueError): record(baseline)
    np.testing.assert_array_equal(baseline["original_vertices"], before["original_vertices"])
    assert baseline[field] == before[field]


def test_vtotal_greater_than_current_nsize_count_is_explicitly_nonrestorable():
    baseline = state(); baseline["vertex_topology"][0, 2] = 3
    with pytest.raises(ValueError, match="vtotal"): record(baseline)


def test_unused_higher_hop_counts_are_restored_without_reading_them():
    baseline=state();baseline['vertex_topology'][2,4:6]=[-1,-7]
    snapshot=record(baseline,defect_vertices=[2],convex_hull_vertices=[])
    candidate=mutate(baseline)
    result,_=restore_candidate_state(surface_state=candidate,snapshot=snapshot)
    assert result['vertex_topology'][2,4:6].tolist()==[-1,-7]
    assert result['neighbor_storage'][2].tolist()==[0,1,3]


@pytest.mark.parametrize("change", ["short_faces", "foreign_vertex_count", "damaged_snapshot", "invalid_mode"])
def test_restore_failure_publishes_no_partial_state(change):
    baseline = state(); snapshot = record(baseline); candidate = mutate(baseline)
    mode = "full"
    if change == "short_faces": candidate["nfaces"] = 1
    elif change == "foreign_vertex_count": snapshot = replace(snapshot, vertex_count=5)
    elif change == "damaged_snapshot": snapshot = replace(snapshot, neighbor_storage=(np.array([9]),) + snapshot.neighbor_storage[1:])
    else: mode = "all"
    before = copy.deepcopy(candidate)
    with pytest.raises(ValueError): restore_candidate_state(surface_state=candidate, snapshot=snapshot, mode=mode)
    for key in ("original_vertices", "vertex_normals", "vertex_topology", "faces", "curvbak"):
        np.testing.assert_array_equal(candidate[key], before[key])
    assert candidate["neighbor_storage"] == before["neighbor_storage"]
    assert candidate["nfaces"] == before["nfaces"]


@pytest.mark.parametrize("kwargs", [dict(profile="unknown"), dict(defect_vertices=[9]),
                                    dict(vertex_translation=[9, 1, 2, 3]), dict(defect_vertices=[0.])])
def test_invalid_selection_rejected(kwargs):
    with pytest.raises(ValueError): record(state(), **kwargs)
