"""无pickle存储容量与逻辑输出比较，不使用真实影像替代benchmark。"""
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    path = Path(__file__).parents[2] / "tools/benchmark_recon_topology_candidate_state.py"
    spec = importlib.util.spec_from_file_location("candidate_state_benchmark",path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def input_state():
    return dict(faces=np.array([[0,1,2],[0,2,1]],np.int32),nfaces=1,
        original_vertices=np.zeros((3,3),np.float32),vertex_normals=np.zeros((3,3),np.float32),
        vertex_rip=np.zeros(3,np.bool_),
        vertex_topology=np.array([[1,2,1,1,2,2,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1]],np.int32),
        neighbor_storage=[np.array([1,2],np.int32),np.array([0],np.int32),np.array([0],np.int32)],
        incident_faces=[np.array([0],np.int32)]*3,
        incident_corners=[np.array([i],np.uint8) for i in range(3)],
        curvbak=np.arange(3,dtype=np.float32))


def test_npz_roundtrip_keeps_max_neighborhood_storage(tmp_path):
    tool=module(); state=input_state(); path=tmp_path/'state.npz'
    tool.dump_state_checkpoint(path,state); loaded=tool.load_state_checkpoint(path)
    assert loaded['neighbor_storage'][0].tolist()==[1,2]  # vtotal is only one.
    assert all(v['byte_equal'] for v in tool.compare_states(loaded,state).values())


def test_inactive_face_capacity_is_diagnostic_not_active_mesh(tmp_path):
    tool=module(); state=input_state(); reference=input_state()
    reference['faces']=reference['faces'][:1]
    comparison=tool.compare_states(state,reference)
    assert comparison['faces']['byte_equal']
    state['faces'][0]=[0,2,1]
    assert not tool.compare_states(state,reference)['faces']['byte_equal']


def test_invalid_offsets_and_unmatched_face_corner_storage_fail(tmp_path):
    tool=module(); state=input_state()
    state['incident_corners'][0]=np.array([0,0],np.uint8)
    with pytest.raises(ValueError,match='lengths'): tool.dump_state_checkpoint(tmp_path/'state.npz',state)
    with pytest.raises(ValueError,match='offsets'):
        tool._rows(np.array([0,1,3]),np.array([1,2]),2,'neighbors')
