"""四轮优化器达到缩步上限时应恢复坐标并完成，不把控制流测试当benchmark。"""
from pathlib import Path
import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np
import pytest
from fnit.recon_all import place_pial_python as stage


@pytest.mark.parametrize("profile", [False, True])
def test_final_rejected_trial_restores_coordinates_and_completes_four_passes(tmp_path, monkeypatch, profile):
    for folder in ('surf', 'mri', 'label'):
        (tmp_path/folder).mkdir()
    xyz = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float32)
    faces = np.array([[0,1,2]], dtype=np.int32)
    fs.write_geometry(str(tmp_path/'surf/lh.white'), xyz, faces, volume_info={
        'head':np.array([20]), 'valid':'1', 'filename':'unit-test.mgz',
        'volume':np.array([4,4,4]), 'voxelsize':np.ones(3),
        'xras':np.array([1.,0,0]), 'yras':np.array([0.,1,0]),
        'zras':np.array([0.,0,1]), 'cras':np.zeros(3)})
    for name in ('cortex', 'cortex+hipamyg'):
        (tmp_path/f'label/lh.{name}.label').write_text('#!ascii label\n3\n0 0 0 0 0\n1 1 0 0 0\n2 0 1 0 0\n')
    for name in ('brain.finalsurfs', 'wm', 'aseg.presurf'):
        nib.save(nib.MGHImage(np.ones((4,4,4), dtype=np.float32), np.eye(4)), str(tmp_path/f'mri/{name}.mgz'))
    (tmp_path/'surf/autodet.gw.stats.lh.dat').write_text('MID_GRAY 50\n'+''.join(
        f'pial_{name} 50\n' for name in ('inside_hi','border_hi','border_low','outside_low','outside_hi')))
    monkeypatch.setattr(stage, 'compute_border_values_first_pass', lambda *a,**k:
        (np.ones(3), None, None, None, np.ones(3,dtype=np.bool_), np.ones(3)))
    monkeypatch.setattr(stage, 'average_marked_values', lambda values,*a: values)
    monkeypatch.setattr(stage, 'intensity_error', lambda *a,**k: (1.,1.,None))
    monkeypatch.setattr(stage, 'intensity_gradient', lambda *a,**k: np.ones((3,3),dtype=np.float32))
    monkeypatch.setattr(stage, 'surface_repulsion_gradient', lambda *a,**k: np.zeros((3,3),dtype=np.float32))
    monkeypatch.setattr(stage, 'average_signed_gradients', lambda values,*a,**k: values)
    monkeypatch.setattr(stage, 'spring_gradient', lambda *a,**k: np.zeros((3,3),dtype=np.float32))
    monkeypatch.setattr(stage, 'quadratic_curvature', lambda *a,**k: np.zeros(3,dtype=np.float32))
    monkeypatch.setattr(stage, 'unconstrained_step_with_offsets', lambda current,*a,**k: (current+.1,np.ones_like(current)*.1))
    monkeypatch.setattr(stage, 'asynchronous_first_step', lambda current,faces,proposal,*a,**k: (proposal,None))
    # 所有试步RMS均上升；第3次拒绝达到终止规则，应还原本步起点。
    monkeypatch.setattr(stage, 'pial_step_decision', lambda ls,lr,s,r,dt,red:
        (dt*.5,red+1,True,True,red+1>2))
    monkeypatch.setattr(stage, 'repair_intersections', lambda vertices,*a: (vertices,{'intersecting_faces_after':0}))
    trace=[]
    report=stage.place_pial_t1(subject=tmp_path, hemisphere='lh', output=tmp_path/'surf/lh.pial.T1',
                              max_steps=4, profile=profile, trace_callback=lambda *args: trace.append(args))
    assert report['pass_ends']==[1,2,3,4]
    actual,actual_faces=fs.read_geometry(report['output'])
    np.testing.assert_array_equal(actual,xyz)
    np.testing.assert_array_equal(actual_faces,faces)
    assert [row[1] for row in trace]==[0,1,2,3]
    assert all(len(row[3]['trials'])==3 for row in trace)
    assert all(row[3]['trials'][-1]['rejected'] and row[3]['trials'][-1]['stop'] for row in trace)
    assert report['profile'] is profile
    if profile:
        assert all(seconds >= 0 for seconds in report['stage_seconds'].values())
        assert sum(report['stage_seconds'].values()) == pytest.approx(report['seconds'])
        assert report['profile_cuda_target'] is None
    else:
        assert 'stage_seconds' not in report
