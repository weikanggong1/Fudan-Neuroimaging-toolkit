"""固定源码停止语义、末态诊断和显式零翻折门，不替代真实benchmark。"""
import numpy as np
import pytest
import torch

from fnit.recon_all.mris_register_overlap import (
    remove_overlap_sphere, SphereOverlapQualityError)
from fnit.recon_all.mris_register_overlap_marked import remove_overlap_sphere_marked
from fnit.recon_all.sphere_standard_finish import finish_standard_sphere


def mesh():
    xyz = torch.tensor([[100.,0,0],[-100.,0,0],[0,100.,0],[0,-100.,0],
                        [0,0,100.],[0,0,-100.]])
    faces = torch.tensor([[0,2,4],[2,1,4],[1,3,4],[3,0,4],
                          [2,0,5],[1,2,5],[3,1,5],[0,3,5]])
    return xyz, faces


@pytest.mark.parametrize('operation', [remove_overlap_sphere, remove_overlap_sphere_marked])
def test_source_limit_keeps_greater_than_and_final_state(operation):
    xyz, faces = mesh(); faces[0] = faces[0].flip(0)
    old_result, old_history = operation(xyz, faces, start_iteration=13, max_iterations=2)
    report = {}
    result, history = operation(xyz, faces, start_iteration=13,
                                max_iterations=2, diagnostics=report)
    assert torch.equal(result, old_result) and history == old_history
    # 固定源使用严格 >；niterations=2允许第三次更新，然后退出。
    assert report['updates'] == 3 and report['final_iteration'] == 16
    assert report['stop_reason'] == 'source_niterations_limit'
    assert report['last_pre_step_negative_float32'] == history[-1]
    assert report['final_negative_float32'] > 0
    assert report['final_negative_double'] > 0
    assert report['negative_area_mm2_sum'] > 0 and report['quality_status'] == 'failed_quality'
    with pytest.raises(SphereOverlapQualityError) as error:
        operation(xyz, faces, start_iteration=13, max_iterations=2,
                  require_unfolded=True)
    assert error.value.diagnostics == report


@pytest.mark.parametrize('operation', [remove_overlap_sphere, remove_overlap_sphere_marked])
def test_no_folds_early_return_and_required_quality(operation):
    xyz, faces = mesh(); report = {}
    result, history = operation(xyz, faces, diagnostics=report, require_unfolded=True)
    assert torch.equal(result, xyz) and history == []
    assert report['stop_reason'] == 'no_negative_faces' and report['updates'] == 0
    assert report['last_pre_step_negative_float32'] is None
    assert report['negative_area_mm2_sum'] == 0 and report['quality_status'] == 'passed'


@pytest.mark.parametrize('operation', [remove_overlap_sphere, remove_overlap_sphere_marked])
@pytest.mark.parametrize('invalid', ['nan', 'face', 'count'])
def test_invalid_inputs_do_not_silently_report_no_negative(operation, invalid):
    xyz, faces = mesh(); kwargs = {}
    if invalid == 'nan': xyz[0,0] = float('nan')
    if invalid == 'face': faces[0,0] = len(xyz)
    if invalid == 'count': kwargs['max_iterations'] = -1
    with pytest.raises(ValueError): operation(xyz, faces, **kwargs)


def test_finish_strict_gate_preserves_normal_coordinates():
    xyz, faces = mesh(); report = {}
    result, history = finish_standard_sphere(xyz.numpy(), faces.numpy(),
        start_iteration=0, diagnostics=report, require_unfolded=True)
    assert np.array_equal(result, xyz.numpy()) and history == []
    assert report['quality_status'] == 'passed'


def test_complete_standard_stage_does_not_write_after_quality_failure(monkeypatch, tmp_path):
    from fnit.recon_all import sphere_standard_run as stage
    xyz, faces = mesh(); xyz = xyz.numpy(); faces = faces.numpy().astype(np.int32)
    monkeypatch.setattr(stage.fsio, 'read_geometry', lambda path: (xyz.copy(),faces.copy()))
    monkeypatch.setattr(stage, 'sample_standard_metric_matrix',
        lambda *a: (np.zeros(7,np.int64),np.zeros(0,np.int32),np.zeros(0,np.float32),None))
    monkeypatch.setattr(stage, 'average_standard_metric', lambda *a: (np.zeros(0,np.float32),None,None))
    monkeypatch.setattr(stage, 'one_ring_metric', lambda *a: (a[2],a[3],a[4],0))
    monkeypatch.setattr(stage, 'first_epoch_gradient', lambda *a,**k: (None,None,np.zeros_like(xyz),None))
    monkeypatch.setattr(stage, 'average_standard_gradient', lambda x,*a: x)
    monkeypatch.setattr(stage, 'first_epoch_line_search',
        lambda *a: {'selected_dt':0,'starting_sse':{'total':0}})
    monkeypatch.setattr(stage, 'first_epoch_sse', lambda *a: {'total':0})
    monkeypatch.setattr(stage, 'next_standard_sphere_scale', lambda *a: None)
    marker = {'final_negative_float32':1,'final_negative_double':1,'stop_reason':'source_niterations_limit'}
    def failed_finish(*a,**kwargs):
        assert kwargs['require_unfolded'] is True and isinstance(kwargs['diagnostics'],dict)
        raise SphereOverlapQualityError(marker)
    monkeypatch.setattr(stage, 'finish_standard_sphere', failed_finish)
    def forbidden_write(*a,**kwargs): pytest.fail('failed sphere must not be written')
    monkeypatch.setattr(stage, 'write_standard_sphere_surface', forbidden_write)
    target = tmp_path/'sphere'
    with pytest.raises(SphereOverlapQualityError):
        stage.run_standard_sphere(inflated=tmp_path/'inflated',smoothwm=tmp_path/'smoothwm',
                                  output=target,require_unfolded=True)
    assert not target.exists()
