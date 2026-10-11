"""候选输入的跨缺陷缓存、两个坐标域、持续几何和失败边界。"""
from unittest.mock import patch
import inspect

import numpy as np
import pytest

from fnit.recon_all.topology_candidate_inputs import (
    prepare_candidate_input_context, prepare_defect_candidate_inputs)
from fnit.recon_all.topology_defect_calibration import calibrated_defect_volume_priors
from fnit.recon_all.topology_edge_mri_score import prepare_topology_edge_inputs


def arguments():
    xyz=np.array([[5,5,5],[7,5,5],[5,7,5],[5,5,7]],np.float32)
    triangles=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32)
    grid=np.indices((20,20,20))
    bins=np.linspace(-2,-.5,100,dtype=np.float32)
    counts=np.full(100,np.float32(.01),np.float32)
    priors=dict(curvature=dict(k1_bins=bins,k2_bins=bins.copy(),
        k1_counts=counts,k2_counts=counts.copy(),k1_width=np.float32(.015),
        k2_width=np.float32(.015),k1_start=bins[0],k2_start=bins[0],
        k1_mean=np.float32(999),k2_mean=np.float32(999), # Metadata is not a trusted input mean.
        joint=np.full((100,100),np.float32(.0001),np.float32)),
        normal_dot=dict(bins=bins.copy(),counts=counts.copy(),bin_size=np.float32(.02)))
    return dict(vertices=xyz,faces=triangles,defect_vertices=np.array([True,False,False,False]),
        vertex_translation=np.arange(4,dtype=np.int32),volume=(sum(grid)+40).astype(np.uint8),
        ras_to_voxel=np.eye(4,dtype=np.float32),global_priors=priors)


def defect_arguments(context, *, current=None):
    return dict(context=context,original_defect_hull=np.array([1,2,3],np.int32),
        candidate_edges=np.array([[0,1],[1,2]],np.int32),
        current_corrected_original_vertices=context.preparation['original_vertices'].copy()
            if current is None else current)


def test_first_defect_matches_mature_full_calibration_and_local_prior():
    args=arguments();context=prepare_candidate_input_context(**args)
    result=prepare_defect_candidate_inputs(**defect_arguments(context))
    expected=calibrated_defect_volume_priors(**{key:args[key] for key in args if key!='global_priors'},
        defect_hull=np.array([1,2,3],np.int32),candidate_edges=np.array([[0,1],[1,2]],np.int32))
    for name,value in expected['priors'].items():
        if isinstance(value,np.ndarray):
            assert result['candidate_kwargs']['local_priors'][name].tobytes()==value.tobytes()
    for name in ('selected','white_raw','gray_raw','white_smooth','gray_smooth'):
        assert result['calibration'][name].tobytes()==expected['calibration'][name].tobytes()
    assert result['calibration']['bin_size']==0


def test_context_preparation_once_for_multiple_defect_calls():
    with patch('fnit.recon_all.topology_candidate_inputs.prepare_topology_edge_inputs',
               wraps=prepare_topology_edge_inputs) as call:
        context=prepare_candidate_input_context(**arguments())
        one=prepare_defect_candidate_inputs(**defect_arguments(context))
        two=prepare_defect_candidate_inputs(**defect_arguments(context))
        assert call.call_count==1
    assert one['candidate_kwargs']['gray_white_pdf'] is two['candidate_kwargs']['gray_white_pdf']
    assert one['candidate_kwargs']['joint_curvature_pdf'] is context.joint_curvature_pdf
    assert one['calibration']['white_smooth'].tobytes()==two['calibration']['white_smooth'].tobytes()


def test_complete_candidate_call_signature_and_shared_mri_inputs():
    # A numerically exact prepared substage must still expose every mandatory
    # argument consumed by the real full-candidate scorer (GA integration bug).
    from fnit.recon_all.topology_candidate_pipeline import evaluate_candidate
    context=prepare_candidate_input_context(**arguments())
    result=prepare_defect_candidate_inputs(**defect_arguments(context))
    kwargs=result['candidate_kwargs']
    bound=inspect.signature(evaluate_candidate).bind(
        surface_state={},retessellation_inputs={},**kwargs)
    assert bound.arguments['volume'] is context.volume
    assert bound.arguments['ras_to_voxel'] is context.ras_to_voxel
    assert not bound.arguments['volume'].flags.writeable
    assert not bound.arguments['ras_to_voxel'].flags.writeable
    assert set(bound.arguments)==set(inspect.signature(evaluate_candidate).parameters)


def test_next_defect_uses_current_corrected_geometry_not_original_reset():
    context=prepare_candidate_input_context(**arguments())
    first=prepare_defect_candidate_inputs(**defect_arguments(context))
    changed=context.preparation['original_vertices'].copy();changed[:,0]+=np.float32(1)
    later=prepare_defect_candidate_inputs(**defect_arguments(context,current=changed))
    a=first['candidate_kwargs']['local_priors'];b=later['candidate_kwargs']['local_priors']
    assert b['volume_origin_mm'][0]==np.float32(a['volume_origin_mm'][0]+1)
    assert a['defect_intensity_volume'].tobytes()!=b['defect_intensity_volume'].tobytes()
    assert first['calibration']['white_smooth'].tobytes()==later['calibration']['white_smooth'].tobytes()


def test_original_hull_and_corrected_edges_remain_distinct():
    a=arguments();first_context=prepare_candidate_input_context(**a)
    first=prepare_defect_candidate_inputs(**defect_arguments(first_context))
    b=arguments();translation=np.array([2,0,3,1],np.int32);b['vertex_translation']=translation
    context=prepare_candidate_input_context(**b)
    args=defect_arguments(context,current=context.preparation['original_vertices'][np.argsort(translation)])
    args['candidate_edges']=translation[args['candidate_edges']]
    second=prepare_defect_candidate_inputs(**args)
    for name in ('defect_intensity_volume','white_log_probability','gray_log_probability','initial_edge_mask'):
        assert first['candidate_kwargs']['local_priors'][name].tobytes()==second['candidate_kwargs']['local_priors'][name].tobytes()
    np.testing.assert_array_equal(first['calibration']['selected'],second['calibration']['selected'])


def test_context_copies_inputs_and_is_readonly_without_precision_changes():
    a=arguments();context=prepare_candidate_input_context(**a)
    before=context.volume.copy();a['volume'][:]=0
    assert context.volume.tobytes()==before.tobytes()
    assert context.volume.dtype==np.uint8
    with pytest.raises(ValueError):context.volume[0,0,0]=1
    with pytest.raises(TypeError):context.k1_histogram['bin_size']=np.float32(0)
    assert context.k1_mean!=999
    args=defect_arguments(context);current=args['current_corrected_original_vertices'].copy()
    prepare_defect_candidate_inputs(**args)
    assert args['current_corrected_original_vertices'].tobytes()==current.tobytes()


@pytest.mark.parametrize('key,value',[
    ('vertex_translation',np.array([0,0,2,3],np.int32)),
    ('defect_vertices',np.array([1,0,0,0],np.int32)),
    ('faces',np.array([[0,1,4]],np.int32)),
])
def test_bad_context_fails_before_sampling(key,value):
    args=arguments();args[key]=value
    with patch('fnit.recon_all.topology_candidate_inputs.prepare_topology_edge_inputs') as call:
        with pytest.raises(ValueError):prepare_candidate_input_context(**args)
        call.assert_not_called()


def test_bad_global_histogram_metadata_fails_before_sampling():
    args=arguments();args['global_priors']['curvature']['k1_width']=None
    with pytest.raises(ValueError):prepare_candidate_input_context(**args)


@pytest.mark.parametrize('key,value',[
    ('original_defect_hull',np.array([],np.int32)),
    ('original_defect_hull',np.array([4],np.int32)),
    ('candidate_edges',np.array([[0,4]],np.int32)),
    ('current_corrected_original_vertices',np.zeros((4,3),np.float64)),
    ('mri_weight',np.inf),
])
def test_bad_defect_input_fails_before_local_mri(key,value):
    context=prepare_candidate_input_context(**arguments());args=defect_arguments(context);args[key]=value
    with patch('fnit.recon_all.topology_candidate_inputs.defect_volume_priors') as call:
        with pytest.raises(ValueError):prepare_defect_candidate_inputs(**args)
        call.assert_not_called()


def test_empty_calibration_and_negative_source_weight_semantics():
    a=arguments();a['defect_vertices'][:]=True
    with pytest.raises(ValueError):prepare_candidate_input_context(**a)
    context=prepare_candidate_input_context(**arguments());args=defect_arguments(context)
    args.update(mri_weight=-1.,volume_unmri_weight=0.,normal_dot_weight=-2.)
    result=prepare_defect_candidate_inputs(**args)
    assert result['candidate_kwargs']['mri_weight']==-1.
    assert result['candidate_kwargs']['volume_unmri_weight']==0.
    assert result['candidate_kwargs']['normal_dot_weight']==-2.
