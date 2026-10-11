"""全局先验组装的域、rip、共享状态与成熟核复用契约。"""
from dataclasses import replace

import numpy as np
import pytest

import fnit.recon_all.topology_global_priors as module


def inputs():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
    return dict(principal_curvature_fields=dict(
        k1=np.array([-.4, .2, .3, .8], np.float32),
        k2=np.array([-.8, -.2, .1, .4], np.float32)),
        smoothed_original_vertices=xyz, faces=faces,
        neighbor_offsets=np.array([0, 3, 6, 9, 12], np.int32),
        neighbors=np.array([2,1,3,0,2,3,0,1,3,0,1,2], np.int32),
        vertex_rip=np.zeros(4, bool))


def test_reuses_metric_normals_and_mature_priors():
    args=inputs();r=module.global_topology_priors(**args)
    expected=module.curvature_histograms(**args['principal_curvature_fields'])
    for key in expected:
        np.testing.assert_array_equal(r['curvature'][key],expected[key])
    metric=r['smoothed_metric']
    dot=module.global_normal_dot_prior(smoothed_original_vertices=metric.positions,
        metric_vertex_normals=metric.normals,neighbor_offsets=args['neighbor_offsets'],
        neighbors=args['neighbors'],vertex_rip=args['vertex_rip'])
    for key in ('bins','counts','raw_counts','projected_values','bin_indices'):
        assert r['normal_dot'][key].tobytes()==dot[key].tobytes()
    assert r['normal_dot']['sampled_edges']==12
    assert not r['original_sff2_included'] and not r['complete_ga']


def test_inputs_and_repeated_output_are_independent():
    args=inputs();before={k:v.copy() for k,v in args.items() if isinstance(v,np.ndarray)}
    before_k={k:v.copy() for k,v in args['principal_curvature_fields'].items()}
    a=module.global_topology_priors(**args);b=module.global_topology_priors(**args)
    for k,v in before.items(): assert args[k].tobytes()==v.tobytes()
    for k,v in before_k.items(): assert args['principal_curvature_fields'][k].tobytes()==v.tobytes()
    assert a['normal_dot']['counts'].tobytes()==b['normal_dot']['counts'].tobytes()
    a['smoothed_metric'].positions[:]=100
    assert not np.array_equal(a['smoothed_metric'].positions,b['smoothed_metric'].positions)


def test_ripped_center_skipped_but_its_neighbor_coordinate_consumed():
    args=inputs();args['vertex_rip'][0]=True
    args['principal_curvature_fields']['k1'][0]=np.nan
    args['principal_curvature_fields']['k2'][0]=np.nan
    r=module.global_topology_priors(**args)
    assert r['normal_dot']['sampled_edges']==9
    assert not r['normal_dot']['consumed'][:3].any()
    assert r['normal_dot']['consumed'][3:].all()
    expected=module.curvature_histograms(
        args['principal_curvature_fields']['k1'][1:],args['principal_curvature_fields']['k2'][1:])
    assert expected['joint'].tobytes()==r['curvature']['joint'].tobytes()


def test_pending_geometry_is_used_and_returned(monkeypatch):
    original=module.repair_vertex_normals
    def changed(**kwargs):
        state=original(**kwargs);xyz=state.positions.copy();xyz[0,0]+=.25
        return replace(state,positions=xyz,changed_vertices=np.array([0]),coordinate_update_count=1)
    monkeypatch.setattr(module,'repair_vertex_normals',changed)
    args=inputs();r=module.global_topology_priors(**args)
    assert r['smoothed_metric'].coordinate_update_count==1
    reference=module.global_normal_dot_prior(
        smoothed_original_vertices=r['smoothed_metric'].positions,
        metric_vertex_normals=r['smoothed_metric'].normals,
        neighbor_offsets=args['neighbor_offsets'],neighbors=args['neighbors'],vertex_rip=args['vertex_rip'])
    assert reference['projected_values'].tobytes()==r['normal_dot']['projected_values'].tobytes()
    assert args['smoothed_original_vertices'][0,0]==0


@pytest.mark.parametrize('change', ['wrong_domain','wrong_k_dtype','bad_rip','missing_k','nonfinite_k','zero_range','all_rip'])
def test_invalid_consumed_state_fails(change):
    args=inputs()
    if change=='wrong_domain':args['smoothed_original_vertices']=args['smoothed_original_vertices'][:-1]
    elif change=='wrong_k_dtype':args['principal_curvature_fields']['k1']=np.array([0,1,2,3],np.float64)
    elif change=='bad_rip':args['vertex_rip']=np.zeros(4,np.float32)
    elif change=='missing_k':args['principal_curvature_fields'].pop('k2')
    elif change=='nonfinite_k':args['principal_curvature_fields']['k1'][1]=np.inf
    elif change=='zero_range':args['principal_curvature_fields']['k2'][:]=1
    else:args['vertex_rip'][:]=True
    with pytest.raises(ValueError):module.global_topology_priors(**args)


def test_bad_ordered_csr_and_status_fail():
    args=inputs();args['neighbors'][0]=7
    with pytest.raises(ValueError,match='vnum'):module.global_topology_priors(**args)
    args=inputs();args['surface_status']='implicit'
    with pytest.raises(ValueError,match='status'):module.global_topology_priors(**args)
