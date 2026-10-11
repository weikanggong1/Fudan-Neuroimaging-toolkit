"""Global-SFF contracts and analytic geometry; these are not MRI benchmarks."""
import numpy as np
import pytest

from fnit.recon_all.topology_global_sff import (
    _global_tangent, _global_state_fields, global_second_fundamental_form,
    global_second_fundamental_form_state, global_curvature_priors,
)


def data():
    xy = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
                   [1, 1], [-1, -1]], np.float32)
    z = np.float32(.25) * xy[:, 0] ** 2 + np.float32(.125) * xy[:, 1] ** 2
    return dict(current_vertices=np.column_stack((xy, z)),
        vertex_normals=np.tile([0., 0., 1.], (7, 1)),
        neighbor_offsets=np.array([0, 6, 6, 6, 6, 6, 6, 6], np.int32),
        neighbors=np.arange(1, 7, dtype=np.int32), vertex_rip=np.zeros(7, np.int32),
        initial_k1=np.full(7, 7., np.float32), initial_k2=np.full(7, -3., np.float32))


def test_analytic_quadratic_and_no_input_mutation():
    inputs=data(); before={k:v.copy() for k,v in inputs.items()}
    r=global_second_fundamental_form(**inputs)
    assert np.allclose([r['k1'][0],r['k2'][0]],[.5,.25],rtol=0,atol=2e-6)
    assert r['fit_kind'].tolist()==[1,0,0,0,0,0,0]
    assert r['valid_neighbors'][0]==6 and r['search_attempts'][0]==1
    assert np.array_equal(r['k1'][1:],inputs['initial_k1'][1:])
    for k,v in before.items():assert np.array_equal(inputs[k],v)
    assert not np.shares_memory(r['k1'],inputs['initial_k1'])


def test_global_tangent_uses_source_permutation_and_fallback():
    a,b=_global_tangent(np.array([0,0,1],np.float32))
    assert np.array_equal(a,[-1,0,0]) and np.array_equal(b,[0,-1,0])
    a,b=_global_tangent(np.array([1,1,1],np.float32)/np.float32(np.sqrt(3)))
    assert np.isclose(np.linalg.norm(a),1,atol=2e-7)
    assert np.isclose(np.linalg.norm(b),1,atol=2e-7)
    assert np.isclose(a@b,0,atol=2e-7)


def test_rip_curvatures_preserved_but_tangent_recomputed():
    inputs=data();inputs['vertex_rip'][0]=1;inputs['initial_k1'][0]=np.nan
    r=global_second_fundamental_form(**inputs)
    assert np.isnan(r['k1'][0]) and r['fit_kind'][0]==-1
    assert np.array_equal(r['tangent_e1'][0],[-1,0,0])
    assert r['search_attempts'][0]==0


def test_zero_global_matrix_uses_identity_directions_not_previous_fit():
    inputs=data();inputs['current_vertices'].fill(0)
    r=global_second_fundamental_form(**inputs)
    assert r['k1'][0]==r['k2'][0]==0 and r['fit_kind'][0]==3
    assert r['condition_number'][0]==np.float32(1e10)
    assert r['search_attempts'][0]==102 and r['valid_neighbors'][0]==0
    assert np.array_equal(r['tangent_e1'][0],[-1,0,0])
    assert np.array_equal(r['tangent_e2'][0],[0,-1,0])


def test_no_neighbor_keeps_initial_nonfinite_curvature():
    inputs=data();inputs['initial_k2'][1]=np.nan
    r=global_second_fundamental_form(**inputs)
    assert np.isnan(r['k2'][1]) and r['fit_kind'][1]==0
    assert r['search_attempts'][1]==0 and np.array_equal(r['tangent_e2'][1],[0,-1,0])
    with pytest.raises(ValueError,match='finite'):
        global_curvature_priors(**inputs)


def test_ripped_neighbors_are_compressed_in_original_order():
    inputs=data();inputs['vertex_rip'][2]=1
    r=global_second_fundamental_form(**inputs)
    other=data();other['neighbors']=np.array([1,3,4,5,6],np.int32)
    other['neighbor_offsets']=np.array([0,5,5,5,5,5,5,5],np.int32)
    r2=global_second_fundamental_form(**other)
    assert r['valid_neighbors'][0]==5
    for key in ['k1','k2','gram','rhs','hessian']:
        assert np.array_equal(r[key][0],r2[key][0])


def test_ill_conditioned_global_fallback_keeps_tangent_and_source_factor():
    inputs=data();x=np.array([0,1,-1,2,-2,3,-3],np.float32)
    inputs['current_vertices']=np.column_stack((x,np.zeros(7,np.float32),np.float32(.25)*x*x))
    r=global_second_fundamental_form(**inputs)
    assert r['fit_kind'][0]==2 and r['k1'][0]==r['k2'][0]==np.float32(.25)
    assert np.array_equal(r['tangent_e1'][0],[-1,0,0])


def test_repeat_all_scored_outputs_exact():
    inputs=data();a=global_second_fundamental_form(**inputs);b=global_second_fundamental_form(**inputs)
    for key,value in a.items():
        if isinstance(value,np.ndarray):assert value.tobytes()==b[key].tobytes()


def state_data():
    args=data()
    args.update(initial_h=np.arange(7,dtype=np.float32)+10,
        initial_gaussian_curvature=np.arange(7,dtype=np.float32)-20,
        vertex_area=np.arange(7,dtype=np.float32)+2)
    return args


def test_state_updates_source_h_k_and_preserves_unupdated_fields():
    args=state_data();before={k:v.copy() for k,v in args.items()}
    r=global_second_fundamental_form_state(**args)
    assert r['h'][0]==np.float32(np.float32(r['k1'][0]+r['k2'][0])/np.float32(2))
    assert r['gaussian_curvature'][0]==np.float32(r['k1'][0]*r['k2'][0])
    assert np.array_equal(r['h'][1:],args['initial_h'][1:])
    assert np.array_equal(r['gaussian_curvature'][1:],args['initial_gaussian_curvature'][1:])
    assert r['surface_statistics']['total_area_mm2']==2
    assert isinstance(r['surface_statistics']['Ktotal'],np.float32)
    for key in ['curv','curvbak','valbak']:assert key in r['unmodified_state_fields']
    for key,value in before.items():assert np.array_equal(args[key],value)


def test_ill_fallback_updates_vertex_curvatures_but_not_surface_summary():
    args=state_data();x=np.array([0,1,-1,2,-2,3,-3],np.float32)
    args['current_vertices']=np.column_stack((x,np.zeros(7,np.float32),np.float32(.25)*x*x))
    r=global_second_fundamental_form_state(**args)
    assert r['fit_kind'][0]==2
    assert r['h'][0]==np.float32(.25) and r['gaussian_curvature'][0]==np.float32(.0625)
    summary=r['surface_statistics']
    assert summary['Kmin']==summary['Hmin']==10000
    assert summary['Kmax']==summary['Hmax']==-10000
    assert summary['Ktotal']==summary['total_area_mm2']==0


def test_source_ktotal_is_rounded_at_every_vertex_not_after_double_sum():
    a=np.array([1,1,-1],np.float32);b=np.ones(3,np.float32)
    fields=_global_state_fields(a,b,np.ones(3,np.int8),b,b,np.array([2**24,1,2**24],np.float32))
    assert fields[6]==0  # Double accumulation then final cast would give 1.
    assert fields[7]==2**25+1


def test_ripped_h_k_nan_payload_retained():
    args=state_data();args['vertex_rip'][0]=1
    args['initial_h'][0]=np.array([0x7fc00017],np.uint32).view(np.float32)[0]
    r=global_second_fundamental_form_state(**args)
    assert r['h'][0].tobytes()==args['initial_h'][0].tobytes()
    assert r['gaussian_curvature'][0].tobytes()==args['initial_gaussian_curvature'][0].tobytes()


@pytest.mark.parametrize('field,value',[
    ('initial_h',np.zeros(8,np.float32)),('vertex_area',np.full(7,np.nan,np.float32)),
])
def test_state_malformed_fields_fail(field,value):
    args=state_data();args[field]=value
    with pytest.raises(ValueError):global_second_fundamental_form_state(**args)


def test_prior_adapter_consumes_computed_k_and_excludes_rip():
    inputs=data();inputs['vertex_rip'][1]=1;inputs['initial_k1'][1]=np.nan
    r=global_curvature_priors(**inputs)
    assert r['sff']['k1'][0]!=inputs['initial_k1'][0]
    assert r['histogram']['joint'].shape==(100,100)
    assert np.isfinite(r['histogram']['joint']).all()
    assert 'complete topology' in r['scope']


def test_zero_width_prior_fails_without_fake_pdf():
    inputs=data();inputs['neighbor_offsets'].fill(0);inputs['neighbors']=np.empty(0,np.int32)
    with pytest.raises(ValueError,match='zero width'):global_curvature_priors(**inputs)


@pytest.mark.parametrize('field,value',[
    ('current_vertices',np.array(1.)),('vertex_normals',np.full((7,3),np.nan)),
    ('neighbor_offsets',np.zeros(8,np.float32)),('neighbor_offsets',np.array([0,6,5,6,6,6,6,6])),
    ('neighbors',np.array([7,1,2,3,4,5])),('neighbors',np.array([-1,1,2,3,4,5])),
    ('vertex_rip',np.zeros(7,np.float32)),('initial_k1',np.ones(8,np.float32)),
])
def test_malformed_stage_fails(field,value):
    inputs=data();inputs[field]=value
    with pytest.raises(ValueError):global_second_fundamental_form(**inputs)
