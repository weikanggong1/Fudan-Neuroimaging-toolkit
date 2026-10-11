"""Dot-prior source-expression/rip/binning/smoothing contracts, not MRI benchmarks."""
import hashlib
import numpy as np
import pytest

from fnit.recon_all.topology_global_dot_prior import global_normal_dot_prior
from fnit.recon_all.topology_intensity_histogram import _smooth_histogram


def data():
    return dict(smoothed_original_vertices=np.array([[0,0,0],[3,4,5],[-1,0,2],[-2,1,4]],np.float32),
        metric_vertex_normals=np.tile([1,2,3],(4,1)).astype(np.float32),
        neighbor_offsets=np.array([0,3,3,3,3],np.int32),neighbors=np.array([1,2,3],np.int32),
        vertex_rip=np.zeros(4,np.int32))


def test_preserves_source_prior_expression_not_candidate_three_term_dot():
    result=global_normal_dot_prior(**data())
    assert np.array_equal(result['projected_values'],[123,-1,22])
    assert result['minimum_projection']==-1 and result['maximum_projection']==123
    assert result['bin_size']==np.float32(np.float32(124)/np.float32(98))
    assert result['sampled_edges']==3
    assert not np.array_equal(result['projected_values'],[26,5,12])
    assert 'mm^2' in result['projection_units']


def test_neighbor_rip_not_excluded_but_center_rip_is():
    args=data();a=global_normal_dot_prior(**args);args['vertex_rip'][1]=1
    b=global_normal_dot_prior(**args)
    for key in ['projected_values','consumed','raw_counts','counts','bins']:
        assert a[key].tobytes()==b[key].tobytes()
    args['vertex_rip'][0]=1
    with pytest.raises(ValueError,match='edge'):global_normal_dot_prior(**args)


def test_input_not_mutated_and_repeat_exact():
    args=data();before={k:v.copy() for k,v in args.items()}
    a=global_normal_dot_prior(**args);b=global_normal_dot_prior(**args)
    for key,value in a.items():
        if isinstance(value,np.ndarray):assert value.tobytes()==b[key].tobytes()
    for key,value in before.items():assert np.array_equal(args[key],value)


def test_zero_range_fails_without_fake_prior():
    args=data();args['smoothed_original_vertices'].fill(0)
    with pytest.raises(ValueError,match='width'):global_normal_dot_prior(**args)


def test_source_01_floor_is_normalized_before_sigma2_smoothing():
    r=global_normal_dot_prior(**data())
    assert np.count_nonzero(r['raw_counts']==np.float32(np.float32(.01)/np.float32(3)))==97
    assert r['counts'].tobytes()==_smooth_histogram(r['raw_counts']).tobytes()
    assert r['bins'].shape==r['counts'].shape==(100,)


def test_generic_smoothing_preserves_frozen_256_oracle():
    raw=np.arange(256,dtype=np.float32)/np.float32(1000)
    # Captured from unchanged mature 256-bin source before this generalization.
    assert hashlib.sha256(_smooth_histogram(raw).tobytes()).hexdigest()=='2b09718866195fab7094c8a8b0ad40cb65e27a14ea8a7f1c0c12fb52ff633918'
    raw=np.maximum(np.random.default_rng(192).poisson(3,256).astype(np.float32),np.float32(.1))/np.float32(700)
    assert hashlib.sha256(_smooth_histogram(raw).tobytes()).hexdigest()=='0f38a683416cbde61beed5609e7a8acf777747ed1e54c56857b102a2630d3f36'


def test_100bin_constant_and_edge_renormalization():
    raw=np.ones(100,np.float32)
    assert _smooth_histogram(raw).tobytes()==raw.tobytes()
    raw=np.arange(100,dtype=np.float32)
    a=_smooth_histogram(raw);b=_smooth_histogram(np.r_[raw,np.zeros(156,np.float32)])
    assert a[:92].tobytes()==b[:92].tobytes()
    assert a[-1]>b[99]  # 100-bin high edge has its own kernel normalization.


def test_nonfinite_projection_reports_error():
    args=data();args['metric_vertex_normals'].fill(np.finfo(np.float32).max)
    with pytest.raises(FloatingPointError):global_normal_dot_prior(**args)


@pytest.mark.parametrize('field,value',[
    ('smoothed_original_vertices',np.array(1.)),('metric_vertex_normals',np.full((4,3),np.nan)),
    ('vertex_rip',np.zeros(4,np.float32)),('neighbors',np.array([1,-1,3],np.int32)),
    ('neighbor_offsets',np.array([0,3,2,3,3],np.int32)),
    ('neighbors',np.array([1,2,4],np.int32)),('neighbor_offsets',np.zeros(5,np.float32)),
])
def test_invalid_state_fails(field,value):
    args=data();args[field]=value
    with pytest.raises(ValueError):global_normal_dot_prior(**args)
