"""完整先验观察器相位和对应关系契约，不用 SFF2 曲率作 SFF1 输入。"""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('global_priors_tool',
    Path(__file__).resolve().parents[2]/'tools/benchmark_recon_topology_global_priors_producer.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def fields():
    current=np.arange(12,dtype=np.float32).reshape(4,3)
    candidate=dict(raw_vertices=current+np.float32(.5),
        smoothed_original_vertices=current,faces=np.array([[0,1,2]],np.int32),
        one_ring_offsets=np.array([0,1,2,3,3],np.int32),
        one_ring_neighbors=np.array([1,2,0],np.int32))
    reference=dict(vertex_fields=np.column_stack((current,current,np.zeros((4,10),np.float32))),
        faces=candidate['faces'].copy(),neighbor_offsets=candidate['one_ring_offsets'].copy(),
        neighbors=candidate['one_ring_neighbors'].copy())
    for name in ('k1_bins','k2_bins','k1_counts','k2_counts','k1_bin_size','k2_bin_size',
                 'k_pdf_geometry','k_pdf','dot_bins','dot_counts','dot_bin_size'):
        candidate[name]=np.array([1,2,3],np.float32);reference[name]=candidate[name].copy()
    return candidate,reference


def test_current_capture_phase_not_raw_input():
    candidate,reference=fields()
    domains,priors=tool.compare_priors(candidate=candidate,reference=reference)
    assert np.any(candidate['raw_vertices']!=reference['vertex_fields'][:,3:6])
    assert all(row['bitwise_equal'] for row in domains.values())
    assert all(row['bitwise_equal'] for row in priors.values())


def test_correspondence_failure_blocks_prior_claim():
    candidate,reference=fields();reference['faces'][0]=[0,2,1]
    domains,priors=tool.compare_priors(candidate=candidate,reference=reference)
    assert not domains['faces']['bitwise_equal'] and priors=={}


def test_prior_failure_kept_when_domain_matches():
    candidate,reference=fields();reference['dot_counts'][1]+=np.float32(.1)
    domains,priors=tool.compare_priors(candidate=candidate,reference=reference)
    assert all(row['bitwise_equal'] for row in domains.values())
    assert priors['dot_counts']['differing_values']==1


def test_signed_zero_is_still_byte_diagnostic():
    r=tool.difference(np.array([0.],np.float32),np.array([-0.],np.float32))
    assert r['differing_values']==0 and not r['bitwise_equal']


def test_joint_geometry_matches_existing_scorer_origin_spacing_contract():
    geometry=tool.curvature_pdf_geometry(dict(k1_start=-3.,k2_start=-2.,
        k1_width=np.float32(.06),k2_width=np.float32(.04)))
    np.testing.assert_array_equal(geometry[:2],np.array([-3,-2],np.float32))
    assert np.all(geometry[2:]>0)
    k=np.array([-2.94,-1.92],np.float32)
    # Existing scorer consumes origin first, spacing second. Reversed export
    # cannot resolve these nonzero coordinates to their intended bins.
    np.testing.assert_array_equal(np.trunc((k-geometry[:2])/geometry[2:]),[0,2])
