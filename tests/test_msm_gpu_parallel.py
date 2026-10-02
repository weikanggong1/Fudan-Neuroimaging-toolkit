"""Exact execution controls; generated meshes are not real-data benchmarks."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import nibabel as nib
import numpy as np
import pytest
from scipy import sparse
from scipy.spatial import cKDTree
import torch

from fnit.msm import _fastpd_native
from fnit.msm._execution import cpu_workers, register_hemispheres
from fnit.msm._ordered_sparse import adaptive_values
from fnit.msm._spatial import ExactCellNearest
from fnit.msm.config import MSMSulcConfig
from fnit.msm.config_msmall import MSMAllConfig
from fnit.msm.msmall import MSMAllInputs, run_msmall
from fnit.msm.msmsulc import _adaptive_resample, _ico, run_msmsulc
from fnit.msm.prepare import MSMSulcInputs

DEVICES = ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))]


def _reference_sparse(fi, fw, ri, rw, values, old_area, new_area):
    m, n = len(fi), len(ri)
    forward = sparse.coo_matrix((fw.ravel(), (np.repeat(np.arange(m), 3), fi.ravel())), shape=(m,n)).tocsr()
    reverse = sparse.coo_matrix((rw.ravel(), (np.repeat(np.arange(n), 3), ri.ravel())), shape=(n,m)).T.tocsr()
    choose = np.diff(reverse.indptr) > np.diff(forward.indptr)
    joined = sparse.diags(choose.astype(float))@reverse+sparse.diags((~choose).astype(float))@forward
    weighted = sparse.diags(new_area)@joined
    correction = np.asarray(weighted.sum(axis=0)).ravel()
    factors = np.divide(old_area, correction, out=np.zeros(n), where=correction>0)
    weighted = weighted@sparse.diags(factors)
    row_sum = np.asarray(weighted.sum(axis=1)).ravel()
    normalized = sparse.diags(np.divide(1.,row_sum,out=np.zeros(m),where=row_sum>0))@weighted
    return np.asarray(normalized@values)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dimensions', [1, 3])
@pytest.mark.parametrize('shape', [(17,29),(29,17),(9,121),(121,9),(3,600)])
def test_area_sparse_weights_and_reductions_match_reference_bitwise(device, dimensions, shape):
    m,n = shape; rng = np.random.default_rng(83)
    fi = np.stack([rng.choice(n,3,replace=False) for _ in range(m)])
    ri = np.stack([rng.choice(m,3,replace=False) for _ in range(n)])
    fw = rng.random((m,3)); fw /= fw.sum(1)[:,None]
    rw = rng.random((n,3)); rw /= rw.sum(1)[:,None]
    fw[0,0] = 0.; rw[0,0] = 0.
    old,new = rng.random(n),rng.random(m)
    values = rng.normal(size=(n,dimensions)) if dimensions>1 else rng.normal(size=n)
    expected = _reference_sparse(fi,fw,ri,rw,values,old,new)
    actual = adaptive_values(*(torch.as_tensor(x,device=device) for x in (fi,fw,ri,rw)), values,old,new)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


@pytest.mark.parametrize('device', DEVICES)
def test_exact_nearest_proof_and_fallback_preserve_original_tree(device):
    vertices,_ = _ico(3); rng = np.random.default_rng(192)
    query = rng.normal(size=(1437,3)); query *= 100/np.linalg.norm(query,axis=1,keepdims=True)
    query = np.r_[query, vertices[:10], np.zeros((1,3)), [[500.,30.,-200.]]]
    search = ExactCellNearest(vertices,device,chunk_size=101)
    nearest, unresolved = search.query(torch.as_tensor(query,device=device))
    expected = cKDTree(vertices).query(query,k=1,workers=1)[1]
    proved = ~unresolved.cpu().numpy()
    assert proved.any() and unresolved.any()
    np.testing.assert_array_equal(nearest.cpu().numpy()[proved],expected[proved])
    completed = nearest.cpu().numpy().copy(); completed[~proved] = cKDTree(vertices).query(query[~proved],k=1)[1]
    np.testing.assert_array_equal(completed,expected)


def test_native_kernels_are_reentrant_and_exception_safe():
    rng = np.random.default_rng(483)
    faces = np.array([[0,1,2],[1,2,3]],np.int32).tobytes()
    costs = rng.normal(size=(2,8)).astype(np.float64).tobytes()
    points = rng.normal(size=(193,3)); centre = np.array([2.,-1.,3.])
    payload = np.stack((rng.random((191,11)),rng.normal(size=(191,11)),np.ones((191,11))),axis=-1)
    expected = (_fastpd_native.optimize(faces,costs,4),
                _fastpd_native.source_rotation_matrices(points,centre,len(points)),
                _fastpd_native.source_wls_cost(payload,191,11,2.))
    def native(index):
        result = (_fastpd_native.optimize(faces,costs,4),
                  _fastpd_native.source_rotation_matrices(points,centre,len(points)),
                  _fastpd_native.source_wls_cost(payload,191,11,2.))
        bad = payload.copy(); bad[0,0,0] = -1
        with pytest.raises(ValueError,match='nonnegative'):
            _fastpd_native.source_wls_cost(bad,191,11,2.)
        assert result == expected
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(native,range(20)))


@pytest.mark.parametrize('device', DEVICES)
def test_parallel_workers_have_distinct_budgets_and_cuda_streams(device):
    barrier = threading.Barrier(2)
    def worker(hemi,budget):
        barrier.wait(timeout=10)
        identity = torch.cuda.current_stream(device).cuda_stream if device.startswith('cuda') else None
        return hemi,budget,cpu_workers(),identity
    actual = register_hemispheres(worker,device,parallel=True,cpu_threads=7)
    assert [(x[0],x[1],x[2]) for x in actual] == [('L',4,4),('R',3,3)]
    if device.startswith('cuda'):
        assert actual[0][3] != actual[1][3]
    assert cpu_workers() == 4


def test_one_thread_budget_is_sequential_and_exception_workers_join():
    seen=[]
    assert register_hemispheres(lambda hemi,n: seen.append(hemi) or n,'cpu',cpu_threads=1) == (1,1)
    assert seen == ['L','R']
    barrier=threading.Barrier(2); completed=threading.Event()
    def worker(hemi,n):
        barrier.wait(timeout=10)
        if hemi=='L': raise ValueError('fixture error')
        completed.set()
    with pytest.raises(ValueError,match='fixture error'):
        register_hemispheres(worker,'cpu',cpu_threads=2)
    assert completed.is_set()


def _inputs(tmp_path, method):
    vertices, faces = _ico(1); inputs={}
    for hemi,offset in [('L',.2),('R',-.7)]:
        sphere=tmp_path/f'{hemi}.surf.gii'; metric=tmp_path/f'{hemi}.shape.gii'
        reference=tmp_path/f'{hemi}.reference.shape.gii'
        nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(vertices.astype(np.float32),intent=1008),
            nib.gifti.GiftiDataArray(faces.astype(np.int32),intent=1009)]),sphere)
        first=vertices[:,0]/100+offset*vertices[:,2]/100
        values=np.stack((first,vertices[:,1]/100,vertices[:,2]/100),1) if method=='msmall' else first[:,None]
        target=values.copy(); target[:,0] += .03*vertices[:,1]/100
        for path,columns in [(metric,values),(reference,target)]:
            nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(x.astype(np.float32),intent=2005)
                                           for x in columns.T]),path)
        inputs[hemi]=(MSMAllInputs(sphere,metric,sphere,reference) if method=='msmall' else
                      MSMSulcInputs(sphere,sphere,metric,sphere,reference,tmp_path/'unused.mat'))
    return inputs


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('method', ['msmsulc','msmall'])
def test_parallel_registration_preserves_full_decoded_outputs_and_schedule(tmp_path,device,method):
    inputs=_inputs(tmp_path,method)
    if method=='msmsulc':
        config=MSMSulcConfig(simval=(1,2,2,2),iterations=(1,1,1,1),control_grid=(1,1,1,1),
                            sampling_grid=(2,2,2,2),data_grid=(1,1,1,1))
        run=run_msmsulc
    else:
        config=MSMAllConfig(simval=(2,),iterations=(1,),control_grid=(1,),sampling_grid=(2,),
                            data_grid=(1,),regularization=(.00001,))
        run=run_msmall
    sequential=run(inputs,tmp_path/'sequential',config=config,device=device,parallel=False,cpu_threads=2)
    parallel=run(inputs,tmp_path/'parallel',config=config,device=device,parallel=True,cpu_threads=2)
    for hemi in 'LR':
        for first,second in zip(nib.load(sequential[hemi]).darrays,nib.load(parallel[hemi]).darrays):
            np.testing.assert_array_equal(first.data,second.data)
    report=json.loads((tmp_path/'parallel/registration_report.json').read_text())
    assert report['execution']['hemispheres_parallel']
    assert report['L']['cpu_threads'] == report['R']['cpu_threads'] == 1
    if device.startswith('cuda'):
        assert report['L']['peak_allocated_gb'] == report['R']['peak_allocated_gb']


def test_empty_sampling_neighborhood_keeps_three_coordinate_columns():
    from fnit.msm.msmsulc import _label_samples
    vertices,faces=_ico(1)
    centre,samples=_label_samples(vertices,faces,0.)
    assert centre.shape == (3,) and samples.shape == (0,3)
    assert np.vstack((centre,samples)).shape == (1,3)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_registration_does_not_reset_callers_allocator_peak(tmp_path,monkeypatch):
    # An earlier stage peak must survive nested MSM calls on the same device.
    previous=torch.empty(8_000_000,dtype=torch.float64,device='cuda:0')
    torch.cuda.synchronize(); before=torch.cuda.max_memory_allocated(0);del previous
    def forbidden(*args,**kwargs):
        raise AssertionError('library must not reset callers global allocator peak')
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',forbidden)
    config=MSMAllConfig(simval=(2,),iterations=(1,),control_grid=(1,),sampling_grid=(2,),
                        data_grid=(1,),regularization=(.00001,))
    run_msmall(_inputs(tmp_path,'msmall'),tmp_path/'peak',config=config,device='cuda:0',cpu_threads=2)
    assert torch.cuda.max_memory_allocated(0)>=before


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dimensions',[1,3])
def test_gpu_adaptive_resampling_matches_full_reference_chain(dimensions):
    # Nonuniform unfolded geometry, shared vertices and exact-zero weights
    # exercise tree ties, radial face selection, CSR arithmetic and transfer.
    vertices,faces=_ico(2); target,target_faces=_ico(1)
    vertices[:,0] += .17*vertices[:,1]
    vertices *=100/np.linalg.norm(vertices,axis=1,keepdims=True)
    rng=np.random.default_rng(3829)
    values=rng.normal(size=len(vertices) if dimensions==1 else (len(vertices),dimensions))
    expected=_adaptive_resample(vertices,faces,values,target,target_faces,device='cuda:0',execution='reference')
    actual=_adaptive_resample(vertices,faces,values,target,target_faces,device='cuda:0',execution='optimized')
    np.testing.assert_array_equal(actual,expected)


def test_pathological_sparse_layout_is_bounded_before_gpu_allocation():
    from fnit.msm._ordered_sparse import _group_table, SparseLayoutTooLarge
    groups=torch.zeros(200,dtype=torch.long)
    with pytest.raises(SparseLayoutTooLarge,match='bounded GPU'):
        _group_table(groups,torch.arange(200),100_000)


def test_unindexed_cuda_is_pinned_before_thread_submission(monkeypatch):
    from contextlib import nullcontext
    observed=[]
    class Stream:
        def __init__(self,device=None):
            self.device=device
            if device is not None: observed.append(device.index)
        def wait_stream(self,other): pass
        def synchronize(self): pass
    monkeypatch.setattr(torch.cuda,'init',lambda:None)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:1)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device=None:Stream())
    monkeypatch.setattr(torch.cuda,'Stream',Stream)
    monkeypatch.setattr(torch.cuda,'device',lambda device:nullcontext())
    monkeypatch.setattr(torch.cuda,'stream',lambda stream:nullcontext())
    assert register_hemispheres(lambda hemi,n:hemi,'cuda',cpu_threads=2)==('L','R')
    assert observed==[1,1]
