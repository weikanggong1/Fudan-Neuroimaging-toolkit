"""Generated numerical regression fixtures only; not performance benchmarks."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import torch

import fnit.bigflica.dicl_torch as d
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def tensors(batch, atoms, features, dtype=torch.float64, seed=94):
    generator=torch.Generator(device='cuda').manual_seed(seed)
    dictionary=torch.randn((atoms,features),generator=generator,device='cuda',dtype=dtype)
    dictionary/=torch.linalg.vector_norm(dictionary,dim=1,keepdim=True)
    samples=torch.randn((batch,features),generator=generator,device='cuda',dtype=dtype)
    if batch >= 4:
        samples[::4]=0
    return samples,dictionary


@pytest.mark.parametrize('batch,atoms,features',[(32,200,500),(1,200,500),(2,200,500),(16,24,8)])
@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
def test_codes_bitwise(batch,atoms,features,dtype):
    samples,dictionary=tensors(batch,atoms,features,dtype)
    for alpha in (1.,.4,1.7):
        expected=d._sparse_codes_lars_compatible_eager(samples,dictionary,alpha,1000)
        actual=d._sparse_codes_lars_compatible(samples,dictionary,alpha,1000)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_final_nodes_reuse_and_nonalias():
    tolerance=2*np.finfo(np.float32).eps
    samples=torch.tensor([[2.,1.+tolerance/2],[2.,1.-tolerance/2],[2.,.5],[1.+tolerance/2,0.],[1.+2*tolerance,0.],[0.,0.]],device='cuda',dtype=torch.float64)
    dictionary=torch.eye(2,device='cuda',dtype=torch.float64)
    prior=None
    for alpha in (1.,.2,4.,1.):
        actual=d._sparse_codes_lars_compatible(samples,dictionary,alpha,1000)
        expected=d._sparse_codes_lars_compatible_eager(samples,dictionary,alpha,1000)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        if prior is not None:
            torch.testing.assert_close(prior,prior_snapshot,atol=0,rtol=0)
            assert actual.data_ptr()!=prior.data_ptr()
        prior=actual
        prior_snapshot=actual.clone()


def test_fractional_float32_alpha_uses_single_rounded_node_threshold(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    alpha = 10.0000005
    tolerance = 500 * np.finfo(np.float32).eps
    # Two separately rounded scalars place this sample on a different node.
    threshold_twice = np.float32(np.float32(alpha) + np.float32(tolerance))
    assert threshold_twice > np.float32(alpha + tolerance)
    samples = torch.zeros((1, 500), device='cuda', dtype=torch.float32)
    dictionary = torch.zeros_like(samples)
    samples[0, 0] = float(threshold_twice)
    dictionary[0, 0] = 1
    expected = d._sparse_codes_lars_compatible_eager(samples, dictionary, alpha, 1000)
    actual = d._sparse_codes_lars_compatible(samples, dictionary, alpha, 1000)
    assert expected[0, 0] > 0
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize('limit',[-1,1,2,3,4,5,6,7,8,9,0,None])
def test_budget_errors_and_reset(limit):
    samples,dictionary=tensors(2,12,20,seed=411)
    def evaluate(function):
        try:
            return function(samples,dictionary,.02,limit)
        except ValueError as exc:
            return str(exc)
    expected=evaluate(d._sparse_codes_lars_compatible_eager)
    actual=evaluate(d._sparse_codes_lars_compatible)
    if isinstance(expected,str):
        assert actual==expected
    else:
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    # A short-budget failure cannot corrupt a later solve with the same key.
    torch.testing.assert_close(d._sparse_codes_lars_compatible(samples,dictionary,1.,1000),
                               d._sparse_codes_lars_compatible_eager(samples,dictionary,1.,1000),atol=0,rtol=0)


def test_dependent_atoms_and_zero_dictionary():
    dictionary=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,0.]],device='cuda',dtype=torch.float64)
    samples=torch.tensor([[2.,0.],[2.,2.],[0.,0.],[-2.,3.]],device='cuda',dtype=torch.float64)
    for current in (dictionary,torch.zeros_like(dictionary),dictionary):
        expected=d._sparse_codes_lars_compatible_eager(samples,current,1.,1000)
        actual=d._sparse_codes_lars_compatible(samples,current,1.,1000)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_thread_local_bounded_cache():
    def worker(seed):
        observed=[]
        for atoms in (5,6,7,8,9):
            samples,dictionary=tensors(2,atoms,12,seed=seed)
            expected=d._sparse_codes_lars_compatible_eager(samples,dictionary,.5,1000)
            actual=d._sparse_codes_lars_compatible(samples,dictionary,.5,1000)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            observed.append(len(d._compatible_lars_graph_cache()))
        return id(d._compatible_lars_graph_cache()),observed
    # Warming separate threads concurrently also checks capture isolation.
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(worker,[71,112]))
    assert results[0][0]!=results[1][0]
    assert all(count<=4 for _,counts in results for count in counts)


def test_precision_flags_select_distinct_graphs():
    samples,dictionary=tensors(2,24,32,torch.float32,seed=917)
    previous=torch.get_float32_matmul_precision()
    try:
        for precision in ('highest','high','highest'):
            torch.set_float32_matmul_precision(precision)
            expected=d._sparse_codes_lars_compatible_eager(samples,dictionary,.4,1000)
            actual=d._sparse_codes_lars_compatible(samples,dictionary,.4,1000)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    finally:
        torch.set_float32_matmul_precision(previous)


def test_same_thread_cross_stream_reuse_waits_for_output_clone():
    samples,dictionary=tensors(2,24,32,torch.float64,seed=23)
    expected=[d._sparse_codes_lars_compatible_eager(samples*scale,dictionary,.4,1000)
              for scale in (1.,1.3,.8)]
    streams=[torch.cuda.Stream(),torch.cuda.Stream()]
    current=torch.cuda.current_stream()
    for stream in streams:
        stream.wait_stream(current)
    results=[]
    for index,scale in enumerate((1.,1.3,.8)):
        with torch.cuda.stream(streams[index%2]):
            results.append(d._sparse_codes_lars_compatible(samples*scale,dictionary,.4,1000))
    for stream in streams:
        current.wait_stream(stream)
    for actual,reference in zip(results,expected):
        torch.testing.assert_close(actual,reference,atol=0,rtol=0)
    assert len({value.data_ptr() for value in results})==len(results)
