"""Candidate batching must not alter order, truncation or tie acceptance."""
import numpy as np
import unittest
from fnit.recon_all.mri_em_register_search_source import score_candidates
from fnit.recon_all.mri_em_register_score_gpu import GCASearchScorer, _density_log_terms
from fnit.recon_all.mri_em_register import StableSamples


class RecordedScorer:
    candidate_chunk = 3
    def __init__(self): self.blocks=[]
    def score_many(self, matrices):
        self.blocks.append(matrices.copy())
        return matrices[:,0,3]


def test_batch_retains_every_candidate_and_order():
    scorer=RecordedScorer()
    def candidates():
        for index in range(8):
            matrix=np.eye(4,dtype=np.float32);matrix[0,3]=index//2
            yield index,matrix
    result=list(score_candidates(None,None,candidates(),scorer))
    assert [int(row[0]) for row in result]==list(range(8))
    assert [float(row[1]) for row in result]==[0,0,1,1,2,2,3,3]
    assert [len(block) for block in scorer.blocks]==[3,3,2]
    maximum=-np.inf;winner=None
    for index,score in result:
        if score>maximum: maximum,winner=score,index
    assert winner==6


def test_empty_candidate_batch():
    scorer=RecordedScorer()
    assert list(score_candidates(None,None,iter(()),scorer))==[]
    assert not scorer.blocks


def test_gpu_backend_rejects_cpu_before_allocating():
    samples=StableSamples(np.zeros((1,3)),np.zeros(1),np.zeros(1),np.ones(1),np.ones(1))
    with unittest.TestCase().assertRaisesRegex(ValueError, "CUDA"):
        GCASearchScorer(samples,np.zeros((2,2,2),np.uint8),device="cpu")

def test_gpu_coordinates_ignore_default_dtype():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for default-dtype regression')
    from fnit.recon_all.mri_em_register_search_jit import log_sample_probability_jit
    samples=StableSamples(np.array([[1,1,0]],np.int32),np.array([2],np.int32),
                          np.array([100],np.float32),np.ones(1,np.float32),np.ones(1,np.float32))
    source=np.zeros((3,3,3),np.uint8);source[1,2,0]=100
    matrix=np.eye(4,dtype=np.float32);matrix[0,:]=[4,2**-25,0,0]
    expected=log_sample_probability_jit(samples,source,matrix)
    previous=torch.get_default_dtype()
    try:
        results=[]
        for dtype in (torch.float32,torch.float64):
            torch.set_default_dtype(dtype)
            scorer=GCASearchScorer(samples,source,device='cuda:0')
            results.append(float(scorer.score_many(matrix[None])[0]))
        assert results==[expected,expected]
    finally:torch.set_default_dtype(previous)


def test_cached_logs_match_numba_reference_not_python_libm():
    from fnit.recon_all.mri_em_register_search_jit import _sample_log_values
    variances=np.array([1.003,17.293,73.182],np.float32)
    priors=np.array([.873,.321,.652],np.float32)
    terms=_density_log_terms(variances,priors)
    means=np.full(3,100,np.float32)
    coordinates=np.zeros((3,3),np.int32)
    source=np.full((2,2,2),100,np.uint8)
    reference=_sample_log_values(coordinates,means,variances,priors,source,np.eye(4,dtype=np.float32))
    assert np.array_equal(terms[:,0]+terms[:,1],reference)


def test_device_reduction_retains_score_order_and_ties():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for reduction regression')
    rng=np.random.default_rng(73)
    coordinates=rng.integers(0,12,(300,3),dtype=np.int32)
    samples=StableSamples(coordinates,np.full(300,2,np.int32),
                          rng.uniform(30,160,300).astype(np.float32),
                          rng.uniform(1,120,300).astype(np.float32),
                          rng.uniform(.1,1,300).astype(np.float32))
    source=rng.integers(0,256,(24,24,24),dtype=np.uint8)
    matrices=np.repeat(np.eye(4,dtype=np.float32)[None],4,axis=0)
    matrices[1,0,3]=1
    matrices[2,1,3]=-1
    matrices[3]=matrices[1]
    ordered=GCASearchScorer(samples,source,device='cuda:0',candidate_chunk=2,sample_chunk=64)
    reduced=GCASearchScorer(samples,source,device='cuda:0',candidate_chunk=2,sample_chunk=64,
                            reduce_on_device=True)
    first=ordered.score_many(matrices)
    second=reduced.score_many(matrices)
    # Device reduction changes the summation tree for very negative
    # out-of-bounds scores; the score can differ by a few 1e-3 while candidate
    # ordering and ties remain stable.
    assert np.allclose(first,second,atol=1e-2,rtol=0)
    assert first[1]==first[3] and second[1]==second[3]
    assert int(np.argmax(first))==int(np.argmax(second))

if __name__ == "__main__":
    test_batch_retains_every_candidate_and_order()
    test_empty_candidate_batch()
    test_gpu_backend_rejects_cpu_before_allocating()
    test_cached_logs_match_numba_reference_not_python_libm()
    print("4 tests passed")
