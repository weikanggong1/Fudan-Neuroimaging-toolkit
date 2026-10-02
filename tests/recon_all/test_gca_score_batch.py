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

if __name__ == "__main__":
    test_batch_retains_every_candidate_and_order()
    test_empty_candidate_batch()
    test_gpu_backend_rejects_cpu_before_allocating()
    test_cached_logs_match_numba_reference_not_python_libm()
    print("4 tests passed")
