"""完整优化器的协议单元测试；不作为真实精度 benchmark。"""
from types import SimpleNamespace
import numpy as np
from fnit.recon_all.mri_em_register_complete import align_em, _minimize

class Quadratic:
    def __init__(self):
        self.samples=SimpleNamespace(labels=np.ones(1,np.int32))
        self.weights=np.array([1.,4.,16.],np.float64)
        self.goal=np.array([2.,3.,4.],np.float64)
    def cost(self,matrix):
        values=matrix.diagonal()[:3].astype(np.float64)-self.goal
        return float(.5*np.sum(self.weights*values*values))
    def gradient(self,matrix):
        result=np.zeros((3,4),np.float32)
        for i in range(3):result[i,i]=self.weights[i]*(matrix[i,i]-self.goal[i])
        return result

def test_complete_lbfgs_advances_beyond_first_direction():
    objective=Quadratic()
    matrix,cost,report=_minimize(objective,np.eye(4,dtype=np.float32),1e-5)
    assert report['optimal_updates']>1
    assert cost<1e-8
    np.testing.assert_allclose(matrix.diagonal()[:3],objective.goal,atol=2e-5,rtol=0)
    assert report['termination'] in ('relative_gradient_tolerance','zero_gradient')

def test_complete_outer_protocol_preserves_affine_row():
    matrix,report=align_em(Quadratic(),np.eye(4,dtype=np.float32),tolerance=1e-5)
    assert matrix.dtype==np.float32
    np.testing.assert_array_equal(matrix[3],[0,0,0,1])
    assert report['sample_count']==1
    assert report['passes']
    assert report['outer_cost_after']<1e-8

def test_line_failure_retains_best_reported_trial():
    class Flat(Quadratic):
        def cost(self,matrix):return 1.0
    matrix,cost,report=_minimize(Flat(),np.eye(4,dtype=np.float32),1e-5)
    np.testing.assert_array_equal(matrix,np.eye(4,dtype=np.float32))
    assert cost==0.0 and report['best_cost']==1.0 and report['optimal_updates']==0
    assert not report['success']
    assert len(report['evaluations'])<=21


def test_missing_classifier_contributes_without_determinant_weight():
    from fnit.recon_all.mri_em_register_em_candidate import _accumulate_gradient
    voxels=np.array([[1,0,0],[2,0,0]],np.int32)
    gradient=_accumulate_gradient(voxels,np.ones((2,3),np.float32),
        np.array([10,0],np.float32),np.array([4,1],np.float32),
        np.array([3,5],np.float32),2,np.array([True,False]))
    expected=np.full((3,4),0,np.float32);expected[:,0]=np.float32(-8.25)*np.float32(1e-5)
    np.testing.assert_array_equal(gradient,expected)


def test_robust_histogram_rejects_nonunit_bin_spacing():
    from fnit.recon_all.mri_em_register import _robust_histogram_gaussian
    try:
        _robust_histogram_gaussian(np.array([0,2,4],np.float32),np.ones(3,np.float32))
    except ValueError:
        return
    raise AssertionError('nonunit fixed-input histogram spacing must fail explicitly')



def test_nonfinite_rejected_trial_preserves_source_line_search():
    class OutsideTrial:
        def cost(self,matrix):
            return 0.0 if np.array_equal(matrix,np.eye(4,dtype=np.float32)) else 10.0
        def gradient(self,matrix):
            return np.full((3,4),1.0 if self.cost(matrix)==0 else np.nan,np.float32)
    original=np.eye(4,dtype=np.float32)
    matrix,cost,report=_minimize(OutsideTrial(),original,1e-7)
    np.testing.assert_array_equal(matrix,original)
    assert report['optimal_updates']==0 and cost==0
    assert any(not row['gradient_finite'] for row in report['evaluations'])
    assert report['termination']=='lower_step_bound'


def test_invalid_affine_or_tolerance_never_evaluates_objective():
    class NoEvaluation:
        @property
        def samples(self):raise AssertionError('invalid input accessed objective')
        def cost(self,matrix):raise AssertionError('invalid input evaluated objective')
        def gradient(self,matrix):raise AssertionError('invalid input evaluated gradient')
    matrices=[]
    for value in (.5,1e-50):
        matrix=np.eye(4,dtype=np.float64);matrix[3,0]=value;matrices.append(matrix)
    for matrix in matrices:
        try:align_em(NoEvaluation(),matrix,tolerance=1e-5)
        except ValueError:pass
        else:raise AssertionError('noncanonical affine row accepted')
    for tolerance in (np.nan,np.inf,-np.inf,0.,-1.,1e-100,1e100):
        try:align_em(NoEvaluation(),np.eye(4),tolerance=tolerance)
        except ValueError:pass
        else:raise AssertionError('invalid FP32 tolerance accepted')


def test_disabled_scaling_low_intensity_preserves_GCA_CA_factor_one():
    from unittest.mock import patch
    import importlib
    preprocessing=importlib.import_module('fnit.recon_all.mri_em_register')
    ca=importlib.import_module('fnit.recon_all.ca_normalize_python')
    # Component contract only: broad, low-intensity uint8 region; not a benchmark.
    image=np.random.default_rng(9).integers(51,101,size=(64,64,64),dtype=np.uint8)
    sites=(None,np.array([2,41],np.int32),None)
    profile={}
    with patch.object(preprocessing,'find_all_sample_sites',return_value=sites),patch.object(preprocessing,'atlas_label_peak',return_value=107):
        peak,_,_=preprocessing.estimate_image_white_matter_peak(object(),image,profile=profile)
    assert profile['scaling_disabled'] and profile['robust_sigma']>profile['width_threshold']
    assert peak==107 and image.max()<peak
    assert ca.estimate_image_white_matter_peak is preprocessing.estimate_image_white_matter_peak
    np.testing.assert_array_equal(ca.scale_masked_input(image,107,peak),image)
