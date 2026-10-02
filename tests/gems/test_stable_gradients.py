import pytest
import torch
from fnit.gems import deformation as module

def _mesh(dtype=torch.float64,device='cpu'):
    reference=torch.tensor([[0.,0,0],[2,0,0],[0,2,0],[0,0,2],[2,2,2],[9,9,9]],dtype=dtype,device=device)
    tetra=torch.tensor([[0,1,2,3],[4,1,2,3],[0,1,2,3]],device=device)
    actual=reference+reference.new_tensor([[.02,-.03,.01],[.15,-.04,.06],[-.03,.05,.04],[.02,.03,-.08],[.03,-.02,.03],[0,0,0]])
    return reference,actual,tetra

def test_fp64_reduction_preserves_fp32_output_and_isolated_vertex():
    ids=torch.tensor([2,0,2,0,2,1])
    values=torch.tensor([[1e10,3,1],[-1e10,1,2],[1,4,2],[1e10,2,3],[-1e10,5,3],[7,8,9]],dtype=torch.float32)
    expected=torch.zeros((4,3),dtype=torch.float64)
    for i,row in zip(ids.tolist(),values):expected[i]=expected[i]+row.double()
    for _ in range(3):
        result=module.ordered_vertex_sum(ids,values,4)
        assert result.dtype==torch.float32 and torch.equal(result,expected.float())
    assert torch.equal(result[3],torch.zeros(3))

def test_zero_contributions_and_empty_ids():
    result=module.ordered_vertex_sum(torch.empty(0,dtype=torch.long),torch.empty((0,3)),5)
    assert result.shape==(5,3) and not torch.count_nonzero(result)

def test_vertex_layout_rejects_inplace_topology_mutation():
    ids=torch.tensor([0,1,0,2]);cache=module.prepare_vertex_reduction(ids,3)
    module.ordered_vertex_sum(ids,torch.ones((4,3)),3,cache)
    ids[0]=1
    with pytest.raises(ValueError,match='rebuild'):
        module.ordered_vertex_sum(ids,torch.ones((4,3)),3,cache)

@pytest.mark.parametrize('analytic',[False,True])
def test_prior_and_jacobian_gradient_matches_autograd_with_shared_vertices(analytic):
    reference,initial,tetra=_mesh();layout=module.prepare_vertex_reduction(tetra.reshape(-1),len(initial));cache=module.prepare_deformation_reference(reference,tetra)
    def evaluate(ordered):
        vertices=initial.clone().requires_grad_(True)
        geometry=module.prepare_current_geometry(vertices,tetra,deterministic_gradient=ordered,vertex_reduction=layout if ordered else None)
        cost,jac=module.ashburner_prior(vertices,reference,tetra,.05,current_geometry=geometry,reference_geometry=cache,analytic_gradient=analytic)
        g,=torch.autograd.grad(cost+.13*jac.sum(),vertices)
        return cost.detach(),jac.detach(),g
    a,b=evaluate(False),evaluate(True)
    for actual,expected in zip(b,a):torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    assert torch.equal(b[2][-1],torch.zeros(3,dtype=b[2].dtype))
    for _ in range(3):
        for actual,expected in zip(evaluate(True),b):assert torch.equal(actual,expected)

def test_ordered_geometry_passes_double_gradient_check():
    reference,initial,tetra=_mesh();layout=module.prepare_vertex_reduction(tetra.reshape(-1),len(initial));cache=module.prepare_deformation_reference(reference,tetra)
    def value(v):
        geometry=module.prepare_current_geometry(v,tetra,deterministic_gradient=True,vertex_reduction=layout)
        cost,jac=module.ashburner_prior(v,reference,tetra,.05,current_geometry=geometry,reference_geometry=cache,analytic_gradient=True)
        return cost+.13*jac.sum()
    assert torch.autograd.gradcheck(value,(initial.requires_grad_(True),),eps=1e-6,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('analytic',[False,True])
@pytest.mark.parametrize('invalid',[False,True])
def test_double_prior_only_promotes_sum_and_keeps_fp32_gradient(analytic,invalid):
    reference,actual,tetra=_mesh(torch.float32)
    if invalid:
        actual[1]=2*actual[0]-actual[1]
    # Independent single-cell costs define the reduction oracle. Repetition
    # makes a rounded FP32 global sum differ from the FP64 sum of FP32 terms.
    repeated=tetra[:2].repeat(1025,1)
    single=[]
    for cell in tetra[:2]:
        cost,_=module.ashburner_prior(actual,reference,cell[None],.05,analytic_gradient=analytic)
        single.append(cost)
    expected=torch.stack(single).double().sum()*1025
    results=[]
    for double in [False,True]:
        v=actual.clone().requires_grad_(True)
        layout=module.prepare_vertex_reduction(repeated.reshape(-1),len(v))
        geometry=module.prepare_current_geometry(v,repeated,deterministic_gradient=True,vertex_reduction=layout)
        cost,jac=module.ashburner_prior(v,reference,repeated,.05,current_geometry=geometry,
                     analytic_gradient=analytic,double_accumulation=double)
        gradient,=torch.autograd.grad(cost,v)
        assert cost.dtype==(torch.float64 if double else torch.float32)
        assert jac.dtype==gradient.dtype==v.dtype==torch.float32
        assert torch.isfinite(cost) and torch.isfinite(gradient).all()
        results.append((cost.detach(),jac.detach(),gradient))
    torch.testing.assert_close(results[1][0],expected,rtol=1e-6,atol=1e-8)
    assert torch.equal(results[0][1],results[1][1])
    assert torch.equal(results[0][2],results[1][2])
    if invalid:assert (results[1][1]<=0).any()


def test_empty_prior_double_sum_still_returns_fp32_zero_gradient():
    reference,actual,_=_mesh(torch.float32)
    v=actual.clone().requires_grad_(True)
    cost,jac=module.ashburner_prior(v,reference,torch.empty((0,4),dtype=torch.long),.05,
                                 double_accumulation=True,analytic_gradient=True)
    gradient,=torch.autograd.grad(cost,v)
    assert cost.dtype==torch.float64 and float(cost)==0
    assert jac.dtype==gradient.dtype==torch.float32 and jac.numel()==0
    assert not torch.count_nonzero(gradient)


@pytest.mark.parametrize('channels',[1,7,19])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fixed_order_sum_supports_alpha_statistics_and_fp64_reference(channels,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    ids=torch.tensor([2,0,2,0,2,1,2,0],device=device)
    factors=torch.arange(1,channels+1,dtype=torch.float32,device=device)
    values=torch.tensor([1e10,-1e10,1,1e10,-1e10,7,3,-2],device=device)[:,None]*factors[None]
    cache=module.prepare_vertex_reduction(ids,4)
    # Small independent per-vertex FP64 reference, including cancellation.
    expected=torch.zeros((4,channels),device=device,dtype=torch.float64)
    for i in range(ids.numel()):expected[ids[i]]=expected[ids[i]]+values[i].double()
    for _ in range(8):
        result=module.ordered_vertex_sum(ids,values,4,cache)
        assert result.dtype==torch.float32 and torch.equal(result,expected.float())
    assert torch.equal(result[3],torch.zeros(channels,device=device))


def test_ordered_sum_rejects_wrong_layout_shape():
    ids=torch.tensor([0,1]);values=torch.zeros((3,7))
    with pytest.raises(ValueError,match='channels'):module.ordered_vertex_sum(ids,values,2)



def test_vertex_layout_rejects_same_pointer_with_different_stride():
    storage=torch.tensor([0,1,2,3])
    ids=storage[:2];other=storage[::2]
    cache=module.prepare_vertex_reduction(ids,4)
    assert ids.data_ptr()==other.data_ptr() and ids.shape==other.shape
    with pytest.raises(ValueError,match='rebuild'):
        module.ordered_vertex_sum(other,torch.ones((2,7)),4,cache)
