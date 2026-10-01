"""Focused CUDA acceptance for optional fixed-order mesh gradients.

This test compares real fused
kernel math to the existing autograd oracle and checks repeated bitwise identity;
these small meshes are unit tests and are not a neuroimaging benchmark.
"""
import numpy as np
import pytest
import torch
from fnit.gems import _raster_triton
from fnit.gems.deformation import (prepare_current_geometry,prepare_deformation_reference,
                                  prepare_vertex_reduction,ashburner_prior)
from fnit.gems.rasterize import build_block_index,compact_data_cost,rasterize_priors_compact

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or _raster_triton.triton is None,
                             reason='CUDA and Triton required')

@pytest.fixture(autouse=True)
def restore_tf32():
    previous = torch.backends.cuda.matmul.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _inputs(classes=3):
    v=torch.tensor([[1.137,1.211,1.317],[15.137,1.211,1.317],[1.137,15.211,1.317],
                    [1.137,1.211,15.317],[15.137,15.211,15.317]],device='cuda')
    tetra=torch.tensor([[0,1,2,3],[4,1,2,3]],device='cuda')
    a=torch.arange(1,5*classes+1,device='cuda',dtype=torch.float32).reshape(5,classes)
    a=a/a.sum(1,keepdim=True);shape=(19,18,17)
    mask=torch.arange(np.prod(shape),device='cuda').reshape(shape)%3!=0
    index=build_block_index(v.cpu().numpy(),tetra.cpu().numpy(),shape,block_size=4)
    ll=-torch.linspace(.1,12,classes*int(mask.sum()),device='cuda').reshape(classes,-1)
    return v,tetra,a,shape,mask,index,ll

@pytest.mark.parametrize('double',[False,True])
@pytest.mark.parametrize('classes',[3,19])
@pytest.mark.parametrize('tf32',[False,True])
def test_ordered_fused_cost_gradient_repeats_and_matches_reference(double,classes,tf32):
    torch.backends.cuda.matmul.allow_tf32=tf32
    initial,tetra,a,shape,mask,index,ll=_inputs(classes)
    layout=prepare_vertex_reduction(tetra.reshape(-1),len(initial));ref=prepare_deformation_reference(initial,tetra)
    def evaluate(ordered,fused,dtype=torch.float32):
        base=initial.to(dtype)
        v=(base+base.new_tensor([[.02,-.03,.01],[.15,-.04,.06],[-.03,.05,.04],[.02,.03,-.08],[.03,-.02,.03]])).detach().requires_grad_(True)
        geometry=prepare_current_geometry(v,tetra,deterministic_gradient=ordered,
                                          vertex_reduction=layout if ordered else None)
        if fused:
            data=compact_data_cost(v,tetra,a,shape,valid_mask=mask,block_index=index,
                likelihood=ll,current_geometry=geometry,double_accumulation=double,
                deterministic_gradient=ordered)
            assert data is not None
        else:
            priors,_=rasterize_priors_compact(v,tetra,a.to(dtype),shape,valid_mask=mask,
                                             block_index=index,current_geometry=geometry)
            data=-(priors.clamp_min(torch.finfo(priors.dtype).tiny).log()+ll.to(dtype)).logsumexp(0).sum(
                dtype=torch.float64 if double else dtype)
        reference_geometry=ref if dtype==torch.float32 else prepare_deformation_reference(base,tetra)
        prior,_=ashburner_prior(v,base,tetra,.05,current_geometry=geometry,
                               reference_geometry=reference_geometry,analytic_gradient=True,double_accumulation=double)
        cost=data+prior;gradient,=torch.autograd.grad(cost,v)
        return cost.detach(),gradient.detach()
    reference=evaluate(False,False);atomic=evaluate(False,True);ordered=evaluate(True,True)
    torch.testing.assert_close(ordered[0],reference[0],atol=.02,rtol=2e-6)
    assert torch.equal(ordered[0],atomic[0])
    relative=torch.linalg.vector_norm(ordered[1]-reference[1])/torch.linalg.vector_norm(reference[1])
    assert float(relative)<2e-4
    assert ordered[1].dtype==torch.float32
    reference64=evaluate(False,False,torch.float64)
    relative64=torch.linalg.vector_norm(ordered[1].double()-reference64[1])/torch.linalg.vector_norm(reference64[1])
    if tf32:
        # The existing TF32 geometry matrix products also differ from full
        # FP64. Ordered accumulation must not worsen that arithmetic error.
        reference_error=torch.linalg.vector_norm(reference[1].double()-reference64[1])/torch.linalg.vector_norm(reference64[1])
        assert float(relative64)<=float(reference_error)+2e-4
    else:
        assert float(relative64)<2e-4
    for _ in range(8):
        again=evaluate(True,True)
        assert torch.equal(again[0],ordered[0]) and torch.equal(again[1],ordered[1])

def test_ordered_fused_all_uncovered_points_give_zero_gradient():
    initial,tetra,a,shape,_,index,_=_inputs()
    mask=torch.zeros(shape,device='cuda',dtype=torch.bool);mask[-1,-1,-1]=True
    ll=torch.zeros((a.shape[1],1),device='cuda');v=initial.clone().requires_grad_(True)
    cost=compact_data_cost(v,tetra,a,shape,valid_mask=mask,block_index=index,likelihood=ll,
                          deterministic_gradient=True)
    gradient,=torch.autograd.grad(cost,v)
    assert float(cost)==0 and not torch.count_nonzero(gradient)


def test_sparse_mask_padding_has_complete_unique_packed_mapping():
    # Irregular sparse masks generate padded lookup blocks plus missing-cell
    # rows. Every retained packed point must be written exactly once.
    initial,tetra,a,_,_,_,_=_inputs()
    shape=(27,26,25)
    index=build_block_index(initial.cpu().numpy(),tetra.cpu().numpy(),shape,block_size=4)
    mask=torch.zeros(shape,device='cuda',dtype=torch.bool)
    mask[2:11:2,2:13:3,2:10:2]=True
    mask[3,3,3]=True
    mask[2,2,3]=True
    mask[-1,-1,-1]=True
    batches,reorder=index.device_compact_batches(mask,initial.device,initial.dtype)
    packed_count=sum(rows.numel() for _,_,_,_,rows in batches)
    mapping=reorder[reorder>=0]
    assert torch.equal(mapping.sort().values,torch.arange(packed_count,device='cuda'))
    assert (reorder<0).any()
    assert any(rows.numel()<points.shape[0]*points.shape[1] for points,_,_,_,rows in batches)
    ll=-torch.linspace(.1,2,a.shape[1]*int(mask.sum()),device='cuda').reshape(a.shape[1],-1)
    def evaluate():
        v=initial.clone().requires_grad_(True)
        geometry=prepare_current_geometry(v,tetra,deterministic_gradient=True)
        cost=compact_data_cost(v,tetra,a,shape,valid_mask=mask,block_index=index,likelihood=ll,
                              current_geometry=geometry,deterministic_gradient=True)
        return cost.detach(),torch.autograd.grad(cost,v)[0]
    cost,gradient=evaluate()
    assert torch.isfinite(cost) and torch.isfinite(gradient).all()
    for _ in range(8):
        again=evaluate()
        assert torch.equal(again[0],cost) and torch.equal(again[1],gradient)
