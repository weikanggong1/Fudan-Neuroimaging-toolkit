"""GPU融合必须保留原传播/卷积算术顺序，覆盖边界和非连续输入。"""
import numpy as np
import pytest
import torch
from fnit.recon_all.normalization.normalize_voronoi_source import voronoi_fill,voronoi_fill_torch
from fnit.recon_all.normalization.normalize_gaussian_source import fs_gaussian_kernel,smooth_bias_torch
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")

@pytest.mark.parametrize("shape",[(2,3,4),(9,11,7),(1,7,5)])
def test_wavefront_exact_order_and_boundary_repetition(shape):
    rng=np.random.default_rng(17)
    values=rng.uniform(1,137,shape).astype(np.float32)
    control=np.zeros(shape,bool);control[0,0,0]=True;control[-1,-1,-1]=True
    expected,_=voronoi_fill(values,control)
    actual,_=voronoi_fill_torch(torch.tensor(values,device="cuda"),torch.tensor(control,device="cuda"))
    np.testing.assert_array_equal(actual.cpu().numpy(),expected)

@pytest.mark.parametrize("sigma",[1.,2.,8.])
def test_smoothing_exact_legacy_torch_and_noncontiguous_input(sigma):
    torch.manual_seed(7)
    values=torch.rand((7,9,11),device="cuda").transpose(0,2)
    source=values*110
    control=torch.zeros_like(values,dtype=torch.uint8);control[0,0,0]=1
    expected=values.float()
    kernel=torch.tensor(fs_gaussian_kernel(sigma),device="cuda")
    for axis in (0,1,2):
        index=torch.arange(expected.shape[axis],device="cuda")
        result=torch.zeros_like(expected)
        for offset in range(len(kernel)):
            neighbor=(index+offset-len(kernel)//2).clamp(0,len(index)-1)
            result+=expected.index_select(axis,neighbor)*kernel[offset]
        expected=result
    expected[control>0]=source[control>0]
    actual,_=smooth_bias_torch(values,source,control,sigma=sigma)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
