"""Independent all26 CPU allocation prototype gates, never performance evidence."""
import numpy as np
import pytest
import torch
from fnit.convertwarp import core as convert
from fnit.invwarp import core as inverse


def bits(value):
    return value.detach().resolve_neg().contiguous().view(torch.int64 if value.dtype==torch.float64 else torch.int32)


@pytest.mark.parametrize('threads',[1,8])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('case',['zero','finite','subnormal','inf','nan','nan_payload','tolerance_ulp','noncontiguous'])
def test_stopping_scalar_exact_bits_and_decision(threads,dtype,case):
    previous=torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        data=torch.tensor([-0.,0.,-0.,0.],dtype=dtype)
        if case=='finite': data=torch.tensor([-1.1,-.001,2.7,1.],dtype=dtype)
        elif case=='subnormal': data=torch.tensor([np.nextafter(np.dtype(str(dtype).split('.')[-1]).type(0),np.dtype(str(dtype).split('.')[-1]).type(1)),-0.,0.],dtype=dtype)
        elif case=='inf': data=torch.tensor([-float('inf'),0.,float('inf')],dtype=dtype)
        elif case=='nan': data=torch.tensor([-0.,float('nan'),1.],dtype=dtype)
        elif case=='nan_payload':
            raw=np.array([0x7ff8000000000037,0xfff8000000000025],dtype=np.uint64) if dtype==torch.float64 else np.array([0x7fc00037,0xffc00025],dtype=np.uint32)
            data=torch.from_numpy(raw.view(np.float64 if dtype==torch.float64 else np.float32))
        elif case=='tolerance_ulp':
            value=torch.tensor(.01,dtype=dtype); data=torch.stack((torch.nextafter(value,torch.tensor(0.,dtype=dtype)),value,-torch.nextafter(value,torch.tensor(1.,dtype=dtype))))
        elif case=='noncontiguous': data=torch.arange(64,dtype=dtype).reshape(8,8).t()[::2]
        expected=data.abs().max(); actual=inverse._correction_max_abs(data)
        assert torch.equal(bits(expected),bits(actual))
        for tolerance in (.01,np.nextafter(.01,0.),np.nextafter(.01,1.),1.,100.):
            assert (float(expected)<tolerance)==(float(actual)<tolerance)
    finally:torch.set_num_threads(previous)


def test_stop_gradient_and_functional_transforms_keep_original(monkeypatch):
    original=torch.aminmax
    def forbidden(*args,**kwargs):raise AssertionError('aminmax changed grad/func path')
    monkeypatch.setattr(torch,'aminmax',forbidden)
    data=torch.tensor([-1.,2.,-3.],dtype=torch.float64,requires_grad=True)
    value=inverse._correction_max_abs(data); value.backward()
    assert torch.equal(data.grad,torch.tensor([0.,0.,-1.],dtype=torch.float64))
    def function(x):return inverse._correction_max_abs(x)
    assert torch.equal(torch.func.vmap(function)(data.detach().repeat(2,1)),torch.tensor([3.,3.],dtype=torch.float64))
    assert torch.equal(torch.func.grad(function)(data.detach()),torch.tensor([0.,0.,-1.],dtype=torch.float64))
    monkeypatch.setattr(torch,'aminmax',original)


@pytest.mark.parametrize('threads',[1,8])
@pytest.mark.parametrize('case',['finite','signed_zero','nonfinite_query','nonfinite_sample','strided','negative_view'])
def test_prepared_embedded_base_keeps_bits_and_no_alias(monkeypatch,threads,case):
    previous=torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        query=torch.randn((3,4,5,6),dtype=torch.float64,generator=torch.Generator().manual_seed(825))
        sampled=torch.randn_like(query)
        matrix=torch.tensor([[1.03,.02,-.001,.3],[-.02,.97,.001,-.4],[.03,.04,1.02,.1],[0.,0.,0.,1.]],dtype=torch.float64)
        if case=='signed_zero':query.zero_();query[0]=-0.;sampled.zero_();sampled[0]=-0.;matrix[:3,3]=-0.
        elif case=='nonfinite_query':query.reshape(-1)[:4]=torch.tensor([float('nan'),float('inf'),-float('inf'),-0.])
        elif case=='nonfinite_sample':sampled.reshape(-1)[:4]=torch.tensor([float('nan'),float('inf'),-float('inf'),-0.])
        elif case=='strided':query=query.transpose(1,2);sampled=sampled.transpose(1,2)
        elif case=='negative_view':query=torch._neg_view(query)
        field=object.__new__(convert._PullField);field.values=torch.zeros((3,7,9,11),dtype=torch.float64)
        field.scaled_inverse=torch.eye(4,dtype=torch.float64);field.embedded_inverse=matrix;field.convention='relative'
        valid=torch.ones(query.shape[1:],dtype=torch.bool)
        monkeypatch.setattr(convert,'_sample_linear',lambda *args,**kw:(sampled,valid))
        prepared=field.values[None].contiguous(memory_format=torch.channels_last_3d)
        qbefore=bits(query).clone(); sbefore=bits(sampled).clone(); mbefore=bits(matrix).clone()
        def old():return (matrix[:3,:3]@query.reshape(3,-1)+matrix[:3,3:4]).reshape(query.shape)+sampled.to(torch.float64)
        expected=old(); first,actual_valid=field.sample(query,prepared_source=prepared); saved=bits(first).clone()
        second,_=field.sample(query,prepared_source=prepared)
        assert torch.equal(bits(first),bits(expected)) and torch.equal(bits(second),bits(expected))
        assert torch.equal(bits(first),saved) and first.data_ptr()!=second.data_ptr()
        for input_value in (query,sampled,matrix,prepared,field.values):
            assert first.untyped_storage().data_ptr()!=input_value.untyped_storage().data_ptr()
        assert torch.equal(bits(query),qbefore) and torch.equal(bits(sampled),sbefore) and torch.equal(bits(matrix),mbefore)
        assert actual_valid is valid
    finally:torch.set_num_threads(previous)


@pytest.mark.parametrize('requires',['query','values','matrix','prepared'])
def test_prepared_base_gradient_stays_functional(monkeypatch,requires):
    query=torch.randn((3,2,3,4),dtype=torch.float64,requires_grad=requires=='query')
    values=torch.ones_like(query,requires_grad=requires=='values')
    matrix=torch.eye(4,dtype=torch.float64,requires_grad=requires=='matrix')
    field=object.__new__(convert._PullField);field.values=values;field.scaled_inverse=torch.eye(4,dtype=torch.float64);field.embedded_inverse=matrix;field.convention='relative'
    source=values[None].contiguous(memory_format=torch.channels_last_3d)
    if requires=='prepared':source=source.detach().requires_grad_()
    sampled=source[0] if requires=='prepared' else values
    monkeypatch.setattr(convert,'_sample_linear',lambda *args,**kw:(sampled,None))
    expected=(matrix[:3,:3]@query.reshape(3,-1)+matrix[:3,3:4]).reshape(query.shape)+sampled
    actual,_=field.sample(query,prepared_source=source)
    assert torch.equal(bits(expected),bits(actual))
    actual.sum().backward()
    leaf={'query':query,'values':values,'matrix':matrix,'prepared':source}[requires]
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_stopping_uses_original_operator(monkeypatch):
    data=torch.tensor([-0.,1.,-3.],dtype=torch.float64,device='cuda')
    def forbidden(*args,**kwargs):raise AssertionError('CUDA aminmax used')
    monkeypatch.setattr(torch,'aminmax',forbidden)
    assert torch.equal(bits(inverse._correction_max_abs(data)),bits(data.abs().max()))


@pytest.mark.parametrize('base_dtype,sampled_dtype',[(torch.float32,torch.float64),(torch.float32,torch.float32),(torch.float64,torch.float32)])
def test_mixed_dtype_keeps_functional_result_promotion(monkeypatch,base_dtype,sampled_dtype):
    query=torch.arange(3*2*3*4,dtype=base_dtype).reshape(3,2,3,4)/7
    sampled=torch.randn(query.shape,dtype=sampled_dtype)
    field=object.__new__(convert._PullField);field.values=sampled;field.scaled_inverse=torch.eye(4,dtype=base_dtype);field.embedded_inverse=torch.eye(4,dtype=base_dtype);field.convention='relative'
    source=sampled[None].contiguous(memory_format=torch.channels_last_3d)
    monkeypatch.setattr(convert,'_diagonal_scaled_coordinates_cpu',lambda *args:None)
    monkeypatch.setattr(convert,'_sample_linear',lambda *args,**kw:(sampled,None))
    matrix=field.embedded_inverse
    expected=(matrix[:3,:3]@query.reshape(3,-1)+matrix[:3,3:4]).reshape(query.shape)+sampled.to(torch.float64)
    def forbidden(*args,**kwargs):raise AssertionError('in-place add changed mixed dtype path')
    monkeypatch.setattr(torch.Tensor,'add_',forbidden)
    actual,_=field.sample(query,prepared_source=source)
    assert actual.dtype==expected.dtype==torch.float64 and torch.equal(bits(actual),bits(expected))


def test_stopping_forward_dual_keeps_original(monkeypatch):
    primal=torch.tensor([1.,-3.,2.],dtype=torch.float64)
    tangent=torch.tensor([.5,.7,.9],dtype=torch.float64)
    def forbidden(*args,**kwargs):raise AssertionError('aminmax changed forward dual route')
    with torch.autograd.forward_ad.dual_level():
        dual=torch.autograd.forward_ad.make_dual(primal,tangent)
        expected=torch.autograd.forward_ad.unpack_dual(dual.abs().max())
        monkeypatch.setattr(torch,'aminmax',forbidden)
        actual=torch.autograd.forward_ad.unpack_dual(inverse._correction_max_abs(dual))
        assert torch.equal(bits(actual.primal),bits(expected.primal))
        assert torch.equal(bits(actual.tangent),bits(expected.tangent))


@pytest.mark.parametrize('which',['query','values','matrix','prepared'])
def test_embedded_forward_dual_keeps_functional_route(monkeypatch,which):
    query=torch.randn((3,2,3,4),dtype=torch.float64)
    values=torch.randn_like(query);matrix=torch.eye(4,dtype=torch.float64)
    source=values[None].contiguous(memory_format=torch.channels_last_3d)
    with torch.autograd.forward_ad.dual_level():
        if which=='query':query=torch.autograd.forward_ad.make_dual(query,torch.ones_like(query))
        elif which=='values':values=torch.autograd.forward_ad.make_dual(values,torch.ones_like(values));source=values[None].contiguous(memory_format=torch.channels_last_3d)
        elif which=='matrix':matrix=torch.autograd.forward_ad.make_dual(matrix,torch.ones_like(matrix))
        else:source=torch.autograd.forward_ad.make_dual(source,torch.ones_like(source))
        field=object.__new__(convert._PullField);field.values=values;field.scaled_inverse=torch.eye(4,dtype=torch.float64);field.embedded_inverse=matrix;field.convention='relative'
        sampled=source[0] if which=='prepared' else values
        monkeypatch.setattr(convert,'_diagonal_scaled_coordinates_cpu',lambda *args:None)
        monkeypatch.setattr(convert,'_sample_linear',lambda *args,**kw:(sampled,None))
        expected=torch.autograd.forward_ad.unpack_dual((matrix[:3,:3]@query.reshape(3,-1)+matrix[:3,3:4]).reshape(query.shape)+sampled.to(torch.float64))
        def forbidden(*args,**kwargs):raise AssertionError('in-place add changed forward dual route')
        monkeypatch.setattr(torch.Tensor,'add_',forbidden)
        actual=torch.autograd.forward_ad.unpack_dual(field.sample(query,prepared_source=source)[0])
        assert torch.equal(bits(actual.primal),bits(expected.primal)) and torch.equal(bits(actual.tangent),bits(expected.tangent))


def test_embedded_func_transform_keeps_functional_route(monkeypatch):
    query=torch.randn((2,3,2,3,4),dtype=torch.float64)
    sampled=torch.randn((3,2,3,4),dtype=torch.float64)
    field=object.__new__(convert._PullField);field.values=sampled;field.scaled_inverse=torch.eye(4,dtype=torch.float64);field.embedded_inverse=torch.eye(4,dtype=torch.float64);field.convention='relative'
    source=sampled[None].contiguous(memory_format=torch.channels_last_3d)
    monkeypatch.setattr(convert,'_diagonal_scaled_coordinates_cpu',lambda *args:None)
    monkeypatch.setattr(convert,'_sample_linear',lambda *args,**kw:(sampled,None))
    def original(q):return (field.embedded_inverse[:3,:3]@q.reshape(3,-1)+field.embedded_inverse[:3,3:4]).reshape(q.shape)+sampled
    def candidate(q):return field.sample(q,prepared_source=source)[0]
    def forbidden(*args,**kwargs):raise AssertionError('in-place add changed func transform path')
    monkeypatch.setattr(torch.Tensor,'add_',forbidden)
    assert torch.equal(bits(torch.func.vmap(original)(query)),bits(torch.func.vmap(candidate)(query)))
    a=torch.func.jvp(original,(query[0],),(torch.ones_like(query[0]),))
    b=torch.func.jvp(candidate,(query[0],),(torch.ones_like(query[0]),))
    assert all(torch.equal(bits(x),bits(y)) for x,y in zip(a,b))
