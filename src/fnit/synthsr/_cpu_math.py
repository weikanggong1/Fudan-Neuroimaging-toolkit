"""Lazy CPU FP32 numerical kernels for the verified SynthSR reference path.

The oneDNN 2.7.3 ELU polynomial uses explicit FMA. Eigen BN uses the SSE
rsqrt seed followed by strictly separate FP32 Newton and affine operations.
These independently written formulas were checked on complete real layers and
both complete CNN outputs; TensorFlow is not a runtime dependency.

Formula provenance:
https://github.com/oneapi-src/oneDNN/blob/v2.7.3/src/cpu/x64/injectors/jit_uni_eltwise_injector.cpp
The public dispatch checks Linux x86-64/SSE2/FMA before importing this module.
"""
import numpy as np
from llvmlite import ir
from numba import njit, prange, types
from numba.extending import intrinsic


@intrinsic
def _fma32(typing_context, first, second, third):
    if any(value != types.float32 for value in (first, second, third)):
        raise TypeError('all operands of this explicit FMA must be FP32')
    signature = types.float32(first, second, third)

    def codegen(context, builder, sig, args):
        function_type = ir.FunctionType(ir.FloatType(), [ir.FloatType()]*3)
        function = builder.module.globals.get('llvm.fma.f32')
        if function is None:
            function = ir.Function(builder.module, function_type, name='llvm.fma.f32')
        return builder.call(function, args)

    return signature, codegen


def _bits(number):
    return np.array(number, dtype=np.uint32).view(np.float32).item()


LOG2E = np.float32(_bits(0x3fb8aa3b))
LN2 = np.float32(_bits(0x3f317218))
LN_MIN = np.float32(_bits(0xc2aeac50))
P1 = np.float32(_bits(0x3f7ffffb))
P2 = np.float32(_bits(0x3efffee3))
P3 = np.float32(_bits(0x3e2aad40))
P4 = np.float32(_bits(0x3d2b9d0d))
P5 = np.float32(_bits(0x3c07cfce))


@njit(parallel=True, fastmath=False, cache=True)
def _elu_flat(source, result):
    for index in prange(source.size):
        value = source[index]
        if value > np.float32(0):
            result[index] = value
        elif value < LN_MIN:
            result[index] = np.float32(-1)
        else:
            power = np.floor(value * LOG2E + np.float32(.5))
            remainder = _fma32(-power, LN2, value)
            polynomial = _fma32(P5, remainder, P4)
            polynomial = _fma32(polynomial, remainder, P3)
            polynomial = _fma32(polynomial, remainder, P2)
            polynomial = _fma32(polynomial, remainder, P1)
            polynomial = _fma32(polynomial, remainder, np.float32(1))
            # oneDNN multiplies by 2^(power-1) and then by two, separately.
            scaling = np.ldexp(np.float32(.5), np.int32(power))
            exponential = polynomial * scaling
            exponential = exponential * np.float32(2)
            result[index] = exponential - np.float32(1)


def _elu_tensor(value):
    import torch
    if value.device.type != 'cpu' or value.dtype != torch.float32 or value.requires_grad:
        raise ValueError('SynthSR CPU ELU requires CPU FP32 without autograd')
    result = torch.empty_like(value)
    if value.is_contiguous(memory_format=torch.channels_last_3d):
        source = value.permute(0,2,3,4,1).numpy().reshape(-1)
        output = result.permute(0,2,3,4,1).numpy().reshape(-1)
    elif value.is_contiguous():
        source = value.numpy().reshape(-1)
        output = result.numpy().reshape(-1)
    else:
        raise ValueError('SynthSR CPU ELU requires contiguous or channels-last 3D input')
    _elu_flat(source, output)
    return result


@intrinsic
def _rsqrt_seed(typingctx, value):
    if value != types.float32:
        return None
    signature = types.float32(types.float32)
    def codegen(context, builder, sig, args):
        vector = ir.VectorType(ir.FloatType(), 4)
        argument = builder.insert_element(ir.Constant(vector, ir.Undefined), args[0], ir.Constant(ir.IntType(32), 0))
        declaration = ir.FunctionType(vector, [vector])
        name = 'llvm.x86.sse.rsqrt.ss'
        function = builder.module.globals.get(name)
        if function is None:
            function = ir.Function(builder.module, declaration, name=name)
        output = builder.call(function, [argument])
        return builder.extract_element(output, ir.Constant(ir.IntType(32), 0))
    return signature, codegen


@njit(fastmath=False, cache=True)
def inverse_factors(variance, epsilon):
    result=np.empty_like(variance)
    for i in range(variance.size):
        value=variance[i]+epsilon
        seed=_rsqrt_seed(value)
        half=seed*np.float32(-.5)
        product=value*seed
        error=product*seed-np.float32(1)
        result[i]=error*half+seed
    return result


@njit(parallel=True,fastmath=False,cache=True)
def _affine_channels(source, target, scale, offset):
    for channel in prange(source.shape[0]):
        s=scale[channel]
        b=offset[channel]
        for i in range(source.shape[1]):
            target[channel,i]=source[channel,i]*s+b


@njit(parallel=True,fastmath=False,cache=True)
def _affine_interleaved(source, target, scale, offset):
    for i in prange(source.shape[0]):
        for channel in range(source.shape[1]):
            target[i,channel]=source[i,channel]*scale[channel]+offset[channel]


def _batch_norm_tensor(value, norm):
    """CPU-only SynthSR CPU affine; rejects autograd and unsupported layouts."""
    import torch
    if value.device.type!='cpu' or value.dtype!=torch.float32 or value.requires_grad:
        raise ValueError('SynthSR CPU BN requires CPU FP32 inference')
    if value.shape[0]!=1:
        raise ValueError('stage SynthSR CPU currently supports one volume')
    gamma=norm.weight.detach().numpy();beta=norm.bias.detach().numpy();mean=norm.running_mean.numpy()
    inverse=inverse_factors(norm.running_var.numpy(),np.float32(norm.eps))
    scale=gamma*inverse;offset=beta-mean*scale
    result=torch.empty_like(value)
    channels=value.shape[1]
    if value.is_contiguous(memory_format=torch.channels_last_3d):
        source=value.permute(0,2,3,4,1).numpy().reshape(-1,channels)
        target=result.permute(0,2,3,4,1).numpy().reshape(-1,channels)
        _affine_interleaved(source,target,scale,offset)
    elif value.is_contiguous():
        _affine_channels(value.numpy().reshape(channels,-1),result.numpy().reshape(channels,-1),scale,offset)
    else:
        raise ValueError('SynthSR CPU BN requires contiguous or channels-last 3D')
    return result
