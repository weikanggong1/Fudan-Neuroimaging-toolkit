"""Diagnostic BN with the observed unfused FP32 graph and rsqrt packet step."""
import platform
from pathlib import Path

if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
    raise RuntimeError('this diagnostic packet implementation requires Linux x86-64; production never imports it')
flags = Path('/proc/cpuinfo').read_text().split('flags', 1)[1].split('\n', 1)[0].split()
if 'sse2' not in flags:
    raise RuntimeError('SSE2 is required for this diagnostic packet implementation')

import numpy as np
from llvmlite import ir
from numba import njit, prange, types
from numba.extending import intrinsic


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


def batch_norm_torch(value, norm):
    """CPU-only diagnostic affine; rejects autograd and unsupported layouts."""
    import torch
    if value.device.type!='cpu' or value.dtype!=torch.float32 or value.requires_grad:
        raise ValueError('diagnostic BN requires CPU FP32 inference')
    if value.shape[0]!=1:
        raise ValueError('stage diagnostic currently supports one volume')
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
        raise ValueError('diagnostic BN requires contiguous or channels-last 3D')
    return result
