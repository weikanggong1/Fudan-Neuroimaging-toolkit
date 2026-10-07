"""Validation-only oneDNN x86 FP32 ELU arithmetic; no production import.

Independent scalar implementation of the numerical formula in oneDNN 2.7.3 /
3.1 x86 eltwise injector, with explicit FP32 FMA. This is a hypothesis until
compared with the actual isolated fused reference op on real complete arrays.
"""
import platform
from pathlib import Path

if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
    raise RuntimeError('this validation helper requires Linux x86-64')
flags = Path('/proc/cpuinfo').read_text().split('flags', 1)[1].split('\n', 1)[0].split()
if 'fma' not in flags:
    raise RuntimeError('the reference oneDNN packet formula requires FMA on this platform')

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


def elu_torch(value):
    import torch
    if value.device.type != 'cpu' or value.dtype != torch.float32 or value.requires_grad:
        raise ValueError('validation ELU requires CPU FP32 without autograd')
    result = torch.empty_like(value)
    if value.is_contiguous(memory_format=torch.channels_last_3d):
        source = value.permute(0,2,3,4,1).numpy().reshape(-1)
        output = result.permute(0,2,3,4,1).numpy().reshape(-1)
    elif value.is_contiguous():
        source = value.numpy().reshape(-1)
        output = result.numpy().reshape(-1)
    else:
        raise ValueError('validation ELU requires contiguous or channels-last 3D input')
    _elu_flat(source, output)
    return result
