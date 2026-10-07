"""Independent finite CPU FP64 arithmetic candidate; no installed BLAS calls.

Ordering follows observed contiguous single-thread Dot arithmetic and the
source-defined two-chain square accumulation. This is a bounded validation
candidate, not a change to FNIT or a claim about every architecture/size.
"""
import math
import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic


@intrinsic
def fma64(typing_context, left, right, accumulator):
    # Same project-owned intrinsic approach already used by _raster_cpu.fma32.
    if left!=types.float64 or right!=left or accumulator!=left:return None
    def emit(context,builder,signature,arguments):
        scalar=ir.DoubleType()
        fn=builder.module.declare_intrinsic('llvm.fma',[scalar],
            fnty=ir.FunctionType(scalar,[scalar]*3))
        return builder.call(fn,arguments)
    return left(left,right,accumulator),emit


@njit(cache=True,fastmath=False)
def paired_dot(left,right):
    even=0.;odd=0.;n=left.size
    for k in range(0,n-1,2):
        even=even+left[k]*right[k]
        odd=odd+left[k+1]*right[k+1]
    if n%2:even=even+left[n-1]*right[n-1]
    return even+odd


@njit(cache=True,fastmath=False)
def norm(values):
    even=0.;odd=0.;n=values.size
    for k in range(0,n-1,2):
        even=even+values[k]*values[k]
        odd=odd+values[k+1]*values[k+1]
    if n%2:even=even+values[n-1]*values[n-1]
    return math.sqrt(even+odd)


@njit(cache=True,fastmath=False)
def dot(left,right):
    n=left.size
    if n<=32:return paired_dot(left,right)
    # Four groups of eight independent FMA chains; unaligned reads are valid.
    lanes=np.zeros((4,8),dtype=np.float64)
    prefix32=(n//32)*32
    for block in range(0,prefix32,32):
        for group in range(4):
            for lane in range(8):
                k=block+group*8+lane
                lanes[group,lane]=fma64(left[k],right[k],lanes[group,lane])
    folded=np.empty((4,4),dtype=np.float64)
    for group in range(4):
        for lane in range(4):
            folded[group,lane]=lanes[group,lane]+lanes[group,lane+4]
    prefix16=(n//16)*16
    for block in range(prefix32,prefix16,16):
        for group in range(4):
            for lane in range(4):
                k=block+group*4+lane
                folded[group,lane]=fma64(left[k],right[k],folded[group,lane])
    sums=np.empty(4,dtype=np.float64)
    for lane in range(4):
        sums[lane]=((folded[0,lane]+folded[1,lane])+folded[2,lane])+folded[3,lane]
    value=(sums[2]+sums[0])+(sums[3]+sums[1])
    # Full eight/four tail products round separately before sequential adds;
    # remaining scalar tail uses explicit FMA. All cutoffs depend on n only.
    tail=n-prefix16
    separate_end=prefix16+(tail//4)*4
    for k in range(prefix16,separate_end):value=value+left[k]*right[k]
    for k in range(separate_end,n):value=fma64(left[k],right[k],value)
    return value
