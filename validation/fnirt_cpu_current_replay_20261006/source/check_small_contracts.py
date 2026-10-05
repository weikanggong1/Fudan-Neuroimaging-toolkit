"""Independent exact-rounding arithmetic and CPU guard contracts, not MRI data."""
from fractions import Fraction
from functools import reduce
import hashlib
import json
import math
import operator
from pathlib import Path
import struct
import numpy as np
from numba import njit
import torch
import own_reductions as own
from pcg_cpu_candidate import preconditioned_conjugate_gradient_cpu as pcg

SIZES=(0,1,3,4,7,8,15,16,17,31,32,33,47,48,63,64,65)
def bits(value):return struct.pack('<d',float(value))
def precise_fma(x,y,z):
    # Rational product+sum is exact, and Python float rounds once to binary64.
    return float(Fraction.from_float(float(x))*Fraction.from_float(float(y))+Fraction.from_float(float(z)))
def serial_add(values):return reduce(operator.add,values,0.)
def norm_definition(values):
    squares=[float(x)*float(x) for x in values]
    return math.sqrt(serial_add(squares[::2])+serial_add(squares[1::2]))
def dot_definition(left,right):
    n=len(left)
    if n<=32:
        products=[float(x)*float(y) for x,y in zip(left,right)]
        return serial_add(products[::2])+serial_add(products[1::2])
    streams=[0.]*32;end32=n-n%32;end16=n-n%16
    for k in range(end32):streams[k%32]=precise_fma(left[k],right[k],streams[k%32])
    folded=[streams[8*g+j]+streams[8*g+j+4] for g in range(4) for j in range(4)]
    for k in range(end32,end16):
        slot=k-end32;folded[slot]=precise_fma(left[k],right[k],folded[slot])
    columns=[reduce(operator.add,[folded[j+4*g] for g in range(4)]) for j in range(4)]
    total=(columns[2]+columns[0])+(columns[3]+columns[1])
    end4=n-(n-end16)%4
    for k in range(end16,end4):total=total+float(left[k])*float(right[k])
    for k in range(end4,n):total=precise_fma(left[k],right[k],total)
    return total

@njit(cache=True,fastmath=False)
def one_fma(x,y,z):return own.fma64(x,y,z)

def run(gradient,diagonal):
    rows=[];guards=[]
    for n in SIZES:
        left,right=gradient[:n].copy(),diagonal[:n].copy()
        rows.append({'length':n,'dot_exact_bits':bits(own.dot(left,right))==bits(dot_definition(left,right)),
            'norm_exact_bits':bits(own.norm(left))==bits(norm_definition(left)),
            'input_sha256':hashlib.sha256(left.tobytes()+right.tobytes()).hexdigest()})
    x,y,z=1.+2.**-27,1.-2.**-27,-1.
    guards.append({'case':'fma_cancellation','passed':bits(one_fma(x,y,z))==bits(-2.**-54) and bits(x*y+z)!=bits(-2.**-54)})
    identity=lambda value:value.clone()
    zero=torch.zeros(3,dtype=torch.float64);solution,report=pcg(identity,zero)
    guards.append({'case':'zero_rhs','passed':report.iterations==0 and report.converged and report.relative_residual==0 and torch.equal(solution,zero)})
    strided=torch.tensor([3.,99.,-5.,88.,2.,77.],dtype=torch.float64)[::2]
    original=strided.clone();solution,report=pcg(identity,strided)
    guards.append({'case':'strided_identity_and_input_unchanged','passed':report.iterations==1 and report.converged and torch.equal(solution,original) and torch.equal(strided,original)})
    for name,operation in (('zero_denominator',lambda v:torch.zeros_like(v)),('negative_denominator',lambda v:-v),
                           ('nan_denominator',lambda v:torch.full_like(v,float('nan'))),('inf_denominator',lambda v:torch.full_like(v,float('inf')))):
        solution,report=pcg(operation,original)
        guards.append({'case':name,'passed':report.iterations==0 and not report.converged and report.relative_residual==1. and torch.count_nonzero(solution).item()==0})
    exceptions=(('device_meta',lambda:pcg(identity,torch.empty(3,device='meta',dtype=torch.float64))),
        ('dtype_fp32',lambda:pcg(identity,original.float())),
        ('rhs_requires_grad',lambda:pcg(identity,original.clone().requires_grad_())),
        ('diagonal_requires_grad',lambda:pcg(identity,original,diagonal=torch.ones_like(original,requires_grad=True))),
        ('diagonal_nonpositive',lambda:pcg(identity,original,diagonal=torch.zeros_like(original))),
        ('diagonal_mismatch',lambda:pcg(identity,original,diagonal=torch.ones(2,dtype=torch.float64))),
        ('output_dtype',lambda:pcg(lambda v:v.float(),original)),
        ('output_requires_grad',lambda:pcg(lambda v:v.clone().requires_grad_(),original)),
        ('output_shape',lambda:pcg(lambda v:v[:2],original)),
        ('tolerance_zero',lambda:pcg(identity,original,tolerance=0.)),
        ('iterations_zero',lambda:pcg(identity,original,max_iterations=0)))
    for name,operation in exceptions:
        try:operation();passed=False
        except (TypeError,ValueError):passed=True
        guards.append({'case':name,'passed':passed})
    return {'scope':'small mathematical contracts derived from saved vector prefixes and explicit guard fixtures; not MRI benchmark or new native oracle',
        'independent_oracle':'exact Fraction multiply+add with one FP64 rounding for FMA; separate FP64 products/adds for norm/tails',
        'rows':rows,'guards':guards,'all_passed':all(r['dot_exact_bits'] and r['norm_exact_bits'] for r in rows) and all(r['passed'] for r in guards),
        'limitations':['Only the declared arithmetic definition is checked; no native BLAS selector/large threaded reduction equivalence is established.',
                       'Meta device rejects the non-CPU contract without allocating CUDA memory. No CUDA tensor or GPU execution is tested.']}
