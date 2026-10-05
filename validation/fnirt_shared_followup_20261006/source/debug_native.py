"""Tiny loader/math contract, isolated from MRI and nonlinear optimization."""
import argparse
import ctypes
import faulthandler
import json
import os

faulthandler.enable()
parser=argparse.ArgumentParser()
parser.add_argument('--library',required=True)
parser.add_argument('--mode',choices=('default','deep'),required=True)
parser.add_argument('--numpy',choices=('no','yes','numba'),required=True)
args=parser.parse_args()
if args.numpy!='no':import numpy
if args.numpy=='numba':import numba;import scipy.sparse
mode=os.RTLD_NOW|os.RTLD_LOCAL
if args.mode=='deep':mode|=getattr(os,'RTLD_DEEPBIND',0)
print('before_library',args.mode,args.numpy,flush=True)
lib=ctypes.CDLL(args.library,mode=mode)
pointer=ctypes.POINTER(ctypes.c_double)
lib.fnit_native_flags.restype=ctypes.c_int
lib.fnit_native_dot.argtypes=[pointer,pointer,ctypes.c_int]
lib.fnit_native_dot.restype=ctypes.c_double
lib.fnit_native_norm.argtypes=[pointer,ctypes.c_int]
lib.fnit_native_norm.restype=ctypes.c_double
print('flags',lib.fnit_native_flags(),flush=True)
values=(ctypes.c_double*5)(1,2,3,4,5)
print('before_dot',flush=True)
dot=lib.fnit_native_dot(values,values,5)
print('dot',dot,flush=True)
norm=lib.fnit_native_norm(values,5)
print(json.dumps({'tiny_contract':dot==55 and norm==55**.5,'dot':dot,'norm':norm}),flush=True)
