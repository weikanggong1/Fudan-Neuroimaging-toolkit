"""Ordered CPU motion resampling with the existing Torch fallback on the caller.

The cubic prefilter uses float64 recurrence and narrows each axis to float32.
Coordinates retain repeated float32 addition; the 64 spline terms retain their
original double-precision order. CUDA uses its separate existing implementation.
"""
import math
import numpy as np
from numba import njit, prange
from ..flirt._cpu import _dispatch_with_cpu_budget, _trilinear


def _axis_impl(data,axis,pole,terms,terminal):
    nx,ny,nz=data.shape
    length=data.shape[axis]
    lines=ny*nz if axis==0 else nx*nz if axis==1 else nx*ny
    out=np.empty_like(data)
    for row in prange(lines):
        a,b=(row//nz,row%nz) if axis<2 else (row//ny,row%ny)
        values=np.empty(length,dtype=np.float64)
        for offset in range(length):
            values[offset]=data[offset,a,b] if axis==0 else data[a,offset,b] if axis==1 else data[a,b,offset]
        initial=values[0];power=pole
        for offset in range(1,terms):
            initial=initial+power*values[offset]
            power=power*pole
        values[0]=initial
        original_last=values[-1]
        for offset in range(1,length):values[offset]=values[offset]+pole*values[offset-1]
        values[-1]=terminal*(2.0*values[-1]-original_last)
        for offset in range(length-2,-1,-1):values[offset]=pole*(values[offset+1]-values[offset])
        for offset in range(length):
            value=np.float32(values[offset]*6.0)
            if axis==0:out[offset,a,b]=value
            elif axis==1:out[a,offset,b]=value
            else:out[a,b,offset]=value
    return out


@njit(cache=True,fastmath=False,inline='always')
def _basis(distance):
    if distance<1.0:
        term=0.5*distance
        term=term*distance
        term=term*(distance-2.0)
        return 2.0/3.0+term
    if distance<2.0:
        term=2.0-distance
        return ((term*term)*term)/6.0
    return 0.0


@njit(cache=True,fastmath=False,inline='always')
def _weight(position,start,index):
    return _basis(abs(position-(start+index)))


def _sample_impl(data,coefficients,pull,shape,background,spline):
    nx,ny,nz=shape
    out=np.empty(shape,dtype=np.float32)
    upperx,uppery,upperz=data.shape[0]-1,data.shape[1]-1,data.shape[2]-1
    for row in prange(nx*nz):
        x,z=row//nz,row%nz
        cx=np.float32(np.float32(np.float32(x)*pull[0,0])+np.float32(np.float32(z)*pull[0,2]))
        cy=np.float32(np.float32(np.float32(x)*pull[1,0])+np.float32(np.float32(z)*pull[1,2]))
        cz=np.float32(np.float32(np.float32(x)*pull[2,0])+np.float32(np.float32(z)*pull[2,2]))
        cx=np.float32(cx+pull[0,3]);cy=np.float32(cy+pull[1,3]);cz=np.float32(cz+pull[2,3])
        for y in range(ny):
            if (not math.isfinite(cx) or not math.isfinite(cy) or not math.isfinite(cz)
                or math.floor(cx)<-1 or math.floor(cy)<-1 or math.floor(cz)<-1
                or math.floor(cx)>=data.shape[0] or math.floor(cy)>=data.shape[1] or math.floor(cz)>=data.shape[2]):
                value=background
            elif not spline:
                value=_trilinear(data,min(max(cx,np.float32(0)),np.float32(upperx)),
                                     min(max(cy,np.float32(0)),np.float32(uppery)),
                                     min(max(cz,np.float32(0)),np.float32(upperz)))
            else:
                px,py,pz=np.float64(cx),np.float64(cy),np.float64(cz)
                rx,ry,rz=int(math.trunc(px+0.5)),int(math.trunc(py+0.5)),int(math.trunc(pz+0.5))
                sx=rx-1 if np.float64(rx)<px else rx-2
                sy=ry-1 if np.float64(ry)<py else ry-2
                sz=rz-1 if np.float64(rz)<pz else rz-2
                value64=np.float64(0)
                for zz in range(4):
                    iz=min(max(sz+zz,0),upperz)
                    wz=_weight(pz,sz,zz)
                    for yy in range(4):
                        iy=min(max(sy+yy,0),uppery)
                        yz=wz*_weight(py,sy,yy)
                        for xx in range(4):
                            ix=min(max(sx+xx,0),upperx)
                            term=np.float64(coefficients[ix,iy,iz])*_weight(px,sx,xx)
                            value64=value64+term*yz
                value=np.float32(value64)
            out[x,y,z]=value
            cx=np.float32(cx+pull[0,1]);cy=np.float32(cy+pull[1,1]);cz=np.float32(cz+pull[2,1])
    return out


_axis_serial=njit(cache=True,fastmath=False)(_axis_impl)
_axis_parallel=njit(cache=True,fastmath=False,parallel=True)(_axis_impl)
_sample_serial=njit(cache=True,fastmath=False,error_model='numpy')(_sample_impl)
_sample_parallel=njit(cache=True,fastmath=False,parallel=True,error_model='numpy')(_sample_impl)


def cubic_coefficients(data):
    coefficients=np.asarray(data,dtype=np.float32)
    pole=math.sqrt(3.0)-2.0
    terminal=-pole/(1.0-pole*pole)
    for axis in range(3):
        length=coefficients.shape[axis]
        if length<2:continue
        terms=min(length,int(math.log(1e-8)/math.log(abs(pole))+1.5))
        arguments=(coefficients,axis,pole,terms,terminal)
        coefficients=(_axis_serial(*arguments) if coefficients.size < 1048576 else
                      _dispatch_with_cpu_budget(_axis_parallel,_axis_serial,arguments))
    return np.ascontiguousarray(coefficients)


def sample(data,pull,shape,background,interpolation):
    data=np.asarray(data,dtype=np.float32)
    if interpolation not in ('linear','spline'):raise ValueError('linear or spline required')
    if interpolation=='linear' and min(data.shape)<2:raise ValueError('Thin-axis linear requires the Torch fallback')
    coefficients=cubic_coefficients(data) if interpolation=='spline' else data
    arguments=(data,coefficients,np.asarray(pull,dtype=np.float32),tuple(shape),
               np.float32(background),interpolation=='spline')
    # At the normal EPI frame size an OpenMP launch costs more than it saves.
    # Keep this independent of global pool settings and use bounded parallelism
    # only for larger reference grids.
    if int(np.prod(shape)) < 1048576:
        return _sample_serial(*arguments)
    return _dispatch_with_cpu_budget(_sample_parallel,_sample_serial,arguments)
