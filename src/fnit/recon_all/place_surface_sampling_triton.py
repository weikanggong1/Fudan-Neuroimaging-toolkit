# Modified FNIT implementation of FreeSurfer placement formulas.
# FreeSurfer Software License: licenses/FreeSurfer.txt; upstream d932c45.
"""Optional fused CUDA MRI sampler; imported only for explicit Triton backend."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _voxel_row(A, x, y, z, row: tl.constexpr):
    value = tl.full(x.shape, 0, tl.float32)
    value = value + tl.load(A + 4*row) * x
    value = value + tl.load(A + 4*row+1) * y
    value = value + tl.load(A + 4*row+2) * z
    value = value + tl.load(A + 4*row+3)
    return value.to(tl.float64)


@triton.jit
def _sample(V, A, x, y, z, mask, W: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    x32, y32, z32 = x.to(tl.float32), y.to(tl.float32), z.to(tl.float32)
    vx = _voxel_row(A, x32, y32, z32, 0)
    vy = _voxel_row(A, x32, y32, z32, 1)
    vz = _voxel_row(A, x32, y32, z32, 2)
    inside = mask & (vx >= -0.5) & (vy >= -0.5) & (vz >= -0.5) & (vx < W-0.5) & (vy < H-0.5) & (vz < D-0.5)
    vx, vy, vz = tl.minimum(tl.maximum(vx, 0), W-1), tl.minimum(tl.maximum(vy, 0), H-1), tl.minimum(tl.maximum(vz, 0), D-1)
    ix, iy, iz = vx.to(tl.int32), vy.to(tl.int32), vz.to(tl.int32)
    fx, fy, fz = vx-ix, vy-iy, vz-iz
    result = tl.full(x.shape, 0, tl.float64)
    for bx in tl.static_range(2):
        for by in tl.static_range(2):
            for bz in tl.static_range(2):
                cx, cy, cz = tl.minimum(ix+bx, W-1), tl.minimum(iy+by, H-1), tl.minimum(iz+bz, D-1)
                wx, wy, wz = fx if bx else 1-fx, fy if by else 1-fy, fz if bz else 1-fz
                result = result + wx*wy*wz * tl.load(V + (cx*H+cy)*D+cz, mask=inside, other=0).to(tl.float64)
    return tl.where(inside, result, 0)


@triton.jit
def sample_kernel(V,A,X,O,N:tl.constexpr,W:tl.constexpr,H:tl.constexpr,D:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N
    x=tl.load(X+3*i,mask=mask,other=0).to(tl.float64)
    y=tl.load(X+3*i+1,mask=mask,other=0).to(tl.float64)
    z=tl.load(X+3*i+2,mask=mask,other=0).to(tl.float64)
    result=_sample(V,A,x,y,z,mask,W,H,D)
    tl.store(O+i,result,mask=mask)


@triton.jit
def gradient_kernel(V,A,X,NORMAL,VALUES,SIGMAS,SKIP,O,N:tl.constexpr,W:tl.constexpr,H:tl.constexpr,D:tl.constexpr,VOXSTEP:tl.constexpr,WEIGHT:tl.constexpr,GLOBAL:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N
    value=tl.load(VALUES+i,mask=mask,other=-1).to(tl.float64)
    active=mask & (tl.load(SKIP+i,mask=mask,other=1)==0) & (value>=0)
    x=tl.load(X+3*i,mask=mask,other=0).to(tl.float64)
    y=tl.load(X+3*i+1,mask=mask,other=0).to(tl.float64)
    z=tl.load(X+3*i+2,mask=mask,other=0).to(tl.float64)
    nx=tl.load(NORMAL+3*i,mask=mask,other=0).to(tl.float64)
    ny=tl.load(NORMAL+3*i+1,mask=mask,other=0).to(tl.float64)
    nz=tl.load(NORMAL+3*i+2,mask=mask,other=0).to(tl.float64)
    sigma=tl.load(SIGMAS+i,mask=mask,other=1).to(tl.float64)
    sigma=tl.where(tl.abs(sigma)<1e-10,GLOBAL,sigma)
    sigma=tl.where(tl.abs(sigma)<1e-10,0.25,sigma)
    sigma=tl.where(active,sigma,1)
    step=tl.minimum(sigma/2.0,VOXSTEP);distance=step
    current=_sample(V,A,x,y,z,active,W,H,D)
    outside=tl.full((B,),0,tl.float64);inside=tl.full((B,),0,tl.float64);total=tl.full((B,),0,tl.float64)
    running=active & (distance<=2*sigma)
    while tl.sum(running.to(tl.int32),0)>0:
        kernel=libdevice.exp(-distance*distance/(2.0*sigma*sigma))
        kernel=tl.where(running,kernel,0)
        outside=outside+kernel*_sample(V,A,x+distance*nx,y+distance*ny,z+distance*nz,running,W,H,D)
        inside=inside+kernel*_sample(V,A,x-distance*nx,y-distance*ny,z-distance*nz,running,W,H,D)
        total=total+kernel
        distance=distance+step
        running=active & (distance<=2*sigma)
    slope=(outside/total-inside/total)/2.0
    sign=tl.where(tl.abs(slope)>=1e-10,slope/tl.abs(slope),-1.0)
    error=tl.minimum(tl.maximum(value-current,-5),5)
    displacement=WEIGHT*error*sign
    tl.store(O+3*i,tl.where(active,nx*displacement,0).to(tl.float32),mask=mask)
    tl.store(O+3*i+1,tl.where(active,ny*displacement,0).to(tl.float32),mask=mask)
    tl.store(O+3*i+2,tl.where(active,nz*displacement,0).to(tl.float32),mask=mask)
