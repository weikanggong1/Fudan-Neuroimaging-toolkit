"""归一化现有PyTorch函数的有序CUDA内核，仅CUDA调用时导入。

逐体素保留offset/邻域顺序，不用原子加法、FMA融合或半精度；
沿同一stream依次启动wavefront和三轴卷积，CPU规则保持原接口。
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _convolve_axis(src,dst,kernel,N:tl.constexpr,SY:tl.constexpr,SZ:tl.constexpr,
                   LENGTH:tl.constexpr,STRIDE:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    index=tl.program_id(0)*B+tl.arange(0,B)
    valid=index<N
    coordinate=(index//STRIDE)%LENGTH
    value=tl.full((B,),0,tl.float32)
    for offset in range(K):
        adjacent=tl.minimum(tl.maximum(coordinate+offset-K//2,0),LENGTH-1)
        sample=tl.load(src+index+(adjacent-coordinate)*STRIDE,mask=valid,other=0)
        weight=tl.load(kernel+offset)
        value=value+sample*weight
    tl.store(dst+index,value,mask=valid)

@triton.jit(do_not_specialize=["COUNT","START","LEVEL"])
def _wavefront(field,distance,ordered,COUNT,START,
               LEVEL,SX:tl.constexpr,SY:tl.constexpr,SZ:tl.constexpr,B:tl.constexpr):
    lane=tl.program_id(0)*B+tl.arange(0,B)
    active=lane<COUNT
    index=tl.load(ordered+START+lane,mask=active,other=0).to(tl.int32)
    x=index//(SY*SZ);y=(index//SZ)%SY;z=index%SZ
    total=tl.full((B,),0,tl.float32)
    count=tl.full((B,),0,tl.int32)
    for dz in tl.static_range(-1,2):
        zi=tl.minimum(tl.maximum(z+dz,0),SZ-1)
        for dy in tl.static_range(-1,2):
            yi=tl.minimum(tl.maximum(y+dy,0),SY-1)
            for dx in tl.static_range(-1,2):
                xi=tl.minimum(tl.maximum(x+dx,0),SX-1)
                other=(xi*SY+yi)*SZ+zi
                prior=tl.load(distance+other,mask=active,other=LEVEL)<LEVEL
                sample=tl.load(field+other,mask=active & prior,other=0)
                total=total+sample
                count=count+prior.to(tl.int32)
    tl.store(field+index,tl.div_rn(total,count.to(tl.float32)),mask=active)

def ordered_smoothing(voronoi:torch.Tensor,kernel:torch.Tensor)->torch.Tensor:
    """同CUDA设备连续float32偏置图和一维核→同shape float32，三轴nearest边界。

    保留0/1/2轴及核offset顺序。分配两个等大缓冲并交替写出，不作强制同步；
    调用方负责恢复控制点及计时同步。属于mri_normalize内部步骤，无独立CLI。
    """
    current=voronoi.float().contiguous()
    first=torch.empty_like(current);second=torch.empty_like(current)
    sx,sy,sz=current.shape
    with torch.cuda.device(current.device):
        for axis,(length,stride) in enumerate(((sx,sy*sz),(sy,sz),(sz,1))):
            result=first if axis%2==0 else second
            _convolve_axis[(triton.cdiv(current.numel(),256),)](
                current,result,kernel,N=current.numel(),SY=sy,SZ=sz,
                LENGTH=length,STRIDE=stride,K=kernel.numel(),B=256,
                num_warps=4,enable_fp_fusion=False)
            current=result
    return current

def ordered_wavefront(field:torch.Tensor,distance:torch.Tensor,
                      ordered:torch.Tensor,boundaries)->torch.Tensor:
    """同设备float32初始field、int32距离、int64排序索引、CPU层边界→原位field。

    所有数组连续；distance来自原chessboard CDT，field仅控制点非零。
    一层一个kernel；只读取严格更早层，保留dz/dy/dx及边界重复计数；
    不将依赖层并行。无新增同步，调用方负责空控制检查及统计。
    """
    sx,sy,sz=field.shape
    with torch.cuda.device(field.device):
        for level in range(1,len(boundaries)):
            start,count=int(boundaries[level-1]),int(boundaries[level]-boundaries[level-1])
            if count:
                _wavefront[(triton.cdiv(count,256),)](
                    field,distance,ordered,COUNT=count,START=start,LEVEL=level,
                    SX=sx,SY=sy,SZ=sz,B=256,num_warps=4,enable_fp_fusion=False)
    return field
