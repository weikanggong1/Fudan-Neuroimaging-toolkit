"""Chunked GPU GCAM coordinate conversion and nearest sampling.

Geometry matrices retain native FP32 coefficients and FP64 accumulation.
The vector field is a target-to-source pull in scanner RAS millimetres.
"""
from __future__ import annotations
import time
import nibabel as nib
import numpy as np
import torch
from fnit._transforms import load_lta, same_geometry
from .ca_register_inverse import _inverse_4x4_native, read_warp_geometries
from .mni_warp_io import native_geometry, write_forward_warp


def _device(device):
    chosen=torch.device(device)
    if chosen.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('explicit CUDA device unavailable; no CPU fallback')
    return chosen


def _affine(matrix, coordinates):
    """Native MatrixMultiplyD: accumulate in double, store each component FP32."""
    matrix=np.asarray(matrix,np.float32)
    inputs=coordinates.to(torch.float64)
    return torch.stack([((inputs[0]*float(matrix[a,0])+inputs[1]*float(matrix[a,1]))+
                          inputs[2]*float(matrix[a,2])+float(matrix[a,3])).float()
                        for a in range(3)])


def _matmul(left,right):
    left,right=np.asarray(left,np.float32),np.asarray(right,np.float32)
    out=np.zeros((4,4),np.float64)
    for k in range(4):out+=left[:,k,None].astype(np.float64)*right[None,k,:].astype(np.float64)
    return out.astype(np.float32)


def _grid(shape,start,stop,device):
    return torch.stack(torch.meshgrid(torch.arange(start,stop,device=device,dtype=torch.float32),
        torch.arange(shape[1],device=device,dtype=torch.float32),
        torch.arange(shape[2],device=device,dtype=torch.float32),indexing='ij'))


def _linear_absolute(field, coordinates):
    shape=field.shape[1:]
    valid=torch.ones_like(coordinates[0],dtype=torch.bool)
    low=[];high=[];lower=[];upper=[]
    for axis,n in enumerate(shape):
        q=coordinates[axis]; valid &= (q>=0)&(q<n)
        m=q.to(torch.long).clamp(0,n-1)
        low.append(m);high.append((m+1).clamp_max(n-1))
        upper.append(q-m.float());lower.append(1.-upper[-1])
    result=torch.zeros((3,*coordinates.shape[1:]),dtype=torch.float32,device=field.device)
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                indices=[(high if v else low)[axis] for axis,v in enumerate((dx,dy,dz))]
                weights=[(upper if v else lower)[axis] for axis,v in enumerate((dx,dy,dz))]
                weight=weights[0]*weights[1]*weights[2]
                result+=weight[None]*field[:,indices[0],indices[1],indices[2]]
    return torch.where(valid[None],result,0.),valid


@torch.inference_mode()
def convert_mni_warp(ras_warp,cropped_source,original,full_target,crop_to_original_lta,
                     crop_to_full_lta,output,*,device='cuda:0',chunk_slices=16):
    """将crop目标RAS位移组合两条LTA，输出full目标FS pull warp。

    六个输入为路径；output为.nii.gz。chunk_slices为X方向分块数，默认16且>=1。
    两条LTA分别crop source→original和crop target→full target；支持type0/1。
    越出crop目标[0,size)的绝对源坐标置0，与GCAMconcat3相同；不延拓位移。
    返回计算/加载/保存秒数；CUDA失败明确报错。显式cpu用于诊断。
    """
    started=time.perf_counter();dev=_device(device)
    if not isinstance(chunk_slices,int) or chunk_slices<1:raise ValueError('chunk_slices must be a positive integer')
    warp=nib.load(str(ras_warp));source=nib.load(str(cropped_source))
    orig=nib.load(str(original));target=nib.load(str(full_target))
    first,second=load_lta(crop_to_original_lta),load_lta(crop_to_full_lta)
    if not (same_geometry(first.source,source,tolerance=1e-4) and
            same_geometry(first.target,orig,tolerance=1e-4) and
            same_geometry(second.source,warp,tolerance=1e-4) and
            same_geometry(second.target,target,tolerance=1e-4)):
        raise ValueError('LTA source/target geometries differ from supplied MNI inputs')
    # load_lta retains supplied voxel matrices. Native inversions round in FP32.
    def vox(transform):
        if transform.space=='voxel':return np.asarray(transform.matrix,np.float32)
        return _matmul(_inverse_4x4_native(transform.target.affine),
                       _matmul(transform.matrix,transform.source.affine))
    source_to_orig=_inverse_4x4_native(_inverse_4x4_native(vox(first)))
    full_to_crop=_inverse_4x4_native(vox(second))
    array=np.asarray(warp.dataobj,np.float32)
    if array.shape != (*warp.shape[:3],3) or not np.isfinite(array).all():
        raise ValueError('expected finite (X,Y,Z,3) RAS displacement')
    displacement=torch.as_tensor(np.ascontiguousarray(array.transpose(3,0,1,2)),device=dev)
    coordinates=torch.empty_like(displacement)
    crop_aff=native_geometry(warp);src_inv=_inverse_4x4_native(native_geometry(source))
    for lo in range(0,warp.shape[0],chunk_slices):
        hi=min(lo+chunk_slices,warp.shape[0]);grid=_grid(warp.shape[:3],lo,hi,dev)
        ras=(_affine(crop_aff,grid).double()+displacement[:,lo:hi].double()).float()
        coordinates[:,lo:hi]=_affine(src_inv,ras)
    output_data=np.empty((*target.shape[:3],3),np.float32)
    source_aff=native_geometry(orig);target_aff=native_geometry(target)
    loaded=time.perf_counter()-started
    tick=time.perf_counter()
    for lo in range(0,target.shape[0],chunk_slices):
        hi=min(lo+chunk_slices,target.shape[0]);grid=_grid(target.shape[:3],lo,hi,dev)
        sampled,valid=_linear_absolute(coordinates,_affine(full_to_crop,grid))
        mapped=torch.where(valid[None],_affine(source_to_orig,sampled),0.)
        output_data[lo:hi]=(_affine(source_aff,mapped).double()-_affine(target_aff,grid).double()).float().movedim(0,-1).cpu().numpy()
    elapsed=time.perf_counter()-tick
    tick=time.perf_counter();write_forward_warp(output_data,orig,target,output)
    return {'load_convert_seconds':loaded,'composition_download_seconds':elapsed,
            'save_seconds':time.perf_counter()-tick,'total_seconds':time.perf_counter()-started,
            'device':str(dev),'chunk_slices':chunk_slices}


def _native_nearest_plan(coordinates, shape):
    """MRIindexNotInVolume uses rint; nint(double) rounds half away from zero."""
    q=coordinates.double()
    nearest=torch.where(q<0,torch.ceil(q-0.5),torch.floor(q+0.5)).long()
    border=q.round()  # C rint, including even ties exactly at the border
    valid=torch.ones_like(q[0],dtype=torch.bool)
    for axis,size in enumerate(shape):
        valid &= (border[axis]>=0)&(border[axis]<size)
    return [nearest[axis].clamp(0,size-1) for axis,size in enumerate(shape)],valid


@torch.inference_mode()
def resample_mni_check(original,forward,output,*,device='cuda:0',chunk_slices=16):
    """最近邻检查图：full目标→source绝对体素；native nint双精度半整数远离零，边界先rint检查再clamp，越界填0。

    original为3D原图、forward为FS (X,Y,Z,1,3) world-mm pull、output为NIfTI。
    保留输入dtype和目标q/sform。device默认CUDA，chunk_slices默认16且>=1。
    无CUDA/扩展/有限坐标或错误源网格时失败，不使用PyTorch ties-to-even。
    """
    started=time.perf_counter();dev=_device(device)
    if not isinstance(chunk_slices,int) or chunk_slices<1:raise ValueError('chunk_slices must be a positive integer')
    image=nib.load(str(original));warp=nib.load(str(forward))
    source_aff,atlas_aff,source_shape=read_warp_geometries(warp)
    if tuple(image.shape)!=source_shape:raise ValueError('forward source grid differs from original')
    array=np.asarray(image.dataobj)
    volume=torch.as_tensor(np.ascontiguousarray(array),device=dev)
    disp=np.asarray(warp.dataobj,np.float32)[:,:,:,0,:]
    out=np.empty(warp.shape[:3],array.dtype);inverse=_inverse_4x4_native(source_aff)
    for lo in range(0,out.shape[0],chunk_slices):
        hi=min(lo+chunk_slices,out.shape[0]);grid=_grid(out.shape,lo,hi,dev)
        delta=torch.as_tensor(np.ascontiguousarray(disp[lo:hi].transpose(3,0,1,2)),device=dev)
        q=_affine(inverse,(_affine(atlas_aff,grid).double()+delta.double()).float())
        if not bool(torch.isfinite(q).all()):raise ValueError('nonfinite forward coordinates')
        idx,valid=_native_nearest_plan(q,source_shape)
        # GCAMmorphToAtlas's pre-sampling domain check (3D original volume).
        valid &= (q[0]>-1)&(q[0]<source_shape[0])&(q[1]>-1)&(q[1]<source_shape[1])
        valid &= ((q[2]==0) if source_shape[2]==1 else ((q[2]>0)&(q[2]<source_shape[2])))
        out[lo:hi]=torch.where(valid,volume[idx[0],idx[1],idx[2]],0).cpu().numpy()
    tick=time.perf_counter()
    header=nib.Nifti1Header();header.set_data_dtype(array.dtype);header.set_xyzt_units('mm','sec')
    result=nib.Nifti1Image(out,warp.affine,header)
    result.set_qform(warp.get_qform(),int(warp.header['qform_code']))
    result.set_sform(warp.get_sform(),int(warp.header['sform_code']))
    nib.save(result,str(output))
    return {'save_seconds':time.perf_counter()-tick,'total_seconds':time.perf_counter()-started,
            'device':str(dev),'chunk_slices':chunk_slices}
