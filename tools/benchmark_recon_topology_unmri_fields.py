"""解析首次真实缺陷的 unMRI 只读检查点；用于隔离 benchmark，不供生产。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load_volume_capture(*, capture_path):
    """FNITVOL1 →完整有序输入、距离和聚合参考字段的独立 NumPy 数组。

    capture_path 为固定源码独立 benchmark 观察器的二进制文件路径。
    顶点 origXYZ/fx/fy/fz 是 surface RAS/mm，法向无量纲，面积 mm²；
    原 status/源顶点→base 顶点映射不去重。四张 volume 的 shape 为 XYZ，
    intensity 为 uint8 的 float32表示，white/gray 为 log-P、distance 为 mm。
    volume geometry 的 size 为源局部网格采样频率，不能当影像 affine。
    返回 dict 只读来源已复制数组，不修改文件。格式、截断、尾部及
    越界 patch 索引错误抛 ValueError。实际 distance 是参考输入，后续
    聚合测试不能据此声称自有几何距离内核已经通过。
    """
    raw=Path(capture_path).read_bytes()
    if raw[:8]!=b'FNITVOL1':
        raise ValueError('FNITVOL1 complete benchmark capture required')
    position=8

    def read(dtype,count,shape=None):
        nonlocal position
        dtype=np.dtype(dtype)
        if count<0 or position+dtype.itemsize*count>len(raw):
            raise ValueError('truncated unMRI capture')
        result=np.frombuffer(raw,dtype,count=count,offset=position).copy()
        position+=dtype.itemsize*count
        return result.reshape(shape) if shape is not None else result

    header=read('<i4',5)
    if np.any(header<0) or np.any(header>100000000):
        raise ValueError('invalid unMRI capture dimensions')
    n,f,k,nv,nf=map(int,header)
    fields=dict(means_and_sentinel=read('<f4',5),vertex_fields=read('<f4',n*13,(n,13)),
        faces=read('<i4',f*3,(f,3)),face_fields=read('<f4',f*4,(f,4)),
        defect_vertices=read('<i4',k*3,(k,3)),patch_vertices=read('<i4',nv),patch_faces=read('<i4',nf))
    for name in ('intensity','white_log_pdf','gray_log_pdf','signed_distance'):
        dims=read('<i4',4)
        if np.any(dims[:3]<=0) or np.any(dims[:3]>100000):
            raise ValueError('invalid local volume dimensions')
        shape=tuple(map(int,dims[:3]))
        fields[name+'_dimensions']=dims
        fields[name+'_geometry']=read('<f4',9)
        fields[name]=read('<f4',int(np.prod(shape,dtype=np.int64)),shape)
    fields['reference_results']=read('<f4',8)
    fields['reference_curvbak']=read('<f4',n)
    if position!=len(raw):
        raise ValueError('unexpected unMRI capture trailing bytes')
    for indices,limit in ((fields['faces'],n),(fields['patch_vertices'],n),(fields['patch_faces'],f),(fields['defect_vertices'][:,1],n)):
        if indices.size and (indices.min()<0 or indices.max()>=limit):
            raise ValueError('unMRI capture mapping or patch index out of range')
    return fields


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('new diagnostic checkpoint required')
    fields=load_volume_capture(capture_path=args.capture)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**fields)
    print(json.dumps(dict(scope='frozen first source unMRI inputs only; no candidate or GA result',
        capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        fields_checkpoint_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        vertices=len(fields['vertex_fields']),faces=len(fields['faces']),defect_vertices=len(fields['defect_vertices']),
        volume_shape=list(fields['signed_distance'].shape)),indent=2))


if __name__=='__main__':
    main()
