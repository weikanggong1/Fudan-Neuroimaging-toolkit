"""解析缺陷候选曲率拟合前的原 vtotal 有序邻域，仅用于冻结阶段诊断。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load_neighbor_capture(*, capture_path):
    """FNITNBR1 → 原 vnum/vtotal/nsizeCur、CSR 与候选几何独立数组。

    capture_path 为只读参考观察器在 computeDefectSecondFundamentalForm
    前记录的文件，无默认值。返回 neighbor_offsets (int32/N+1)、neighbors
    (int32/entries)、vertex_topology (int32/Nx3)、vertex_fields (float32/Nx15)
    及 selected_vertices (int32/K)。vertex_fields 列依次为 origXYZ(mm，
    tkregister surface RAS)、法向、e1/e2切向量、k1/k2(mm^-1)、有符号面积
    (mm²)。后面两主曲率是重算前值，不能冒充自产拟合结果。
    CSR 保留原 vt.v[0:vtotal] 次序，不从 faces 重建或排序；拓扑前三列
    为 vnum/vtotal/nsizeCur，vnum 一环可短于拟合所用的 vtotal。
    不供生产读取，格式/尺寸/索引/度数或尾部异常抛 ValueError。
    """
    raw=Path(capture_path).read_bytes()
    if raw[:8]!=b'FNITNBR1':
        raise ValueError('complete FNITNBR1 candidate neighborhood required')
    offset=8
    def read(dtype,count,shape=None):
        nonlocal offset
        dtype=np.dtype(dtype)
        if count<0 or offset+count*dtype.itemsize>len(raw):
            raise ValueError('truncated candidate neighborhood')
        result=np.frombuffer(raw,dtype,count=count,offset=offset).copy()
        offset+=count*dtype.itemsize
        return result.reshape(shape) if shape is not None else result
    header=read('<i4',3);n,entries,k=map(int,header)
    if n<=0 or max(n,entries,k)>100000000 or min(entries,k)<0:
        raise ValueError('invalid candidate neighborhood dimensions')
    fields=dict(header=header,neighbor_offsets=read('<i4',n+1),
        vertex_topology=read('<i4',n*3,(n,3)),neighbors=read('<i4',entries),
        vertex_fields=read('<f4',n*15,(n,15)),selected_vertices=read('<i4',k))
    if offset!=len(raw):raise ValueError('unexpected neighborhood trailing data')
    csr,topo=fields['neighbor_offsets'],fields['vertex_topology']
    if csr[0]!=0 or csr[-1]!=entries or np.any(np.diff(csr)<0):
        raise ValueError('invalid original vtotal CSR')
    if np.any(topo<0) or np.any(topo[:,0]>topo[:,1]) or np.any(np.diff(csr)!=topo[:,1]):
        raise ValueError('vnum/vtotal topology and CSR mismatch')
    for a in [fields['neighbors'],fields['selected_vertices']]:
        if a.size and (a.min()<0 or a.max()>=n):raise ValueError('candidate neighborhood index out of range')
    return fields


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('new candidate neighborhood diagnostic output required')
    fields=load_neighbor_capture(capture_path=args.capture)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**fields)
    print(json.dumps(dict(scope='frozen candidate curvature input; no own curvature or GA claim',
        capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        vertices=int(fields['header'][0]),entries=int(fields['header'][1]),selected=int(fields['header'][2]))))


if __name__=='__main__':main()
