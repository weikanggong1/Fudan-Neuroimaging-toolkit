"""解析局部缺陷 volume/PDF 创建与 MRI 采样前后的只读诊断输入。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load_volume_preparation_capture(*,capture_path):
    """FNITPDF1 → 全 origXYZ/ETable/白灰PDF及前mask或后intensity/logPDF。

    capture_path 是源观察器记录的完整文件路径，无默认值。返回独立
    header(int32/8)，algorithm_volume_geometry(float32/9：start、索引
    每mm乘数、end)，parameters(float32/7：resolution、两bin_size、
    两mean、两mean_ll)，ras_to_voxel(float64/4x4，原MRI-only矩阵)，
    original_vertices(float32/Nx3，tkregister surface RAS/mm)，
    potential_edges(int32/Kx2，原表顺序)，white/gray_bins、white/gray_pdf
    (float32)，intensity(uint8/WxHxD，phase0是初始mask，phase1是MRI强度)。
    phase1另含 white/gray_log_pdf(float32/WxHxD)。MRI 原体素不嵌入，
    调用方必须另核对同原brain的SHA。错误格式/尺寸/索引/尾部抛ValueError；
    此参考只用于同输入回归，不用于生产求解或补齐输出。
    """
    raw=Path(capture_path).read_bytes()
    if raw[:8]!=b'FNITPDF1':raise ValueError('complete FNITPDF1 required')
    pos=8
    def read(dtype,count,shape=None):
        nonlocal pos
        dt=np.dtype(dtype)
        if count<0 or pos+dt.itemsize*count>len(raw):raise ValueError('truncated volume preparation capture')
        data=np.frombuffer(raw,dt,count=count,offset=pos).copy();pos+=dt.itemsize*count
        return data.reshape(shape) if shape else data
    h=read('<i4',8);n,k,w,d1,d2,hw,hg,phase=map(int,h)
    if min(n,w,d1,d2,hw,hg)<1 or k<0 or max(h)>100000000 or phase not in (0,1):raise ValueError('invalid volume preparation dimensions')
    fields=dict(header=h,algorithm_volume_geometry=read('<f4',9),parameters=read('<f4',7),
        ras_to_voxel=read('<f8',16,(4,4)),original_vertices=read('<f4',n*3,(n,3)),
        potential_edges=read('<i4',k*2,(k,2)),white_bins=read('<f4',hw),white_pdf=read('<f4',hw),gray_bins=read('<f4',hg),gray_pdf=read('<f4',hg),intensity=read('u1',w*d1*d2,(w,d1,d2)))
    if phase:
        fields['white_log_pdf']=read('<f4',w*d1*d2,(w,d1,d2));fields['gray_log_pdf']=read('<f4',w*d1*d2,(w,d1,d2))
    if pos!=len(raw):raise ValueError('unexpected volume preparation trailing data')
    edges=fields['potential_edges']
    if edges.size and (edges.min()<0 or edges.max()>=n):raise ValueError('potential edge endpoint out of range')
    if not np.isfinite(fields['original_vertices']).all() or not np.isfinite(fields['ras_to_voxel']).all():raise ValueError('nonfinite preparation coordinates')
    return fields


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError('new volume preparation diagnostic checkpoint required')
    f=load_volume_preparation_capture(capture_path=a.capture);a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**f)
    print(json.dumps(dict(scope='frozen volume/PDF preparation reference',capture_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest(),phase=int(f['header'][7]),shape=list(f['intensity'].shape))))


if __name__=='__main__':main()
