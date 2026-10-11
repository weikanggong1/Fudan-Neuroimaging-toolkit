"""灰白组织校准的真实入口、原PDF及平滑PDF完整只读解析。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load_calibration_capture(*, capture_path):
    """FNITHST1→原nofix序的几何/强度/完整chull/actual selected/PDF。

    capture_path为完整源观测；返回header i32[12](N,F,邻接项,chullK,insideK,
    borderK,phase,actualSelectedK,4histBins)、vertex_fields f32(N,9)(origXYZ,
    normalXYZ,val,val2,val2bak；surface RAS/mm及MRI强度)、flags i32(N,2)
    (marked,rip)、topology i32(N,3)(vnum,vtotal,nsizeCur)、faces i32(F,3)、
    ordered vtotal neighbor CSR、hull、inside_status i32(K,2)、border及actual
    selected i32，4类white/gray/border/gradient bin_size及bins/counts f32。
    phase0入口/1原PDF/2平滑PDF；不是排序边或corrected编号，映射由调用方
    自产。返回数组独立，非法header/CSR/索引/截断/尾部报ValueError；只供
    同输入诊断，不能把参考PDF传入生产来补齐自产统计。
    """
    raw=Path(capture_path).read_bytes()
    if raw[:8]!=b'FNITHST1':raise ValueError('complete FNITHST1 required')
    pos=8
    def read(dtype,count,shape=None):
        nonlocal pos
        dt=np.dtype(dtype)
        if count<0 or pos+dt.itemsize*count>len(raw):raise ValueError('truncated calibration capture')
        data=np.frombuffer(raw,dt,count=count,offset=pos).copy();pos+=dt.itemsize*count
        return data.reshape(shape) if shape is not None else data
    h=read('<i4',12);n,f,ne,nh,nd,nb,phase,ns,hw,hg,hb,hd=map(int,h)
    if min(n,hw,hg,hb,hd)<1 or min(f,ne,nh,nd,nb,ns)<0 or max(h)>100000000 or phase not in (0,1,2):raise ValueError('invalid calibration header')
    fields=dict(header=h,vertex_fields=read('<f4',n*9,(n,9)),flags=read('<i4',n*2,(n,2)),
                topology=read('<i4',n*3,(n,3)),faces=read('<i4',f*3,(f,3)),
                neighbor_offsets=read('<i4',n+1),neighbors=read('<i4',ne),
                hull=read('<i4',nh),inside_status=read('<i4',nd*2,(nd,2)),border=read('<i4',nb),selected=read('<i4',ns))
    for name,count in zip(['white','gray','border','gradient'],[hw,hg,hb,hd]):
        fields[name+'_bin_size']=read('<f4',1)
        fields[name+'_bins']=read('<f4',count)
        fields[name+'_counts']=read('<f4',count)
    if pos!=len(raw):raise ValueError('unexpected calibration trailing data')
    off=fields['neighbor_offsets'];top=fields['topology']
    if off[0]!=0 or off[-1]!=ne or np.any(np.diff(off)<0) or np.any(top<0) or not np.array_equal(np.diff(off),top[:,1]) or np.any(top[:,0]>top[:,1]):raise ValueError('invalid calibration ordered CSR')
    for name in ['faces','neighbors','hull','border','selected']:
        a=fields[name]
        if a.size and (a.min()<0 or a.max()>=n):raise ValueError(name+' index out of range')
    if fields['inside_status'].size and (fields['inside_status'][:,0].min()<0 or fields['inside_status'][:,0].max()>=n):raise ValueError('inside index out of range')
    if not np.isfinite(fields['vertex_fields']).all():raise ValueError('nonfinite calibration geometry/intensity')
    return fields


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError('new calibration checkpoint required')
    f=load_calibration_capture(capture_path=a.capture);a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**f)
    print(json.dumps(dict(scope='frozen histogram calibration reference, not production input',capture_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest(),header=f['header'].tolist())))


if __name__=='__main__':main()
