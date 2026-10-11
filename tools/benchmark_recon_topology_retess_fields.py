"""读取完整动态retess前后状态；冻结参考仅用于隔离同输入验证。"""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib,json
import numpy as np


def load_retessellation_capture(*,capture_path):
    """FNITRTS1→完整有序基底/边/域/标记，单位surface RAS mm。

    输入为固定源只读观察器的文件；返回独立arrays；格式、长度、CSR、
    顶点/边索引不合法抛ValueError。阶段before/after由header[-1]声明，
    仅after包含selected_vertices/selected_edges。此函数不执行拓扑修复。
    """
    raw=Path(capture_path).read_bytes()
    if raw[:8]!=b'FNITRTS1':raise ValueError('complete FNITRTS1 required')
    pos=8
    def read(dt,count,shape=None):
        nonlocal pos
        dt=np.dtype(dt)
        if count<0 or pos+dt.itemsize*count>len(raw):raise ValueError('truncated retessellation capture')
        a=np.frombuffer(raw,dt,count=count,offset=pos).copy();pos+=dt.itemsize*count
        return a.reshape(shape) if shape else a
    h=read('<i4',11);n,f,m,k,ni,nb,nh,mode,mapping,no,phase=map(int,h)
    if n<1 or min(f,m,k,ni,nb,nh,no)<0 or max(h)>100000000 or mode not in (0,1) or phase not in (0,1):raise ValueError('invalid retessellation header')
    fields=dict(header=h,vertex_fields=read('<f4',n*9,(n,9)),vertex_flags=read('<i4',n*3,(n,3)),faces=read('<i4',f*3,(f,3)),neighbor_offsets=read('<i4',n+1),neighbors=read('<i4',m),edge_fields=read('<i4',k*4,(k,4)),ordering=read('<i4',k),overlap_offsets=read('<i4',k+1),overlaps=read('<i4',no),inside_fields=read('<i4',ni*2,(ni,2)),border_vertices=read('<i4',nb),hull_vertices=read('<i4',nh),selected_counts=read('<i4',4))
    c=fields['selected_counts']
    if c.min()<0 or c.max()>max(n,k):raise ValueError('invalid retess selected dimensions')
    fields['selected_vertices']=read('<i4',int(c[0])) if phase else np.empty(0,np.int32)
    fields['selected_edges']=read('<i4',int(c[1])) if phase else np.empty(0,np.int32)
    if pos!=len(raw):raise ValueError('retessellation trailing data')
    for start,end,name in [(fields['neighbor_offsets'],m,'neighbor'),(fields['overlap_offsets'],no,'overlap')]:
        if start[0]!=0 or start[-1]!=end or np.any(np.diff(start)<0):raise ValueError(name+' CSR invalid')
    for a,lim in [(fields['faces'],n),(fields['neighbors'],n),(fields['edge_fields'][:,:2],n),(fields['inside_fields'][:,0],n),(fields['border_vertices'],n),(fields['hull_vertices'],n),(fields['ordering'],k),(fields['overlaps'],k),(fields['selected_vertices'],n),(fields['selected_edges'],k)]:
        if a.size and (a.min()<0 or a.max()>=lim):raise ValueError('retessellation index out of range')
    if not np.array_equal(np.sort(fields['ordering']),np.arange(k)):raise ValueError('complete edge permutation required')
    if not np.isfinite(fields['vertex_fields']).all():raise ValueError('nonfinite retess coordinate')
    return fields


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('new diagnostic checkpoint required')
    fields=load_retessellation_capture(capture_path=a.capture);a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**fields)
    print(json.dumps(dict(capture_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest(),header=fields['header'].tolist())))


if __name__=='__main__':main()
