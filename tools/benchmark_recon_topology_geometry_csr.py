"""实验：当前候选有序面→完整CSR，串行编译计数/填充，不接入生产。

只比较整数拓扑准备；不缓存任何跨候选面、坐标、法线或MRI反馈。
真实benchmark必须显式提供当前自产候选faces文件及source/input hashes。
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numba import njit
from fnit.recon_all import place_surface_normals as mature_normals
from fnit.recon_all.place_surface_normals import ordered_face_csr


@njit(cache=True, fastmath=False)
def _counted_csr(faces, nvertices):
    offsets=np.zeros(nvertices+1,np.int64)
    for face in range(len(faces)):
        for corner in range(3):offsets[faces[face,corner]+1]+=1
    for vertex in range(nvertices):offsets[vertex+1]+=offsets[vertex]
    cursor=offsets[:-1].copy()
    face_ids=np.empty(3*len(faces),np.int64)
    corners=np.empty(3*len(faces),np.int64)
    for face in range(len(faces)):
        for corner in range(3):
            vertex=faces[face,corner];entry=cursor[vertex]
            face_ids[entry]=face;corners[entry]=corner;cursor[vertex]+=1
    return offsets,face_ids,corners


def counted_face_csr(faces, nvertices):
    """实验完整整数CSR，保留每个顶点的源face/corner次序及重复项。

    faces:当前候选有序整数(F,3)，nvertices:非负Python/NumPy整数N。
    输出独立int64 offsets(N+1)、face_ids(3F)、corners(3F)，同成熟接口。
    每次读取当前faces，输入不变；空行/空面合法；非法输入分配前报错。
    只允许实验驱动显式调用，不改变FaceNormalTopology或生产默认。
    """
    if not isinstance(nvertices,(int,np.integer))or isinstance(nvertices,(bool,np.bool_))or nvertices<0:
        raise ValueError('nvertices must be a nonnegative integer')
    tri=np.asarray(faces)
    if tri.ndim!=2 or tri.shape[1]!=3 or not np.issubdtype(tri.dtype,np.integer):
        raise ValueError('faces must be integer (F,3)')
    if np.any(tri<0)or np.any(tri>=nvertices):raise ValueError('face index outside vertex domain')
    if nvertices>np.iinfo(np.int64).max-1 or 3*len(tri)>np.iinfo(np.int64).max:
        raise ValueError('CSR exceeds int64 capacity')
    return _counted_csr(np.ascontiguousarray(tri,np.int64),int(nvertices))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-npz',type=Path,required=True)
    parser.add_argument('--candidate-sha256',required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--source-manifest-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pairs',type=int,default=3)
    args=parser.parse_args()
    if args.pairs<1:raise ValueError('positive pairs required')
    if sha(args.candidate_npz)!=args.candidate_sha256 or sha(args.source_manifest)!=args.source_manifest_sha256:raise ValueError('frozen source/input identity mismatch')
    manifest=json.loads(args.source_manifest.read_text())
    for name,record in manifest['payload'].items():
        if sha(args.source_manifest.parent/name)!=record['sha256']:raise ValueError('frozen payload changed: '+name)
    actual_module_sha=sha(Path(mature_normals.__file__))
    mature_record=manifest['payload'].get('src/fnit/recon_all/place_surface_normals.py')
    driver_record=manifest['payload'].get('tools/benchmark_recon_topology_geometry_csr.py')
    if mature_record is None or driver_record is None or actual_module_sha!=mature_record['sha256'] or sha(Path(__file__))!=driver_record['sha256']:
        raise ValueError('actual imported mature module or experiment driver differs from frozen payload')
    with np.load(args.candidate_npz,allow_pickle=False)as data:
        faces=data['faces'].copy();nvertices=len(data['original_vertices'])
    args.output.mkdir(parents=True,exist_ok=False)
    records=[];reference=None
    # Fresh process first calls include this process JIT/cache loading; caches may exist.
    orders=[('baseline','compiled')]+[('compiled','baseline')if i%2==0 else('baseline','compiled')for i in range(args.pairs)]
    for pair,order in enumerate(orders):
        for backend in order:
            tick=time.perf_counter();values=(ordered_face_csr(faces,nvertices)if backend=='baseline'else counted_face_csr(faces,nvertices));seconds=time.perf_counter()-tick
            if reference is None:reference=tuple(x.copy()for x in values)
            exact=all(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()for a,b in zip(values,reference))
            if not exact:raise AssertionError('complete CSR mismatch')
            records.append(dict(pair=pair,backend=backend,seconds=seconds,all_arrays_byte_equal=exact))
    report=dict(scope='complete current-candidate integer CSR API only; not geometry/fitness/GA/MRI benchmark',input_sha256=args.candidate_sha256,source_manifest_sha256=args.source_manifest_sha256,driver_sha256=sha(Path(__file__)),actual_mature_module_sha256=actual_module_sha,faces=len(faces),vertices=nvertices,records=records,production_default_changed=False,full_geometry='not_assessed',whole_GA='not_assessed')
    result=args.output/'report.private.json';result.write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(status='complete',report_sha256=sha(result)))+'\n')


if __name__=='__main__':main()
