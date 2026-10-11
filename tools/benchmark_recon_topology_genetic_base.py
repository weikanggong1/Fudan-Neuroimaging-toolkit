"""完整初始 GA base 的冻结输入回归；after 捕获只在候选保存后比较。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np

from fnit.recon_all.topology_genetic_base import (
    create_genetic_base,COPIED_VECTOR_FIELDS,COPIED_SCALAR_FIELDS)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(offsets,values,n,name):
    if (offsets.shape!=(n+1,) or offsets[0]!=0 or offsets[-1]!=len(values)
            or np.any(np.diff(offsets)<0)):
        raise ValueError(name+' inconsistent ordered offsets')
    return [values[int(offsets[i]):int(offsets[i+1])].copy() for i in range(n)]


def load_base_capture(*,capture_path):
    """只读 FNITBAS1→显式完整 copied/zero attrs、原序 CSR、maps 与 metadata。

    binary必须完整；原 one-ring vnum 行与入射face/corner 行按原访问顺序
    保留。phase0是源 TMP restore+metric 后，phase1是新 base 初始化之后；
    解析器不运行/修改原算法。face_normals 未求值、不虚构参考法向数组。
    不消费额外未用 storage/capacity，元数据保留实际容量。
    """
    raw=Path(capture_path).read_bytes()
    if raw[:8]!=b'FNITBAS1':raise ValueError('complete FNITBAS1 capture required')
    offset=8
    def read(dtype,count,shape=None):
        nonlocal offset
        dt=np.dtype(dtype)
        if count<0 or offset+count*dt.itemsize>len(raw):raise ValueError('truncated initial base capture')
        a=np.frombuffer(raw,dt,count=count,offset=offset).copy();offset+=count*dt.itemsize
        return a.reshape(shape) if shape else a
    header=read('<i4',18)
    n,nf,en,ei,phase,sn,sf,vcap,fcap,_,_,_,_,_,_,_,nd,selected=map(int,header)
    if (min(n,nf,en,ei,sn,sf,vcap,fcap,nd,selected)<0 or n==0 or sn==0
            or max(n,nf,en,ei,sn,sf,vcap,fcap,nd,selected)>100000000
            or phase not in (0,1) or vcap<n or fcap<nf):
        raise ValueError('invalid base dimensions/capacity/phase')
    vg_dimensions=read('<i4',5);vg_fields=read('<f8',18)
    filename_length=int(read('<i4',1)[0])
    if filename_length<0 or filename_length>4096:raise ValueError('invalid volume geometry fname length')
    filename_bytes=read('u1',filename_length);cache_flags=read('<i4',3)
    if np.any((cache_flags!=0)&(cache_flags!=1)):raise ValueError('invalid VG cache flags')
    cache_i_to_r=read('<f4',16) if cache_flags[0] else None
    caches=[]
    for present in cache_flags[1:]:
        if not present:caches.append(None);continue
        dimensions=read('<i4',3);rows,cols,kind=map(int,dimensions)
        if rows<1 or cols<1 or rows*cols>1000000:raise ValueError('invalid cached MATRIX size')
        caches.append(dict(dimensions=dimensions,values=read('<f8',rows*cols,(rows,cols))))
    fields=read('<f4',n*42,(n,42));flags=read('<i4',n*3,(n,3));topology=read('<i4',n*7,(n,7))
    faces=np.empty((nf,3),np.int32);face_rip=np.empty(nf,np.int32);face_area=np.empty(nf,np.float32)
    for f in range(nf):
        a=read('<i4',4);faces[f]=a[:3];face_rip[f]=a[3];face_area[f]=read('<f4',1)[0]
    neighbor_offsets=read('<i4',n+1);neighbors=read('<i4',en)
    incident_offsets=read('<i4',n+1)
    if (incident_offsets[0]!=0 or incident_offsets[-1]!=ei or np.any(np.diff(incident_offsets)<0)):
        raise ValueError('invalid initial incident offsets')
    incident_faces=np.empty(ei,np.int32);incident_corners=np.empty(ei,np.uint8)
    for v in range(n):
        a,b=map(int,incident_offsets[v:v+2]);inc=read('<i4',b-a);corner=read('<i4',b-a)
        if corner.size and (corner.min()<0 or corner.max()>2):raise ValueError('invalid original triangle corner')
        incident_faces[a:b]=inc;incident_corners[a:b]=corner
    labels=read('<i4',sn);status=read('<i4',sn);defect_sizes=read('<i4',nd)
    if np.any(defect_sizes<0) or int(defect_sizes.sum(dtype=np.int64))!=selected:
        raise ValueError('invalid source ordered defect sizes')
    defect_order=read('<i4',selected);vtrans=read('<i4',sn);ftrans=read('<i4',sf)
    if offset!=len(raw):raise ValueError('unexpected initial base trailing bytes')
    if (np.any(labels<0) or np.any(labels>nd) or selected!=np.count_nonzero(labels)
            or (defect_order.size and (defect_order.min()<0 or defect_order.max()>=sn))):
        raise ValueError('invalid initial source labels/order')
    vectors={name:fields[:,3*i:3*i+3].copy() for i,name in enumerate(COPIED_VECTOR_FIELDS)}
    scalars={name:fields[:,15+i].copy() for i,name in enumerate(COPIED_SCALAR_FIELDS)}
    state=dict(**vectors,**scalars,nfaces=nf,faces=faces,face_rip=face_rip,face_area=face_area,
        flattened_vertices=fields[:,30:33].copy(),tangent_e1=fields[:,33:36].copy(),
        tangent_e2=fields[:,36:39].copy(),vertex_area=fields[:,39].copy(),
        undefval=fields[:,40].copy(),old_undefval=fields[:,41].copy(),
        vertex_marks=flags[:,0].copy(),vertex_rip=flags[:,1].copy(),vertex_border=flags[:,2].copy(),
        vertex_topology=topology,
        neighbor_storage=_rows(neighbor_offsets,neighbors,n,'initial one-ring'),
        incident_faces=_rows(incident_offsets,incident_faces,n,'initial incident faces'),
        incident_corners=_rows(incident_offsets,incident_corners,n,'initial incident corners'))
    metadata=dict(use_real_ras=int(header[11]),hemisphere=int(header[12]),
        volume_geometry=dict(dimensions=vg_dimensions,fields=vg_fields,filename_bytes=filename_bytes,
            i_to_r__=cache_i_to_r,r_to_i__=caches[0],register_mat=caches[1]),
        status=int(header[14]),origxyz_status=int(header[15]))
    return dict(header=header,surface_state=state,source_metadata=metadata,
        source_defect_labels=labels,source_defect_status=status,
        source_defect_sizes=defect_sizes,source_defect_order=defect_order,
        vertex_translation=vtrans,face_translation=ftrans,capture_sha256=sha(capture_path))


def _compare(a,b):
    x,y=np.asarray(a),np.asarray(b)
    if x.shape!=y.shape or x.dtype!=y.dtype:
        return dict(shape_equal=x.shape==y.shape,dtype_equal=x.dtype==y.dtype,bitwise_equal=False)
    rawx,rawy=np.frombuffer(x.tobytes(),np.uint8),np.frombuffer(y.tobytes(),np.uint8)
    if x.dtype.kind=='f':
        finite=np.isfinite(x)&np.isfinite(y);err=np.abs(x[finite].astype(np.float64)-y[finite].astype(np.float64))
        equal=(x==y)|(np.isnan(x)&np.isnan(y))
    else:err=np.abs(x.astype(np.float64)-y.astype(np.float64)).reshape(-1);equal=x==y
    return dict(shape_equal=True,dtype_equal=True,bitwise_equal=x.tobytes()==y.tobytes(),
        differing_values=int(np.count_nonzero(~equal)),differing_bytes=int(np.count_nonzero(rawx!=rawy)),
        max_abs=float(err.max(initial=0)),p99_abs=float(np.percentile(err,99)) if err.size else 0.)


def compare_states(a,b):
    result={}
    for key,value in b.items():
        if key in ('neighbor_storage','incident_faces','incident_corners'):
            aa,bb=a[key],value
            rows=[_compare(x,y) for x,y in zip(aa,bb)]
            result[key]=dict(bitwise_equal=len(aa)==len(bb) and all(r['bitwise_equal'] for r in rows),
                differing_rows=sum(not r['bitwise_equal'] for r in rows),
                differing_values=sum(r.get('differing_values',0) for r in rows),
                differing_bytes=sum(r.get('differing_bytes',0) for r in rows),rows_equal=len(aa)==len(bb))
        else:result[key]=_compare(a[key],value)
    return result


def save_checkpoint(path,result):
    """完整私有候选无pickle NPZ；logical数组与有序 CSR，不发布参考体积。"""
    state=result['surface_state'];values={k:np.asarray(v) for k,v in state.items()
        if k not in ('neighbor_storage','incident_faces','incident_corners')}
    for name,off in [('neighbor_storage','neighbor_offsets'),('incident_faces','incident_offsets'),('incident_corners','incident_offsets')]:
        rows=state[name];o=np.r_[0,np.cumsum([len(v) for v in rows])].astype(np.int64)
        if off in values and not np.array_equal(values[off],o):raise ValueError('candidate face/corner offsets differ')
        values[off]=o;values[name]=np.concatenate(rows)
    values.update(vertex_translation=result['vertex_translation'],face_translation=result['face_translation'],source_order=result['source_order'])
    np.savez_compressed(path,**values)


def run(*,before,after,before_repeat,after_repeat,source_manifest,
        producer_benchmark,observer_build_receipt,observer_run_receipt,output,threads=4):
    """before显式冻结输入→两次完整 producer；保存后才读after/repeat比较。

    全路径具名传入、output新目录；threads≥1，总预算复用成熟thread_budget。
    before全属性是声明的阶段输入（并非自产完整rawMRI链）；after仅诊断。
    校验all copied字段、有序neighbors/incidents、源零分配字段、metadata及
    mappings，保留NaNbits。完整GA/整例/最终质量/整体等效均未判定。
    """
    if output.exists():raise FileExistsError('fresh initial base benchmark output required')
    if threads<1:raise ValueError('positive explicit thread budget required')
    from fnit.recon_all.thread_budget import thread_budget
    started=time.perf_counter();source=load_base_capture(capture_path=before)
    if int(source['header'][4])!=0:raise ValueError('before construction source required')
    state=source['surface_state']
    kwargs=dict(faces=state['faces'],defect_labels=source['source_defect_labels'],
        defect_status=source['source_defect_status'],
        source_vertex_fields={key:state[key] for key in COPIED_VECTOR_FIELDS},
        source_scalar_fields={key:state[key] for key in COPIED_SCALAR_FIELDS},
        source_neighbors=state['neighbor_storage'],source_incident_faces=state['incident_faces'],
        source_incident_corners=state['incident_corners'],source_vertex_rip=state['vertex_rip'],
        source_metadata=source['source_metadata'])
    with thread_budget(threads=threads) as budget:
        first=create_genetic_base(**kwargs);second=create_genetic_base(**kwargs)
    output.mkdir(parents=True);save_checkpoint(output/'candidate_state.private.npz',first)
    candidate_wall=time.perf_counter()-started
    reference=load_base_capture(capture_path=after)
    refbefore=load_base_capture(capture_path=before_repeat);refafter=load_base_capture(capture_path=after_repeat)
    if int(reference['header'][4])!=1 or int(refbefore['header'][4])!=0 or int(refafter['header'][4])!=1:
        raise ValueError('actual source phase0/1 reference pairs required')
    fields=compare_states(first['surface_state'],reference['surface_state'])
    mappings={k:_compare(first[k],reference[k]) for k in ('vertex_translation','face_translation')}
    repeat=compare_states(first['surface_state'],second['surface_state'])
    source_repeat={
        'before':dict(fields=compare_states(source['surface_state'],refbefore['surface_state']),
            binary_equal=source['capture_sha256']==refbefore['capture_sha256']),
        'after':dict(fields=compare_states(reference['surface_state'],refafter['surface_state']),
            binary_equal=reference['capture_sha256']==refafter['capture_sha256'])}
    metadata={k:_compare(first['metadata'][k],reference['source_metadata'][k])
        for k in ('use_real_ras','hemisphere','status','origxyz_status')}
    def compare_geometry(a,b,prefix='volume_geometry'):
        if isinstance(b,dict):
            for k,value in b.items():compare_geometry(a[k],value,prefix+'.'+k)
        elif b is None or a is None:
            metadata[prefix]=dict(bitwise_equal=a is None and b is None)
        else:metadata[prefix]=_compare(a,b)
    compare_geometry(first['metadata']['volume_geometry'],reference['source_metadata']['volume_geometry'])
    metadata.update(surface_type_code=_compare(np.int32(first['metadata']['surface_type_code']),reference['header'][13]),
        vertex_capacity=_compare(np.int32(first['metadata']['vertex_capacity']),reference['header'][7]),
        face_capacity=_compare(np.int32(first['metadata']['face_capacity']),reference['header'][8]),
        nsize=_compare(np.int32(first['metadata']['nsize']),reference['header'][9]),
        max_nsize=_compare(np.int32(first['metadata']['max_nsize']),reference['header'][10]))
    report=dict(status='complete_same_input_initial_genetic_base_regression',
        stage_execution_complete=True,strict_base_passed=all(r['bitwise_equal'] for group in (fields,mappings,metadata) for r in group.values()),
        comparisons=fields,mapping_comparisons=mappings,metadata_comparisons=metadata,
        candidate_repeat=repeat,candidate_repeat_exact=all(x['bitwise_equal'] for x in repeat.values()),
        reference_repeat=source_repeat,reference_repeat_exact=all(g['binary_equal'] for g in source_repeat.values()),
        source_vertices=int(source['header'][0]),source_faces=int(source['header'][1]),
        kept_vertices=len(first['source_order']),kept_faces=first['surface_state']['nfaces'],
        removed_faces=int(np.count_nonzero(first['face_translation']<0)),
        defects=int(source['header'][16]),marked_vertices=int(np.count_nonzero(source['source_defect_labels'])),
        discard_vertices=int(np.count_nonzero(first['surface_state']['vertex_rip'])),
        candidate_seconds=first['seconds'],candidate_repeat_seconds=second['seconds'],
        candidate_wall_including_before_and_checkpoint=candidate_wall,tool_wall_before_report=time.perf_counter()-started,
        input_capture_sha256=source['capture_sha256'],reference_capture_sha256=reference['capture_sha256'],
        reference_repeat_capture_sha256={k:v['capture_sha256'] for k,v in [('before',refbefore),('after',refafter)]},
        source_manifest_sha256=sha(source_manifest),producer_benchmark_sha256=sha(producer_benchmark),
        observer_build_receipt_sha256=sha(observer_build_receipt),observer_run_receipt_sha256=sha(observer_run_receipt),
        source_module_sha256=sha(Path(__file__).parents[1]/'src/fnit/recon_all/topology_genetic_base.py'),
        benchmark_sha256=sha(__file__),after_state_inputs_consumed=False,
        declared_frozen_stage_fields_consumed=True,raw_mri_to_base_fully_self_generated=False,
        face_normal_cache_zero_is_allocator_contract_not_observer_evaluation=True,
        environment=dict(python=platform.python_version(),numpy=np.__version__,thread_budget=budget),
        whole_ga=False,whole_recon_all=False,mesh_quality='not_assessed',overall_equivalence='not_assessed')
    (output/'report.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],
        strict_base_passed=report['strict_base_passed'],report_sha256=sha(output/'report.private.json')),indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['before','after','before-repeat','after-repeat','source-manifest','producer-benchmark',
                 'observer-build-receipt','observer-run-receipt','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    run(**vars(parser.parse_args()))


if __name__=='__main__':main()
