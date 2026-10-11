"""只读审查 pial 残余组中的不可移动面/线段交会，不扩展优化搜索。"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import resource
import time

FACE_EDGE_KINDS={"none","proper_interior","closed_contact","coplanar_contact"}
EDGE_EDGE_KINDS={"none","transverse_interiors","closed_contact","collinear_overlap"}
INPUTS={"surf/lh.white","surf/autodet.gw.stats.lh.dat","label/lh.cortex.label","label/lh.cortex+hipamyg.label", "mri/brain.finalsurfs.mgz","mri/wm.mgz","mri/aseg.presurf.mgz"}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def points(array):return [tuple(Fraction.from_float(float(v)) for v in row) for row in array]


def segment_triangle_relation(segment,triangle):
    """有限IEEE→精确固定线段/三角面关系，无长度容差、无索引豁免。

    proper_interior要求端点严格异侧且交点在三角形开内部；固定面与
    固定边满足此条件时，另一非退化三角面无法靠移动第三角点消除
    proper交叉。closed/coplanar只证明持久接触，不冒称内部穿越。
    """
    a,b=points(segment);tri=points(triangle);normal=cross(sub(tri[1],tri[0]),sub(tri[2],tri[0]))
    if a==b or not any(normal):raise ValueError("nondegenerate segment/triangle required")
    da,db=dot(sub(a,tri[0]),normal),dot(sub(b,tri[0]),normal)
    if da==db==0:
        lower,upper=Fraction(0),Fraction(1)
        for p,q in zip(tri,tri[1:]+tri[:1]):
            va=dot(cross(sub(q,p),sub(a,p)),normal);vb=dot(cross(sub(q,p),sub(b,p)),normal)
            delta=vb-va
            if delta==0:
                if va<0:return "none"
            elif delta>0:lower=max(lower,-va/delta)
            else:upper=min(upper,-va/delta)
        return "coplanar_contact" if lower<=upper else "none"
    if da*db>0:return "none"
    t=da/(da-db)
    if not 0<=t<=1:return "none"
    p=tuple(a[i]+t*(b[i]-a[i]) for i in range(3))
    signs=[dot(cross(sub(q,s),sub(p,s)),normal) for s,q in zip(tri,tri[1:]+tri[:1])]
    if min(signs)<0:return "none"
    return "proper_interior" if da*db<0 and min(signs)>0 else "closed_contact"


def segment_segment_relation(first,second):
    """精确三维固定线段关系；交叉只证明持久接触，未判三角面proper。"""
    a,b=points(first);c,d=points(second);u,v,w=sub(b,a),sub(d,c),sub(c,a)
    if not any(u) or not any(v):raise ValueError("nondegenerate segments required")
    normal=cross(u,v)
    if not any(normal):
        if any(cross(w,u)):return "none"
        axis=max(range(3),key=lambda i:abs(u[i]));t0,t1=(c[axis]-a[axis])/u[axis],(d[axis]-a[axis])/u[axis]
        lower,upper=max(Fraction(0),min(t0,t1)),min(Fraction(1),max(t0,t1))
        return "none" if lower>upper else "collinear_overlap" if lower<upper else "closed_contact"
    if dot(w,normal)!=0:return "none"
    drop=max(range(3),key=lambda i:abs(normal[i]));i,j=[k for k in range(3) if k!=drop]
    determinant=u[i]*v[j]-u[j]*v[i]
    t=(w[i]*v[j]-w[j]*v[i])/determinant;s=(w[i]*u[j]-w[j]*u[i])/determinant
    if not (0<=t<=1 and 0<=s<=1):return "none"
    return "transverse_interiors" if 0<t<1 and 0<s<1 else "closed_contact"


def run(*,diagnostic_case:Path,candidate:Path,input_hashes:Path,terminal_sha256:str,candidate_sha256:str,
        exact_report:Path,expected_exact_report_sha256:str,output:Path,threads:int=4):
    """只读七输入/两自产末态/已完整精确名单，逐pair审查全部固定子结构。

    8个Path/str参数均必填；output须不存在。threads默认4正整数，为
    资源声明，实际亲和性另记录。几何float32(N,3)、ordered faces
    整数(F,3)，空间surface RAS/mm。两掩膜按rip或cortex外定义固定，
    输出全部pair的固定来源/同ownwhite坐标一致性、固定面–边与边–边
    类别及聚合/输入/源码SHA、墙钟/RSS。无优化/GPU/原阶段程序；
    哈希漂移、非法名单/几何或既有文件直接失败，不更改MRI或质量门。
    """
    started=time.perf_counter()
    if output.exists():raise FileExistsError(output)
    if type(threads) is not int or threads<1:raise ValueError("positive integer threads required")
    expected=json.loads(input_hashes.read_text())
    if set(expected)!=INPUTS:raise ValueError("frozen seven input map required")
    terminal=diagnostic_case/"cleanup-result.private.npy"
    before={k:sha(diagnostic_case/"subject"/k) for k in sorted(INPUTS)}
    if before!=expected or sha(terminal)!=terminal_sha256 or sha(candidate)!=candidate_sha256 or sha(exact_report)!=expected_exact_report_sha256:raise ValueError("frozen input/list differs")
    d=json.loads(exact_report.read_text())
    if d["execution_status"]!="complete_readonly_exact_pair_audit" or d["input_sha256"]!=expected or d["terminal_sha256"]!=terminal_sha256 or d["candidate_sha256"]!=candidate_sha256:raise ValueError("actual exact report provenance differs")
    import numpy as np
    import nibabel.freesurfer as fs
    white,faces=fs.read_geometry(str(diagnostic_case/"subject/surf/lh.white"));white=white.astype(np.float32);faces=faces.astype(np.int32)
    cortex=np.zeros(len(white),bool);rip=np.ones(len(white),bool)
    cortex[fs.read_label(str(diagnostic_case/"subject/label/lh.cortex.label")).astype(np.int64)]=True
    rip[fs.read_label(str(diagnostic_case/"subject/label/lh.cortex+hipamyg.label")).astype(np.int64)]=False
    fixed=rip|~cortex;states=[]
    if d["faces"]!=len(faces) or d["vertices"]!=len(white) or len(d["states"])!=2:raise ValueError("same frozen grid/list required")
    for record,state,path in zip(d["states"],("entry","candidate"),(terminal,candidate)):
        if record["state"]!=state:raise ValueError("two known states required")
        xyz=np.load(path)
        if xyz.dtype!=np.float32 or xyz.shape!=white.shape or not np.isfinite(xyz).all():raise ValueError("finite ordered float32(N,3) required")
        tick=time.perf_counter();rows=[]
        for observed in record["rows"]:
            pair=observed["pair"]
            if type(pair) is not list or len(pair)!=2 or any(type(v) is not int for v in pair) or not 0<=pair[0]<pair[1]<len(faces):raise ValueError("valid complete pair IDs required")
            fa,fb=pair;face_edge=[];edge_edge=[]
            edges=lambda f:[(int(f[i]),int(f[j])) for i,j in ((0,1),(1,2),(2,0)) if fixed[f[i]] and fixed[f[j]]]
            for a,b in ((fa,fb),(fb,fa)):
                if fixed[faces[a]].all():
                    for edge in edges(faces[b]):face_edge.append({"fixed_face":a,"fixed_edge":list(edge),"kind":segment_triangle_relation(xyz[list(edge)],xyz[faces[a]])})
            for a in edges(faces[fa]):
                for b in edges(faces[fb]):edge_edge.append({"first_edge":list(a),"second_edge":list(b),"kind":segment_segment_relation(xyz[list(a)],xyz[list(b)])})
            corners=[]
            for v in sorted(set(map(int,faces[[fa,fb]].ravel()))):
                if fixed[v]:corners.append({"vertex":v,"fixed_source":"rip_and_outside_cortex" if rip[v] and not cortex[v] else "rip" if rip[v] else "outside_cortex", "equals_own_white":bool(np.array_equal(xyz[v],white[v]))})
            rows.append({"pair":pair,"fixed_corners":corners,"face_edge_checks":face_edge,"edge_edge_checks":edge_edge, "immutable_proper_obstruction":any(v["kind"]=="proper_interior" for v in face_edge), "immutable_contact_obstruction":any(v["kind"]!="none" for v in face_edge+edge_edge)})
        states.append({"state":state,"pairs":len(rows),"pairs_with_immutable_proper_obstruction":sum(r["immutable_proper_obstruction"] for r in rows), "pairs_with_any_immutable_contact":sum(r["immutable_contact_obstruction"] for r in rows), "face_edge_classes":{k:sum(v["kind"]==k for r in rows for v in r["face_edge_checks"]) for k in FACE_EDGE_KINDS}, "edge_edge_classes":{k:sum(v["kind"]==k for r in rows for v in r["edge_edge_checks"]) for k in EDGE_EDGE_KINDS}, "exact_checks_seconds":time.perf_counter()-tick,"rows":rows})
    if before!={k:sha(diagnostic_case/"subject"/k) for k in sorted(INPUTS)} or sha(terminal)!=terminal_sha256 or sha(candidate)!=candidate_sha256 or sha(exact_report)!=expected_exact_report_sha256:raise RuntimeError("frozen sources changed")
    result={"schema_version":1,"execution_status":"complete_readonly_fixed_structure_audit", "scope":"all fixed substructures of existing frozen exact selfpair lists; no joint optimizer or new quality gate", "script_sha256":sha(__file__),"exact_report_sha256":expected_exact_report_sha256,"input_sha256":before, "terminal_sha256":terminal_sha256,"candidate_sha256":candidate_sha256,"inputs_unchanged":True, "coordinate_space":"surface_RAS","coordinate_unit":"mm","vertices":len(white),"faces":len(faces),"fixed_vertices":int(fixed.sum()), "threads":threads,"cpu_affinity":sorted(os.sched_getaffinity(0)),"states":states, "wall_seconds_before_report_write":time.perf_counter()-started,"peak_process_cpu_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024), "gpu_computation":False,"gpu_memory_bytes":None,"original_stage_program_called":False,"quality_gate_changed":False, "overall_metric_equivalence":"not_assessed"}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic-case","candidate","input-hashes","exact-report","output"):parser.add_argument("--"+name,type=Path,required=True)
    for name in ("terminal-sha256","candidate-sha256","expected-exact-report-sha256"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--threads",type=int,default=4);d=run(**vars(parser.parse_args()))
    print(json.dumps({"execution_status":d["execution_status"],"immutable_proper_pairs":[v["pairs_with_immutable_proper_obstruction"] for v in d["states"]]}))
