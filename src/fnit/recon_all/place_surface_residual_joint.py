"""可选单残余组联合投影；一次候选族 LP 与一次 SLSQP，不接生产。"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import linprog, minimize

from .place_surface_residual_cleanup import _checked, _face_normals, _pair_sets
from .place_surface_residual_projection import _separating_planes, _skew
from .white_pial_crossing_python import WhitePialCrossingIndex, _rational_triangle, _exact_coplanar_kind


def exact_affected_self_pairs(*, vertices, faces, query_faces, exact_relation):
    """受影响面×全网格inclusive bbox→完整独立精确局部面对及计时。

    vertices/faces为空间surface RAS/mm有限float32(N,3)/整数(F,3)，
    query_faces有效一维整数面编号，exact_relation独立逐边裁剪callable。
    min/max直接取原IEEE角点，F64转换不改变float32值，不用距离
    容差/半径索引；无共享ID排除，只去同一面和重复unordered pair。
    返回proper、共面正面积、所有精确阳性三个set及类别/时间dict。
    这是局部完备门，不宣称全图exact验收。非法域/退化面明确失败。
    """
    xyz,tri,_,_=_checked(vertices,faces,np.zeros(len(vertices),bool),vertices)
    query=np.asarray(query_faces)
    if query.ndim!=1 or not np.issubdtype(query.dtype,np.integer) or len(query) and (query.min()<0 or query.max()>=len(tri)) or not callable(exact_relation):
        raise ValueError("valid integer query face domain and independent classifier required")
    query=np.unique(query);points=xyz.astype(float)[tri]
    if np.any(np.linalg.norm(_face_normals(xyz,tri),axis=1)==0):raise ValueError("nondegenerate faces required")
    tick=time.perf_counter();low,high=points.min(axis=1),points.max(axis=1);candidates=set();selected=set(map(int,query))
    for fa in query:
        ids=np.flatnonzero(np.all(low[fa]<=high,axis=1)&np.all(low<=high[fa],axis=1))
        candidates.update((min(int(fa),int(fb)),max(int(fa),int(fb))) for fb in ids if fa!=fb and not (fb in selected and fb<fa))
    bbox_seconds=time.perf_counter()-tick;tick=time.perf_counter()
    proper,area,positive=set(),set(),set()
    classes={k:0 for k in ('none','proper','edge_point_contact','coplanar_area_overlap','coplanar_edge_point_contact')}
    shared_positive=0
    for fa,fb in sorted(candidates):
        relation=exact_relation(points[fa],points[fb])
        if type(relation) is not int or relation not in (0,1,2,3):raise ValueError("known exact relation required")
        kind=('none','proper','edge_point_contact','coplanar_edge_point_contact')[relation]
        if relation==3 and _exact_coplanar_kind(_rational_triangle(points[fa]),_rational_triangle(points[fb]))==3:
            kind='coplanar_area_overlap';area.add((fa,fb))
        if relation==1:proper.add((fa,fb))
        if relation:positive.add((fa,fb));shared_positive+=bool(set(map(int,tri[fa]))&set(map(int,tri[fb])))
        classes[kind]+=1
    return proper,area,positive,{"query_faces":len(query),"complete_bbox_pairs":len(candidates),"exact_tested_pairs":len(candidates),
        "exact_classes":classes,"positive_pairs_with_shared_vertex_ids":shared_positive,
        "bbox_enumeration_seconds":bbox_seconds,"exact_classification_seconds":time.perf_counter()-tick,
        "candidate_rule":"inclusive original-IEEE bounds; all grid faces; no shared-ID exclusion", "global_exact_scope":"not_assessed"}


def _exact_planes(triangle):
    """复用五个单位半空间次序，并返回有限IEEE对应的精确方向。"""
    points, normal = _rational_triangle(triangle)
    axes = [normal, tuple(-v for v in normal)]
    for start, end, inside in ((points[0], points[1], points[2]),
                               (points[1], points[2], points[0]),
                               (points[2], points[0], points[1])):
        edge = tuple(end[i]-start[i] for i in range(3))
        axis = (edge[1]*normal[2]-edge[2]*normal[1],
                edge[2]*normal[0]-edge[0]*normal[2],
                edge[0]*normal[1]-edge[1]*normal[0])
        if sum((inside[i]-start[i])*axis[i] for i in range(3)) > 0:
            axis = tuple(-v for v in axis)
        axes.append(axis)
    return [(name, origin, axis, exact_axis) for (name, origin, axis), exact_axis
            in zip(_separating_planes(triangle), axes)]


def build_fixed_face_family(*, vertices, faces, fixed_vertices, pairs,
                            clearance_mm: float = .002):
    """同网格单连通残余组→一次预选的线性半空间族。

    vertices有限float32(N,3)，faces整数(F,3)，fixed_vertices为bool(N,)，
    空间surface RAS/mm；pairs完整升序无向面对列表，不得重复。
    clearance_mm默认.002正有限mm，仅移动角点要求此间距；固定角点
    精确接触保持，不添加新的margin。返回active整数(V,)、affected
    整数(A,)、矩阵/右端float64(C,3V)/(C,)及完整方向诊断dict。
    每个完全固定面仅预选原有五半空间中固定角点不处于负侧且预估
    移动最小的一项。固定角点跨面异侧不强制法向分离，可考虑已有
    外侧边半空间；这只是选定候选族，未覆盖不能证明几何无解。
    不修改坐标；非法几何/名单/多连通组抛ValueError。
    """
    xyz, tri, fixed, _ = _checked(vertices, faces, fixed_vertices, vertices)
    if not np.isfinite(clearance_mm) or clearance_mm <= 0:
        raise ValueError("positive finite clearance required")
    normals = _face_normals(xyz, tri)
    if np.any(np.linalg.norm(normals, axis=1) == 0):
        raise ValueError("nondegenerate initial faces required")
    if type(pairs) is not list:
        raise ValueError("complete ordered pair list required")
    checked = []
    for pair in pairs:
        if type(pair) is not tuple or len(pair) != 2 or any(type(v) is not int for v in pair) or not 0 <= pair[0] < pair[1] < len(tri):
            raise ValueError("ordered distinct pair IDs required")
        checked.append(pair)
    if checked != sorted(set(checked)):
        raise ValueError("sorted unique pair list required")
    corner_sets = [set(map(int, tri[list(pair)].ravel())) for pair in pairs]
    connected = set() if not corner_sets else corner_sets[0].copy()
    pending = corner_sets[1:]
    while pending:
        linked = [s for s in pending if connected & s]
        if not linked: raise ValueError("exactly one residual component required")
        for s in linked: connected.update(s)
        pending = [s for s in pending if not connected.issuperset(s)]
    active = np.array(sorted(v for v in connected if not fixed[v]), np.int64)
    slot = {int(v): i for i, v in enumerate(active)}
    affected = np.flatnonzero(np.isin(tri, active).any(axis=1))
    rows, rhs, selections = [], [], []
    for pair in pairs:
        for fixed_face, moving_face in (pair, pair[::-1]):
            if not fixed[tri[fixed_face]].all(): continue
            moving_ids = tri[moving_face]
            rational, _ = _rational_triangle(xyz[moving_ids])
            options = []
            for name, origin, axis, exact_axis in _exact_planes(xyz[tri[fixed_face]]):
                # origin为输入角点的精确IEEE值，不按浮点近零阈值选侧。
                rational_origin = _rational_triangle(np.tile(origin, (3,1)))[0][0]
                signs = [sum((p[i]-rational_origin[i])*exact_axis[i] for i in range(3)) for p in rational]
                fixed_signs = [int((value>0)-(value<0)) for value, v in zip(signs, moving_ids) if fixed[v]]
                distance = (xyz[moving_ids].astype(float)-origin)@axis
                move = ~fixed[moving_ids]
                cost = float(np.sum(np.maximum(0., clearance_mm-distance[move])**2))
                valid = all(s >= 0 for s in fixed_signs)
                options.append({"plane": name, "fixed_corner_signs": fixed_signs,
                                "decision": "eligible_fixed_domain" if valid else "fixed_corner_wrong_side",
                                "optimistic_squared_projection_mm2": cost,
                                "moving_corners": int(move.sum())})
            eligible = [i for i, row in enumerate(options) if row["decision"] == "eligible_fixed_domain"]
            selected = min(eligible, key=lambda i: (options[i]["optimistic_squared_projection_mm2"], i)) if eligible else None
            record = {"pair": list(pair), "fixed_face": fixed_face, "moving_face": moving_face,
                      "candidate_planes": options, "selected_plane": None if selected is None else options[selected]["plane"],
                      "decision": "no_fixed_compatible_halfspace_in_selected_family" if selected is None else "selected_once"}
            selections.append(record)
            if selected is None: continue
            _, origin, axis, _ = _exact_planes(xyz[tri[fixed_face]])[selected]
            for vertex in moving_ids:
                if fixed[vertex]: continue
                row = np.zeros(3*len(active));row[3*slot[int(vertex)]:3*slot[int(vertex)]+3] = axis
                rows.append(row);rhs.append(float(clearance_mm-(xyz[vertex].astype(float)-origin)@axis))
    matrix = np.array(rows, dtype=float).reshape(len(rows), 3*len(active))
    report = {"group_vertices": len(connected), "fixed_group_vertices": len(connected)-len(active),
              "moving_vertices": len(active), "slsqp_dimensions": 3*len(active), "affected_faces": len(affected),
              "halfspace_rows": len(rows), "fixed_face_selections": selections,
              "family_completeness": "selected fixed-face halfspaces only; not all collision-free geometries",
              "existing_fixed_contact_margin_changed": False}
    return active, affected, matrix, np.asarray(rhs), report


def solve_joint_family(*, vertices, faces, active, affected, matrix, rhs,
                       max_displacement_mm: float = 1., minimum_area_ratio: float = .2,
                       optimizer_iterations: int = 100):
    """一次L1最小位移盒约束LP后一次3V维SLSQP，返回trial或None及报告。

    vertices/faces及active/affected同上；matrix/rhs为预选族输出。
    最大欧氏位移默认1mm、每面面积比至少.2、原法向点积正、最多
    100迭代；这些是候选约束，不是整例等效标准。LP采用每坐标
    ±1mm松弛盒及辅助绝对值变量，失败仅针对此族。SLSQP再约束
    欧氏距离及全部入射面；float32量化后必须另经完整质量门。
    LP/SLSQP无重选方向、多seed或重试；不修改输入。非法参数报错。
    """
    if not np.isfinite(max_displacement_mm) or max_displacement_mm <= 0 or not np.isfinite(minimum_area_ratio) or not 0 < minimum_area_ratio <= 1:
        raise ValueError("positive displacement and area ratio in (0,1] required")
    if type(optimizer_iterations) is not int or optimizer_iterations < 1:
        raise ValueError("positive integer iterations required")
    xyz = np.asarray(vertices);tri = np.asarray(faces);active = np.asarray(active);affected = np.asarray(affected)
    xyz,tri,_,_ = _checked(xyz,tri,np.ones(len(xyz),bool),xyz)
    for ids,limit in ((active,len(xyz)),(affected,len(tri))):
        if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or len(ids) and (ids.min()<0 or ids.max()>=limit) or not np.array_equal(ids,np.unique(ids)):
            raise ValueError("sorted unique integer active/affected IDs in domain required")
    if not np.array_equal(affected,np.flatnonzero(np.isin(tri,active).any(axis=1))):
        raise ValueError("affected must contain every incident face and no others")
    if np.any(np.linalg.norm(_face_normals(xyz,tri),axis=1)==0):
        raise ValueError("nondegenerate initial faces required")
    matrix,rhs = np.asarray(matrix),np.asarray(rhs)
    count = len(active);dimensions = 3*count
    if rhs.ndim!=1 or matrix.shape != (len(rhs), dimensions) or not np.issubdtype(matrix.dtype,np.floating) or not np.issubdtype(rhs.dtype,np.floating) or not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise ValueError("finite corresponding halfspace matrix/rhs required")
    if not count or not len(rhs):
        return None, {"decision": "empty_selected_family", "lp_calls": 0, "slsqp_calls": 0}
    tick = time.perf_counter();identity = np.eye(dimensions)
    lp_matrix = np.vstack((np.column_stack((-matrix, np.zeros_like(matrix))),
                           np.column_stack((identity, -identity)), np.column_stack((-identity, -identity))))
    lp_rhs = np.concatenate((-rhs, np.zeros(2*dimensions)))
    lp = linprog(np.concatenate((np.zeros(dimensions), np.ones(dimensions))), A_ub=lp_matrix, b_ub=lp_rhs,
                 bounds=[(-max_displacement_mm,max_displacement_mm)]*dimensions+[(0,max_displacement_mm)]*dimensions,
                 method="highs")
    report = {"lp_calls": 1, "slsqp_calls": 0, "lp_status": int(lp.status), "lp_success": bool(lp.success),
              "lp_seconds": time.perf_counter()-tick, "lp_dimensions_with_l1_auxiliary": 2*dimensions,
              "lp_infeasible_claim_scope": "this preselected halfspace family under coordinate box only"}
    if not lp.success or not np.isfinite(lp.x).all():
        return None, report|{"decision": "selected_family_lp_failed"}
    base = xyz.astype(float)[tri[affected]];n0 = _face_normals(xyz,tri[affected]);scale = np.sum(n0*n0,axis=1)
    slot = np.full(len(xyz),-1,np.int64);slot[active] = np.arange(count);slots = slot[tri[affected]];valid = slots >= 0
    def geometry(delta):
        local = delta.reshape(count,3);surface = base.copy();surface[valid] += local[slots[valid]]
        u,v = surface[:,1]-surface[:,0],surface[:,2]-surface[:,0]
        return local,np.cross(u,v),(_skew(v-u),-_skew(v),_skew(u))
    def constraints(delta):
        local,normal,_ = geometry(delta)
        return np.concatenate((np.sum(normal*n0,axis=1)/scale-1e-8,
                               np.sum(normal*normal,axis=1)/scale-minimum_area_ratio**2,
                               1.-np.sum(local*local,axis=1)/max_displacement_mm**2,matrix@delta-rhs))
    def jacobian(delta):
        local,normal,derivatives = geometry(delta);size = len(affected)
        jac = np.zeros((2*size+count+len(rhs),dimensions))
        for corner,derivative in enumerate(derivatives):
            live = slots[:,corner]>=0;row = np.flatnonzero(live);col = slots[live,corner]
            orient = np.einsum('fji,fj->fi',derivative[live],n0[live])/scale[live,None]
            area = 2*np.einsum('fji,fj->fi',derivative[live],normal[live])/scale[live,None]
            for axis in range(3):
                jac[row,3*col+axis] += orient[:,axis];jac[size+row,3*col+axis] += area[:,axis]
        for axis in range(3):jac[2*size+np.arange(count),3*np.arange(count)+axis] = -2*local[:,axis]/max_displacement_mm**2
        jac[-len(rhs):] = matrix
        return jac
    initial = lp.x[:dimensions];tick = time.perf_counter()
    answer = minimize(lambda d:float(d@d),initial,jac=lambda d:2*d,method="SLSQP",
                      constraints=[{"type":"ineq","fun":constraints,"jac":jacobian}],
                      options={"maxiter":optimizer_iterations,"ftol":1e-9,"disp":False})
    report.update(slsqp_calls=1,slsqp_dimensions=dimensions,optimizer_status=int(answer.status),
                  optimizer_success=bool(answer.success),optimizer_iterations=int(answer.nit),
                  optimizer_seconds=time.perf_counter()-tick,minimum_optimizer_constraint=None)
    answer_values=np.asarray(answer.x)
    if answer_values.shape!=(dimensions,) or not np.issubdtype(answer_values.dtype,np.floating) or not np.isfinite(answer_values).all():
        return None,report|{"decision":"selected_family_optimizer_invalid_or_nonfinite"}
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        values=constraints(answer_values)
    if not np.isfinite(values).all():
        return None,report|{"decision":"selected_family_optimizer_constraints_nonfinite"}
    report['minimum_optimizer_constraint']=float(values.min())
    if values.min() < -1e-9:
        return None, report|{"decision":"selected_family_optimizer_infeasible"}
    candidate = xyz.copy()
    with np.errstate(over='ignore',invalid='ignore'):
        candidate[active]=(xyz[active].astype(float)+answer_values.reshape(count,3)).astype(np.float32)
    if not np.isfinite(candidate).all():
        return None,report|{"decision":"nonfinite_quantized_trial_not_retained"}
    return candidate,report|{"decision":"finite_quantized_trial"}


def resolve_joint_residual_intersections(*, vertices, faces, fixed_vertices, white_vertices,
                                        exact_relation, max_displacement_mm: float = 1.,
                                        minimum_area_ratio: float = .2, clearance_mm: float = .002,
                                        optimizer_iterations: int = 100):
    """单残余组联合候选→accepted结果、独立trial/None、完整失败报告。

    四网格参数同旧resolver；exact_relation为冻结独立裁剪callable，
    返回0/1/2/3，原名单必须逐对为1。参数默认1mm/.2/.002/100，
    仅一次LP及SLSQP。trial为float32(N,3)私有诊断，不因拒绝删除；
    result拒绝时逐元素等进入坐标，接受时等trial。报告无坐标/路径。
    全源/F64面对严格减少且无新pair，全部fixed不动、入射面朝向/
    面积/欧氏位移达门，无新增white proper或非固定精确接触；所有
    surviving/new detector pair独立精确分类。全部受影响面×全网格
    前后inclusive bbox另逐一独立exact，禁止新增proper/共面正面积/
    精确阳性pair，不依赖浮点检测器覆盖证明。未变面坐标身份不变。
    该局部门不宣称所有未变全图已有自相交均exact审查。初态已有
    穿越单列。
    残余失败保留，不接production/原阶段程序。无原名单返回未改
    输入；多连通组、非proper名单及输入非法明确失败。
    """
    wall=time.perf_counter();original,tri,fixed,white = _checked(vertices,faces,fixed_vertices,white_vertices)
    if not callable(exact_relation):raise ValueError("frozen independent exact classifier required")
    source,pairs = _pair_sets(original,tri);union=sorted(pairs|{tuple(sorted(p)) for p in source})
    def exact_rows(xyz,positive,source_membership,float64_membership):
        rows=[]
        for fa,fb in positive:
            relation=exact_relation(xyz[tri[fa]].astype(float),xyz[tri[fb]].astype(float))
            if type(relation) is not int or relation not in (0,1,2,3):raise ValueError("known exact relation required")
            rows.append({"pair":[fa,fb],"exact_relation":relation,
                         "source_directed_orientations":int((fa,fb) in source_membership)+int((fb,fa) in source_membership),
                         "float64_positive":(fa,fb) in float64_membership})
        return rows
    def self_qc(source_membership,float64_membership):
        source_faces={a for a,b in source_membership};float64_faces={v for pair in float64_membership for v in pair}
        return {"source_directed_pairs":len(source_membership),"float64_pairs":len(float64_membership),
                "source_faces":len(source_faces),"float64_faces":len(float64_faces),
                "source_vertices":len({int(v) for f in source_faces for v in tri[f]}),
                "float64_vertices":len({int(v) for f in float64_faces for v in tri[f]})}
    before_rows=exact_rows(original,union,source,pairs)
    if any(r['exact_relation']!=1 for r in before_rows):raise ValueError("complete exact proper input list required")
    active,affected,matrix,rhs,family = build_fixed_face_family(vertices=original,faces=tri,fixed_vertices=fixed,pairs=union,clearance_mm=clearance_mm)
    trial,optimizer = solve_joint_family(vertices=original,faces=tri,active=active,affected=affected,matrix=matrix,rhs=rhs,
                                        max_displacement_mm=max_displacement_mm,minimum_area_ratio=minimum_area_ratio,
                                        optimizer_iterations=optimizer_iterations)
    report={"schema_version":1,"diagnostic_contract_revision":2,"trial_evaluation":"no_finite_trial","trial_intersection_qc":"not_assessed",
            "algorithm":"independent_joint_fixed_face_family_v1","family":family,"optimizer":optimizer,
            "parameters":{"max_displacement_mm":float(max_displacement_mm),"minimum_area_ratio":float(minimum_area_ratio),
                          "clearance_mm":float(clearance_mm),"optimizer_iterations":optimizer_iterations},
            "source_directed_pairs_before":len(source),"float64_pairs_before":len(pairs),"exact_before":before_rows,
            "self_qc_before":self_qc(source,pairs),
            "strict_reproduction":"independent_algorithm_not_source_reproduction","overall_metric_equivalence":"not_assessed",
            "original_stage_program_called":False,"production_default_change":False,"white_pial_guard":None,"exact_affected_self_guard":None}
    result=original.copy();after_source,after_pairs=source,pairs
    if trial is not None:
        if trial.dtype!=np.float32 or trial.shape!=original.shape or not np.isfinite(trial).all():
            report['optimizer']=optimizer|{'decision':'nonfinite_or_invalid_quantized_trial_not_retained'}
            report.update(decision=report['optimizer']['decision'],status='failed_residual' if source or pairs else 'already_zero_in_source_and_float64',
                source_directed_pairs_after=len(source),float64_pairs_after=len(pairs),final_coordinates_equal_entry=True,fixed_different_elements=0,
                full_resolver_seconds=time.perf_counter()-wall,self_qc_final=self_qc(source,pairs))
            return result,None,report
        displacement=np.linalg.norm(trial.astype(float)-original,axis=1);normal0=_face_normals(original,tri);normal1=_face_normals(trial,tri)
        area_ratio=np.linalg.norm(normal1,axis=1)/np.linalg.norm(normal0,axis=1)
        outside=np.setdiff1d(np.arange(len(trial)),active)
        basic={"finite_coordinates":True,"fixed_coordinates_equal":np.array_equal(trial[fixed].view(np.uint32),original[fixed].view(np.uint32)),
            "only_group_movable_vertices_changed":np.array_equal(trial[outside].view(np.uint32),original[outside].view(np.uint32)),
            "euclidean_displacement_within_limit":displacement.max(initial=0.)<=max_displacement_mm,
            "all_initial_normal_dots_positive":bool(np.all(np.sum(normal0*normal1,axis=1)>0)),
            "all_area_ratios_within_limit":bool(np.all(area_ratio>=minimum_area_ratio)),
            "no_zero_area_faces":bool(np.all(np.linalg.norm(normal1,axis=1)>0))}
        report.update(basic_trial_checks={k:bool(v) for k,v in basic.items()},
            trial_changed_vertices=int(np.count_nonzero(displacement)),trial_max_displacement_mm=float(displacement.max(initial=0.)),
            trial_p99_displacement_mm=float(np.quantile(displacement,.99)),
            trial_changed_vertex_p99_mm=float(np.quantile(displacement[displacement!=0],.99)) if np.any(displacement) else 0.,
            trial_minimum_area_ratio=float(area_ratio.min()),trial_zero_area_faces=int(np.count_nonzero(np.linalg.norm(normal1,axis=1)==0)),
            trial_nonpositive_initial_normal_dot_faces=int(np.count_nonzero(np.sum(normal0*normal1,axis=1)<=0)),
            trial_fixed_different_elements=int(np.count_nonzero(trial[fixed].view(np.uint32)!=original[fixed].view(np.uint32))))
        if not all(basic.values()):
            report.update(trial_evaluation='basic_geometry_rejected_before_intersection_qc',decision='rejected_by_basic_geometry_before_intersection_qc',
                status='failed_residual' if source or pairs else 'already_zero_in_source_and_float64',source_directed_pairs_after=len(source),
                float64_pairs_after=len(pairs),final_coordinates_equal_entry=True,fixed_different_elements=0,
                full_resolver_seconds=time.perf_counter()-wall,self_qc_final=self_qc(source,pairs))
            return result,trial,report
        report.update(trial_evaluation='complete_quality_checks',trial_intersection_qc='complete')
        after_source,after_pairs = _pair_sets(trial,tri)
        report['exact_trial']=exact_rows(trial,sorted(after_pairs|{tuple(sorted(p)) for p in after_source}),after_source,after_pairs)
        report['self_qc_trial']=self_qc(after_source,after_pairs)
        before_exact,before_area,before_positive,before_exact_qc=exact_affected_self_pairs(vertices=original,faces=tri,query_faces=affected,exact_relation=exact_relation)
        after_exact,after_area,after_positive,after_exact_qc=exact_affected_self_pairs(vertices=trial,faces=tri,query_faces=affected,exact_relation=exact_relation)
        index=WhitePialCrossingIndex(white_vertices=white,faces=tri,fixed_vertices=fixed)
        before_cross,before_contact,before_legal,before_qc=index.query(original)
        after_cross,after_contact,after_legal,after_qc=index.query(trial)
        before_nonmedial,after_nonmedial=before_contact-before_legal,after_contact-after_legal
        checks={key:basic[key] for key in ('fixed_coordinates_equal','only_group_movable_vertices_changed','euclidean_displacement_within_limit','all_initial_normal_dots_positive','all_area_ratios_within_limit')}
        checks.update({
                "source_strict_subset":after_source<source,"float64_strict_subset":after_pairs<pairs,
                "no_new_exact_affected_proper_pairs":after_exact<=before_exact,
                "no_new_exact_affected_coplanar_area_pairs":after_area<=before_area,
                "no_new_exact_affected_positive_pairs":after_positive<=before_positive,
                "no_new_white_proper_pairs":after_cross<=before_cross,"no_new_nonmedial_exact_contacts":after_nonmedial<=before_nonmedial})
        report.update(trial_checks={k:bool(v) for k,v in checks.items()},
                      source_directed_pairs_trial=len(after_source),float64_pairs_trial=len(after_pairs),
                      new_source_pairs=[list(p) for p in sorted(after_source-source)],new_float64_pairs=[list(p) for p in sorted(after_pairs-pairs)],
                      trial_changed_vertices=int(np.count_nonzero(displacement)),trial_max_displacement_mm=float(displacement.max(initial=0.)),
                      trial_p99_displacement_mm=float(np.quantile(displacement,.99)),
                      trial_changed_vertex_p99_mm=float(np.quantile(displacement[displacement!=0],.99)) if np.any(displacement) else 0.,
                      trial_minimum_area_ratio=float(area_ratio.min()),
                      trial_nonpositive_initial_normal_dot_faces=int(np.count_nonzero(np.sum(normal0*normal1,axis=1)<=0)),
                      trial_fixed_different_elements=int(np.count_nonzero(trial[fixed].view(np.uint32)!=original[fixed].view(np.uint32))))
        report['white_pial_guard']={"initial":before_qc,"trial":after_qc,
            "initial_nonmedial_exact_contacts":len(before_nonmedial),"trial_nonmedial_exact_contacts":len(after_nonmedial),
            "new_proper_pairs":[list(p) for p in sorted(after_cross-before_cross)],
            "new_nonmedial_exact_contact_pairs":[list(p) for p in sorted(after_nonmedial-before_nonmedial)],
            "trial_zero_proper":not after_cross,"trial_zero_nonmedial_contact":not after_nonmedial}
        if all(checks.values()):result=trial.copy();report['decision']='accepted_once'
        else:report['decision']='rejected_by_complete_quality_gates'
        accepted=report['decision']=='accepted_once'
        report['exact_affected_self_guard']={'initial':before_exact_qc,'trial':after_exact_qc,
            'initial_proper_pairs':len(before_exact),'trial_proper_pairs':len(after_exact),'final_proper_pairs':len(after_exact if accepted else before_exact),
            'initial_coplanar_area_pairs':len(before_area),'trial_coplanar_area_pairs':len(after_area),'final_coplanar_area_pairs':len(after_area if accepted else before_area),
            'new_proper_pairs':[list(p) for p in sorted(after_exact-before_exact)],
            'new_coplanar_area_pairs':[list(p) for p in sorted(after_area-before_area)],
            'new_positive_pairs':[list(p) for p in sorted(after_positive-before_positive)],
            'unchanged_faces_retained_bitwise':bool(checks['only_group_movable_vertices_changed']),
            'global_existing_exact_intersections':'not_assessed'}
        report['white_pial_guard'].update(final_zero_proper=not (after_cross if accepted else before_cross),
                                          final_zero_nonmedial_contact=not (after_nonmedial if accepted else before_nonmedial))
    else:report['decision']=optimizer['decision']
    accepted=report['decision']=='accepted_once'
    report.update(status='passed_zero_in_source_float64_and_affected_exact' if accepted and not after_source and not after_pairs and not after_exact and not after_area else 'failed_residual' if source or pairs else 'already_zero_in_source_and_float64',
                  source_directed_pairs_after=len(after_source if accepted else source),float64_pairs_after=len(after_pairs if accepted else pairs),
                  final_coordinates_equal_entry=bool(np.array_equal(result.view(np.uint32),original.view(np.uint32))),fixed_different_elements=int(np.count_nonzero(result[fixed].view(np.uint32)!=original[fixed].view(np.uint32))),
                  full_resolver_seconds=time.perf_counter()-wall)
    report['self_qc_final']=self_qc(after_source if accepted else source,after_pairs if accepted else pairs)
    return result,trial,report
