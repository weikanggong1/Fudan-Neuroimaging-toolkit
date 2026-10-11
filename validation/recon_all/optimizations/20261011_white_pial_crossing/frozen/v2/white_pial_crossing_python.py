"""自有 white–pial 穿越检查；不同表面不排除同索引或共享角点。

完整查询和局部查询使用同一缓存的自产 white；不调用外部命令。
这是几何检查，不声明 FreeSurfer 逐位复现或整例指标等效。
"""
from __future__ import annotations

import time
from fractions import Fraction
import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from .place_surface_collision import _interval
from .place_surface_collision_torch import _source_pairs


@njit(cache=True)
def _strict_relations(first, second):
    """F64无距离截断：1为开平面两侧的正长度交线，2为边/点接触。

    3表示精确共面，须进一步检测二维重叠；0为未证明相交。极短但
    正长度的交线仍记为1，不以近接触标签豁免。共面判据不使用原源
    谓词的1e-6/1e-5保护；原源谓词另行记录，不替代本几何语义。
    """
    relation = np.zeros(len(first), np.int8)
    overlap = np.zeros(len(first), np.float64)
    depth_first = np.zeros(len(first), np.float64)
    depth_second = np.zeros(len(first), np.float64)
    ambiguous = np.zeros(len(first), np.bool_)
    shared_corners = np.zeros(len(first), np.int8)
    for slot in range(len(first)):
        a, b = first[slot], second[slot]
        na = np.cross(a[1]-a[0], a[2]-a[0])
        nb = np.cross(b[1]-b[0], b[2]-b[0])
        da = np.empty(3, np.float64); db = np.empty(3, np.float64)
        for corner in range(3):
            da[corner] = np.dot(a[corner]-b[0], nb)
            db[corner] = np.dot(b[corner]-a[0], na)
        # 坐标逐元素相同是数学身份：这个点恰在对方平面上。不能因
        # normal-dot尾差把共享边判作开平面内部穿越；仍检测整个面对。
        known_a = np.zeros(3,np.bool_); known_b = np.zeros(3,np.bool_)
        for ca in range(3):
            for cb in range(3):
                if np.all(a[ca] == b[cb]):
                    da[ca] = 0.; db[cb] = 0.; known_a[ca] = True; known_b[cb] = True
                    shared_corners[slot] += 1
        for corner in range(3):
            for vector,edge1,edge2,distance,known in (
                    (a[corner]-b[0],b[1]-b[0],b[2]-b[0],da[corner],known_a[corner]),
                    (b[corner]-a[0],a[1]-a[0],a[2]-a[0],db[corner],known_b[corner])):
                permanent = 0.
                for axis in range(3):
                    j,k=(axis+1)%3,(axis+2)%3
                    permanent += abs(vector[axis])*(abs(edge1[j]*edge2[k])+abs(edge1[k]*edge2[j]))
                if not known and abs(distance) <= 64*np.finfo(np.float64).eps*(permanent+1.):
                    ambiguous[slot] = True
        amin, amax = da.min(), da.max(); bmin, bmax = db.min(), db.max()
        depth_first[slot] = max(0., min(amax, -amin))/np.sqrt(np.dot(nb, nb))
        depth_second[slot] = max(0., min(bmax, -bmin))/np.sqrt(np.dot(na, na))
        direction = np.cross(na, nb)
        if amin == amax == bmin == bmax == 0:
            relation[slot] = 3
            continue
        if np.max(np.abs(direction)) == 0:
            if np.max(np.abs(da)) == 0 and np.max(np.abs(db)) == 0:
                relation[slot] = 3
            continue
        if amin > 0 or amax < 0 or bmin > 0 or bmax < 0:
            continue
        axis = int(np.argmax(np.abs(direction)))
        a0, a1, a_coplanar = _interval(a[:, axis], da)
        b0, b1, b_coplanar = _interval(b[:, axis], db)
        if a_coplanar or b_coplanar:
            continue
        length = min(a1, b1)-max(a0, b0)
        endpoint_error = 64*np.finfo(np.float64).eps*(abs(a0)+abs(a1)+abs(b0)+abs(b1)+1.)
        if abs(length) <= endpoint_error:
            ambiguous[slot] = True
        overlap[slot] = length/abs(direction[axis])*np.sqrt(np.dot(direction, direction))
        if length < 0:
            continue
        if length > 0 and amin < 0 < amax and bmin < 0 < bmax:
            relation[slot] = 1
        else:
            relation[slot] = 2
    return relation, overlap, depth_first, depth_second, ambiguous, shared_corners


def _rational_triangle(triangle):
    points = [tuple(Fraction.from_float(float(value)) for value in row) for row in triangle]
    u = tuple(points[1][i]-points[0][i] for i in range(3))
    v = tuple(points[2][i]-points[0][i] for i in range(3))
    normal = (u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
    return points,normal


def _exact_relation(first,second):
    """有限IEEE坐标的精确有理数determinant/interval，仅用于误差界内。"""
    a,na=first;b,nb=second
    da=[sum((point[i]-b[0][i])*nb[i] for i in range(3)) for point in a]
    db=[sum((point[i]-a[0][i])*na[i] for i in range(3)) for point in b]
    amin,amax,bmin,bmax=min(da),max(da),min(db),max(db)
    depth_a=float(max(0,min(amax,-amin)))/np.linalg.norm([float(value) for value in nb])
    depth_b=float(max(0,min(bmax,-bmin)))/np.linalg.norm([float(value) for value in na])
    if amin == amax == bmin == bmax == 0:return 3,0.,depth_a,depth_b
    direction=(na[1]*nb[2]-na[2]*nb[1],na[2]*nb[0]-na[0]*nb[2],na[0]*nb[1]-na[1]*nb[0])
    if not any(direction) or amin>0 or amax<0 or bmin>0 or bmax<0:return 0,0.,depth_a,depth_b
    axis=max(range(3),key=lambda i:abs(direction[i]))
    # 复用已验证的interval分支次序；py_func的标量算术在Fraction上精确。
    a0,a1,ac=_interval.py_func([point[axis] for point in a],da)
    b0,b1,bc=_interval.py_func([point[axis] for point in b],db)
    if ac or bc:return 0,0.,depth_a,depth_b
    length=min(a1,b1)-max(a0,b0)
    overlap=float(length)*np.linalg.norm([float(value/direction[axis]) for value in direction])
    if length<0:return 0,overlap,depth_a,depth_b
    relation=1 if length>0 and amin<0<amax and bmin<0<bmax else 2
    return relation,overlap,depth_a,depth_b


def _exact_coplanar_kind(first,second):
    """精确凸多边形裁剪：3为正面积重叠、4为边/点、0为不接触。"""
    a,normal=first;b,_=second
    drop=max(range(3),key=lambda i:abs(normal[i]));axes=[i for i in range(3) if i!=drop]
    polygon=[tuple(point[i] for i in axes) for point in a]
    clip=[tuple(point[i] for i in axes) for point in b]
    def orient(p,q,r):return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    sign=1 if orient(*clip)>0 else -1
    for start,end in zip(clip,clip[1:]+clip[:1]):
        previous=polygon;polygon=[]
        if not previous:break
        for p,q in zip(previous,previous[1:]+previous[:1]):
            dp,dq=sign*orient(start,end,p),sign*orient(start,end,q)
            if dp>=0:polygon.append(p)
            if dp*dq<0:polygon.append(tuple(p[i]+(q[i]-p[i])*dp/(dp-dq) for i in range(2)))
    if not polygon:return 0
    area=sum(p[0]*q[1]-p[1]*q[0] for p,q in zip(polygon,polygon[1:]+polygon[:1]))
    return 3 if area!=0 else 4


def _coplanar_area(first, second, normal):
    """复用既有残余几何诊断的凸三角形裁剪，返回真实共面重叠面积。"""
    drop = int(np.argmax(np.abs(normal))); axes = [i for i in range(3) if i != drop]
    polygon = [p.copy() for p in first[:, axes]]; clip = second[:, axes]
    orient = np.cross(clip[1]-clip[0], clip[2]-clip[0])
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        previous = polygon; polygon = []
        if not previous: break
        def distance(point): return float(np.cross(b-a, point-a))*np.sign(orient)
        for p, q in zip(previous, previous[1:]+previous[:1]):
            dp, dq = distance(p), distance(q)
            if dp >= 0: polygon.append(p)
            if dp*dq < 0: polygon.append(p+(q-p)*dp/(dp-dq))
    if len(polygon) < 3: return 0.
    points = np.asarray(polygon); after = np.roll(points, -1, axis=0)
    return float(abs(np.sum(points[:, 0]*after[:, 1]-points[:, 1]*after[:, 0]))/2/abs(normal[drop]))


def _validate_mesh(vertices, faces, name):
    xyz, triangles = np.asarray(vertices), np.asarray(faces)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.issubdtype(xyz.dtype, np.floating) or not np.isfinite(xyz).all():
        raise ValueError(name+" requires finite floating vertices (N,3), surface RAS/mm")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError("ordered integer faces (F,3) required")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(xyz)):
        raise ValueError("face vertex outside mesh")
    points = xyz.astype(np.float64)[triangles]
    normals = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    if not np.isfinite(normals).all() or np.any(np.linalg.norm(normals, axis=1) == 0):
        raise ValueError(name+" has a zero-area triangle")
    return xyz, triangles, points


class WhitePialCrossingIndex:
    """缓存自产white，检查同网格pial；局部查询完整覆盖受影响面。

    white_vertices为有限浮点(N,3)，faces为有序整数(F,3)，二者surface
    RAS/mm；fixed_vertices为bool(N,)，用于标记不移动的真实内侧壁接触，
    不改变任一坐标。不同表面即使面索引/角点索引相同也必须检查。
    query(pial_vertices, query_faces=None)返回严格穿越面对集合、非穿越
    精确边/点/共面接触集合、合法固定内侧壁接触集合及完整数值摘要。所有集合
    的二元组依次为(pial面,white面)，不是自相交的无向面对。

    proper_crossings要求两个三角面都严格跨越对方平面且交线正长度；
    没有毫米下限。原源谓词的近接触、边/点接触和共面重叠单独统计，
    不能称为穿越消除。legal_fixed_contacts仅要求该pial面全部固定且
    坐标与自产white同索引面完全相同，标识不变的内侧壁接触；其它
    共面/近接触不被自动豁免。输入非法或退化面明确抛ValueError。
    此检查不保证嵌套/厚度正确，不替代自相交、拓扑和最终指标验收。
    """
    def __init__(self, *, white_vertices, faces, fixed_vertices):
        white, triangles, points = _validate_mesh(white_vertices, faces, "white")
        fixed = np.asarray(fixed_vertices)
        if fixed.dtype != np.bool_ or fixed.shape != (len(white),):
            raise ValueError("fixed_vertices must be boolean (N,)")
        self.white = white.copy(); self.faces = triangles.copy(); self.fixed = fixed.copy()
        self.points = points; self.centers = points.mean(axis=1)
        self.radii = np.nextafter(np.linalg.norm(points-self.centers[:, None], axis=2).max(axis=1), np.inf)
        self.low, self.high = points.min(axis=1), points.max(axis=1)
        self.tree = cKDTree(self.centers)
        self._exact_white = {}

    def query(self, pial_vertices, query_faces=None):
        wall = time.perf_counter()
        pial, _, points = _validate_mesh(pial_vertices, self.faces, "pial")
        if pial.shape != self.white.shape:
            raise ValueError("same ordered white/pial vertex domain required")
        if query_faces is None:
            query = np.arange(len(self.faces), dtype=np.int64)
        else:
            raw = np.asarray(query_faces)
            if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer) or raw.size and (raw.min() < 0 or raw.max() >= len(self.faces)):
                raise ValueError("query_faces must contain valid integer face indices")
            query = np.unique(raw)
        centers = points.mean(axis=1)
        radii = np.nextafter(np.linalg.norm(points-centers[:, None], axis=2).max(axis=1), np.inf)
        low, high = points.min(axis=1), points.max(axis=1)
        proper, contacts, legal = set(), set(), set()
        classes = {"proper_crossing": 0, "noncoplanar_edge_or_point_contact": 0,
                   "exact_coplanar_area_overlap": 0, "exact_coplanar_edge_or_point_contact": 0,
                   "source_tolerance_contact": 0}
        broad, tested, short_crossings, source_positive, source_negative_proper = 0, 0, 0, 0, 0
        max_overlap = max_depth_a = max_depth_b = 0.
        exact_pairs = shared_corner_proofs = 0
        exact_pial = {}
        for start in range(0, len(query), 256):
            ids = query[start:start+256]
            # 4mm是沿用源碰撞的保守候选余量，不是穿越容差；不截断候选。
            rounding = 64*np.finfo(np.float64).eps*(np.max(np.abs(centers[ids]), axis=1)+
                       np.max(np.abs(self.centers), initial=0.)+radii[ids]+self.radii.max(initial=0.)+1.)
            query_radius = np.nextafter(radii[ids]+self.radii.max(initial=0.)+4.+rounding, np.inf)
            rows = self.tree.query_ball_point(centers[ids], query_radius, eps=0.)
            for local_query, (fa, neighbors) in enumerate(zip(ids, rows)):
                near = np.asarray(neighbors, np.int64); broad += len(near)
                distance = np.sum((self.centers[near]-centers[fa])**2, axis=1)
                radius_sum = np.nextafter(self.radii[near]+radii[fa]+4.+rounding[local_query], np.inf)
                keep = distance <= np.nextafter(radius_sum**2, np.inf)
                keep &= np.all(np.nextafter(low[fa]-1e-5,-np.inf) <= np.nextafter(self.high[near],np.inf), axis=1)
                keep &= np.all(np.nextafter(self.low[near]-1e-5,-np.inf) <= np.nextafter(high[fa],np.inf), axis=1)
                near = near[keep]
                if not len(near): continue
                a = np.broadcast_to(points[fa], (len(near),3,3)).copy(); b = self.points[near]
                relation, overlap, depth_a, depth_b, ambiguous, shared = _strict_relations(a, b)
                shared_corner_proofs += int(shared.sum())
                for slot in np.flatnonzero(ambiguous):
                    fb = int(near[slot])
                    if int(fa) not in exact_pial: exact_pial[int(fa)] = _rational_triangle(points[fa])
                    if fb not in self._exact_white: self._exact_white[fb] = _rational_triangle(self.points[fb])
                    relation[slot],overlap[slot],depth_a[slot],depth_b[slot] = _exact_relation(exact_pial[int(fa)],self._exact_white[fb])
                    exact_pairs += 1
                source = _source_pairs(a.astype(np.float32), b.astype(np.float32)) | _source_pairs(b.astype(np.float32), a.astype(np.float32))
                tested += len(near)
                source_positive += int(source.sum())
                for slot in np.flatnonzero((relation != 0) | source):
                    pair = (int(fa), int(near[slot]))
                    if relation[slot] == 1:
                        proper.add(pair); classes['proper_crossing'] += 1
                        source_negative_proper += int(not source[slot])
                        short_crossings += int(overlap[slot] <= 1e-5)
                        max_overlap = max(max_overlap, float(overlap[slot]))
                        max_depth_a = max(max_depth_a, float(depth_a[slot])); max_depth_b = max(max_depth_b, float(depth_b[slot]))
                    else:
                        if relation[slot] == 2: label = 'noncoplanar_edge_or_point_contact'
                        elif relation[slot] == 3:
                            if shared[slot] == 3: kind = 3
                            else:
                                fb = int(near[slot])
                                if int(fa) not in exact_pial: exact_pial[int(fa)] = _rational_triangle(points[fa])
                                if fb not in self._exact_white: self._exact_white[fb] = _rational_triangle(self.points[fb])
                                kind = _exact_coplanar_kind(exact_pial[int(fa)],self._exact_white[fb])
                            if kind == 3: label = 'exact_coplanar_area_overlap'
                            elif kind == 4: label = 'exact_coplanar_edge_or_point_contact'
                            elif source[slot]: label = 'source_tolerance_contact'
                            else: continue
                        else: label = 'source_tolerance_contact'
                        classes[label] += 1
                        if label != 'source_tolerance_contact':
                            contacts.add(pair)
                            if np.all(self.fixed[self.faces[fa]]) and np.array_equal(pial[self.faces[fa]], self.white[self.faces[fa]]):
                                legal.add(pair)
        return proper, contacts, legal, {
            "query_faces": len(query), "broad_phase_candidates": broad, "tested_pairs": tested,
            "proper_crossing_pairs": len(proper), "exact_contact_pairs": len(contacts),
            "source_predicate_positive_pairs": source_positive,
            "source_negative_proper_crossings": source_negative_proper,
            "legal_fixed_contact_pairs": len(legal), "classes": classes,
            "short_positive_crossings_not_exempted": short_crossings,
            "maximum_intersection_length_mm": max_overlap, "maximum_pial_plane_depth_mm": max_depth_a,
            "maximum_white_plane_depth_mm": max_depth_b, "query_seconds": time.perf_counter()-wall,
            "coordinate_space": "surface_RAS", "coordinate_unit": "mm",
            "proper_crossing_minimum_length_mm": 0.0, "near_contact_display_length_mm": 1e-5,
            "source_predicate_contacts_are_not_strict_crossings": True,
            "strict_relation_backend": "F64_with_exact_IEEE_fraction_fallback",
            "exact_fallback_pairs": exact_pairs, "exact_shared_corner_plane_zero_proofs": shared_corner_proofs,
            "adaptive_error_bound_epsilon_multiple": 64,
        }
