"""只读复核自产 white/pial：独立有理数逐边裁剪，不调用源 interval。"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_clip_relation(first, second):
    """有限 IEEE 三角形→0无接触/1开平面正交线/2边点/3共面接触。

    直接枚举三条边与对方平面的有理数交点，再比较闭区间。没有源
    interval 的分支，也没有长度容差；共面使用二维凸多边形分离轴。
    只用于独立审查，不参与生产或候选的坐标更新。
    """
    def prepare(triangle):
        p = [tuple(Fraction.from_float(float(v)) for v in row) for row in triangle]
        u = tuple(p[1][i]-p[0][i] for i in range(3))
        v = tuple(p[2][i]-p[0][i] for i in range(3))
        n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        if not any(n): raise ValueError('nondegenerate triangles required')
        return p, n
    a, na = prepare(first); b, nb = prepare(second)
    da = [sum((p[i]-b[0][i])*nb[i] for i in range(3)) for p in a]
    db = [sum((p[i]-a[0][i])*na[i] for i in range(3)) for p in b]
    if not any(da) and not any(db):
        drop = max(range(3), key=lambda i: abs(na[i])); axes = [i for i in range(3) if i != drop]
        aa = [tuple(p[i] for i in axes) for p in a]; bb = [tuple(p[i] for i in axes) for p in b]
        for polygon in (aa, bb):
            for p, q in zip(polygon, polygon[1:]+polygon[:1]):
                normal = (q[1]-p[1], p[0]-q[0])
                va = [sum(x[i]*normal[i] for i in range(2)) for x in aa]
                vb = [sum(x[i]*normal[i] for i in range(2)) for x in bb]
                if max(va) < min(vb) or max(vb) < min(va): return 0
        return 3
    def section(points, distances):
        hits = {p for p, d in zip(points, distances) if d == 0}
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if distances[i]*distances[j] < 0:
                t = distances[i]/(distances[i]-distances[j])
                hits.add(tuple(points[i][k]+t*(points[j][k]-points[i][k]) for k in range(3)))
        return hits
    sa, sb = section(a, da), section(b, db)
    if not sa or not sb: return 0
    direction = (na[1]*nb[2]-na[2]*nb[1], na[2]*nb[0]-na[0]*nb[2], na[0]*nb[1]-na[1]*nb[0])
    if not any(direction): return 0
    axis = max(range(3), key=lambda i: abs(direction[i]))
    lower = max(min(p[axis] for p in sa), min(p[axis] for p in sb))
    upper = min(max(p[axis] for p in sa), max(p[axis] for p in sb))
    if lower > upper: return 0
    return 1 if lower < upper and min(da) < 0 < max(da) and min(db) < 0 < max(db) else 2


def run(*, installed_directory: Path, overlay_directory: Path, diagnostic_case: Path,
        input_hashes: Path, terminal_sha256: str, expected_helper_sha256: str,
        output: Path, global_proper_sample: int = 1024, threads: int = 4):
    """冻结七输入/末态，独立精确复核全局均匀样本和全部残余局部面对。

    参数路径仅作只读输入；output须不存在。global_proper_sample默认
    1024正整数：排序全集上的确定性等距样本，不根据结果挑选。threads
    默认4正整数仅限制Numba。空间为surface RAS/mm；有序网格不得变。
    返回仅哈希、类别计数、计时/RSS的dict，不输出路径、影像或坐标。
    哈希漂移、输入非法、精确分类不符明确失败并保留数值失败报告；
    无GPU/优化器/官方程序，不是整例或最终网格质量验收。
    """
    started = time.perf_counter()
    if output.exists(): raise FileExistsError(output)
    for value in (global_proper_sample, threads):
        if type(value) is not int or value < 1: raise ValueError('positive integer budgets required')
    expected = json.loads(input_hashes.read_text())
    required = {'surf/lh.white','surf/autodet.gw.stats.lh.dat','label/lh.cortex.label',
                'label/lh.cortex+hipamyg.label','mri/brain.finalsurfs.mgz','mri/wm.mgz','mri/aseg.presurf.mgz'}
    if set(expected) != required: raise ValueError('exact frozen seven input map required')
    before = {name: _sha(diagnostic_case/'subject'/name) for name in sorted(required)}
    terminal_path = diagnostic_case/'cleanup-result.private.npy'
    helper = overlay_directory/'white_pial_crossing_python.py'
    if before != expected or _sha(terminal_path) != terminal_sha256 or _sha(helper) != expected_helper_sha256:
        raise ValueError('frozen input/source differs')
    sys.path.insert(0, str(installed_directory))
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(overlay_directory))
    import numpy as np
    import nibabel.freesurfer as fs
    import numba
    numba.set_num_threads(threads)
    from fnit.recon_all.white_pial_crossing_python import WhitePialCrossingIndex
    from fnit.recon_all.place_surface_intersection_marking import mark_source_intersections
    white, faces = fs.read_geometry(str(diagnostic_case/'subject/surf/lh.white'))
    white = white.astype(np.float32); faces = faces.astype(np.int32); pial = np.load(terminal_path)
    cortex = np.zeros(len(white), bool); rip = np.ones(len(white), bool)
    cortex[fs.read_label(str(diagnostic_case/'subject/label/lh.cortex.label')).astype(np.int64)] = True
    rip[fs.read_label(str(diagnostic_case/'subject/label/lh.cortex+hipamyg.label')).astype(np.int64)] = False
    index = WhitePialCrossingIndex(white_vertices=white, faces=faces, fixed_vertices=rip | ~cortex)
    proper, contacts, legal, summary = index.query(pial)
    ordered = sorted(proper)
    positions = np.unique(np.linspace(0, len(ordered)-1, min(global_proper_sample, len(ordered)), dtype=np.int64)) if ordered else []
    sampled = {ordered[i] for i in positions}
    marked, _ = mark_source_intersections(pial, faces, predicate_backend='numba')
    affected = np.flatnonzero(marked[faces].any(axis=1)); selected = set(int(v) for v in affected)
    # 独立完整局部候选：直接从white缓存所有bbox相交面生成，不调用KDTree。
    # 最后仍精确裁剪；bbox只用于安全排除，不使用公差截断真正的穿越。
    wp, pp = white.astype(np.float64)[faces], pial.astype(np.float64)[faces]
    low, high = wp.min(axis=1), wp.max(axis=1)
    local_pairs = set()
    for fa in affected:
        ids = np.flatnonzero(np.all(pp[fa].min(axis=0) <= high, axis=1) & np.all(low <= pp[fa].max(axis=0), axis=1))
        local_pairs.update((int(fa), int(fb)) for fb in ids)
    pairs = sorted(sampled | local_pairs)
    classes = {'none': 0, 'proper': 0, 'edge_point_contact': 0, 'coplanar_contact': 0}
    mismatches = []
    exact_proper = set(); exact_contact = set()
    audit_started = time.perf_counter()
    for fa, fb in pairs:
        relation = exact_clip_relation(pp[fa], wp[fb]); classes[('none','proper','edge_point_contact','coplanar_contact')[relation]] += 1
        if relation == 1: exact_proper.add((fa, fb))
        elif relation in (2, 3): exact_contact.add((fa, fb))
        observed = 1 if (fa, fb) in proper else 2 if (fa, fb) in contacts else 0
        if (relation == 1) != (observed == 1) or (relation in (2, 3)) != (observed == 2):
            mismatches.append((fa, fb, relation, observed))
    after = {name: _sha(diagnostic_case/'subject'/name) for name in sorted(required)}
    if before != after or _sha(terminal_path) != terminal_sha256 or _sha(helper) != expected_helper_sha256:
        raise RuntimeError('frozen inputs/source changed during audit')
    local_proper_equal = {p for p in exact_proper if p[0] in selected} == {p for p in proper if p[0] in selected}
    local_contact_equal = {p for p in exact_contact if p[0] in selected} == {p for p in contacts if p[0] in selected}
    result = {'schema_version': 1, 'execution_status': 'pass_independent_exact_audit' if not mismatches and local_proper_equal and local_contact_equal else 'failed_independent_exact_audit',
              'scope': 'frozen own pial; deterministic global sample plus complete affected-face bbox cross-product',
              'script_sha256': _sha(__file__), 'helper_sha256': expected_helper_sha256,
              'input_sha256': before, 'terminal_sha256': terminal_sha256, 'inputs_unchanged': True,
              'vertices': len(white), 'faces': len(faces), 'global_proper_pairs': len(proper),
              'global_proper_sample_requested': global_proper_sample, 'global_proper_sample_actual': len(sampled),
              'affected_faces': len(affected), 'independent_local_bbox_pairs': len(local_pairs),
              'audited_pairs': len(pairs), 'exact_classes': classes, 'mismatch_pairs': len(mismatches),
              'mismatch_pair_set_sha256': hashlib.sha256(json.dumps(mismatches).encode()).hexdigest(),
              'complete_local_proper_set_equal': local_proper_equal, 'complete_local_contact_set_equal': local_contact_equal,
              'independent_exact_seconds': time.perf_counter()-audit_started,
              'wall_seconds_before_report_write': time.perf_counter()-started,
              'peak_process_cpu_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
              'gpu_computation': False, 'gpu_memory_bytes': None, 'original_stage_program_called': False,
              'overall_metric_equivalence': 'not_assessed'}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('installed-directory','overlay-directory','diagnostic-case','input-hashes','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('terminal-sha256','expected-helper-sha256'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--global-proper-sample', type=int, default=1024)
    parser.add_argument('--threads', type=int, default=4)
    report = run(**vars(parser.parse_args()))
    print(json.dumps({'execution_status': report['execution_status'], 'mismatch_pairs': report['mismatch_pairs']}))
    raise SystemExit(0 if report['execution_status']=='pass_independent_exact_audit' else 1)
