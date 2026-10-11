"""逐一精确审查既有自产 pial 自相交面对；不移动表面、不改质量门。"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time


KINDS = ("none", "proper", "edge_point_contact", "coplanar_area_overlap", "coplanar_edge_point_contact")
INPUTS = {"surf/lh.white", "surf/autodet.gw.stats.lh.dat", "label/lh.cortex.label",
          "label/lh.cortex+hipamyg.label", "mri/brain.finalsurfs.mgz", "mri/wm.mgz", "mri/aseg.presurf.mgz"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _coplanar_has_area(first, second):
    """精确二维逐边裁剪后判断面积非零，不以近接触距离代替面积。"""
    a = [tuple(Fraction.from_float(float(v)) for v in row) for row in first]
    b = [tuple(Fraction.from_float(float(v)) for v in row) for row in second]
    u = tuple(a[1][i]-a[0][i] for i in range(3)); v = tuple(a[2][i]-a[0][i] for i in range(3))
    normal = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    drop = max(range(3), key=lambda i: abs(normal[i])); axes = [i for i in range(3) if i != drop]
    polygon = [tuple(p[i] for i in axes) for p in a]; clip = [tuple(p[i] for i in axes) for p in b]
    def orient(p, q, r): return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    sign = 1 if orient(*clip) > 0 else -1
    for p, q in zip(clip, clip[1:]+clip[:1]):
        output = []
        if not polygon: return False
        for s, e in zip(polygon, polygon[1:]+polygon[:1]):
            ds, de = sign*orient(p, q, s), sign*orient(p, q, e)
            if ds >= 0: output.append(s)
            if (ds < 0 <= de) or (de < 0 <= ds):
                t = ds/(ds-de); output.append(tuple(s[i]+t*(e[i]-s[i]) for i in range(2)))
        polygon = output
    return bool(polygon) and sum(p[0]*q[1]-p[1]*q[0] for p, q in zip(polygon, polygon[1:]+polygon[:1])) != 0


def audit_pairs(*, vertices, faces, fixed_vertices, source_pairs, float64_pairs, exact_relation):
    """同索引 (N,3) RAS/mm、(F,3)、bool(N,) 与完整面对集合→逐对数值表。

    集合只提供待审查名单，不决定精确类别。共有顶点/边仅单列，不
    自动排除真正的内部交叉；共面精确面积重叠与边点接触分别报告。
    source_pairs 是有向集合，float64_pairs 必须为严格升序无向集合。
    exact_relation 必须为冻结的独立有理数逐边裁剪函数。返回全部
    面编号/类别/固定点/连通组计数，不返回坐标，不改变输入。
    """
    import numpy as np
    xyz, tri, fixed = np.asarray(vertices), np.asarray(faces), np.asarray(fixed_vertices)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.issubdtype(xyz.dtype, np.floating) or not np.isfinite(xyz).all():
        raise ValueError("finite floating vertices (N,3) required")
    if tri.ndim != 2 or tri.shape[1] != 3 or not np.issubdtype(tri.dtype, np.integer) or tri.size and (tri.min() < 0 or tri.max() >= len(xyz)):
        raise ValueError("ordered integer faces (F,3) required")
    if fixed.dtype != np.bool_ or fixed.shape != (len(xyz),): raise ValueError("boolean fixed_vertices (N,) required")
    if not isinstance(source_pairs, set) or not isinstance(float64_pairs, set): raise ValueError("complete pair sets required")
    for values, ordered in ((source_pairs, False), (float64_pairs, True)):
        for row in values:
            if type(row) is not tuple or len(row) != 2 or any(type(v) is not int or v < 0 or v >= len(tri) for v in row) or row[0] == row[1] or ordered and row[0] > row[1]:
                raise ValueError("valid distinct face pair integers required")
    pairs = sorted(float64_pairs | {tuple(sorted(row)) for row in source_pairs})
    triangles = xyz.astype(np.float64)[tri]; rows = []
    for fa, fb in pairs:
        relation = exact_relation(triangles[fa], triangles[fb])
        if type(relation) is not int or relation not in (0, 1, 2, 3): raise ValueError("known exact relation required")
        kind = ("none", "proper", "edge_point_contact", "coplanar_area_overlap")[relation]
        if relation == 3 and not _coplanar_has_area(triangles[fa], triangles[fb]): kind = "coplanar_edge_point_contact"
        corner_ids = set(map(int, tri[fa])) | set(map(int, tri[fb]))
        rows.append({"pair": [fa, fb], "exact_kind": kind,
                     "source_directed_orientations": int((fa, fb) in source_pairs)+int((fb, fa) in source_pairs),
                     "float64_positive": (fa, fb) in float64_pairs,
                     "shared_vertex_ids": len(set(map(int, tri[fa])) & set(map(int, tri[fb]))),
                     "shared_coordinate_corners": int(np.all(triangles[fa, :, None] == triangles[fb, None, :], axis=2).any(axis=1).sum()),
                     "unique_corner_vertices": len(corner_ids), "fixed_corner_vertices": sum(bool(fixed[v]) for v in corner_ids),
                     "all_corners_fixed": all(bool(fixed[v]) for v in corner_ids)})
    # 两个待审面对的任何面角点相连即归入同一残余组件；不扩成全脑网格。
    remaining = set(range(len(rows))); components = []
    while remaining:
        selected = {min(remaining)}; vertices_in_group = set()
        while True:
            for i in selected: vertices_in_group.update(map(int, tri[rows[i]["pair"]].ravel()))
            added = {i for i in remaining-selected if vertices_in_group.intersection(map(int, tri[rows[i]["pair"]].ravel()))}
            if not added: break
            selected.update(added)
        remaining.difference_update(selected); index = len(components)
        for i in selected: rows[i]["component"] = index
        components.append({"component": index, "pairs": len(selected), "unique_vertices": len(vertices_in_group),
                           "fixed_vertices": sum(bool(fixed[v]) for v in vertices_in_group),
                           "exact_classes": {kind: sum(rows[i]["exact_kind"] == kind for i in selected) for kind in KINDS}})
    return {"source_directed_pairs": len(source_pairs), "float64_pairs": len(float64_pairs), "union_pairs": len(rows),
            "exact_classes": {kind: sum(row["exact_kind"] == kind for row in rows) for kind in KINDS},
            "float64_nonproper_pairs": sum(row["float64_positive"] and row["exact_kind"] != "proper" for row in rows),
            "source_nonproper_unordered_pairs": sum(row["source_directed_orientations"] > 0 and row["exact_kind"] != "proper" for row in rows),
            "pairs_with_shared_vertex_ids": sum(row["shared_vertex_ids"] > 0 for row in rows),
            "pairs_with_fixed_corners": sum(row["fixed_corner_vertices"] > 0 for row in rows),
            "rows": rows, "components": components}


def run(*, installed_directory: Path, overlay_directory: Path, diagnostic_case: Path,
        candidate: Path, input_hashes: Path, terminal_sha256: str, candidate_sha256: str,
        exact_audit_source: Path, expected_exact_audit_sha256: str, expected_pair_helper_sha256: str,
        expected_entry_float64_pairs: int, expected_candidate_float64_pairs: int, output: Path, threads: int = 4):
    """冻结七输入、进入/候选末态与源码，完整枚举两套阳性面对再精确审计。

    全部路径及哈希/预期面对数必填；threads默认4正整数。candidate为
    本轮实际自产float32(N,3) .npy，不读官方。output须不存在，报告仅
    数值/固定枚举/SHA，不含影像、坐标、私人路径/主机/argv。哈希或
    预期计数不符、输入/分类非法明确失败，不生成伪造报告；固定字节
    前后校验。时间包含加载、两次完整枚举及裁剪，外层CLI另外计时。
    """
    started = time.perf_counter()
    if output.exists(): raise FileExistsError(output)
    for value in (threads, expected_entry_float64_pairs, expected_candidate_float64_pairs):
        if type(value) is not int or value < 0: raise ValueError("nonnegative integer budgets required")
    if not threads: raise ValueError("positive threads required")
    expected = json.loads(input_hashes.read_text())
    if set(expected) != INPUTS: raise ValueError("exact frozen seven-input map required")
    terminal = diagnostic_case/"cleanup-result.private.npy"; helper = overlay_directory/"place_surface_residual_cleanup.py"
    observed = {name: sha(diagnostic_case/"subject"/name) for name in sorted(INPUTS)}
    if observed != expected or sha(terminal) != terminal_sha256 or sha(candidate) != candidate_sha256 or sha(helper) != expected_pair_helper_sha256 or sha(exact_audit_source) != expected_exact_audit_sha256:
        raise ValueError("frozen source or input hash differs")
    sys.path.insert(0, str(installed_directory)); import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(overlay_directory))
    import numpy as np
    import nibabel.freesurfer as fs
    import numba
    numba.set_num_threads(threads)
    from fnit.recon_all.place_surface_residual_cleanup import _pair_sets
    spec = importlib.util.spec_from_file_location("self_pair_independent_exact", exact_audit_source)
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    white, faces = fs.read_geometry(str(diagnostic_case/"subject/surf/lh.white")); faces = faces.astype(np.int32)
    cortex = np.zeros(len(white), bool); rip = np.ones(len(white), bool)
    cortex[fs.read_label(str(diagnostic_case/"subject/label/lh.cortex.label")).astype(np.int64)] = True
    rip[fs.read_label(str(diagnostic_case/"subject/label/lh.cortex+hipamyg.label")).astype(np.int64)] = False
    fixed = rip | ~cortex; results = []
    for state, path, count in (("entry", terminal, expected_entry_float64_pairs), ("candidate", candidate, expected_candidate_float64_pairs)):
        xyz = np.load(path)
        if xyz.dtype != np.float32 or xyz.shape != white.shape: raise ValueError("same ordered float32 (N,3) required")
        tick = time.perf_counter(); source, pairs = _pair_sets(xyz, faces); enumeration = time.perf_counter()-tick
        if len(pairs) != count: raise ValueError("actual frozen positive pair count differs")
        tick = time.perf_counter(); result = audit_pairs(vertices=xyz, faces=faces, fixed_vertices=fixed,
                                                        source_pairs=source, float64_pairs=pairs, exact_relation=audit.exact_clip_relation)
        result.update(state=state, enumeration_seconds=enumeration, exact_audit_seconds=time.perf_counter()-tick)
        results.append(result)
    after = {name: sha(diagnostic_case/"subject"/name) for name in sorted(INPUTS)}
    if after != observed or sha(terminal) != terminal_sha256 or sha(candidate) != candidate_sha256 or sha(helper) != expected_pair_helper_sha256 or sha(exact_audit_source) != expected_exact_audit_sha256:
        raise RuntimeError("frozen input/source changed during audit")
    result = {"schema_version": 1, "execution_status": "complete_readonly_exact_pair_audit",
              "scope": "all existing self-detector positive pairs on frozen own entry and candidate; no whole or new optimizer",
              "script_sha256": sha(__file__), "pair_helper_sha256": expected_pair_helper_sha256,
              "exact_audit_sha256": expected_exact_audit_sha256, "input_sha256": observed,
              "terminal_sha256": terminal_sha256, "candidate_sha256": candidate_sha256,
              "inputs_unchanged": True, "coordinate_space": "surface_RAS", "coordinate_unit": "mm",
              "vertices": len(white), "faces": len(faces), "fixed_vertices": int(fixed.sum()),
              "threads": threads, "cpu_affinity": sorted(os.sched_getaffinity(0)), "states": results, "source_version_bindings": {
                  name: sha(module.__file__) for name, module in sorted(sys.modules.copy().items())
                  if name.startswith("fnit.recon_all.") and getattr(module, "__file__", None) and Path(module.__file__).is_file()},
              "wall_seconds_before_report_write": time.perf_counter()-started,
              "peak_process_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
              "gpu_computation": False, "gpu_memory_bytes": None, "original_stage_program_called": False,
              "overall_metric_equivalence": "not_assessed", "quality_gate_changed": False}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("installed-directory", "overlay-directory", "diagnostic-case", "candidate", "input-hashes", "exact-audit-source", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("terminal-sha256", "candidate-sha256", "expected-exact-audit-sha256", "expected-pair-helper-sha256"):
        parser.add_argument("--"+name, required=True)
    for name in ("expected-entry-float64-pairs", "expected-candidate-float64-pairs"):
        parser.add_argument("--"+name, type=int, required=True)
    parser.add_argument("--threads", type=int, default=4)
    report = run(**vars(parser.parse_args()))
    print(json.dumps({"execution_status": report["execution_status"], "classes": [row["exact_classes"] for row in report["states"]]}))
