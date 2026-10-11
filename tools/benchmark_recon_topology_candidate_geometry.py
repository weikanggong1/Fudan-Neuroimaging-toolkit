"""唯一 retess 前状态→自产重连、25次平滑、40次MRI拟合的真实回归。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from benchmark_recon_topology_retess_fields import load_retessellation_capture
from benchmark_recon_topology_match_fields import load_match_capture
from benchmark_recon_topology_match_chain import _compare
from fnit.recon_all.topology_candidate_geometry import candidate_geometry
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*, before, mri, white_mean, gray_mean, k1_mean, k2_mean,
        smooth_reference, after_reference, output):
    """唯一 before 及显式校准均值→候选几何，保存后才读取参考后态。

    before 是 FNITRTS1 phase0，包含完整 ordered base/canonical/origXYZ、
    全边表、原域和状态；mri 是同一 uint8(X,Y,Z) brain.mgz。四个均值
    是明确的冻结校准先验：white/gray无单位、k1/k2为mm⁻¹。目标 MRI-only
    tkregister 矩阵由自身header生成。output 必须不存在。参考仅在全部
    候选保存后用于比较，无参考新增面、邻接或拟合坐标进入算法。
    返回/落盘完整计时、源码/输入SHA及数组严格比较；这是冻结候选阶段，
    没有校准先验生成、DVS、完整fitness/GA/最终QC或整例验收声明。
    """
    tick = time.perf_counter()
    before, mri, smooth_reference, after_reference, output = map(Path, (before, mri, smooth_reference, after_reference, output))
    if output.exists():
        raise FileExistsError('fresh candidate geometry output required')
    b = load_retessellation_capture(capture_path=before)
    if int(b['header'][-1]) != 0:
        raise ValueError('actual pre-retess phase0 input required')
    means = np.array([white_mean, gray_mean, k1_mean, k2_mean], np.float32)
    if not np.isfinite(means).all() or np.any(means[2:] == 0):
        raise ValueError('finite explicit calibration means; nonzero curvature means required')
    image = nib.load(mri)
    volume = np.asarray(image.dataobj)
    if volume.ndim != 3 or volume.dtype != np.uint8:
        raise ValueError('same uint8 XYZ brain.mgz required')
    matrix = source_topology_ras_to_voxel(volume_header=image.header)
    n, k = int(b['header'][0]), int(b['header'][3])
    off, ov = b['neighbor_offsets'], b['overlap_offsets']
    rows = [b['neighbors'][off[v]:off[v+1]] for v in range(n)]
    overlaps = [b['overlaps'][ov[e]:ov[e+1]] for e in range(k)] if int(b['header'][9]) else None
    args = dict(canonical=b['vertex_fields'][:, 6:9], base_faces=b['faces'], neighbors=rows,
        candidate_edges=b['edge_fields'][:, :2], edge_used=b['edge_fields'][:, 2], ordering=b['ordering'],
        inside_vertices=b['inside_fields'][:, 0], inside_discarded=b['inside_fields'][:, 1] == 1,
        border_vertices=b['border_vertices'], hull_vertices=b['hull_vertices'], vertex_rip=b['vertex_flags'][:, 1] != 0,
        fixed_mapping=b['vertex_flags'][:, 2] == 0 if int(b['header'][8]) else None,
        mode='all' if int(b['header'][7]) else 'some', overlaps=overlaps,
        overlap_complete=b['edge_fields'][:, 3] != 0 if overlaps is not None else None)
    loading_seconds = time.perf_counter()-tick
    output.mkdir(parents=True)
    candidates, times = [], []
    for label in ['cold', 'repeat']:
        r = candidate_geometry(original_vertices=b['vertex_fields'][:, :3], volume=volume,
            ras_to_voxel=matrix, white_mean=float(means[0]), gray_mean=float(means[1]),
            k1_mean=float(means[2]), k2_mean=float(means[3]), retessellation_inputs=args)
        elapsed = r['stage_seconds'].copy()
        value = dict(faces=r['faces'], smooth=r['smoothed_original_vertices'], matched=r['original_vertices'],
            patch_normals=r['selected_vertex_normals'],
            selected_vertices=r['selected_vertices'], selected_faces=r['selected_faces'],
            neighbor_offsets=r['neighbor_offsets'], neighbors=r['ordered_neighbors'],
            incident_offsets=r['incident_offsets'], incident_faces=r['incident_faces'], incident_corners=r['incident_corners'],
            face_normals=r['face_normals'], edge_used=r['edge_used'], ras_to_voxel=matrix)
        tick = time.perf_counter(); np.savez_compressed(output/(label+'.npz'), **value)
        elapsed['write'] = time.perf_counter()-tick
        candidates.append(value); times.append(dict(label=label, seconds=elapsed))
    # All geometry and candidate files exist before any post-retess reference is opened.
    smooth_ref = load_match_capture(capture_path=smooth_reference)
    after_ref = load_match_capture(capture_path=after_reference)
    checks = []
    for value in candidates:
        arrays = {key: dict(shape_same=value[key].shape == after_ref[key].shape,
            bytes_exact=value[key].tobytes() == after_ref[key].tobytes())
            for key in ['faces', 'selected_vertices', 'selected_faces', 'neighbor_offsets', 'neighbors',
                        'incident_offsets', 'incident_faces', 'incident_corners']}
        checks.append(dict(arrays=arrays, smooth=_compare(value['smooth'], smooth_ref['vertex_fields'][:, :3]),
            matched=_compare(value['matched'], after_ref['vertex_fields'][:, :3]),
            selected_normals=_compare(value['patch_normals'], after_ref['vertex_fields'][value['selected_vertices'], 9:12])))
    source_root = Path(__file__).resolve().parents[1]
    code = ['tools/benchmark_recon_topology_candidate_geometry.py', 'tools/benchmark_recon_topology_match_fields.py',
        'src/fnit/recon_all/topology_candidate_geometry.py',
        'tools/benchmark_recon_topology_retess_fields.py', 'src/fnit/recon_all/topology_retessellation_dynamic.py',
        'src/fnit/recon_all/topology_retessellation_python.py', 'src/fnit/recon_all/topology_preflight_python.py',
        'src/fnit/recon_all/topology_defect_smooth.py', 'src/fnit/recon_all/topology_defect_mri_match.py',
        'src/fnit/recon_all/topology_candidate_normals.py', 'src/fnit/recon_all/topology_edge_mri_score.py']
    passed = all(all(x['bytes_exact'] for x in check['arrays'].values()) and all(check[key]['exact']
        for key in ['smooth', 'matched', 'selected_normals']) for check in checks)
    report = dict(status='strict_candidate_geometry_passed' if passed else 'strict_candidate_geometry_failed',
        scope='same initial pre-retess candidate→ordered face detection→25smooth→40match→normals; frozen calibrated means, no full GA',
        initial_input_sha256={'before': _sha(before), 'brain': _sha(mri)},
        calibrated_means=means.tolist(), reference_input_consumed=False,
        comparison_reference_sha256={'smooth': _sha(smooth_reference), 'after': _sha(after_reference)},
        source_sha256={path: _sha(source_root/path) for path in code},
        loading_seconds=loading_seconds, results=times, checks=checks,
        complete_ga=False, final_mesh_quality='not_assessed', overall_metric_equivalence='not_assessed')
    (output/'report.private.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['before', 'mri', 'smooth-reference', 'after-reference', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ['white-mean', 'gray-mean', 'k1-mean', 'k2-mean']:
        parser.add_argument('--'+name, type=float, required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
