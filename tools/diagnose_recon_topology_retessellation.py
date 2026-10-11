#!/usr/bin/env python3
"""只读解释首patch strict失败的边界与球面包含证据，不修补候选。"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
from fnit.recon_all.smooth_surface_python import ordered_neighbors
from fnit.recon_all.topology_preflight_python import topology_counts
from fnit.recon_all.topology_preflight_python import center_sphere, project_and_smooth_sphere
from fnit.recon_all.topology_retessellation_python import _contains_neighbor_on_sphere
from fnit.recon_all import topology_retessellation_python as producer
from fnit.recon_all.topology_preflight_python import _edge_intersects


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct_candidate(*, faces: np.ndarray, nvertices: int, report: dict):
    """按完整源索引和面决策只读重建失败候选，并核对全部V/E/F计数。"""
    retained = report["retained_inside_source_indices"]
    deleted = report["removed_vertex_source_indices"]
    # 本只读诊断只处理报告明确无retain丢失、仅删除初始DISCARD的首patch。
    if report["missing_retained_source_indices"] or len(retained) + len(deleted) != report["inside_vertices"]:
        raise ValueError("cannot infer full translation from this producer report")
    inside = set(retained) | set(deleted)
    old_neighbors = ordered_neighbors(faces, nvertices)
    source_order, found = [min(inside)], {min(inside)}
    for vertex in source_order:
        for other in old_neighbors[vertex]:
            if other in inside and other not in found:
                found.add(other); source_order.append(other)
    if found != inside:
        raise ValueError("reported inside does not reproduce one complete source component")
    output_map = report["output_vertex_source_indices"]
    first_retained = next(vertex for vertex in source_order if vertex in set(retained))
    offset = output_map.index(first_retained)
    if output_map[offset:offset + len(retained)] != [v for v in source_order if v in set(retained)]:
        raise ValueError("reported output map does not retain source BFS order")
    inverse = np.asarray(output_map[:offset] + source_order + output_map[offset + len(retained):], np.int32)
    if len(inverse) != nvertices or len(np.unique(inverse)) != nvertices:
        raise ValueError("incomplete source translation")
    decisions = report["source_face_discovery"]["complete_ordered_face_decisions"]
    translated = np.asarray([row["oriented_face"] for row in decisions if row["decision"] == "accepted"], np.int32).reshape(-1, 3)
    combined = np.concatenate((faces[report["outside_face_source_indices"]], inverse[translated]))
    used = np.unique(combined)
    reconstructed = topology_counts(np.searchsorted(used, combined), len(used))
    if reconstructed.__dict__ != report["mesh_output"]:
        raise ValueError("source reconstruction differs from preserved failed candidate")
    return combined, inverse, reconstructed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-surface", type=Path, required=True)
    parser.add_argument("--input-qsphere", type=Path, required=True)
    parser.add_argument("--intensity-volume", type=Path, required=True)
    parser.add_argument("--stage-report", type=Path, required=True)
    parser.add_argument("--canonical-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-edge-decisions", action="store_true",
                        help="以原参数重跑失败原语，仅捕获完整边表与阻挡顺序，不修补面")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("new diagnostic directory required")
    args.output_dir.mkdir(parents=True)
    report = json.loads(args.stage_report.read_text())
    checkpoint_report = json.loads(args.checkpoint_report.read_text())
    original, faces, metadata = fsio.read_geometry(str(args.input_surface), read_metadata=True)
    if sha256(args.input_surface) != report["input_sha256"]["orig_nofix"] or sha256(args.intensity_volume) != report["input_sha256"]["MRI"]:
        raise ValueError("original same-input hashes required")
    if sha256(args.input_qsphere) != report["input_sha256"]["qsphere_nofix"] or checkpoint_report["input_sha256"] != report["input_sha256"]:
        raise ValueError("checkpoint and candidate must have the same three original inputs")
    combined, inverse, reconstructed = reconstruct_candidate(faces=faces,nvertices=len(original),report=report)
    decisions = report["source_face_discovery"]["complete_ordered_face_decisions"]
    pairs = np.concatenate((combined[:, [0, 1]], combined[:, [1, 2]], combined[:, [2, 0]]))
    edges, counts = np.unique(np.sort(pairs, axis=1), axis=0, return_counts=True)
    boundary = edges[counts == 1]
    with np.load(args.canonical_checkpoint, allow_pickle=False) as checkpoint:
        canonical_full = checkpoint["canonical"]
    qsphere, qfaces = fsio.read_geometry(str(args.input_qsphere))
    if not np.array_equal(qfaces, faces):
        raise ValueError("ordered qsphere faces must equal the original mesh")
    recomputed, iterations = center_sphere(project_and_smooth_sphere(qsphere, faces))
    if canonical_full.shape != original.shape or not np.array_equal(canonical_full, recomputed[inverse]):
        raise ValueError("full same-source canonical checkpoint required")
    canonical = canonical_full.astype(np.float32)
    linked = {}
    for face in combined:
        for a, b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            linked.setdefault(int(a),set()).add(int(b));linked.setdefault(int(b),set()).add(int(a))
    # 被拒候选也是当前边图中的3-clique；这给出邻接的可证明子集而非重建全部边。
    for row in decisions:
        vertices = inverse[row["vertices"]]
        for a,b in ((vertices[0],vertices[1]),(vertices[1],vertices[2]),(vertices[2],vertices[0])):
            linked.setdefault(int(a),set()).add(int(b));linked.setdefault(int(b),set()).add(int(a))
    translation = np.argsort(inverse)
    edge_diagnosis = None
    if args.replay_edge_decisions:
        captured = {}
        original_accept = producer.accept_first_candidate_edges
        def capture(canonical, candidates, used, local_edges):
            selected = original_accept(canonical, candidates, used, local_edges)
            captured.update(canonical=canonical.copy(), candidates=candidates.copy(),
                            used=used.copy(), local_edges=local_edges.copy(), selected=selected.copy())
            return selected
        producer.accept_first_candidate_edges = capture
        try:
            replay = producer.run_first_retessellation(
                input_surface=args.input_surface, input_qsphere=args.input_qsphere,
                intensity_volume=args.intensity_volume,
                output_surface=args.output_dir/'replay.rejected-candidate',
                report_path=args.output_dir/'replay.private.json',
                tissue_scale_mm=report['tissue_scale_mm'],
                edge_sample_spacing_mm=report['edge_sample_spacing_mm'], require_quality=False)
        finally:
            producer.accept_first_candidate_edges = original_accept
        unchanged_keys = ('input_sha256','source_sha256','candidate_edge_order_sha256',
                          'candidate_edge_cost_sha256','accepted_edge_flags_sha256',
                          'output_vertex_source_indices','outside_face_source_indices',
                          'source_face_discovery','mesh_output')
        if any(replay[key] != report[key] for key in unchanged_keys):
            raise ValueError("replayed full source decisions differ from preserved failure")
        if replay['output_written']:
            raise RuntimeError("unexpected successful producer: this diagnosis is for preserved failures")
        boundary_set = set(map(int, boundary.ravel()))
        boundary_keys = {tuple(row) for row in boundary}
        probes = []
        for first, second in itertools.combinations(sorted(boundary_set), 2):
            if (first, second) in boundary_keys:
                continue
            query = np.sort(translation[[first,second]])
            found_rows = np.flatnonzero(np.all(np.sort(captured['candidates'],axis=1)==query,axis=1))
            row = {'source_vertices':[first,second], 'candidate_table_rows':found_rows.tolist()}
            if len(found_rows) == 1:
                index=int(found_rows[0]); a,b=captured['candidates'][index]
                row.update(used_flag=int(captured['used'][index]), accepted=bool(captured['selected'][index]))
                active = np.concatenate((captured['local_edges'],captured['candidates'][:index][captured['selected'][:index]]))
                hits=[]
                for position,(c,d) in enumerate(active):
                    if _edge_intersects(captured['canonical'],a,b,c,d):
                        hits.append({'active_row':position,'source_vertices':inverse[[c,d]].tolist(),
                                     'origin':'preserved external blocking edge' if position < len(captured['local_edges']) else 'earlier accepted candidate'})
                row['complete_current_blocking_edges']=hits
            probes.append(row)
        edge_diagnosis={'full_replay_decision_hashes_identical':True,
                        'replay_stage_report_sha256':sha256(args.output_dir/'replay.private.json'),
                        'nonboundary_pairs_among_boundary_vertices':probes,
                        'scope':'edge-state diagnosis only; no new edges/faces accepted to fill a hole'}
    witnesses = []
    boundary_set = set(map(int,boundary.ravel()))
    for row in decisions:
        if row["decision"] != "contains_neighbor_on_sphere":
            continue
        tri_source = inverse[row["vertices"]]
        if not set(map(int,tri_source)).issubset(boundary_set):
            continue
        candidates = set().union(*(linked.get(int(v),set()) for v in tri_source)) - set(map(int,tri_source))
        found = []
        for vertex in sorted(candidates):
            if _contains_neighbor_on_sphere(canonical, np.asarray(row["vertices"],np.int32), np.asarray([translation[vertex]],np.int32)):
                found.append(vertex)
        witnesses.append({"rejected_triangle_source_indices":tri_source.tolist(),
                          "proven_contained_neighbor_source_indices":found,
                          "scope":"witnesses from proven adjacency subset; not all graph neighbors or exact first source return"})
    image = nib.load(str(args.intensity_volume))
    matrix = surface_ras_to_voxel(image.header,metadata)
    points = np.c_[original[np.unique(boundary)],np.ones(len(np.unique(boundary)))] @ matrix.T
    result = {"scope":"preserved failed producer output diagnosis; no surface/output repair",
              "stage_report_sha256":sha256(args.stage_report),"source_sha256":sha256(Path(__file__)),
              "input_sha256":report["input_sha256"],"producer_source_sha256":report["source_sha256"],
              "canonical_checkpoint_sha256":sha256(args.canonical_checkpoint),
              "checkpoint_report_sha256":sha256(args.checkpoint_report),
              "canonical_checkpoint_recomputed_all_elements_identical":True,
              "canonical_center_iterations":iterations,
              "reconstructed_mesh_counts":reconstructed.__dict__,
              "boundary_edges_source_indices":boundary.tolist(),"contained_neighbor_witnesses":witnesses,
              "full_edge_replay_diagnosis":edge_diagnosis,
              "original_removed_patch_domain":report["original_removed_patch_domain"],
              "whole_topology_complete":False,"overall_metric_equivalence":"not_assessed"}
    (args.output_dir/'diagnosis.private.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    center = np.rint(points[:,:3].mean(axis=0)).astype(int)
    volume = np.asarray(image.dataobj)
    boundary_voxel = np.c_[original[boundary.reshape(-1)],np.ones(boundary.size)] @ matrix.T
    fig,axes = plt.subplots(1,3,figsize=(12,4))
    for ax,cut,dims in zip(axes,range(3),((1,2),(0,2),(0,1))):
        sl=np.take(volume,int(center[cut]),axis=cut)
        ax.imshow(sl.T,origin='lower',cmap='gray',vmin=0,vmax=150)
        lines=boundary_voxel[:,:3].reshape(-1,2,3)[:,:,list(dims)]
        ax.add_collection(LineCollection(lines,colors='red',linewidths=1.5))
        ax.set_xlim(center[dims[0]]-20,center[dims[0]]+20)
        ax.set_ylim(center[dims[1]]-20,center[dims[1]]+20)
        ax.set_xlabel(f'voxel axis {dims[0]}');ax.set_ylabel(f'voxel axis {dims[1]}')
        ax.set_title(f'source MRI slice axis {cut}={center[cut]}')
    fig.suptitle('Public ds000114 frozen first patch: rejected candidate boundary (projection)')
    fig.tight_layout();fig.savefig(args.output_dir/'failed_boundary_T1.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    main()
