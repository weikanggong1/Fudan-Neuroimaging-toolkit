"""复用成熟距离与顶点指标，核对真实链输出；只导出统计，不导出坐标。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from fnit.recon_all.place_surface_quality import mesh_structure, proper_pairs, placement_structure_passed
from fnit.recon_all.thread_budget import thread_budget


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(array, unit):
    values = np.asarray(array, np.float64)
    return dict(unit=unit, count=values.size, mean=float(values.mean()),
                p99=float(np.quantile(values, .99)), maximum=float(values.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--hemisphere', choices=['lh', 'rh'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparator = Path(__file__).parents[1] / 'validation/recon_all/accuracy_20261003/task_04/reference_surface_distance.py'
    spec = importlib.util.spec_from_file_location('fnit_frozen_mesh_distance', comparator)
    distance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(distance)
    report = dict(scope='actual written surface outputs; failed terminal arrays excluded',
                  script_sha256=sha(__file__), distance_comparator_sha256=sha(comparator),
                  stages={}, volumes={}, downstream_thickness='not_assessed',
                  segmentation_dice='not_assessed: no segmentation algorithm changed',
                  roi_statistics='not_assessed: annotation and downstream aggregation outside this stage',
                  overall_equivalence='not_assessed', status='running_output_audit')
    def save():
        tmp = args.output.with_suffix('.pending')
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        tmp.replace(args.output)
    save()
    with thread_budget(threads=args.threads):
        for name in ['brain.finalsurfs', 'wm', 'aseg.presurf']:
            path = args.case / 'candidate/mri' / (name + '.mgz')
            image = nib.load(path)
            report['volumes'][name] = dict(sha256=sha(path), shape=[int(size) for size in image.shape],
                dtype=str(image.get_data_dtype()), scanner_affine=image.affine.tolist(),
                vox2ras_tkr=image.header.get_vox2ras_tkr().tolist())
        for stage, suffix in [('preaparc', 'white.preaparc'), ('white', 'white'), ('pial', 'pial.T1')]:
            candidate = args.case / 'candidate/surf' / f'{args.hemisphere}.{suffix}'
            reference = args.case / f'{stage}-conda/surf' / f'{args.hemisphere}.{suffix}'
            row = dict(candidate_exists=candidate.is_file(), reference_exists=reference.is_file())
            report['stages'][stage] = row
            if not candidate.is_file() or not reference.is_file():
                row['comparison'] = 'not_assessed_missing_written_output'
                save()
                continue
            first, faces = nib.freesurfer.read_geometry(str(candidate))
            second, other_faces = nib.freesurfer.read_geometry(str(reference))
            exact = np.array_equal(first, second) and np.array_equal(faces, other_faces)
            ordered = np.array_equal(faces, other_faces) and first.shape == second.shape
            row.update(candidate_sha256=sha(candidate), reference_sha256=sha(reference),
                       ordered_faces_exact=ordered, coordinates_exact=exact,
                       candidate_structure=mesh_structure(vertices=first, faces=faces),
                       reference_structure=mesh_structure(vertices=second, faces=other_faces))
            _, row['candidate_self_proper'] = proper_pairs(first_vertices=first, faces=faces)
            if exact:
                row['reference_self_proper'] = dict(row['candidate_self_proper'])
                row['reference_quality_proof'] = 'complete mesh identity'
            else:
                _, row['reference_self_proper'] = proper_pairs(first_vertices=second, faces=other_faces)
            if stage == 'pial':
                white, white_faces = nib.freesurfer.read_geometry(str(args.case / 'candidate/surf' / f'{args.hemisphere}.white'))
                if not np.array_equal(faces, white_faces):
                    raise ValueError('pial/white ordered faces differ')
                _, row['candidate_white_pial_proper'] = proper_pairs(first_vertices=first, second_vertices=white, faces=faces)
                if exact:
                    row['reference_white_pial_proper'] = dict(row['candidate_white_pial_proper'])
                else:
                    if not np.array_equal(other_faces, white_faces):
                        raise ValueError('reference pial/white ordered faces differ')
                    _, row['reference_white_pial_proper'] = proper_pairs(first_vertices=second, second_vertices=white, faces=other_faces)
            row['candidate_quality_passed'] = (row['candidate_self_proper']['proper_pairs'] == 0
                and row.get('candidate_white_pial_proper', {}).get('proper_pairs', 0) == 0
                and placement_structure_passed(row['candidate_structure']))
            if ordered:
                row['indexed_distance'] = summary(np.linalg.norm(first-second, axis=1), 'mm')
            if exact:
                # 相同完整坐标和面保证双向点到闭三角面距离为零。
                row['point_to_triangle_proof'] = 'complete mesh identity'
                forward = backward = np.zeros(len(first), np.float64)
            else:
                forward = distance._point_to_mesh(first, second, other_faces)
                backward = distance._point_to_mesh(second, first, faces)
            row['candidate_to_reference_triangle'] = summary(forward, 'mm')
            row['reference_to_candidate_triangle'] = summary(backward, 'mm')
            row['distance_scope'] = 'bidirectional sampled vertex to complete closed triangles; not continuous Hausdorff'
            save()
    report['status'] = 'complete_output_audit'
    save()


if __name__ == '__main__':
    main()
