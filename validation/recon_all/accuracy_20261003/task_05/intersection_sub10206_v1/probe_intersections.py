"""Read-only CPU diagnosis of saved meshes; no repair or GPU execution."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ['NUMBA_DISABLE_JIT'] = '1'
import nibabel.freesurfer.io as fs
import numpy as np
from scipy.spatial import cKDTree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_edge_hits(a, b):
    """Independent segment-plane/barycentric witnesses without native tolerance."""
    witnesses = []
    for source, target in ((a, b), (b, a)):
        normal = np.cross(target[1] - target[0], target[2] - target[0])
        length = np.linalg.norm(normal)
        if length == 0:
            continue
        distances = (source - target[0]) @ (normal / length)
        for edge in range(3):
            other = (edge + 1) % 3
            d0, d1 = distances[edge], distances[other]
            if d0 * d1 >= 0 or d0 == d1:
                continue
            t = d0 / (d0 - d1)
            point = source[edge] + t * (source[other] - source[edge])
            matrix = np.column_stack((target[1] - target[0], target[2] - target[0]))
            uv = np.linalg.lstsq(matrix, point - target[0], rcond=None)[0]
            bary = np.array([1 - uv.sum(), *uv])
            if np.min(bary) >= -1e-12:
                witnesses.append({'segment_fraction': float(t), 'barycentric': bary.tolist(),
                                  'signed_endpoint_distances_mm': [float(d0), float(d1)]})
    return witnesses


def inspect(path, predicate, placement_predicate):
    start = time.monotonic()
    if not path.exists():
        return {'status': 'missing', 'path': str(path), 'result': None}
    vertices, faces = fs.read_geometry(str(path))
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    pairs = cKDTree(centers).query_pairs(2 * radii.max() + 1e-5, output_type='ndarray')
    a, b = pairs.T
    pairs = pairs[np.sum((centers[a] - centers[b]) ** 2, axis=1) <=
                  (radii[a] + radii[b] + 1e-5) ** 2]
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    a, b = pairs.T
    pairs = pairs[np.all(low[a] <= high[b] + 1e-5, axis=1) &
                  np.all(low[b] <= high[a] + 1e-5, axis=1)]
    a, b = pairs.T
    shared = ~np.all(faces[a, :, None] != faces[b, None, :], axis=(1, 2))
    hits = []
    for fa, fb in pairs[~shared]:
        ta, tb = triangles[fa], triangles[fb]
        if predicate(ta, tb):
            witnesses = strict_edge_hits(ta, tb)
            hits.append({'faces': [int(fa), int(fb)],
                         'vertex_ids': [faces[fa].tolist(), faces[fb].tolist()],
                         'shared_vertex_ids': np.intersect1d(faces[fa], faces[fb]).tolist(),
                         'areas_mm2': [float(areas[fa]), float(areas[fb])],
                         'equal_coordinate_corner_pairs': int(np.all(ta[:, None] == tb[None], axis=2).sum()),
                         'placement_predicate': bool(placement_predicate(ta, tb)),
                         'strict_edge_witnesses': witnesses})
    hit_faces = sorted({f for h in hits for f in h['faces']})
    return {'status': 'complete', 'path': str(path), 'sha256': sha(path),
            'vertices': len(vertices), 'faces': len(faces),
            'ordered_faces_sha256': hashlib.sha256(faces.astype('<i4').tobytes()).hexdigest(),
            'zero_area_faces': int((areas == 0).sum()), 'minimum_face_area_mm2': float(areas.min()),
            'shared_vertex_pairs_excluded': int(shared.sum()), 'intersecting_face_count': len(hit_faces),
            'hit_face_ids': hit_faces, 'hit_pairs': hits,
            'strict_witness_pair_count': sum(bool(h['strict_edge_witnesses']) for h in hits),
            'seconds_cpu_diagnostic': time.monotonic() - start}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--hemispheres', nargs='+', choices=['lh', 'rh'], default=['lh'])
    p.add_argument('--surfaces', nargs='+', default=['white.preaparc', 'white', 'pial'])
    args = p.parse_args()
    sys.path.insert(0, str(args.source_root / 'src'))
    import fnit
    from fnit.recon_all import mris_remove_intersection_python as detector
    from fnit.recon_all import place_surface_collision as placement
    result = {'python': sys.executable, 'fnit_import': fnit.__file__,
              'detector_sha256': sha(detector.__file__), 'placement_predicate_sha256': sha(placement.__file__),
              'script_sha256': sha(__file__) if __file__ != '<stdin>' else None,
              'device': 'cpu', 'numba_jit': False, 'meshes': {}}
    for role, subject in [('candidate', args.candidate), ('official', args.reference)]:
        for hemi in args.hemispheres:
            for surface in args.surfaces:
                name = hemi + '.' + surface
                result['meshes'][role + '/' + name] = inspect(subject / 'surf' / name,
                    detector._triangles_intersect, placement.triangles_intersect)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
