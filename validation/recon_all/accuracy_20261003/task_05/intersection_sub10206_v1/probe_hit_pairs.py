"""CPU-only refinement of previously reported face pairs; no mesh mutation."""
import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import nibabel.freesurfer.io as fs
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--subject', type=Path, required=True)
p.add_argument('--native-source', type=Path, required=True)
p.add_argument('--pairs-json', required=True, help='Previously detected surface→face-pair list JSON')
args = p.parse_args()
native = args.native_source / 'utils/tritri.cpp'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()

def exact_witnesses(a, b):
    """Exact rational Cramer test of saved float coordinates, with strict interiors."""
    def determinant(m):
        return (m[0][0] * (m[1][1]*m[2][2]-m[1][2]*m[2][1])
                -m[0][1] * (m[1][0]*m[2][2]-m[1][2]*m[2][0])
                +m[0][2] * (m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    result = []
    a = [[Fraction(float(x)) for x in row] for row in a]
    b = [[Fraction(float(x)) for x in row] for row in b]
    for side, (source, target) in enumerate(((a, b), (b, a))):
        for edge in range(3):
            other = (edge+1)%3
            matrix = [[target[1][i]-target[0][i], target[2][i]-target[0][i],
                       source[edge][i]-source[other][i]] for i in range(3)]
            det = determinant(matrix)
            if not det:
                continue
            rhs = [source[edge][i]-target[0][i] for i in range(3)]
            uv_t = []
            for column in range(3):
                copy = [row.copy() for row in matrix]
                for i in range(3):
                    copy[i][column] = rhs[i]
                uv_t.append(determinant(copy)/det)
            u,v,t = uv_t
            if 0<t<1 and u>0 and v>0 and u+v<1:
                result.append({'side': side, 'edge':edge, 'segment_fraction_exact':str(t),
                    'barycentric_exact': [str(1-u-v),str(u),str(v)]})
    return result
output = {'native_source_sha256': sha(native), 'subject': str(args.subject), 'surfaces': {}}
with tempfile.TemporaryDirectory(prefix='fnit-tritri-diagnostic-') as tmp:
    wrapper = Path(tmp) / 'wrapper.cpp'
    wrapper.write_text('#include "tritri.h"\nextern "C" __attribute__((visibility("default"))) int fnit_test(double* a,double* b){return tri_tri_intersect(a,a+3,a+6,b,b+3,b+6);}\n')
    library = Path(tmp) / 'test.so'
    cmd = ['g++', '-O2', '-fPIC', '-shared', '-ffunction-sections', '-fdata-sections',
           '-fvisibility=hidden', '-Wl,--gc-sections', '-I' + str(args.native_source / 'include'),
           str(wrapper), str(native), '-o', str(library)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output['compile_returncode'] = proc.returncode
    output['compile_stderr'] = proc.stderr
    output['compile_command'] = cmd
    test = None
    if proc.returncode == 0:
        output['diagnostic_library_sha256'] = sha(library)
        lib = ctypes.CDLL(str(library))
        test = lib.fnit_test
        ptr = ctypes.POINTER(ctypes.c_double)
        test.argtypes = [ptr, ptr]
        test.restype = ctypes.c_int
    for name, pairs in json.loads(args.pairs_json).items():
        path = args.subject / 'surf' / name
        vertices, faces = fs.read_geometry(str(path))
        cortex = fs.read_label(str(args.subject / 'label' / (name[:2] + '.cortex.label')))
        ids = sorted({v for pair in pairs for f in pair for v in faces[f].tolist()})
        metrics = []
        for fa, fb in pairs:
            a, b = np.ascontiguousarray(vertices[faces[fa]], dtype=np.float64), np.ascontiguousarray(vertices[faces[fb]], dtype=np.float64)
            n1, n2 = np.cross(a[1]-a[0],a[2]-a[0]), np.cross(b[1]-b[0],b[2]-b[0])
            metrics.append({'faces':[fa,fb], 'native_tritri': bool(test(a.ctypes.data_as(ptr),b.ctypes.data_as(ptr))) if test else None,
                'exact_rational_strict_crossings':exact_witnesses(a,b),
                'corner_distance_max_mm':float(np.linalg.norm(a[:,None]-b[None],axis=2).max()),
                'plane_distances_mm': [((b-a[0])@n1/np.linalg.norm(n1)).tolist(),((a-b[0])@n2/np.linalg.norm(n2)).tolist()]})
        orig_path = args.subject/'surf'/(name[:2]+'.orig')
        orig_vertices, orig_faces = fs.read_geometry(str(orig_path))
        orig_metrics = None
        if np.array_equal(faces, orig_faces):
            triangle = orig_vertices[orig_faces[sorted({f for pair in pairs for f in pair})]]
            orig_areas = np.linalg.norm(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),axis=1)/2
            orig_metrics = {'minimum_area_mm2_at_hit_faces':float(orig_areas.min()),
                'maximum_area_mm2_at_hit_faces':float(orig_areas.max())}
        output['surfaces'][name] = {'surface_sha256':sha(path),'cortex_label_sha256':sha(args.subject/'label'/(name[:2]+'.cortex.label')),
            'orig_surface_sha256':sha(orig_path),'ordered_faces_equal_orig':bool(np.array_equal(faces,orig_faces)),
            'orig_hit_face_metrics':orig_metrics,
            'hit_vertex_ids': ids, 'hit_vertices_inside_cortex':int(np.isin(ids,cortex).sum()),
            'hit_vertices_outside_cortex':int((~np.isin(ids,cortex)).sum()),'pairs':metrics}
print(json.dumps(output,indent=2))
