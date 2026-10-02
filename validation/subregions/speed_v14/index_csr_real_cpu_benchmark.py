import argparse
import ast
import gc
import hashlib
import json
import math
import platform
import socket
import statistics
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
from fnit.gems.atlas import GEMSAtlas
from fnit.gems.rasterize import BlockIndex


def identity(path):
    path = Path(path)
    content = path.read_bytes()
    return {'path': str(path), 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


def load_function(path):
    source = Path(path).read_text()
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_block_index')
    namespace = {'np': np, 'math': math, 'BlockIndex': BlockIndex}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['build_block_index'], hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()


def digest(index):
    hasher = hashlib.sha256()
    for values in index.candidates:
        hasher.update(np.asarray([len(values)], dtype='<i8').tobytes())
        hasher.update(values.astype('<i8', copy=False).tobytes())
    return hasher.hexdigest()


def equal(a, b):
    return a.shape == b.shape and a.block_size == b.block_size and len(a.candidates) == len(b.candidates) and all(
        x.dtype == y.dtype == np.int64 and np.array_equal(x, y) for x, y in zip(a.candidates, b.candidates))


def traced_peak(function, args):
    gc.collect()
    tracemalloc.start()
    index = function(*args)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del index
    return {'current_bytes': current, 'peak_bytes': peak}


parser = argparse.ArgumentParser()
parser.add_argument('--old-raster', required=True, type=Path)
parser.add_argument('--new-raster', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
old, old_sha = load_function(args.old_raster)
new, new_sha = load_function(args.new_raster)
root = Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_subregions_unified_20260930')
prepared = Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_subregions_plus_20260928/samseg_native_build_20260929')
report = {'mode': 'real_saved_mesh_cpu_block_index_comparison', 'host': socket.gethostname(), 'started_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(), 'numpy': np.__version__, 'gpu_used': False, 'repeats': 3, 'timing_scope': 'build_block_index only; image loading, hashes, equality and tracemalloc excluded', 'old_function_sha256': old_sha, 'new_function_sha256': new_sha, 'source_files': {'old': identity(args.old_raster), 'new': identity(args.new_raster)}, 'cases': {}}
for name, directory in [('thalamus', 'port_thalamus_debug'), ('hippo-amygdala-left', 'port_hippo_integrated_fullfix_left_tmp')]:
    stage = prepared / directory
    image_path, aligned_path, mesh_path = [stage / value for value in ('processedImageMasked.mgz', 'alignedAtlasImage.mgz', 'warpedOriginalMesh.txt.gz')]
    lut_path = root / 'atlases' / name / 'compressionLookupTable.txt'
    image, aligned = nib.load(image_path), nib.load(aligned_path)
    transform = np.linalg.inv(image.affine) @ aligned.affine
    atlas = GEMSAtlas.from_freesurfer(mesh_path, lut_path).transformed(transform, transform_reference=True)
    shape = tuple(np.asarray(image.dataobj).squeeze().shape)
    vertices = atlas.vertices.astype(np.float32)
    call_args = (vertices, atlas.tetrahedra, shape, 8, 3.)
    a, b = old(*call_args), new(*call_args)
    same = equal(a, b)
    pairs = sum(len(values) for values in a.candidates)
    row = {'inputs': [identity(path) for path in (mesh_path, image_path, aligned_path, lut_path)], 'coordinate_transform': transform.tolist(), 'vertex_input_dtype': str(vertices.dtype), 'shape': list(shape), 'vertices': len(vertices), 'tetrahedra': len(atlas.tetrahedra), 'block_size': 8, 'margin': 3., 'blocks': len(a.candidates), 'candidate_pairs': pairs, 'candidate_bytes': sum(values.nbytes for values in a.candidates), 'max_candidates_per_block': max(map(len, a.candidates), default=0), 'every_block_array_equal': same, 'old_candidate_sha256': digest(a), 'new_candidate_sha256': digest(b), 'times_seconds': {'old': [], 'new': []}, 'vectorized_path': pairs <= 8_388_608, 'coordinate_chunk_pairs_limit': 1_048_576, 'global_sort_pairs_limit': 8_388_608}
    del a, b
    for repeat in range(3):
        for tag, function in ([('old', old), ('new', new)] if repeat % 2 == 0 else [('new', new), ('old', old)]):
            start = perf_counter()
            index = function(*call_args)
            row['times_seconds'][tag].append(perf_counter() - start)
            del index
    row['median_seconds'] = {tag: statistics.median(values) for tag, values in row['times_seconds'].items()}
    row['observed_median_ratio_old_over_new'] = row['median_seconds']['old'] / row['median_seconds']['new']
    row['tracemalloc_separate_pass'] = {tag: traced_peak(function, call_args) for tag, function in [('old', old), ('new', new)]}
    report['cases'][name] = row
    print(json.dumps({'case': name, 'pairs': pairs, 'equal': same, 'median_seconds': row['median_seconds'], 'observed_ratio': row['observed_median_ratio_old_over_new'], 'tracemalloc': row['tracemalloc_separate_pass']}), flush=True)
    if not same or row['old_candidate_sha256'] != row['new_candidate_sha256']:
        raise RuntimeError('candidate mismatch')
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(str(args.output), flush=True)
