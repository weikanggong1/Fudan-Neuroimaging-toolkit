"""真实末态全网格面对配对回归：已有编译CSR替代Python宽相枚举。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time


def run(*, frozen_source: Path, overlay_source: Path, baseline_module: Path,
        diagnostic_case: Path, input_hashes: Path, output_directory: Path, threads: int = 4):
    """原/新完整面对集合配对，不移动顶点、不消费原软件参考。

    frozen_source是FNIT冻结src；overlay_source为新求解器/成熟marker。
    baseline_module为本轮v1固定FNIT helper文件；diagnostic_case含自产
    white和terminal快照。input_hashes是七前置SHA。新output_directory
    保存聚合JSON和服务器原位面对NPZ。threads正整数默认4。
    返回complete或failed，对不同面对保留报告并抛异常。RAS/mm，
    全部面/阳性面对均比较，不以计数替代集合。没有完整solver/整例。
    """
    started = time.perf_counter()
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1: raise ValueError('positive threads required')
    if output_directory.exists(): raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    expected = json.loads(input_hashes.read_text())
    if len(expected) != 7 or any(sha(diagnostic_case/'subject'/name) != digest for name, digest in expected.items()):
        raise ValueError('seven real input SHA differ')
    if sha(baseline_module) != '058edba6cbf02c215c1b1b09bf050a66c35f8a9332f89dfa54b71dfa6f713e87':
        raise ValueError('exact frozen v1 baseline module required')
    sys.path.insert(0, str(frozen_source))
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(overlay_source))
    import numpy as np
    import nibabel.freesurfer as fs
    import numba
    numba.set_num_threads(threads)
    from fnit.recon_all.place_surface_residual_cleanup import _pair_sets
    spec = importlib.util.spec_from_file_location('fnit.recon_all._residual_pair_baseline_v1', baseline_module)
    baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
    xyz = np.load(diagnostic_case/'cleanup-result.private.npy')
    white, faces = fs.read_geometry(str(diagnostic_case/'subject/surf/lh.white')); faces = faces.astype(np.int32)
    if xyz.shape != white.shape or xyz.dtype != np.float32 or not np.isfinite(xyz).all(): raise ValueError('ordered real terminal required')
    rows = []
    artifacts = {}
    for name, function in (('python_v1', baseline._pair_sets), ('existing_compiled_csr', _pair_sets)):
        tick = time.perf_counter(); source, independent = function(xyz, faces)
        seconds = time.perf_counter()-tick
        source_rows = np.array(sorted(source), np.int64).reshape(-1, 2)
        independent_rows = np.array(sorted(independent), np.int64).reshape(-1, 2)
        artifacts[name+'_source'] = source_rows; artifacts[name+'_float64'] = independent_rows
        rows.append({'implementation': name, 'full_pair_call_seconds': seconds,
                     'source_directed_pairs': len(source), 'float64_pairs': len(independent),
                     'source_faces': len({a for a, b in source}), 'float64_faces': len({v for pair in independent for v in pair}),
                     'source_pair_rows_sha256': hashlib.sha256(source_rows.tobytes()).hexdigest(),
                     'float64_pair_rows_sha256': hashlib.sha256(independent_rows.tobytes()).hexdigest()})
    equal = all(np.array_equal(artifacts['python_v1_'+kind], artifacts['existing_compiled_csr_'+kind]) for kind in ('source', 'float64'))
    np.savez(output_directory/'pair_sets.private.npz', **artifacts)
    report = {'schema_version': 1, 'status': 'complete_exact_pair_sets' if equal else 'failed_pair_difference',
              'scope': 'same-input full widephase/narrowphase pair-set regression only; not full residual solver or raw T1',
              'all_source_directed_and_float64_pairs_exact': equal, 'input_sha256': expected,
              'terminal_sha256': sha(diagnostic_case/'cleanup-result.private.npy'), 'baseline_module_sha256': sha(baseline_module),
              'candidate_module_sha256': sha(Path(sys.modules[_pair_sets.__module__].__file__)),
              'marker_module_sha256': sha(Path(sys.modules['fnit.recon_all.place_surface_intersection_marking'].__file__)),
              'script_sha256': sha(__file__), 'rows': rows, 'threads': threads,
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'gpu_memory_bytes': None, 'gpu_computation': False,
              'peak_cpu_memory_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
              'pair_artifact_sha256': sha(output_directory/'pair_sets.private.npz'),
              'complete_api_wall_seconds': time.perf_counter()-started,
              'timing_scope': 'single sequential real pair regression; shared environment; no ABBA or whole acceleration claim',
              'whole_pipeline_speedup': None, 'overall_metric_equivalence': 'not_assessed'}
    (output_directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if not equal: raise RuntimeError('complete pair sets changed; candidate solver is blocked')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-source','overlay-source','baseline-module','diagnostic-case','input-hashes','output-directory'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args(); report = run(**vars(a)); print(json.dumps({'status': report['status']}))


if __name__ == '__main__': main()
