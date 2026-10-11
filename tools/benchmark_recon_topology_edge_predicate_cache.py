"""显式几何memo的真实首identity候选诊断；两臂完成后才比较输出。

--config私有JSON指定三项自产输入及SHA、冻结源码manifest/SHA、source_root、
output、hemisphere、threads、cpu_affinity、maximum_entries。--arm=baseline/memo
只执行该臂cold/warm/audit；--compare读取两份完整收据后比全字段/控制。
不运行完整GA、参考程序或原始T1整例，不读另一臂结果到候选计算。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

CANDIDATE_FIELDS = frozenset(('faces','patch_faces','selected_faces','selected_vertices','selected_edges','edge_used',
    'marks','removed_vertices','temporary_faces','original_vertices','smoothed_original_vertices',
    'face_normals','selected_vertex_normals','cached_original_face_area_mm2','incident_offsets',
    'incident_faces','incident_corners','neighbor_offsets','ordered_neighbors','k1','k2','tangent_e1','tangent_e2',
    'vertex_normals','vertex_curvbak','flattened_vertices','face_area','displacement_vertices','displacement_mm',
    'distance_mm','fitness','core_fields','feedback_vertex_rip','feedback_use_counts','feedback_vertex_fitness','reset_edge_used'))
STATE_FIELDS = frozenset(('current_vertices','original_vertices','temporary_vertices','canonical_vertices','vertex_normals',
    'val','val2','valbak','val2bak','imag_val','curv','curvbak','stat','mean','mean_imag','std_error','H','K','k1','k2',
    'nfaces','faces','face_rip','face_normals','face_area','flattened_vertices','tangent_e1','tangent_e2','vertex_area',
    'undefval','old_undefval','vertex_marks','vertex_border','vertex_rip','vertex_topology','neighbor_storage',
    'incident_faces','incident_corners','neighbor_offsets','incident_offsets'))
CONTROL_FIELDS = frozenset(('retessellation_trace','face_decisions','rounds','ninside','ndiscarded','face_check'))
ARTIFACT_NAMES = frozenset(name + suffix for name in ('cold','warm','audit')
    for suffix in ('.candidate.private.npz','.state.private.npz','.control.private.json'))
TOOL_RELATIVE = 'tools/benchmark_recon_topology_edge_predicate_cache.py'
REQUIRED_HELPERS = frozenset(('benchmark_recon_topology_candidate_state', 'benchmark_recon_topology_prepared_cache'))
REQUIRED_FNIT = frozenset(('fnit', 'fnit.recon_all', 'fnit.recon_all.thread_budget',
    'fnit.recon_all.topology_edge_mri_score', 'fnit.recon_all.topology_genetic_base_prepare',
    'fnit.recon_all.topology_global_priors', 'fnit.recon_all.topology_candidate_inputs',
    'fnit.recon_all.topology_genetic_defect_inputs', 'fnit.recon_all.topology_genetic_search',
    'fnit.recon_all.topology_ga_population', 'fnit.recon_all.topology_edge_predicate_cache',
    'fnit.recon_all.topology_edge_blocker_scan', 'fnit.recon_all.topology_retessellation_dynamic',
    'fnit.recon_all.topology_candidate_cycle', 'fnit.recon_all.topology_candidate_pipeline',
    'fnit.recon_all.topology_preflight_python'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_protocol(protocol, manifest, manifest_sha256):
    """启动前固定完整成熟输出schema；不从本轮双方的共同子集推断验收。"""
    if protocol['source_manifest_sha256'] != manifest_sha256 or protocol['git_base'] != manifest['git_base']:
        raise ValueError('protocol source manifest/version mismatch')
    gates = protocol['strict_gates']
    if (set(gates['candidate_npz_fields']) != CANDIDATE_FIELDS or gates['candidate_npz_field_count'] != 36
            or set(gates['state_npz_fields']) != STATE_FIELDS or gates['state_npz_field_count'] != 40
            or set(gates['full_control_fields']) != CONTROL_FIELDS):
        raise ValueError('complete fixed 36/40/6 output schema required')


def module_relative(name, manifest):
    """只接受manifest声明的常规FNIT Python模块/包路径和对应具名SHA。"""
    if name != 'fnit' and not name.startswith('fnit.'):
        raise ValueError('only FNIT module source names are allowed')
    base = 'src/' + name.replace('.', '/')
    paths = [p for p in (base + '.py', base + '/__init__.py') if p in manifest['files']]
    expected = manifest['expected_fnit_module_sha256'].get(name)
    if len(paths) != 1 or expected != manifest['files'][paths[0]]['sha256']:
        raise ValueError('FNIT module name/path/hash missing or inconsistent in manifest')
    return paths[0]


def validate_report_binding(report, manifest, manifest_sha256, protocol_sha256):
    """完整收据后的离线来源门；不读取另一臂后态到候选计算。"""
    if (report['source_manifest_sha256'] != manifest_sha256 or report['protocol_sha256'] != protocol_sha256
            or report['actual_tool_sha256'] != manifest['files'][TOOL_RELATIVE]['sha256']
            or report['all_fnit_modules_inside_frozen_root'] is not True):
        raise ValueError('actual tool/protocol/frozen source binding mismatch')
    loaded = report['actual_loaded_fnit_sha256']; sources = report['actual_loaded_fnit_sources']
    if not REQUIRED_FNIT.issubset(loaded) or set(loaded) != set(sources):
        raise ValueError('complete actual FNIT source/module binding required')
    for name, actual in loaded.items():
        relative = module_relative(name, manifest); record = manifest['files'][relative]
        if (actual != record['sha256'] or sources[name] != dict(relative_path=relative,
                origin_relative_path=relative, size=record['size'], sha256=record['sha256'])):
            raise ValueError('actual FNIT module path/origin/hash binding mismatch')
    helpers = report['actual_loaded_benchmark_tool_sources']
    if not REQUIRED_HELPERS.issubset(helpers):
        raise ValueError('complete actual benchmark helper binding required')
    for name, actual in helpers.items():
        relative = 'tools/' + name + '.py'
        if not name.startswith('benchmark_recon_topology_') or relative not in manifest['files']:
            raise ValueError('actual benchmark helper missing in manifest')
        record = manifest['files'][relative]
        if actual != dict(relative_path=relative, origin_relative_path=relative,
                          size=record['size'], sha256=record['sha256']):
            raise ValueError('actual benchmark helper path/origin/hash binding mismatch')


def validate_source(config):
    """校验全部声明字节；FNIT实际导入另核路径和SHA，不使用安装目录兜底。"""
    root = Path(config['source_root']).resolve()
    manifest_path = Path(config['source_manifest'])
    if sha(manifest_path) != config['source_manifest_sha256']:
        raise ValueError('frozen source manifest SHA mismatch')
    manifest = json.loads(manifest_path.read_text())
    for name, record in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != record['sha256'] or path.stat().st_size != record['size']:
            raise ValueError('actual frozen source payload mismatch')
    if (sha(config['protocol']) != config['protocol_sha256']):
        raise ValueError('frozen protocol SHA mismatch')
    protocol = json.loads(Path(config['protocol']).read_text())
    validate_protocol(protocol, manifest, config['source_manifest_sha256'])
    if (protocol['resources']['requested_threads'] != config['threads']
            or protocol['resources']['candidate_cpu_affinity'] != config['cpu_affinity']
            or protocol['input_sha256'] != config['input_sha256']
            or protocol['cache']['maximum_entries'] != config['maximum_entries']):
        raise ValueError('config inputs/resource/cache cap differs from frozen protocol')
    tool = (root / TOOL_RELATIVE).resolve()
    if Path(__file__).resolve() != tool or sha(tool) != manifest['files'][TOOL_RELATIVE]['sha256']:
        raise ValueError('actual diagnostic tool loaded outside or different from frozen source')
    return root, manifest


def loaded_source(root, manifest):
    """全部已载入FNIT Python模块必须来自本次freeze并逐字节绑定。"""
    actual = {}
    for name, module in tuple(sys.modules.items()):
        if name != 'fnit' and not name.startswith('fnit.'):
            continue
        if not getattr(module, '__file__', None):
            raise ValueError('FNIT namespace without explicit frozen source is unsupported')
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(root):
            raise ValueError('FNIT module loaded outside the declared freeze')
        relative = str(path.relative_to(root))
        expected_relative = module_relative(name, manifest)
        origin = Path(getattr(getattr(module, '__spec__', None), 'origin', '')).resolve()
        record = manifest['files'].get(relative)
        if (relative != expected_relative or origin != path or record is None or sha(path) != record['sha256']):
            raise ValueError('actual FNIT module missing or different in frozen manifest')
        actual[name] = record['sha256']
    return actual


def source_records(root, manifest, actual):
    """实际导入路径/origin已由loaded_source核验；仅写相对源码名和hash。"""
    return {name: dict(relative_path=module_relative(name, manifest),
        origin_relative_path=module_relative(name, manifest),
        size=manifest['files'][module_relative(name, manifest)]['size'], sha256=value)
        for name, value in actual.items()}


def loaded_helpers(root, manifest, functions):
    """所有实际benchmark工具按路径/origin/hash绑定，两个数值/状态helper核函数定义位置。"""
    if set(functions) != REQUIRED_HELPERS:
        raise ValueError('both explicit benchmark helper functions required')
    actual = {}
    for name, module in tuple(sys.modules.items()):
        if not name.startswith('benchmark_recon_topology_'):
            continue
        relative = 'tools/' + name + '.py'
        expected = (root / relative).resolve()
        path = Path(getattr(module, '__file__', '')).resolve()
        origin = Path(getattr(getattr(module, '__spec__', None), 'origin', '')).resolve()
        record = manifest['files'].get(relative)
        if (path != expected or origin != path or record is None or sha(path) != record['sha256']):
            raise ValueError('actual benchmark helper loaded outside/different from freeze')
        actual[name] = dict(relative_path=relative, origin_relative_path=relative,
                            size=record['size'], sha256=record['sha256'])
    if not REQUIRED_HELPERS.issubset(actual):
        raise ValueError('complete loaded benchmark helper modules required')
    for name, function in functions.items():
        code = getattr(function, '__code__', None)
        if (getattr(function, '__module__', None) != name or code is None
                or Path(code.co_filename).resolve() != (root / ('tools/' + name + '.py')).resolve()):
            raise ValueError('actual benchmark helper function source mismatch')
    return actual


@contextmanager
def diagnostic_scanner(memo, observations):
    """仅本诊断新进程的类factory观测接线；原类总在finally恢复。"""
    from fnit.recon_all import topology_edge_blocker_scan as module
    from fnit.recon_all.topology_edge_predicate_cache import MemoizedOrderedBlockerScan
    original = module.OrderedBlockerScan
    parent = original if memo is None else MemoizedOrderedBlockerScan
    class TimedScanner(parent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.complete_scan_api_seconds = 0.0
            observations.append(self)
        def first_blocker(self, **kwargs):
            tick = time.perf_counter()
            try:
                return super().first_blocker(**kwargs)
            finally:
                self.complete_scan_api_seconds += time.perf_counter() - tick
    def factory(**kwargs):
        return TimedScanner(**kwargs) if memo is None else TimedScanner(memo=memo, **kwargs)
    module.OrderedBlockerScan = factory
    try:
        yield
    finally:
        module.OrderedBlockerScan = original


def run_arm(config, arm):
    """三项自产输入→首缺陷相同identity候选；不消费任何随机量。"""
    started = time.perf_counter()
    if arm not in ('baseline', 'memo'):
        raise ValueError('arm must be baseline or memo')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('this CPU-only diagnostic requires CUDA_VISIBLE_DEVICES empty')
    threads = config['threads']
    if isinstance(threads, bool) or not isinstance(threads, int) or threads not in (2, 4):
        raise ValueError('declared total thread budget must be 2 or 4')
    if sorted(os.sched_getaffinity(0)) != config['cpu_affinity']:
        raise ValueError('actual CPU affinity differs from frozen protocol')
    thread_names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'NUMBA_NUM_THREADS')
    if any(os.environ.get(name) != str(threads) for name in thread_names):
        raise ValueError('all fresh-process thread environments must match declared budget')
    cache_dir = Path(config['numba_cache_dir']).resolve()
    if (os.environ.get('NUMBA_CACHE_DIR') is None
            or Path(os.environ['NUMBA_CACHE_DIR']).resolve() != cache_dir
            or cache_dir.exists() and any(cache_dir.iterdir())):
        raise ValueError('one empty isolated Numba cache per fresh arm required')
    root, manifest = validate_source(config)
    paths = {key: Path(config['inputs'][key]) for key in ('nofix', 'qsphere', 'brain')}
    input_hashes = {key: sha(path) for key, path in paths.items()}
    if input_hashes != config['input_sha256']:
        raise ValueError('declared own raw input SHA mismatch')
    output = Path(config['output'])
    if output.exists():
        raise FileExistsError('fresh diagnostic output required')
    output.mkdir(parents=True)
    import nibabel as nib
    import nibabel.freesurfer.io as fsio
    from fnit.recon_all.thread_budget import thread_budget
    from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
    from fnit.recon_all.topology_genetic_base_prepare import prepare_nofix_genetic_base
    from fnit.recon_all.topology_global_priors import global_topology_priors
    from fnit.recon_all.topology_candidate_inputs import prepare_candidate_input_context, prepare_defect_candidate_inputs
    from fnit.recon_all.topology_genetic_defect_inputs import ordered_genetic_defect_inputs
    from fnit.recon_all.topology_genetic_search import prepare_genetic_defect_evaluator
    from fnit.recon_all.topology_ga_population import generate_initial_ordering
    from fnit.recon_all.topology_edge_predicate_cache import EdgePredicateMemo
    from benchmark_recon_topology_candidate_state import dump_state_checkpoint
    from benchmark_recon_topology_prepared_cache import candidate_payload
    helper_functions = dict(benchmark_recon_topology_candidate_state=dump_state_checkpoint,
                            benchmark_recon_topology_prepared_cache=candidate_payload)
    helpers_before = loaded_helpers(root, manifest, helper_functions)
    loaded_before = loaded_source(root, manifest)
    stages = {}
    tick = time.perf_counter()
    raw, faces, geometry = fsio.read_geometry(str(paths['nofix']), read_metadata=True)
    sphere, sphere_faces = fsio.read_geometry(str(paths['qsphere']))
    raw, faces, sphere = np.asarray(raw, np.float32), np.asarray(faces, np.int32), np.asarray(sphere, np.float32)
    if raw.shape != sphere.shape or not np.array_equal(faces, sphere_faces):
        raise ValueError('same ordered own nofix/qsphere required')
    if config['hemisphere'] not in ('lh', 'rh'):
        raise ValueError('explicit lh or rh required')
    image = nib.load(str(paths['brain'])); volume = np.asarray(image.dataobj)
    matrix = source_topology_ras_to_voxel(volume_header=image.header)
    metadata = dict(use_real_ras=0, hemisphere=0 if config['hemisphere'] == 'lh' else 1,
                    status=0, origxyz_status=0, volume_geometry=dict(geometry))
    stages['load_seconds'] = time.perf_counter() - tick
    records = []
    with thread_budget(threads=threads) as actual_thread_policy:
        tick = time.perf_counter()
        base = prepare_nofix_genetic_base(vertices=raw, faces=faces, qsphere_vertices=sphere,
            volume=volume, ras_to_voxel=matrix, source_metadata=metadata)
        state = base['surface_state']; source = base['prepared']['base_arguments']; metric = base['metric']
        stages['own_base_seconds'] = time.perf_counter() - tick
        tick = time.perf_counter()
        priors = global_topology_priors(principal_curvature_fields=base['sff1'],
            smoothed_original_vertices=source['source_vertex_fields']['original_vertices'], faces=faces,
            neighbor_offsets=metric['one_ring_offsets'], neighbors=metric['one_ring_neighbors'],
            vertex_rip=metric['sff_arguments']['vertex_rip'], surface_status='sphere')
        context = prepare_candidate_input_context(vertices=raw, faces=faces, defect_vertices=base['merge']['labels'] != 0,
            vertex_translation=base['vertex_translation'], volume=volume, ras_to_voxel=matrix, global_priors=priors)
        p = context.preparation
        edges = ordered_genetic_defect_inputs(canonical_vertices=source['source_vertex_fields']['canonical_vertices'],
            original_vertices=p['original_vertices'], original_normals=p['original_normals'], faces=faces,
            defect_labels=base['merge']['labels'], ordered_regions=base['merge']['regions'], retention_status=base['retention_status'],
            vertex_translation=base['vertex_translation'], white_values=p['white_values'], gray_values=p['gray_values'],
            volume=volume, ras_to_voxel=matrix, defect_number=0, use_overlap=False)
        candidate = prepare_defect_candidate_inputs(context=context, original_defect_hull=edges['original_hull_vertices'],
            candidate_edges=edges['edge_vertices'], current_corrected_original_vertices=state['original_vertices'])
        stages['own_priors_edges_local_seconds'] = time.perf_counter() - tick
        def forbid_draw(*_):
            raise RuntimeError('first identity candidate must not consume RNG')
        for name in ('cold', 'warm', 'audit'):
            tick = time.perf_counter()
            prepared = prepare_genetic_defect_evaluator(surface_state=state, defect_inputs=edges,
                candidate_inputs=candidate['candidate_kwargs'], face_detection_cache=True,
                blocker_scan_backend='numba', distance_backend='serial')
            evaluator = prepared['evaluator']; r = evaluator.retessellation_inputs
            r['blocker_scan_audit'] = name == 'audit'
            memo = EdgePredicateMemo(canonical=r['canonical'], maximum_entries=config['maximum_entries']) if arm == 'memo' else None
            preparation = time.perf_counter() - tick
            try:
                tick = time.perf_counter()
                ordering = generate_initial_ordering(edge_count=len(edges['edge_vertices']), population_index=0,
                    initial_selection=True, segment_edge_numbers=prepared['segment_edge_numbers'], segment_rows=prepared['segment_rows'],
                    **prepared['overlap_arguments'], draw_random_number=forbid_draw)
                ordering_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                observations = []
                with diagnostic_scanner(memo, observations):
                    feedback = evaluator(ordering=ordering['ordering'], discarded_defect_vertices=r['inside_discarded'],
                        vertex_rip=np.asarray(state['vertex_rip'], np.int32), use_counts=np.zeros(len(r['inside_vertices']), np.int32),
                        vertex_fitness=np.zeros(len(r['inside_vertices']), np.float32))
                elapsed = time.perf_counter() - tick
                last = evaluator.last_result
                tick = time.perf_counter()
                arrays, control = candidate_payload(last['candidate'])
                arrays.update(fitness=np.asarray(feedback['fitness'], np.float64),
                    feedback_vertex_rip=feedback['vertex_rip'], feedback_use_counts=feedback['use_counts'],
                    feedback_vertex_fitness=feedback['vertex_fitness'], reset_edge_used=last['edge_used'])
                np.savez_compressed(output / (name + '.candidate.private.npz'), **arrays)
                dump_state_checkpoint(output / (name + '.state.private.npz'), evaluator.surface_state)
                write(output / (name + '.control.private.json'), control)
                writing = time.perf_counter() - tick
                g = last['candidate']['geometry']
                records.append(dict(run=name, complete_cycle_seconds=elapsed,
                    preparation_seconds=preparation, preparation_and_cycle_seconds=preparation + elapsed,
                    ordering_seconds=ordering_seconds,
                    full_candidate_api_seconds=preparation + ordering_seconds + elapsed,
                    candidate_api_and_output_seconds=preparation + ordering_seconds + elapsed + writing,
                    cycle_subseconds=last['seconds'], candidate_subseconds=last['candidate']['seconds'],
                    retessellation_seconds=g.get('retessellation_seconds'),
                    complete_scan_api_seconds=sum(v.complete_scan_api_seconds for v in observations),
                    scanner_preparation_seconds=sum(v.preparation_seconds for v in observations),
                    output_npz_control_write_seconds=writing,
                    logical_predicate_calls=g['blocker_predicate_calls'], sequence_sha256=g['blocker_predicate_sequence_sha256'],
                    cache=memo.report() if memo is not None else None))
            finally:
                if memo is not None:
                    memo.close()
    loaded_after = loaded_source(root, manifest)
    helpers_after = loaded_helpers(root, manifest, helper_functions)
    if any(helpers_after.get(name) != value for name, value in helpers_before.items()):
        raise ValueError('actual benchmark helper source binding changed during calculation')
    if any(loaded_after.get(name) != value for name, value in loaded_before.items()):
        raise ValueError('FNIT source binding changed during calculation')
    if {key: sha(path) for key, path in paths.items()} != input_hashes:
        raise ValueError('frozen MRI input changed')
    artifact_names = [name + suffix for name in ('cold', 'warm', 'audit')
                      for suffix in ('.candidate.private.npz', '.state.private.npz', '.control.private.json')]
    report = dict(status='complete_fixed_first_identity_candidate_diagnostic', arm=arm, records=records,
        stages_seconds=stages, input_sha256=input_hashes, source_manifest_sha256=config['source_manifest_sha256'],
        actual_loaded_fnit_sha256=loaded_after, actual_tool_sha256=sha(__file__),
        actual_loaded_fnit_sources=source_records(root, manifest, loaded_after),
        actual_loaded_benchmark_tool_sources=helpers_after,
        all_fnit_modules_inside_frozen_root=True, protocol_sha256=config['protocol_sha256'],
        artifact_sha256={name: sha(output / name) for name in artifact_names},
        threads=threads, cpu_affinity=sorted(os.sched_getaffinity(0)), cuda_hidden=True,
        actual_thread_policy=actual_thread_policy,
        fresh_thread_environment={name: os.environ[name] for name in thread_names},
        peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        cold_tool_seconds_before_report=time.perf_counter() - started,
        actual_rng_draws=0, population_index=0, identity_ordering=True,
        global_function_replacement=True, production_wiring=False,
        observer_same_outer_scan_timing_both_arms=True, isolated_numba_cache_per_fresh_arm=True,
        cache_shared_between_repeats=False, cross_candidate_real_search='not_assessed',
        whole_recon_all='not_assessed', full_ga=False, final_mesh_quality='not_assessed')
    write(output / 'report.private.json', report)
    write(output / 'COMPLETE.json', dict(status=report['status'], report_sha256=sha(output / 'report.private.json')))
    return report


def compare_completed(*, baseline, candidate, output, source_manifest, protocol):
    """两臂完整收据就绪后才读NPZ，比较完整字段集/shape/dtype/bytes/控制。"""
    started = time.perf_counter()
    baseline, candidate, output = Path(baseline), Path(candidate), Path(output)
    if output.exists():
        raise FileExistsError('fresh comparison JSON required')
    manifest_sha256 = sha(source_manifest); protocol_sha256 = sha(protocol)
    manifest = json.loads(Path(source_manifest).read_text()); declared = json.loads(Path(protocol).read_text())
    validate_protocol(declared, manifest, manifest_sha256)
    reports = []
    for path in (baseline, candidate):
        receipt = json.loads((path / 'COMPLETE.json').read_text())
        if receipt['report_sha256'] != sha(path / 'report.private.json'):
            raise ValueError('complete report receipt mismatch')
        report = json.loads((path / 'report.private.json').read_text())
        if (receipt['status'] != 'complete_fixed_first_identity_candidate_diagnostic'
                or report['status'] != receipt['status']
                or [row['run'] for row in report['records']] != ['cold', 'warm', 'audit']
                or report['actual_rng_draws'] != 0):
            raise ValueError('complete fixed identity candidate report required')
        validate_report_binding(report, manifest, manifest_sha256, protocol_sha256)
        if (report['input_sha256'] != declared['input_sha256']
                or report['threads'] != declared['resources']['requested_threads']
                or report['cpu_affinity'] != declared['resources']['candidate_cpu_affinity']):
            raise ValueError('actual frozen input/resource binding mismatch')
        if set(report['artifact_sha256']) != ARTIFACT_NAMES:
            raise ValueError('complete nine candidate/state/control artifacts required')
        for name, expected in report['artifact_sha256'].items():
            if sha(path / name) != expected:
                raise ValueError('complete candidate/state/control artifact mismatch')
        reports.append(report)
    if ([r['arm'] for r in reports] != ['baseline', 'memo']
            or reports[0]['input_sha256'] != reports[1]['input_sha256']
            or reports[0]['source_manifest_sha256'] != reports[1]['source_manifest_sha256']
            or reports[0]['actual_loaded_fnit_sha256'] != reports[1]['actual_loaded_fnit_sha256']
            or reports[0]['actual_loaded_benchmark_tool_sources'] != reports[1]['actual_loaded_benchmark_tool_sources']
            or reports[0]['threads'] != reports[1]['threads']
            or reports[0]['cpu_affinity'] != reports[1]['cpu_affinity']):
        raise ValueError('same frozen inputs/source/resource pair required')
    checks = []
    for index, name in enumerate(('cold', 'warm', 'audit')):
        for kind in ('candidate', 'state'):
            with np.load(baseline / f'{name}.{kind}.private.npz', allow_pickle=False) as left, \
                    np.load(candidate / f'{name}.{kind}.private.npz', allow_pickle=False) as right:
                keys_equal = set(left.files) == set(right.files)
                expected_fields = CANDIDATE_FIELDS if kind == 'candidate' else STATE_FIELDS
                if set(left.files) != expected_fields or set(right.files) != expected_fields:
                    raise ValueError('complete fixed candidate/state field schema required')
                fields = {key: bool(left[key].shape == right[key].shape and left[key].dtype == right[key].dtype
                          and left[key].tobytes() == right[key].tobytes()) for key in left.files} if keys_equal else {}
            checks.append(dict(run=name, kind=kind, keys_equal=keys_equal, field_count=len(fields),
                               all_shape_dtype_bytes_equal=keys_equal and all(fields.values())))
        for path in (baseline, candidate):
            if set(json.loads((path / f'{name}.control.private.json').read_text())) != CONTROL_FIELDS:
                raise ValueError('complete fixed six-field control schema required')
        checks.append(dict(run=name, kind='control', all_shape_dtype_bytes_equal=
            (baseline / f'{name}.control.private.json').read_bytes() == (candidate / f'{name}.control.private.json').read_bytes()))
        l, r = (report['records'][index] for report in reports)
        checks.append(dict(run=name, kind='logical_audit', all_shape_dtype_bytes_equal=
            l['logical_predicate_calls'] == r['logical_predicate_calls'] and l['sequence_sha256'] == r['sequence_sha256']))
        if name == 'audit' and (not isinstance(l['sequence_sha256'], str) or len(l['sequence_sha256']) != 64):
            raise ValueError('full ordered audit SHA required')
        if r['cache']['logical_predicate_calls'] != r['logical_predicate_calls']:
            raise ValueError('memo misses/hits must cover every logical predicate')
    for arm, path in zip(('baseline', 'memo'), (baseline, candidate)):
        for name in ('warm', 'audit'):
            for kind in ('candidate', 'state'):
                with np.load(path / f'cold.{kind}.private.npz', allow_pickle=False) as left, \
                        np.load(path / f'{name}.{kind}.private.npz', allow_pickle=False) as right:
                    same = set(left.files) == set(right.files) and all(
                        left[k].shape == right[k].shape and left[k].dtype == right[k].dtype
                        and left[k].tobytes() == right[k].tobytes() for k in left.files)
                checks.append(dict(run=name, kind=arm + '_self_repeat_' + kind, all_shape_dtype_bytes_equal=same))
            checks.append(dict(run=name, kind=arm + '_self_repeat_control', all_shape_dtype_bytes_equal=
                (path / 'cold.control.private.json').read_bytes() == (path / f'{name}.control.private.json').read_bytes()))
    passed = all(c['all_shape_dtype_bytes_equal'] for c in checks)
    result = dict(status='passed_fixed_candidate_pair' if passed else 'failed_fixed_candidate_pair',
                  checks=checks, reports_sha256=[sha(p / 'report.private.json') for p in (baseline, candidate)],
                  comparison_hash_npz_control_read_seconds=time.perf_counter() - started,
                  full_ga=False, production_wiring=False, whole_recon_all='not_assessed')
    write(output, result)
    if not passed:
        raise RuntimeError('complete fixed candidate/state/control/audit mismatch')
    return result


def run_pair(*, baseline_config, memo_config, output):
    """两个fresh exec顺序完成后才比较；CLI外层墙钟包含导入、读写和退出。"""
    output = Path(output)
    if output.exists():
        raise FileExistsError('fresh pair controller directory required')
    configs = [json.loads(Path(p).read_text()) for p in (baseline_config, memo_config)]
    for key in ('input_sha256', 'source_manifest_sha256', 'protocol_sha256', 'threads', 'cpu_affinity', 'maximum_entries'):
        if configs[0][key] != configs[1][key]:
            raise ValueError('same input/source/CPU budget/cache cap required')
    if Path(configs[0]['numba_cache_dir']).resolve() == Path(configs[1]['numba_cache_dir']).resolve():
        raise ValueError('isolated Numba cache directories must differ')
    output.mkdir(parents=True)
    rows = []
    for arm, config_path, config in zip(('baseline', 'memo'), (baseline_config, memo_config), configs):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', NUMBA_CACHE_DIR=config['numba_cache_dir'])
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'NUMBA_NUM_THREADS'):
            environment[name] = str(config['threads'])
        tick = time.perf_counter()
        with (output / (arm + '.private.log')).open('w') as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--config', str(config_path), '--arm', arm],
                env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        rows.append(dict(arm=arm, exit_code=result.returncode,
                         complete_cold_cli_wall_seconds=time.perf_counter() - tick))
        write(output / 'cli.private.json', dict(records=rows, raw_t1_whole=False,
            includes_imports_preflight_raw_preparation_three_candidates_io_exit=True,
            no_other_arm_outputs_read_before_both_complete=True))
        if result.returncode:
            raise RuntimeError('fixed candidate arm failed; inspect preserved private log/exit')
    return compare_completed(baseline=configs[0]['output'], candidate=configs[1]['output'], output=output / 'pair.private.json',
        source_manifest=configs[0]['source_manifest'], protocol=configs[0]['protocol'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--arm', choices=('baseline', 'memo'))
    parser.add_argument('--compare', nargs=5, metavar=('BASELINE', 'CANDIDATE', 'OUTPUT', 'SOURCE_MANIFEST', 'PROTOCOL'))
    parser.add_argument('--pair', nargs=3, metavar=('BASELINE_CONFIG', 'MEMO_CONFIG', 'OUTPUT'))
    args = parser.parse_args()
    if args.pair:
        if args.config or args.arm or args.compare:
            parser.error('--pair is separate from --config/--arm/--compare')
        run_pair(baseline_config=args.pair[0], memo_config=args.pair[1], output=args.pair[2])
    elif args.compare:
        if args.config or args.arm:
            parser.error('--compare is separate from --config/--arm')
        compare_completed(baseline=args.compare[0], candidate=args.compare[1], output=args.compare[2],
                          source_manifest=args.compare[3], protocol=args.compare[4])
    else:
        if args.config is None or args.arm is None:
            parser.error('--config and --arm are required')
        run_arm(json.loads(args.config.read_text()), args.arm)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
