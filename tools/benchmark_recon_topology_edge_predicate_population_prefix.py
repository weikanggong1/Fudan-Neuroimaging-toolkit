"""独立前三初始种群memo诊断；保留成熟控制器、连续反馈和完整RNG流。

只执行显式诊断p=3/max_unchanged=0/niters=0的初始前缀，不执行完整GA。
源码、三项自产输入和资源由私有config及冻结protocol/manifest指定。
两臂完成前不读另一臂输出；不修改生产后端、cap或既有v5文件。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import benchmark_recon_topology_edge_predicate_cache as identity

TOOL_RELATIVE = 'tools/benchmark_recon_topology_edge_predicate_population_prefix.py'
RUNS = ('cold', 'warm', 'audit')
SLOTS = (0, 1, 2)
ORDER_FIELDS = frozenset(('ordering', 'segment_order', 'selected_segment_index', 'mutated',
    'ordering_random_draws', 'input_vertex_rip', 'input_use_counts', 'input_vertex_fitness',
    'input_discarded_defect_vertices', 'population_slot', 'rng_call_count_before',
    'rng_call_count_after', 'canonical_sha256', 'base_context_state_sha256'))
PREFIX_FIELDS = frozenset(('best_ordering', 'best_discarded', 'best_fitness', 'population_orderings',
    'population_fitness', 'population_ranks', 'best_population_index', 'discarded_defect_vertices',
    'vertex_rip', 'use_counts', 'vertex_fitness', 'deletion_count', 'elimination_threshold', 'generations'))
PREFIX_CONTROL_FIELDS = frozenset(('evaluation_trace', 'generation_trace', 'termination', 'counts',
    'final_tessellation_complete', 'complete_ga', 'overall_metric_equivalence'))
COUNT_FIELDS = frozenset(('evaluated', 'removed_vertices', 'best_removed_vertices', 'best_patch_number',
    'mutation_attempts', 'mutation_improvements', 'crossover_attempts', 'crossover_improvements'))
RNG_DTYPE = np.dtype([('ordinal', '<u8'), ('low', '<f8'), ('high', '<f8'), ('value', '<f8')])
ARTIFACT_NAMES = frozenset(
    [f'{run}.slot{slot}.{kind}.private.{ext}' for run in RUNS for slot in SLOTS
     for kind, ext in (('candidate', 'npz'), ('state', 'npz'), ('ordering', 'npz'), ('control', 'json'))]
    + [f'{run}.{kind}.private.{ext}' for run in RUNS
       for kind, ext in (('rng', 'npy'), ('prefix', 'npz'), ('prefix_control', 'json'))])
THREAD_NAMES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'NUMBA_NUM_THREADS')
sha, write = identity.sha, identity.write


class RandomTrace:
    """包住现有VNL实例；原参数/返回值不改，记录完整序号和F64区间/值。"""
    def __init__(self, source):
        self.source = source
        self.frames = []

    def __call__(self, low, high):
        value = self.source(low, high)
        self.frames.append((len(self.frames), float(low), float(high), float(value)))
        return value

    def array(self):
        return np.asarray(self.frames, dtype=RNG_DTYPE)


@contextmanager
def observe_orderings(trace):
    """只在新诊断进程观察成熟ordering函数；finally恢复，无额外随机调用。"""
    from fnit.recon_all import topology_ga_population as population
    original = population.generate_initial_ordering
    records = []
    def observed(**kwargs):
        before = len(trace.frames); tick = time.perf_counter()
        result = original(**kwargs)
        seconds = time.perf_counter() - tick
        records.append(dict(population_slot=kwargs['population_index'],
            rng_call_count_before=before, rng_call_count_after=len(trace.frames),
            ordering_seconds=seconds, result={k: v.copy() if isinstance(v, np.ndarray) else v
                                             for k, v in result.items()}))
        return result
    population.generate_initial_ordering = observed
    try:
        yield records
    finally:
        population.generate_initial_ordering = original


def run_initial_prefix(*, search_arguments, draw_random_number, evaluate_patch,
                       normal_population_size=10, normal_max_unchanged=10):
    """调用未改的成熟控制器，截取初始3项；诊断参数不属于生产配方。

    search_arguments包含完整E域、分组/overlap、RP/rip、删除状态和策略；
    不允许暗含种群/停止参数或回调。先检查标准裁剪后仍有至少3项。
    p只作为初始循环界；max_unchanged=0跳过各代，niters=0明确不跑代。
    调用本身不生成最终网格，也不复位传入评估器的跨候选状态。
    """
    from fnit.recon_all.topology_genetic_search import source_population_limits
    from fnit.recon_all.topology_ga_population import run_population_search
    forbidden = {'population_size', 'max_unchanged', 'niters', 'draw_random_number', 'evaluate_patch'}
    if forbidden.intersection(search_arguments):
        raise ValueError('diagnostic prefix policy/callback must be explicit')
    normal = source_population_limits(edge_count=search_arguments['edge_count'],
        population_size=normal_population_size, max_unchanged=normal_max_unchanged)
    if normal['population_size'] < 3:
        raise ValueError('normal initial population has fewer than three candidates')
    return run_population_search(**search_arguments, population_size=3, max_unchanged=0, niters=0,
        draw_random_number=draw_random_number, evaluate_patch=evaluate_patch)


def validate_protocol(protocol, manifest, manifest_sha256):
    identity.validate_protocol(protocol, manifest, manifest_sha256)
    gates = protocol['strict_gates']
    expected = {'ordering_npz_fields': ORDER_FIELDS, 'prefix_npz_fields': PREFIX_FIELDS,
                'prefix_control_fields': PREFIX_CONTROL_FIELDS, 'prefix_count_fields': COUNT_FIELDS,
                'artifact_names': ARTIFACT_NAMES}
    if any(set(gates[k]) != v for k, v in expected.items()):
        raise ValueError('complete fixed prefix/ordering/artifact schema required')
    if (protocol['prefix_slots'] != list(SLOTS) or protocol['arm_run_order'] != list(RUNS)
            or protocol['diagnostic_policy'] != dict(population_size=3, max_unchanged=0, niters=0)
            or protocol['normal_policy'] != dict(population_size=10, max_unchanged=10, niters=-1)
            or protocol['seed'] != 1234 or protocol['cache']['maximum_entries'] != 250000
            or protocol['hemisphere'] != 'lh'
            or gates['comparison_count'] != 114 or gates['rng_comparison'] != 'ordinal_low_high_value_bytes'):
        raise ValueError('fixed first-three prefix/source RNG policy required')


def validate_source(config):
    # Reuse full v5 payload/name/path/hash checks without changing its globals.
    root, manifest = identity.validate_source(config)
    protocol = json.loads(Path(config['protocol']).read_text())
    validate_protocol(protocol, manifest, config['source_manifest_sha256'])
    if config.get('hemisphere') != protocol['hemisphere']:
        raise ValueError('actual configured hemisphere differs from frozen protocol')
    entry = (root / TOOL_RELATIVE).resolve()
    if Path(__file__).resolve() != entry or sha(entry) != manifest['files'][TOOL_RELATIVE]['sha256']:
        raise ValueError('actual prefix entry loaded outside/different from freeze')
    return root, manifest, protocol


def validate_report_binding(report, manifest, manifest_sha256, protocol_sha256):
    if (report['source_manifest_sha256'] != manifest_sha256 or report['protocol_sha256'] != protocol_sha256
            or report['actual_tool_sha256'] != manifest['files'][TOOL_RELATIVE]['sha256']
            or report['all_fnit_modules_inside_frozen_root'] is not True):
        raise ValueError('actual prefix tool/protocol/source binding mismatch')
    sources, loaded = report['actual_loaded_fnit_sources'], report['actual_loaded_fnit_sha256']
    if not identity.REQUIRED_FNIT.issubset(loaded) or set(sources) != set(loaded):
        raise ValueError('complete actual FNIT binding required')
    for name, digest in loaded.items():
        relative = identity.module_relative(name, manifest); r = manifest['files'][relative]
        if digest != r['sha256'] or sources[name] != dict(relative_path=relative,
                origin_relative_path=relative, size=r['size'], sha256=r['sha256']):
            raise ValueError('actual FNIT source path/origin/hash binding mismatch')
    helpers = report['actual_loaded_benchmark_tool_sources']
    if not (identity.REQUIRED_HELPERS | {'benchmark_recon_topology_edge_predicate_cache'}).issubset(helpers):
        raise ValueError('complete actual prefix helper binding required')
    for name, source in helpers.items():
        relative = 'tools/' + name + '.py'; r = manifest['files'].get(relative)
        if not name.startswith('benchmark_recon_topology_') or r is None or source != dict(
                relative_path=relative, origin_relative_path=relative, size=r['size'], sha256=r['sha256']):
            raise ValueError('actual helper source path/origin/hash binding mismatch')


def canonical_contract(evaluator):
    """检查同一prepared的完整不可变值域；不依赖可变object id缓存。"""
    from fnit.recon_all.topology_edge_predicate_cache import _immutable_layout
    r = evaluator.retessellation_inputs; canonical = r['canonical']
    if canonical is not r['base_context'].canonical:
        raise ValueError('same prepared canonical object required')
    layout, buffer = _immutable_layout(canonical)
    return canonical, layout, buffer, hashlib.sha256(buffer).hexdigest(), r['base_context'].state_sha256


def same_canonical_domain(current, original):
    """按对象/完整只读链元信息核验，不对ndarray使用含糊的tuple相等。"""
    return (current[0] is original[0] and current[2] is original[2]
        and len(current[1]) == len(original[1])
        and all(a[0] is b[0] and a[1:] == b[1:] for a,b in zip(current[1],original[1]))
        and current[3:] == original[3:])


def ordering_payload(*, observed, inputs, canonical_sha256, base_sha256):
    o = observed['result']
    if np.asarray(inputs['ordering']).tobytes() != o['ordering'].tobytes():
        raise ValueError('evaluation ordering differs from actual mature generator')
    return dict(ordering=o['ordering'], segment_order=o['segment_order'],
        selected_segment_index=np.asarray(-1 if o['selected_segment_index'] is None else o['selected_segment_index'], np.int32),
        mutated=np.asarray(o['mutated'], np.bool_), ordering_random_draws=o['random_draws'],
        input_vertex_rip=inputs['vertex_rip'], input_use_counts=inputs['use_counts'],
        input_vertex_fitness=inputs['vertex_fitness'], input_discarded_defect_vertices=inputs['discarded_defect_vertices'],
        population_slot=np.asarray(observed['population_slot'], np.int32),
        rng_call_count_before=np.asarray(observed['rng_call_count_before'], np.int64),
        rng_call_count_after=np.asarray(observed['rng_call_count_after'], np.int64),
        canonical_sha256=np.frombuffer(bytes.fromhex(canonical_sha256), np.uint8),
        base_context_state_sha256=np.frombuffer(bytes.fromhex(base_sha256), np.uint8))


def prefix_payload(result):
    integer = {'best_population_index', 'deletion_count', 'generations'}
    arrays = {k: np.asarray(result[k], np.int32 if k in integer else
        np.float64 if k == 'best_fitness' else None) for k in PREFIX_FIELDS}
    return arrays, {k: result[k] for k in sorted(PREFIX_CONTROL_FIELDS)}


def prepare_own_first_defect(config):
    """复用成熟自产上游函数；不加载旧候选、官方结果或模型权重。"""
    import nibabel as nib
    from nibabel.freesurfer import io as fsio
    from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
    from fnit.recon_all.topology_genetic_base_prepare import prepare_nofix_genetic_base
    from fnit.recon_all.topology_global_priors import global_topology_priors
    from fnit.recon_all.topology_candidate_inputs import prepare_candidate_input_context, prepare_defect_candidate_inputs
    from fnit.recon_all.topology_genetic_defect_inputs import ordered_genetic_defect_inputs
    tick = time.perf_counter()
    raw, faces, geometry = fsio.read_geometry(config['inputs']['nofix'], read_metadata=True)
    sphere, sf = fsio.read_geometry(config['inputs']['qsphere'])
    raw, faces, sphere = np.asarray(raw, np.float32), np.asarray(faces, np.int32), np.asarray(sphere, np.float32)
    if raw.shape != sphere.shape or not np.array_equal(faces, sf) or config['hemisphere'] not in ('lh', 'rh'):
        raise ValueError('same ordered own nofix/qsphere and explicit hemisphere required')
    image = nib.load(config['inputs']['brain']); volume = np.asarray(image.dataobj)
    matrix = source_topology_ras_to_voxel(volume_header=image.header)
    metadata = dict(use_real_ras=0, hemisphere=0 if config['hemisphere'] == 'lh' else 1,
        status=0, origxyz_status=0, volume_geometry=dict(geometry))
    stages = dict(load_seconds=time.perf_counter()-tick); tick = time.perf_counter()
    base = prepare_nofix_genetic_base(vertices=raw, faces=faces, qsphere_vertices=sphere,
        volume=volume, ras_to_voxel=matrix, source_metadata=metadata)
    state = base['surface_state']; source = base['prepared']['base_arguments']; metric = base['metric']
    stages['own_base_seconds'] = time.perf_counter()-tick; tick = time.perf_counter()
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
    stages['own_priors_edges_local_seconds'] = time.perf_counter()-tick
    return state, edges, candidate['candidate_kwargs'], stages


def run_arm(config, arm):
    if arm not in ('baseline', 'memo') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('explicit CPU-only baseline/memo arm required')
    threads = config['threads']
    if type(threads) is not int or threads != 2 or sorted(os.sched_getaffinity(0)) != config['cpu_affinity']:
        raise ValueError('declared two-thread CPU affinity required')
    if any(os.environ.get(n) != str(threads) for n in THREAD_NAMES):
        raise ValueError('all thread environments must match fixed budget')
    cache = Path(config['numba_cache_dir']).resolve()
    if (os.environ.get('NUMBA_CACHE_DIR') is None or Path(os.environ['NUMBA_CACHE_DIR']).resolve() != cache
            or cache.exists() and any(cache.iterdir())):
        raise ValueError('fresh isolated empty Numba cache required')
    root, manifest, protocol = validate_source(config)
    paths = {k: Path(config['inputs'][k]) for k in ('nofix', 'qsphere', 'brain')}
    if {k: sha(p) for k, p in paths.items()} != protocol['input_sha256']:
        raise ValueError('own raw inputs SHA mismatch')
    output = Path(config['output'])
    if output.exists(): raise FileExistsError('fresh diagnostic output required')
    output.mkdir(parents=True)
    from fnit.recon_all.thread_budget import thread_budget
    from fnit.recon_all.topology_genetic_search import prepare_genetic_defect_evaluator, TopologyRandomStream
    from fnit.recon_all.topology_edge_predicate_cache import EdgePredicateMemo
    from fnit.recon_all import topology_ga_population as population, topology_edge_blocker_scan as scanner
    from benchmark_recon_topology_candidate_state import dump_state_checkpoint
    from benchmark_recon_topology_prepared_cache import candidate_payload
    helpers = dict(benchmark_recon_topology_candidate_state=dump_state_checkpoint,
                   benchmark_recon_topology_prepared_cache=candidate_payload)
    loaded_before = identity.loaded_source(root, manifest)
    helpers_before = identity.loaded_helpers(root, manifest, helpers)
    records = []
    with thread_budget(threads=threads) as actual_thread_policy:
        state, edges, candidate, stages = prepare_own_first_defect(config)
        for run in RUNS:
            prefix_tick = time.perf_counter(); tick = time.perf_counter()
            # Independent full initial storage per chain, never per candidate.
            own_initial_state = copy.deepcopy(state)
            prepared = prepare_genetic_defect_evaluator(surface_state=own_initial_state,
                defect_inputs=edges, candidate_inputs=candidate, face_detection_cache=True,
                blocker_scan_backend='numba', distance_backend='serial')
            evaluator = prepared['evaluator']; r = evaluator.retessellation_inputs
            r['blocker_scan_audit'] = run == 'audit'
            canonical, layout, buffer, canonical_sha, base_sha = canonical_contract(evaluator)
            original_domain = (canonical, layout, buffer, canonical_sha, base_sha)
            original_ordering, original_scanner = population.generate_initial_ordering, scanner.OrderedBlockerScan
            memo = EdgePredicateMemo(canonical=canonical, maximum_entries=250000) if arm == 'memo' else None
            preparation_seconds = time.perf_counter()-tick
            random = RandomTrace(TopologyRandomStream(seed=1234))
            slots = []
            try:
                with observe_orderings(random) as orderings:
                    def evaluate(**inputs):
                        slot = len(slots)
                        if slot not in SLOTS or len(orderings) != slot+1 or orderings[slot]['population_slot'] != slot:
                            raise ValueError('actual initial slot/order callback sequence mismatch')
                        now = canonical_contract(evaluator)
                        if not same_canonical_domain(now, original_domain):
                            raise ValueError('canonical metadata/domain changed between candidates')
                        cache_before = memo.report() if memo is not None else None
                        observations = []; tick = time.perf_counter()
                        with identity.diagnostic_scanner(memo, observations):
                            feedback = evaluator(**inputs)
                        cycle_seconds = time.perf_counter()-tick
                        last = evaluator.last_result; g = last['candidate']['geometry']
                        tick = time.perf_counter()
                        arrays, control = candidate_payload(last['candidate'])
                        arrays.update(fitness=np.asarray(feedback['fitness'], np.float64),
                            feedback_vertex_rip=feedback['vertex_rip'], feedback_use_counts=feedback['use_counts'],
                            feedback_vertex_fitness=feedback['vertex_fitness'], reset_edge_used=last['edge_used'])
                        np.savez_compressed(output/f'{run}.slot{slot}.candidate.private.npz', **arrays)
                        dump_state_checkpoint(output/f'{run}.slot{slot}.state.private.npz', evaluator.surface_state)
                        np.savez_compressed(output/f'{run}.slot{slot}.ordering.private.npz',
                            **ordering_payload(observed=orderings[slot], inputs=inputs,
                                               canonical_sha256=canonical_sha, base_sha256=base_sha))
                        write(output/f'{run}.slot{slot}.control.private.json', control)
                        writing = time.perf_counter()-tick
                        slots.append(dict(slot=slot, complete_cycle_seconds=cycle_seconds,
                            ordering_seconds=orderings[slot]['ordering_seconds'],
                            complete_scan_api_seconds=sum(x.complete_scan_api_seconds for x in observations),
                            scanner_preparation_seconds=sum(x.preparation_seconds for x in observations),
                            retessellation_seconds=g.get('retessellation_seconds'),
                            output_npz_control_write_seconds=writing,
                            logical_predicate_calls=g['blocker_predicate_calls'],
                            sequence_sha256=g['blocker_predicate_sequence_sha256'],
                            cache_before=cache_before, cache_after=memo.report() if memo is not None else None,
                            cycle_subseconds=last['seconds'], candidate_subseconds=last['candidate']['seconds']))
                        return feedback
                    args = dict(edge_count=len(r['candidate_edges']), initial_selection=True,
                        segment_edge_numbers=prepared['segment_edge_numbers'], segment_rows=prepared['segment_rows'],
                        **prepared['overlap_arguments'], translated_defect_vertices=r['inside_vertices'],
                        discarded_defect_vertices=r['inside_discarded'],
                        vertex_rip=np.asarray(own_initial_state['vertex_rip'], np.int32),
                        use_counts=np.zeros(len(r['inside_vertices']), np.int32),
                        vertex_fitness=np.zeros(len(r['inside_vertices']), np.float32),
                        vertex_eliminate=True, deletion_count=1, elimination_threshold=np.float32(4))
                    tick = time.perf_counter()
                    prefix = run_initial_prefix(search_arguments=args, draw_random_number=random, evaluate_patch=evaluate)
                    controller_with_output = time.perf_counter()-tick
                restored = (population.generate_initial_ordering is original_ordering
                            and scanner.OrderedBlockerScan is original_scanner)
                if not restored:
                    raise ValueError('actual diagnostic function restoration required')
                if len(slots) != 3 or random.source.draws != len(random.frames):
                    raise ValueError('complete three-slot feedback/RNG accounting required')
                now = canonical_contract(evaluator)
                if not same_canonical_domain(now, original_domain):
                    raise ValueError('canonical changed at prefix completion')
                tick = time.perf_counter()
                np.save(output/f'{run}.rng.private.npy', random.array(), allow_pickle=False)
                arrays, control = prefix_payload(prefix)
                np.savez_compressed(output/f'{run}.prefix.private.npz', **arrays)
                write(output/f'{run}.prefix_control.private.json', control)
                summary_write = time.perf_counter()-tick
                records.append(dict(run=run, slots=slots, preparation_seconds=preparation_seconds,
                    controller_with_output_seconds=controller_with_output,
                    summary_output_write_seconds=summary_write,
                    complete_prefix_api_and_output_seconds=time.perf_counter()-prefix_tick,
                    source_controller_seconds=prefix['seconds'], actual_rng_draws=len(random.frames),
                    full_rng_stream_sha256=hashlib.sha256(random.array().tobytes()).hexdigest(),
                    canonical_sha256=canonical_sha, base_context_state_sha256=base_sha,
                    canonical_domain_constant=True, hooks_restored_observed=restored,
                    memo_shared_across_slots=True if memo is not None else False))
            finally:
                release_tick = time.perf_counter()
                if memo is not None: memo.close()
                if records and records[-1]['run'] == run:
                    records[-1]['memo_release_seconds'] = time.perf_counter()-release_tick
                    records[-1]['complete_prefix_api_and_output_seconds'] = time.perf_counter()-prefix_tick
    loaded_after = identity.loaded_source(root, manifest)
    helpers_after = identity.loaded_helpers(root, manifest, helpers)
    if any(loaded_after.get(k) != v for k,v in loaded_before.items()) or any(helpers_after.get(k) != v for k,v in helpers_before.items()):
        raise ValueError('actual frozen source binding changed during calculation')
    if {k: sha(p) for k,p in paths.items()} != protocol['input_sha256']:
        raise ValueError('own raw input changed during calculation')
    report = dict(status='complete_fixed_initial3_prefix_diagnostic', arm=arm, records=records,
        stages_seconds=stages, input_sha256=protocol['input_sha256'], hemisphere=config['hemisphere'],
        source_manifest_sha256=config['source_manifest_sha256'], protocol_sha256=config['protocol_sha256'],
        actual_tool_sha256=sha(__file__), actual_loaded_fnit_sha256=loaded_after,
        actual_loaded_fnit_sources=identity.source_records(root, manifest, loaded_after),
        actual_loaded_benchmark_tool_sources=helpers_after, all_fnit_modules_inside_frozen_root=True,
        artifact_sha256={n: sha(output/n) for n in ARTIFACT_NAMES},
        threads=threads, cpu_affinity=sorted(os.sched_getaffinity(0)), cuda_hidden=True,
        actual_thread_policy=actual_thread_policy,
        fresh_thread_environment={n: os.environ[n] for n in THREAD_NAMES},
        peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
        ordering_function_replacement=True, scanner_factory_replacement=True, restore_in_finally=True,
        global_function_replacement=True,
        memo_shared_between_repeats=False, original_math_changed=False, production_wiring=False,
        diagnostic_population_size=3, seed=1234, full_ga=False, final_mesh_quality='not_assessed',
        whole_recon_all='not_assessed', overall_equivalence='not_assessed')
    write(output/'report.private.json', report)
    write(output/'COMPLETE.json', dict(status=report['status'], report_sha256=sha(output/'report.private.json')))
    return report


def _arrays(path, expected):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != expected: raise ValueError('complete fixed NPZ field schema required')
        return {k: z[k] for k in z.files}


def validate_ordering_and_rng(path, run):
    rng = np.load(path/f'{run}.rng.private.npy', allow_pickle=False)
    if (rng.ndim != 1 or rng.dtype != RNG_DTYPE or not np.array_equal(rng['ordinal'], np.arange(len(rng),dtype=np.uint64))
            or any(not np.isfinite(rng[k]).all() for k in ('low','high','value'))
            or np.any(rng['high'] < rng['low']) or np.any(rng['value'] < rng['low']) or np.any(rng['value'] > rng['high'])):
        raise ValueError('complete ordered finite RNG stream required')
    end = 0; canonical = base = None; previous_feedback = initial_discarded = None
    for slot in SLOTS:
        z = _arrays(path/f'{run}.slot{slot}.ordering.private.npz', ORDER_FIELDS)
        ints = ('ordering','segment_order','input_vertex_rip','input_use_counts','selected_segment_index','population_slot')
        if (any(z[k].dtype != np.int32 for k in ints) or z['ordering'].ndim != 1 or z['segment_order'].ndim != 1
                or not np.array_equal(np.sort(z['ordering']), np.arange(len(z['ordering'])))
                or any(z[k].shape != () for k in ('selected_segment_index','population_slot','mutated','rng_call_count_before','rng_call_count_after'))
                or z['mutated'].dtype != np.bool_ or z['input_vertex_fitness'].dtype != np.float32
                or z['input_discarded_defect_vertices'].dtype != np.bool_
                or z['input_vertex_rip'].ndim != 1 or z['input_use_counts'].ndim != 1
                or z['input_vertex_fitness'].shape != z['input_use_counts'].shape
                or z['input_discarded_defect_vertices'].shape != z['input_use_counts'].shape
                or not np.isfinite(z['input_vertex_fitness']).all()
                or z['ordering_random_draws'].dtype != np.float64 or z['ordering_random_draws'].ndim != 2
                or z['ordering_random_draws'].shape[1:] != (3,)
                or any(z[k].dtype != np.int64 for k in ('rng_call_count_before','rng_call_count_after'))
                or any(z[k].dtype != np.uint8 or z[k].shape != (32,) for k in ('canonical_sha256','base_context_state_sha256'))
                or int(z['population_slot']) != slot or int(z['rng_call_count_before']) != end):
            raise ValueError('actual ordering/RNG/feedback schema required')
        if previous_feedback is not None and any(not _equal_arrays(
                {'value':z['input_'+k]}, {'value':previous_feedback[k]})
                for k in ('vertex_rip','use_counts','vertex_fitness')):
            raise ValueError('source feedback must accumulate without reset between slots')
        if initial_discarded is None: initial_discarded = z['input_discarded_defect_vertices']
        elif not _equal_arrays({'value':initial_discarded},{'value':z['input_discarded_defect_vertices']}):
            raise ValueError('initial prefix cannot perform generation-time elimination')
        feedback = _arrays(path/f'{run}.slot{slot}.candidate.private.npz',identity.CANDIDATE_FIELDS)
        previous_feedback = {k:feedback['feedback_'+k] for k in ('vertex_rip','use_counts','vertex_fitness')}
        if any(v.shape != z['input_'+k].shape or v.dtype != z['input_'+k].dtype
               for k,v in previous_feedback.items()):
            raise ValueError('complete source feedback dimensions/types required')
        start, end = int(z['rng_call_count_before']), int(z['rng_call_count_after'])
        if start < 0 or end < start or end > len(rng): raise ValueError('complete RNG slot boundaries required')
        actual = np.column_stack([rng[k][start:end] for k in ('low','high','value')])
        if actual.shape != z['ordering_random_draws'].shape or actual.tobytes() != z['ordering_random_draws'].tobytes():
            raise ValueError('generator draws differ from full RNG stream')
        if slot == 0 and (end != 0 or bool(z['mutated']) or int(z['selected_segment_index']) != -1
                          or not np.array_equal(z['ordering'], np.arange(len(z['ordering'])))):
            raise ValueError('source slot0 identity/no RNG required')
        if canonical is None: canonical, base = z['canonical_sha256'].tobytes(), z['base_context_state_sha256'].tobytes()
        if z['canonical_sha256'].tobytes() != canonical or z['base_context_state_sha256'].tobytes() != base:
            raise ValueError('constant immutable canonical/prepared domain required')
    if end != len(rng): raise ValueError('unaccounted RNG tail is forbidden')
    return rng


def _equal_arrays(left, right):
    return set(left) == set(right) and all(left[k].shape == right[k].shape and left[k].dtype == right[k].dtype
                                        and left[k].tobytes() == right[k].tobytes() for k in left)


def validate_cache_accounting(record, arm):
    """闭合每slot累计计数；满cap后未知独特key不可由miss数替代。"""
    previous = None
    counts = ('entries','logical_predicate_calls','cache_hits','actual_predicate_evaluations',
              'inserted_unique_keys','capacity_bypass_evaluations')
    for slot in record['slots']:
        if type(slot['logical_predicate_calls']) is not int or slot['logical_predicate_calls'] < 0:
            raise ValueError('nonnegative actual logical predicate count required')
        before, after = slot['cache_before'], slot['cache_after']
        if arm == 'baseline':
            if before is not None or after is not None: raise ValueError('baseline must not consume memo')
            continue
        for cache in (before, after):
            if (any(type(cache[k]) is not int or cache[k] < 0 for k in counts)
                    or cache['maximum_entries'] != 250000 or cache['entries'] > 250000
                    or cache['entries'] != cache['inserted_unique_keys']
                    or cache['logical_predicate_calls'] != cache['cache_hits']+cache['actual_predicate_evaluations']
                    or cache['actual_predicate_evaluations'] != cache['inserted_unique_keys']+cache['capacity_bypass_evaluations']
                    or cache['geometry_only'] is not True or cache['concurrent_writes_allowed'] is not False
                    or cache['complete_unique_key_count'] != (None if cache['capacity_bypass_evaluations'] else cache['entries'])
                    or cache['canonical_sha256'] != record['canonical_sha256']):
                raise ValueError('complete bounded pure predicate memo accounting required')
        if previous is None:
            if any(before[k] != 0 for k in counts): raise ValueError('each repeat must start with empty memo')
        elif before != previous: raise ValueError('same memo cumulative state must continue across slots')
        if (any(after[k] < before[k] for k in counts)
                or after['logical_predicate_calls']-before['logical_predicate_calls'] != slot['logical_predicate_calls']):
            raise ValueError('slot logical calls must equal memo cumulative delta')
        previous = after


def validate_prefix_summary(path, run):
    """初始前缀汇总必须来自三个已保存slot及最后一份反馈，不能省字段。"""
    z=_arrays(path/f'{run}.prefix.private.npz',PREFIX_FIELDS)
    orders=[_arrays(path/f'{run}.slot{s}.ordering.private.npz',ORDER_FIELDS) for s in SLOTS]
    candidates=[_arrays(path/f'{run}.slot{s}.candidate.private.npz',identity.CANDIDATE_FIELDS) for s in SLOTS]
    expected_orders=np.stack([o['ordering'] for o in orders])
    scores=np.asarray([c['fitness'].item() for c in candidates],np.float64)
    best=int(np.argmax(scores));expected=dict(population_orderings=expected_orders,population_fitness=scores,
        best_ordering=expected_orders[best],best_fitness=np.asarray(scores[best],np.float64),
        best_population_index=np.asarray(best,np.int32),best_discarded=orders[best]['input_discarded_defect_vertices'],
        discarded_defect_vertices=orders[-1]['input_discarded_defect_vertices'],
        vertex_rip=candidates[-1]['feedback_vertex_rip'],use_counts=candidates[-1]['feedback_use_counts'],
        vertex_fitness=candidates[-1]['feedback_vertex_fitness'],generations=np.asarray(0,np.int32),
        deletion_count=np.asarray(1,np.int32),elimination_threshold=np.asarray(4,np.float32))
    if any(not _equal_arrays({'value':z[k]},{'value':v}) for k,v in expected.items()):
        raise ValueError('prefix summary must equal actual slot outputs and accumulated final feedback')
    if (not np.isfinite(scores).all() or z['population_ranks'].dtype!=np.int32
            or not np.array_equal(np.sort(z['population_ranks']),np.arange(3))):
        raise ValueError('finite source fitness and complete population ranks required')
    control=json.loads((path/f'{run}.prefix_control.private.json').read_text())
    improved=-np.inf
    for slot,trace in enumerate(control['evaluation_trace']):
        if (set(trace)!={'generation','phase','slot','fitness','strict_improvement'} or trace['generation']!=0
                or trace['phase']!='initial' or trace['slot']!=slot or trace['fitness']!=float(scores[slot])
                or trace['strict_improvement'] is not bool(scores[slot]>improved)):
            raise ValueError('complete actual mature initial evaluation trace required')
        improved=max(improved,scores[slot])


def compare_completed(*, baseline, candidate, output, source_manifest, protocol):
    started = time.perf_counter(); baseline, candidate, output = map(Path,(baseline,candidate,output))
    if output.exists(): raise FileExistsError('fresh prefix comparison required')
    manifest = json.loads(Path(source_manifest).read_text()); declared = json.loads(Path(protocol).read_text())
    validate_protocol(declared, manifest, sha(source_manifest)); reports = []
    for path in (baseline,candidate):
        receipt = json.loads((path/'COMPLETE.json').read_text()); report = json.loads((path/'report.private.json').read_text())
        if (receipt['report_sha256'] != sha(path/'report.private.json') or receipt['status'] != 'complete_fixed_initial3_prefix_diagnostic'
                or report['status'] != receipt['status'] or [r['run'] for r in report['records']] != list(RUNS)):
            raise ValueError('complete initial3 receipt required')
        validate_report_binding(report,manifest,sha(source_manifest),sha(protocol))
        if (set(report['artifact_sha256']) != ARTIFACT_NAMES
                or any(sha(path/n) != v for n,v in report['artifact_sha256'].items())):
            raise ValueError('complete 45-artifact receipt required')
        if (report['input_sha256'] != declared['input_sha256'] or report['threads'] != declared['resources']['requested_threads']
                or report['cpu_affinity'] != declared['resources']['candidate_cpu_affinity'] or report['seed'] != 1234
                or report.get('hemisphere') != declared['hemisphere']):
            raise ValueError('actual fixed input/resource/seed binding mismatch')
        policy = report['actual_thread_policy']
        if (report['cuda_hidden'] is not True or report['global_function_replacement'] is not True
                or report['memo_shared_between_repeats'] is not False or report['production_wiring'] is not False
                or policy['requested_threads'] != 2 or policy['restoration_complete'] is not True
                or any(policy[k]['effective'] != 2 or policy[k]['restored'] != policy[k]['before']
                       for k in ('torch','numba'))
                or report['fresh_thread_environment'] != {n:'2' for n in THREAD_NAMES}):
            raise ValueError('actual fixed thread budget/restoration/diagnostic scope required')
        for record in report['records']:
            run = record['run']; rng = validate_ordering_and_rng(path,run)
            validate_cache_accounting(record, report['arm'])
            if ([s['slot'] for s in record['slots']] != list(SLOTS) or record['actual_rng_draws'] != len(rng)
                    or record['full_rng_stream_sha256'] != hashlib.sha256(rng.tobytes()).hexdigest()
                    or record['canonical_domain_constant'] is not True or record['hooks_restored_observed'] is not True
                    or record['memo_shared_across_slots'] != (report['arm']=='memo')):
                raise ValueError('complete source RNG/slot/domain accounting required')
            if run == 'audit' and any(not isinstance(s['sequence_sha256'],str)
                    or len(s['sequence_sha256']) != 64 or any(x not in '0123456789abcdef' for x in s['sequence_sha256'])
                    for s in record['slots']):
                raise ValueError('actual full ordered predicate audit digest required')
            for slot in SLOTS:
                for kind, fields in (('candidate',identity.CANDIDATE_FIELDS),('state',identity.STATE_FIELDS)):
                    _arrays(path/f'{run}.slot{slot}.{kind}.private.npz',fields)
                if set(json.loads((path/f'{run}.slot{slot}.control.private.json').read_text())) != identity.CONTROL_FIELDS:
                    raise ValueError('complete six-field control required')
            _arrays(path/f'{run}.prefix.private.npz',PREFIX_FIELDS)
            c = json.loads((path/f'{run}.prefix_control.private.json').read_text())
            if (set(c) != PREFIX_CONTROL_FIELDS or set(c['counts']) != COUNT_FIELDS or c['counts']['evaluated'] != 3
                    or c['counts']['mutation_attempts'] != 0 or c['counts']['crossover_attempts'] != 0
                    or c['generation_trace'] != [] or len(c['evaluation_trace']) != 3
                    or [e['phase'] for e in c['evaluation_trace']] != ['initial']*3
                    or [e['slot'] for e in c['evaluation_trace']] != list(SLOTS)
                    or c['complete_ga'] is not False or c['final_tessellation_complete'] is not False):
                raise ValueError('complete initial-only source controller required')
            validate_prefix_summary(path,run)
        reports.append(report)
    if ([r['arm'] for r in reports] != ['baseline','memo']
            or reports[0]['actual_loaded_fnit_sha256'] != reports[1]['actual_loaded_fnit_sha256']
            or reports[0]['actual_loaded_benchmark_tool_sources'] != reports[1]['actual_loaded_benchmark_tool_sources']):
        raise ValueError('same actual source pair required')
    checks = []
    def compare_run(left, left_run, right, right_run, label):
        for slot in SLOTS:
            for kind, fields in (('candidate',identity.CANDIDATE_FIELDS),('state',identity.STATE_FIELDS),('ordering',ORDER_FIELDS)):
                a = _arrays(left/f'{left_run}.slot{slot}.{kind}.private.npz',fields)
                b = _arrays(right/f'{right_run}.slot{slot}.{kind}.private.npz',fields)
                checks.append(dict(scope=label,run=right_run,slot=slot,kind=kind,field_count=len(fields),equal=_equal_arrays(a,b)))
            checks.append(dict(scope=label,run=right_run,slot=slot,kind='control',equal=
                (left/f'{left_run}.slot{slot}.control.private.json').read_bytes() == (right/f'{right_run}.slot{slot}.control.private.json').read_bytes()))
        ra = validate_ordering_and_rng(left,left_run); rb = validate_ordering_and_rng(right,right_run)
        checks.append(dict(scope=label,run=right_run,kind='rng_stream',equal=ra.dtype==rb.dtype and ra.shape==rb.shape and ra.tobytes()==rb.tobytes()))
        checks.append(dict(scope=label,run=right_run,kind='prefix',field_count=14,equal=_equal_arrays(
            _arrays(left/f'{left_run}.prefix.private.npz',PREFIX_FIELDS),_arrays(right/f'{right_run}.prefix.private.npz',PREFIX_FIELDS))))
        checks.append(dict(scope=label,run=right_run,kind='prefix_control',equal=
            (left/f'{left_run}.prefix_control.private.json').read_bytes() == (right/f'{right_run}.prefix_control.private.json').read_bytes()))
    for run in RUNS:
        compare_run(baseline,run,candidate,run,'cross_arm')
        a,b = [next(r for r in x['records'] if r['run']==run) for x in reports]
        for slot in SLOTS:
            x,y = a['slots'][slot],b['slots'][slot]
            checks.append(dict(scope='cross_arm',run=run,slot=slot,kind='predicate_sequence',equal=
                x['logical_predicate_calls']==y['logical_predicate_calls'] and (run!='audit' or x['sequence_sha256']==y['sequence_sha256'])))
    for path, arm in ((baseline,'baseline'),(candidate,'memo')):
        for run in ('warm','audit'): compare_run(path,'cold',path,run,arm+'_self_repeat')
    if len(checks) != 114: raise ValueError('complete fixed 114 comparison groups required')
    passed = all(c['equal'] for c in checks)
    result = dict(status='passed_fixed_initial3_prefix_pair' if passed else 'failed_fixed_initial3_prefix_pair',
        checks=checks,reports_sha256=[sha(p/'report.private.json') for p in (baseline,candidate)],
        comparison_hash_npz_control_read_seconds=time.perf_counter()-started,
        full_ga=False,production_wiring=False,whole_recon_all='not_assessed')
    write(output,result)
    if not passed: raise RuntimeError('complete prefix/RNG/state/control/sequence mismatch')
    return result


def run_pair(*, baseline_config, memo_config, output):
    output = Path(output)
    if output.exists(): raise FileExistsError('fresh pair controller required')
    configs = [json.loads(Path(p).read_text()) for p in (baseline_config,memo_config)]
    for key in ('input_sha256','source_manifest_sha256','protocol_sha256','threads','cpu_affinity','maximum_entries'):
        if configs[0][key] != configs[1][key]: raise ValueError('same fixed input/source/resource/cap required')
    if Path(configs[0]['numba_cache_dir']).resolve()==Path(configs[1]['numba_cache_dir']).resolve():
        raise ValueError('independent fresh JIT caches required')
    output.mkdir(parents=True); rows=[]
    for arm,path,config in zip(('baseline','memo'),(baseline_config,memo_config),configs):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NUMBA_CACHE_DIR=config['numba_cache_dir'])
        env.update({n:str(config['threads']) for n in THREAD_NAMES});tick=time.perf_counter()
        with (output/(arm+'.private.log')).open('w') as log:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--config',str(path),'--arm',arm],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        rows.append(dict(arm=arm,exit_code=result.returncode,complete_cold_cli_wall_seconds=time.perf_counter()-tick))
        write(output/'cli.private.json',dict(records=rows,no_other_arm_outputs_read_before_both_complete=True,
            includes_imports_preflight_raw_preparation_nine_candidates_io_exit=True,raw_t1_whole=False))
        if result.returncode: raise RuntimeError('prefix arm failed; private exit/log preserved')
    return compare_completed(baseline=configs[0]['output'],candidate=configs[1]['output'],output=output/'pair.private.json',
        source_manifest=configs[0]['source_manifest'],protocol=configs[0]['protocol'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path);p.add_argument('--arm',choices=('baseline','memo'))
    p.add_argument('--pair',nargs=3);p.add_argument('--compare',nargs=5);a=p.parse_args()
    if a.pair:
        if a.config or a.arm or a.compare:p.error('--pair is separate')
        run_pair(baseline_config=a.pair[0],memo_config=a.pair[1],output=a.pair[2])
    elif a.compare:
        if a.config or a.arm:p.error('--compare is separate')
        compare_completed(baseline=a.compare[0],candidate=a.compare[1],output=a.compare[2],source_manifest=a.compare[3],protocol=a.compare[4])
    else:
        if a.config is None or a.arm is None:p.error('--config and --arm required')
        run_arm(json.loads(a.config.read_text()),a.arm)


if __name__=='__main__': main()
