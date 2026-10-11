"""不可变四顶点memo合同；小固定几何不作为真实MRI提速证据。"""
import hashlib
import itertools
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from fnit.recon_all.topology_edge_blocker_scan import OrderedBlockerScan
from fnit.recon_all.topology_edge_predicate_cache import EdgePredicateMemo, MemoizedOrderedBlockerScan
from fnit.recon_all.topology_retessellation_dynamic import (
    prepare_retessellation_base, retessellate_ordered_edges,
)


def frozen(points):
    value = np.asarray(points, np.float32)
    return np.frombuffer(value.tobytes(), np.float32).reshape(value.shape)


def scanners(xyz, rows, scan, cap=250_000, audit=True):
    memo = EdgePredicateMemo(canonical=xyz, maximum_entries=cap)
    return (OrderedBlockerScan(scan_vertices=scan, neighbor_rows=rows, audit=audit),
            MemoizedOrderedBlockerScan(scan_vertices=scan, neighbor_rows=rows, memo=memo, audit=audit), memo)


def paired(left, right, xyz, scan, valid, a=0, b=1):
    kwargs = dict(xyz=xyz, a=a, b=b, scan_vertices=scan, valid=valid)
    x, y = left.first_blocker(**kwargs), right.first_blocker(**kwargs)
    assert x == y
    assert left.calls == right.calls
    assert left.predicate_calls == right.predicate_calls
    if left.audit:
        assert left.sequence_digest.hexdigest() == right.sequence_digest.hexdigest()
    return y


def test_cross_candidate_hits_keep_full_dynamic_scan_and_valid_order():
    xyz = frozen([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1], [0, 2, 1], [0, -2, 1]])
    rows = [[], [], [3], [2], [5], [4]]
    scan = np.asarray([4, 2], np.int32)
    left, right, memo = scanners(xyz, rows, scan)
    valid = np.ones(6, np.bool_)
    assert paired(left, right, xyz, scan, valid) == (4, 5)
    # New candidate has its own dynamic rows; only the bool memo is shared.
    left2 = OrderedBlockerScan(scan_vertices=scan, neighbor_rows=rows, audit=True)
    right2 = MemoizedOrderedBlockerScan(scan_vertices=scan, neighbor_rows=rows, memo=memo, audit=True)
    assert paired(left2, right2, xyz, scan, valid) == (4, 5)
    left2.sync_vertex(4, []); right2.sync_vertex(4, [])
    assert paired(left2, right2, xyz, scan, valid) == (2, 3)
    # valid is re-read on every call; the cached first hit cannot bypass it.
    valid[3] = False
    assert paired(left2, right2, xyz, scan, valid) is None
    valid[3] = True
    assert paired(left2, right2, xyz, scan, valid) == (2, 3)
    left2.sync_vertex(4, [5]); right2.sync_vertex(4, [5])
    assert paired(left2, right2, xyz, scan, valid) == (4, 5)
    report = memo.report()
    assert report['cache_hits'] == 3
    assert report['actual_predicate_evaluations'] == 2
    assert report['complete_unique_key_count'] == 2
    assert report['logical_predicate_calls'] == left.predicate_calls + left2.predicate_calls


def test_directed_keys_nearzero_degenerate_shared_ids_and_extreme_f32():
    t = np.float32(5e-8)
    for scale in (np.float32(1), np.float32(1e-30), np.float32(1e30)):
        xyz = frozen(np.asarray([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1],
            [.5, 0, 1], [-.5, 0, 1], [0, np.nextafter(t, np.float32(0)), 1],
            [0, np.nextafter(t, np.float32(1)), 1]], np.float32) * scale)
        with EdgePredicateMemo(canonical=xyz) as memo:
            for a, b, v, w in itertools.permutations(range(8), 4):
                rows = [[] for _ in xyz]; rows[v] = [a, w, b]
                scan = np.asarray([v], np.int32)
                left = OrderedBlockerScan(scan_vertices=scan, neighbor_rows=rows, audit=True)
                right = MemoizedOrderedBlockerScan(scan_vertices=scan, neighbor_rows=rows, memo=memo, audit=True)
                for _ in range(2): paired(left, right, xyz, scan, np.ones(8, np.bool_), a, b)
            report = memo.report()
            assert report['inserted_unique_keys'] == 8 * 7 * 6 * 5
            assert report['cache_hits'] == report['actual_predicate_evaluations']
            assert report['complete_unique_key_count'] == 1680


def test_direction_is_not_sorted_and_signed_zero_values_stay_unchanged():
    xyz = frozen([[1, -0.0, 1], [-1, 0.0, 1], [0, 1, 1], [-0.0, -1, 1]])
    before = xyz.tobytes()
    memo = EdgePredicateMemo(canonical=xyz)
    for a, b, v, w in ((0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        rows = [[] for _ in xyz]; rows[v] = [w]; scan = np.asarray([v], np.int32)
        cached = MemoizedOrderedBlockerScan(scan_vertices=scan, neighbor_rows=rows, memo=memo)
        cached.first_blocker(xyz=xyz, a=a, b=b, scan_vertices=scan, valid=np.ones(4, np.bool_))
    assert memo.report()['entries'] == 4
    assert xyz.tobytes() == before
    assert memo.canonical_sha256 == hashlib.sha256(before).hexdigest()


@pytest.mark.parametrize('cap', [0, 1, 2])
def test_capacity_changes_only_cache_strategy_and_never_skips_predicates(cap):
    xyz = frozen([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, 2, 1], [1, 2, 1], [2, 3, 1]])
    rows = [[], [], [3, 4, 5], [], [], []]; scan = np.asarray([2, 2], np.int32)
    left, right, memo = scanners(xyz, rows, scan, cap=cap)
    for _ in range(3): paired(left, right, xyz, scan, np.ones(6, np.bool_))
    report = memo.report()
    assert report['entries'] == cap
    assert report['capacity_bypass_evaluations'] > 0
    assert report['complete_unique_key_count'] is None
    assert report['cache_hits'] + report['actual_predicate_evaluations'] == left.predicate_calls


def test_mutable_readonly_owner_and_partial_or_wrong_dtype_buffers_rejected():
    writable = np.ones((4, 3), np.float32)
    readonly_owner = writable.copy(); readonly_owner.flags.writeable = False
    readonly_view = writable.view(); readonly_view.flags.writeable = False
    for value in (writable, readonly_owner, readonly_view, frozen(writable)[1:],
                  np.frombuffer(np.ones((4, 3), np.float64).tobytes(), np.float64).reshape(4, 3)):
        with pytest.raises(ValueError): EdgePredicateMemo(canonical=value)


def test_prepared_original_input_changes_do_not_change_cached_immutable_coordinates():
    original = np.asarray([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1]], np.float32)
    base = prepare_retessellation_base(canonical=original, base_faces=np.empty((0, 3), np.int32),
                                       neighbors=[[], [], [3], [2]])
    scan = np.asarray([2], np.int32)
    left, right, memo = scanners(base.canonical, base.neighbors, scan)
    assert paired(left, right, base.canonical, scan, np.ones(4, np.bool_)) == (2, 3)
    original[1] = [1, 2, 1]
    assert paired(left, right, base.canonical, scan, np.ones(4, np.bool_)) == (2, 3)
    with pytest.raises(ValueError, match='exact immutable'):
        right.first_blocker(xyz=original, a=0, b=1, scan_vertices=scan, valid=np.ones(4, np.bool_))
    assert memo.report()['cache_hits'] == 1


def test_context_identity_metadata_lifetime_and_concurrent_use_are_strict():
    xyz = frozen(np.ones((4, 3), np.float32)); scan = np.asarray([2], np.int32)
    _, right, memo = scanners(xyz, [[], [], [3], []], scan)
    same_bits_other_object = frozen(xyz)
    for wrong in (same_bits_other_object, xyz.copy()):
        with pytest.raises(ValueError, match='exact immutable'):
            right.first_blocker(xyz=wrong, a=0, b=1, scan_vertices=scan, valid=np.ones(4, np.bool_))
    shape = xyz.shape; xyz.shape = (2, 6)
    try:
        with pytest.raises(ValueError): memo.report()
    finally: xyz.shape = shape
    with memo._exclusive(xyz):
        with pytest.raises(RuntimeError, match='concurrent or reentrant'):
            right.first_blocker(xyz=xyz, a=0, b=1, scan_vertices=scan, valid=np.ones(4, np.bool_))
        with pytest.raises(RuntimeError): right.sync_vertex(2, [])
        with pytest.raises(RuntimeError): memo.close()
    memo.close(); memo.close()
    assert memo._memo is None
    with pytest.raises(RuntimeError, match='closed'): memo.report()


def test_complete_dynamic_retessellation_and_audit_identical_with_cache(monkeypatch):
    import fnit.recon_all.topology_edge_blocker_scan as module
    xyz = np.asarray([[1, 0, 1], [0, 1, 1], [-1, -1, 1], [0, 0, 1],
                      [.25, -.2, 1], [-.25, .2, 1]], np.float32)
    rows = [[] for _ in xyz]
    base = prepare_retessellation_base(canonical=xyz, base_faces=np.empty((0, 3), np.int32), neighbors=rows)
    edges = np.asarray(list(itertools.combinations(range(6), 2)), np.int32)
    kwargs = dict(canonical=base.canonical, base_faces=base.base_faces, neighbors=base.neighbors,
        base_context=base, candidate_edges=edges, edge_used=np.zeros(15, np.int32),
        ordering=np.asarray([4, 0, 2, 10, 5, 12, 3, 6, 1, 14, 8, 13, 11, 7, 9]),
        inside_vertices=np.arange(6, dtype=np.int32), inside_discarded=np.zeros(6, np.bool_),
        border_vertices=np.empty(0, np.int32), hull_vertices=np.empty(0, np.int32), mode='some',
        blocker_scan_backend='numba', blocker_scan_audit=True)
    expected = retessellate_ordered_edges(**kwargs)
    with EdgePredicateMemo(canonical=base.canonical) as memo:
        def factory(**kw): return MemoizedOrderedBlockerScan(memo=memo, **kw)
        monkeypatch.setattr(module, 'OrderedBlockerScan', factory)
        for _ in range(2):
            actual = retessellate_ordered_edges(**kwargs)
            for key in ('faces', 'edge_used', 'marks', 'selected_edges', 'selected_vertices',
                        'removed_vertices', 'temporary_faces'):
                assert expected[key].dtype == actual[key].dtype
                assert expected[key].shape == actual[key].shape
                assert expected[key].tobytes() == actual[key].tobytes()
            for key in ('neighbors', 'retessellation_trace', 'rounds', 'ninside', 'ndiscarded',
                        'blocker_scan_calls', 'blocker_predicate_calls', 'blocker_predicate_sequence_sha256'):
                assert expected[key] == actual[key], key
        assert expected['rounds'] == 2
        assert expected['removed_vertices'].tolist() == [5]
        assert memo.report()['cache_hits'] > 0


def test_invalid_cap_finite_domain_and_fixed_mapping_fail_clearly():
    for cap in (True, -1, 1.5, 2**63):
        with pytest.raises(ValueError): EdgePredicateMemo(canonical=frozen(np.ones((4, 3))), maximum_entries=cap)
    for xyz in (frozen([[0, 0, 0]]), frozen([[np.nan, 0, 1]]), frozen([[np.inf, 0, 1]])):
        with pytest.raises(ValueError): EdgePredicateMemo(canonical=xyz)
    xyz = frozen(np.ones((4, 3))); scan = np.asarray([2], np.int32)
    _, right, memo = scanners(xyz, [[], [], [3], []], scan)
    for valid in (np.ones(3, np.bool_), np.ones(4, np.int32)):
        with pytest.raises(ValueError): right.first_blocker(xyz=xyz, a=0, b=1, scan_vertices=scan, valid=valid)
    for a in (-1, True, 2**32, .5):
        with pytest.raises(ValueError): right.first_blocker(xyz=xyz, a=a, b=1, scan_vertices=scan, valid=np.ones(4, np.bool_))
    with pytest.raises(ValueError):
        MemoizedOrderedBlockerScan(scan_vertices=scan, neighbor_rows=[[], [], []], memo=memo)


def tool():
    path = Path(__file__).resolve().parents[2] / 'tools/benchmark_recon_topology_edge_predicate_cache.py'
    spec = importlib.util.spec_from_file_location('fnit_predicate_memo_diagnostic_contract', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_diagnostic_factory_observer_restores_on_error_and_preserves_baseline():
    import fnit.recon_all.topology_edge_blocker_scan as module
    original = module.OrderedBlockerScan
    xyz = frozen([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1]])
    scan = np.asarray([2], np.int32); records = []
    for memo in (None, EdgePredicateMemo(canonical=xyz)):
        with pytest.raises(RuntimeError, match='injected'):
            with tool().diagnostic_scanner(memo, records):
                scanner = module.OrderedBlockerScan(scan_vertices=scan, neighbor_rows=[[], [], [3], []], audit=True)
                assert scanner.first_blocker(xyz=xyz, a=0, b=1, scan_vertices=scan, valid=np.ones(4, np.bool_)) == (2, 3)
                raise RuntimeError('injected observer failure')
        assert module.OrderedBlockerScan is original
    assert len(records) == 2
    assert all(v.complete_scan_api_seconds > 0 for v in records)
    assert records[0].sequence_digest.hexdigest() == records[1].sequence_digest.hexdigest()


def fake_completed(root, arm, *, changed=None):
    """仅测试报告比较器，不执行影像或算法，不作为MRI benchmark。"""
    api = tool(); root.mkdir(); artifacts = {}; records = []
    binding = fake_binding(root.parent)
    for name in ('cold', 'warm', 'audit'):
        for kind in ('candidate', 'state'):
            value = np.asarray([1, -0.0], np.float32)
            if changed == (name, kind): value[0] = 2
            path = root / f'{name}.{kind}.private.npz'
            fields = api.CANDIDATE_FIELDS if kind == 'candidate' else api.STATE_FIELDS
            np.savez_compressed(path, **{key: value for key in fields})
            artifacts[path.name] = api.sha(path)
        path = root / f'{name}.control.private.json'
        api.write(path, {key: [1, 2] for key in sorted(api.CONTROL_FIELDS)})
        artifacts[path.name] = api.sha(path)
        records.append(dict(run=name, logical_predicate_calls=3,
            sequence_sha256='a' * 64 if name == 'audit' else None,
            cache={'logical_predicate_calls': 3} if arm == 'memo' else None))
    status = 'complete_fixed_first_identity_candidate_diagnostic'
    api.write(root / 'report.private.json', dict(status=status, arm=arm, records=records,
        artifact_sha256=artifacts, input_sha256={'same_input': 'b' * 64}, source_manifest_sha256=api.sha(binding['source_manifest']),
        protocol_sha256=api.sha(binding['protocol']), actual_tool_sha256='e' * 64,
        actual_loaded_fnit_sha256={name: 'd' * 64 for name in api.REQUIRED_FNIT},
        actual_loaded_fnit_sources={name: dict(relative_path=fake_module_path(name),
            origin_relative_path=fake_module_path(name), size=1, sha256='d' * 64) for name in api.REQUIRED_FNIT},
        all_fnit_modules_inside_frozen_root=True,
        actual_loaded_benchmark_tool_sources={name:dict(relative_path='tools/' + name + '.py',
            origin_relative_path='tools/' + name + '.py', size=1, sha256='1' * 64) for name in api.REQUIRED_HELPERS},
        threads=2, cpu_affinity=[1, 2], actual_rng_draws=0))
    api.write(root / 'COMPLETE.json', dict(status=status, report_sha256=api.sha(root / 'report.private.json')))


def fake_binding(root):
    """固定schema与来源收据的测试夹具；不作为真实程序/影像验证。"""
    api = tool(); manifest = root / 'binding_manifest.json'; protocol = root / 'binding_protocol.json'
    if not manifest.exists():
        files = {
            api.TOOL_RELATIVE: dict(size=1, sha256='e' * 64),
            'src/fnit/recon_all/topology_edge_predicate_cache.py': dict(size=1, sha256='d' * 64)}
        files.update({'tools/' + name + '.py': dict(size=1, sha256='1' * 64) for name in api.REQUIRED_HELPERS})
        files.update({fake_module_path(name): dict(size=1, sha256='d' * 64) for name in api.REQUIRED_FNIT})
        api.write(manifest, dict(git_base='a' * 40, files=files,
            expected_fnit_module_sha256={name: 'd' * 64 for name in api.REQUIRED_FNIT}))
        api.write(protocol, dict(source_manifest_sha256=api.sha(manifest), git_base='a' * 40,
            input_sha256={'same_input': 'b' * 64}, resources=dict(requested_threads=2,candidate_cpu_affinity=[1,2]),
            strict_gates=dict(candidate_npz_fields=sorted(api.CANDIDATE_FIELDS), candidate_npz_field_count=36,
            state_npz_fields=sorted(api.STATE_FIELDS), state_npz_field_count=40,
            full_control_fields=sorted(api.CONTROL_FIELDS))))
    return dict(source_manifest=manifest, protocol=protocol)


def fake_module_path(name):
    return 'src/' + name.replace('.', '/') + ('/__init__.py' if name in ('fnit', 'fnit.recon_all') else '.py')


def refresh_report(root, mutation):
    api = tool(); path = root / 'report.private.json'; value = json.loads(path.read_text())
    mutation(value); api.write(path, value)
    api.write(root / 'COMPLETE.json', dict(status=value['status'], report_sha256=api.sha(path)))


def test_actual_helper_module_and_function_paths_checked_before_calculation(monkeypatch, tmp_path):
    api = tool(); functions = {}; manifest = {'files': {}}
    for name in api.REQUIRED_HELPERS:
        path = tmp_path / ('tools/' + name + '.py'); path.parent.mkdir(exist_ok=True)
        path.write_text('def target():\n    return None\n')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, name, module); functions[name] = module.target
        manifest['files']['tools/' + name + '.py'] = dict(size=path.stat().st_size, sha256=api.sha(path))
    assert set(api.loaded_helpers(tmp_path, manifest, functions)) == api.REQUIRED_HELPERS
    name = next(iter(api.REQUIRED_HELPERS)); module = sys.modules[name]; original = module.__file__
    module.__file__ = str(tmp_path / 'outside.py')
    with pytest.raises(ValueError, match='outside/different'): api.loaded_helpers(tmp_path, manifest, functions)
    module.__file__ = original
    def wrong_function(): return None
    wrong_function.__module__ = name
    with pytest.raises(ValueError, match='function source'):
        api.loaded_helpers(tmp_path, manifest, dict(functions, **{name: wrong_function}))
    Path(original).write_text('def target():\n    return None\n# changed bytes\n')
    with pytest.raises(ValueError, match='outside/different'): api.loaded_helpers(tmp_path, manifest, functions)


def test_completed_pair_compares_full_fields_and_self_repeat_not_only_cross_arm(tmp_path):
    baseline, candidate = tmp_path / 'base', tmp_path / 'memo'
    fake_completed(baseline, 'baseline'); fake_completed(candidate, 'memo')
    assert tool().compare_completed(baseline=baseline, candidate=candidate, output=tmp_path / 'pair.json', **fake_binding(tmp_path))['status'] == 'passed_fixed_candidate_pair'
    other = tmp_path / 'wrong'; fake_completed(other, 'memo', changed=('warm', 'state'))
    with pytest.raises(RuntimeError, match='mismatch'):
        tool().compare_completed(baseline=baseline, candidate=other, output=tmp_path / 'failed.json', **fake_binding(tmp_path))
    assert json.loads((tmp_path / 'failed.json').read_text())['status'] == 'failed_fixed_candidate_pair'


def test_failed_or_missing_receipt_and_invalid_frozen_source_do_not_pass(tmp_path):
    api = tool(); base, memo = tmp_path / 'base', tmp_path / 'memo'
    fake_completed(base, 'baseline'); fake_completed(memo, 'memo')
    (memo / 'COMPLETE.json').write_text('{"status":"failed","report_sha256":"wrong"}')
    with pytest.raises(ValueError): api.compare_completed(baseline=base, candidate=memo, output=tmp_path / 'pair.json', **fake_binding(tmp_path))
    assert not (tmp_path / 'pair.json').exists()
    manifest = tmp_path / 'source.json'; manifest.write_text('{"files":{}}')
    with pytest.raises(ValueError, match='manifest SHA'):
        api.validate_source(dict(source_root=tmp_path, source_manifest=manifest, source_manifest_sha256='wrong'))


@pytest.mark.parametrize('kind,missing', [('candidate','distance_mm'), ('state','canonical_vertices')])
def test_same_missing_npz_field_in_both_arms_is_rejected(tmp_path, kind, missing):
    api = tool(); base, memo = tmp_path / 'base', tmp_path / 'memo'
    fake_completed(base, 'baseline'); fake_completed(memo, 'memo')
    for root in (base, memo):
        path = root / f'cold.{kind}.private.npz'
        with np.load(path, allow_pickle=False) as z: fields = {k: z[k] for k in z.files if k != missing}
        np.savez_compressed(path, **fields)
        refresh_report(root, lambda r: r['artifact_sha256'].update({path.name: api.sha(path)}))
    with pytest.raises(ValueError, match='field schema'):
        api.compare_completed(baseline=base, candidate=memo, output=tmp_path / 'pair.json', **fake_binding(tmp_path))
    assert not (tmp_path / 'pair.json').exists()


def test_empty_artifact_dict_and_jointly_missing_control_cannot_pass(tmp_path):
    api = tool(); base, memo = tmp_path / 'base', tmp_path / 'memo'
    fake_completed(base, 'baseline'); fake_completed(memo, 'memo')
    for root in (base, memo): refresh_report(root, lambda r: r.update(artifact_sha256={}))
    with pytest.raises(ValueError, match='nine'):
        api.compare_completed(baseline=base, candidate=memo, output=tmp_path / 'pair.json', **fake_binding(tmp_path))
    for root in (base, memo):
        artifacts = {name: api.sha(root / name) for name in api.ARTIFACT_NAMES}
        control = root / 'cold.control.private.json'; value = json.loads(control.read_text()); value.pop('face_check')
        api.write(control, value); artifacts[control.name] = api.sha(control)
        refresh_report(root, lambda r: r.update(artifact_sha256=artifacts))
    with pytest.raises(ValueError, match='six-field control'):
        api.compare_completed(baseline=base, candidate=memo, output=tmp_path / 'pair.json', **fake_binding(tmp_path))


@pytest.mark.parametrize('mismatch', ['tool', 'module_hash', 'module_origin', 'empty_modules', 'helper_path', 'module_missing'])
def test_source_binding_mismatches_are_rejected_even_when_both_reports_agree(tmp_path, mismatch):
    api = tool(); base, memo = tmp_path / 'base', tmp_path / 'memo'
    fake_completed(base, 'baseline'); fake_completed(memo, 'memo')
    def change(r):
        name = 'fnit.recon_all.topology_edge_predicate_cache'
        if mismatch == 'tool': r['actual_tool_sha256'] = 'f' * 64
        elif mismatch == 'module_hash': r['actual_loaded_fnit_sha256'][name] = 'f' * 64
        elif mismatch == 'module_origin': r['actual_loaded_fnit_sources'][name]['origin_relative_path'] = 'different/source.py'
        elif mismatch == 'helper_path': r['actual_loaded_benchmark_tool_sources']['benchmark_recon_topology_candidate_state']['relative_path'] = 'different/helper.py'
        elif mismatch == 'module_missing':
            missing = 'fnit.recon_all.thread_budget'
            r['actual_loaded_fnit_sha256'].pop(missing); r['actual_loaded_fnit_sources'].pop(missing)
        else: r['actual_loaded_fnit_sha256'] = {}; r['actual_loaded_fnit_sources'] = {}
    for root in (base, memo): refresh_report(root, change)
    with pytest.raises(ValueError, match='binding'):
        api.compare_completed(baseline=base, candidate=memo, output=tmp_path / 'pair.json', **fake_binding(tmp_path))


def test_pair_failure_preserves_cold_cli_and_never_starts_second_arm(monkeypatch, tmp_path):
    from types import SimpleNamespace
    api = tool()
    configs = []
    for name in ('baseline', 'memo'):
        path = tmp_path / (name + '.json')
        api.write(path, dict(input_sha256={'same': 'a' * 64}, source_manifest_sha256='b' * 64,
            protocol_sha256='c' * 64,
            threads=2, cpu_affinity=[1, 2], maximum_entries=10, numba_cache_dir=str(tmp_path / (name + '_jit')),
            output=str(tmp_path / (name + '_out'))))
        configs.append(path)
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert kwargs['env']['NUMBA_CACHE_DIR'] == str(tmp_path / 'baseline_jit')
        assert all(kwargs['env'][name] == '2' for name in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'NUMBA_NUM_THREADS'))
        return SimpleNamespace(returncode=3)
    monkeypatch.setattr(api.subprocess, 'run', fail)
    compared = []
    monkeypatch.setattr(api, 'compare_completed', lambda **kwargs: compared.append(kwargs))
    with pytest.raises(RuntimeError, match='arm failed'):
        api.run_pair(baseline_config=configs[0], memo_config=configs[1], output=tmp_path / 'pair')
    assert len(calls) == 1 and not compared
    receipt = json.loads((tmp_path / 'pair/cli.private.json').read_text())
    assert receipt['records'][0]['exit_code'] == 3
    assert receipt['records'][0]['complete_cold_cli_wall_seconds'] > 0
