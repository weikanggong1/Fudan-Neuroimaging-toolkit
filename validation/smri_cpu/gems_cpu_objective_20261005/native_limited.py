"""Bounded installed-native right HA optimizer on the exact captured frame.

This isolated benchmark retains the validated v5 recipe transform, topology,
image and Gaussian definition. It performs at most 40 synthetic updates, no EM
or complete recipe, then scores three saved Python accepted points separately.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from time import perf_counter

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    spec = importlib.util.spec_from_file_location('validated_native_v5', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--prior-native', type=Path, required=True)
    parser.add_argument('--python-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=37)
    args = parser.parse_args()
    if not 1 <= args.steps <= 40:
        raise ValueError('at most 40 synthetic updates')
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    if stat.S_IMODE(args.output.stat().st_mode) != 0o700:
        raise RuntimeError('real private arrays require actual0700 directory')
    if digest(args.helper) != 'bd0dcbcd995e330849cd4cdd311481bb7db0038a5ba348c2267af7b6a70e437d':
        # Exact tested helper, no unreviewed native setup implementation.
        # The hard binding is checked before any native computation.
        raise RuntimeError('validated native v5 helper changed')
    helper = load(args.helper)
    sys.path.insert(0, '/public/software/apps/Freesurfer/8.2.0-1/python/lib/python3.8/site-packages/samseg/gems')
    import gemsbindings as gems
    gems.setGlobalDefaultNumberOfThreads(8)
    if digest(gems.__file__) != '8125a39cd9885ada66ac2945dc5d7e38737e3bb707bce26ad6d796340fadbf82':
        raise RuntimeError('installed native binary changed')
    core_path = Path('/public/software/apps/Freesurfer/8.2.0-1/python/lib/python3.8/site-packages/samseg/subregions/core.py')
    if digest(core_path) != 'beb64fa1f2a9d6b6946388fc6641b92ffc5a100bef47b96b55877d4df9d0d700':
        raise RuntimeError('installed synthetic recipe/options changed')
    capture = json.loads((args.capture/'report.public.json').read_text())
    if capture['structure'] != 'hippo-amygdala-right':
        raise RuntimeError('bounded experiment is right HA only')
    for name, expected in capture['outputs'].items():
        if digest(args.capture/name) != expected:
            raise RuntimeError('capture changed: '+name)
    if digest(args.mesh) != capture['inputs']['mesh']:
        raise RuntimeError('captured atlas changed')
    with np.load(args.capture/'shared_input.npz', allow_pickle=False) as loaded:
        data = {name: loaded[name].copy() for name in loaded.files}
    python_report = json.loads((args.python_run/'summary.public.json').read_text())
    if python_report['capture_sha256'] != digest(args.capture/'shared_input.npz'):
        raise RuntimeError('Python trajectory uses a different capture')
    if not all(all(mode['gate'].values()) for mode in python_report['modes'].values()):
        raise RuntimeError('strict Python initial state gate failed')
    image = gems.KvlImage(np.asfortranarray(data['image']))
    if not np.array_equal(image.getImageBuffer(), data['image']):
        raise RuntimeError('native image roundtrip changed')
    matrix = np.eye(4)
    matrix[:3, :3] = data['boundary_transform'].astype(np.float32).astype(np.float64)
    transform = gems.KvlTransform(np.asfortranarray(matrix))
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not np.isfinite(determinant) or determinant == 0:
        raise RuntimeError('captured transform invalid')
    raw_ids, raw_cells, raw_tetra, _, compressed_cells = helper.read_mesh_topology(args.mesh)
    if not np.array_equal(raw_tetra, data['tetrahedra']):
        raise RuntimeError('captured original tet rows differ')
    expected_tetra = raw_tetra.copy()
    if determinant < 0:
        expected_tetra[:, [0, 1]] = expected_tetra[:, [1, 0]]
    orientation, _ = helper.orientation(data['reference'].astype(np.float64), expected_tetra)
    if orientation['negative_tetrahedra'] or orientation['zero_tetrahedra']:
        raise RuntimeError('captured reference is not recipe-normalized')
    calculator = gems.KvlCostAndGradientCalculator(typeName='AtlasMeshToIntensityImage',
        images=[image], boundaryCondition='Sliding', transform=transform,
        means=np.asfortranarray(data['means']), variances=np.asfortranarray(data['variances']),
        mixtureWeights=np.ones(len(data['means']), dtype=np.float32),
        numberOfGaussiansPerClass=np.ones(len(data['means']), dtype=np.int32))
    topology_gate = {}

    def fresh(points, check_topology=False):
        collection = gems.KvlMeshCollection()
        collection.read(str(args.mesh))
        if not np.array_equal(collection.reference_mesh.can_moves, data['can_move']):
            raise RuntimeError('native mobility masks differ')
        collection.transform(transform)
        collection.set_positions(np.asfortranarray(data['reference'].astype(np.float64)),
                                 [np.asfortranarray(points.astype(np.float64))])
        collection.k = float(data['stiffness'])
        mesh = collection.get_mesh(0)
        mesh.alphas = np.asfortranarray(data['alphas'].astype(np.float32))
        if not (np.array_equal(mesh.points, points)
                and np.array_equal(collection.reference_position, data['reference'])
                and np.array_equal(mesh.alphas, data['alphas'])):
            raise RuntimeError('native in-memory exact shared frame changed')
        if check_topology:
            prefix = args.output/'normalized_atlas.private'
            collection.write(str(prefix))
            written_ids, written_cells, written_tet, _, _ = helper.read_mesh_topology(str(prefix)+'.gz')
            topology_gate.update(tet_rows_exact=bool(np.array_equal(written_tet, expected_tetra)),
                compressed_point_ids_exact=bool(np.array_equal(written_ids, np.arange(len(raw_ids)))),
                compressed_cell_ids_exact=bool(np.array_equal(written_cells, compressed_cells)),
                in_memory_reference_exact=True, in_memory_current_exact=True,
                alphas_exact=True, node_flags_exact=True)
            if not all(topology_gate.values()):
                raise RuntimeError('v5 native reference/topology gate failed')
        return collection, mesh

    options = {'Verbose': False, 'MaximalDeformationStopCriterion': 1e-10,
        'LineSearchMaximalDeformationIntervalStopCriterion': 1e-10,
        'MaximumNumberOfIterations': 1000, 'BFGS-MaximumMemoryLength': 12}
    collection, mesh = fresh(data['vertices'], True)
    initial_cost, initial_gradient = calculator.evaluate_mesh_position(mesh)
    old = json.loads((args.prior_native/'summary.public.json').read_text())
    if (old['program_sha256'] != digest(args.helper)
            or old['shared_input_sha256'] != digest(args.capture/'shared_input.npz')
            or abs(float(initial_cost)-old['native_step']['initial_cost']) > 1e-6):
        raise RuntimeError('earlier validated actual native initial gate changed')
    initial_comparison = helper.difference(initial_gradient, np.load(args.capture/'fnit_gradient.npy'))
    if initial_comparison['relative_l2'] > 1e-5:
        raise RuntimeError('native initial gradient gate failed')
    optimizer = gems.KvlOptimizer('L-BFGS', mesh, calculator, options)
    rows = []
    reason = 'diagnostic_step_limit'
    begin = perf_counter()
    native_directory = args.output/'native_trajectory'
    native_directory.mkdir(mode=0o700)
    for step in range(args.steps):
        before = np.asarray(mesh.points).copy()
        started = perf_counter()
        cost, deformation = optimizer.step_optimizer_samseg()
        points = np.asarray(mesh.points).copy()
        accepted_cost, accepted_gradient = calculator.evaluate_mesh_position(mesh)
        if not (np.isfinite(cost) and np.isfinite(points).all()
                and np.isfinite(accepted_gradient).all() and np.isfinite(accepted_cost)):
            raise RuntimeError('nonfinite actual native accepted state')
        row = {'step': step+1, 'returned_cost': float(cost), 'accepted_cost': float(accepted_cost),
            'returned_maximal_deformation': float(deformation),
            'actual_maximal_deformation': float(np.max(np.linalg.norm(points-before, axis=1))),
            'native_update_and_extra_gradient_observation_seconds': perf_counter()-started,
            'native_internal_trial_sequence_available': False}
        path = native_directory/('accepted-%03d.private.npz' % (step+1))
        np.savez_compressed(path, points=points, gradient=accepted_gradient)
        row['private_state_sha256'] = digest(path)
        if step == 0:
            with np.load(args.prior_native/'native_step.private.npz', allow_pickle=False) as old_points:
                row['first_points_exact_to_validated_v5_native_step'] = bool(np.array_equal(points, old_points['accepted_points']))
            if not row['first_points_exact_to_validated_v5_native_step']:
                raise RuntimeError('actual native first step changed')
        rows.append(row)
        if deformation == 0:
            reason = 'actual_native_returned_zero_deformation'
        write(args.output/'progress.public.json', {'rows': rows, 'stop_reason': reason})
        if deformation == 0:
            break
    native_seconds = perf_counter()-begin
    same_point = {}
    trajectory = {}
    for mode, info in python_report['modes'].items():
        last = info['steps']
        selected = sorted(set([1, min(3, last), last]))
        mode_same_point = {}
        mode_trajectory = {}
        for step in selected:
            path = args.python_run/mode/('accepted-%03d.private.npz' % step)
            with np.load(path, allow_pickle=False) as loaded:
                points, gradient = loaded['points'].copy(), loaded['gradient'].copy()
            same_collection, same_mesh = fresh(points)
            same_cost, same_gradient = calculator.evaluate_mesh_position(same_mesh)
            python_cost = info['rows'][step-1]['cost' if mode=='existing_armijo' else 'accepted_cost']
            error = helper.difference(same_gradient, gradient)
            delta = float(same_cost-python_cost)
            mode_same_point[str(step)] = {'native_cost': float(same_cost), 'python_cost': python_cost,
                'cost_minus_python': delta, 'gradient': error, 'private_python_state_sha256': digest(path),
                'gate': {'absolute_cost_pass': abs(delta)<=.01,
                         'gradient_relative_l2_pass': error.get('relative_l2', float('inf'))<=1e-5}}
            if step <= len(rows):
                with np.load(native_directory/('accepted-%03d.private.npz' % step), allow_pickle=False) as native:
                    actual_native = native['points'].copy()
                mode_trajectory[str(step)] = {'points_at_distinct_accepted_states': helper.difference(points, actual_native),
                    'native_points_rounded_to_float32_exact': bool(np.array_equal(points, actual_native.astype(np.float32))),
                    'native_minus_python_cost_at_distinct_accepted_states': float(rows[step-1]['accepted_cost']-python_cost),
                    'this_is_not_a_same_point_gradient_test': True}
        same_point[mode], trajectory[mode] = mode_same_point, mode_trajectory
    report = {'status': 'completed_bounded_native_fixed_likelihood_trajectory',
        'scope': 'Right HA synthetic stage1 only, at most40 updates, no EM or full recipe or default adoption',
        'native_options': options, 'native_rows': rows, 'native_stop_reason': reason,
        'native_trajectory_observation_seconds_including_extra_gradients_and_save': native_seconds,
        'same_point_tolerances': {'absolute_total_cost': .01, 'gradient_relative_l2': 1e-5},
        'same_point': same_point, 'trajectory_at_distinct_accepted_states': trajectory,
        'exact_native_topology_reference_gate': topology_gate, 'transform_determinant': determinant,
        'initial_gradient_comparison': initial_comparison,
        'native_binding_sha256': digest(gems.__file__), 'native_options_source_sha256': digest(core_path),
        'validated_native_helper_sha256': digest(args.helper), 'program_sha256': digest(__file__),
        'capture_sha256': digest(args.capture/'shared_input.npz'), 'mesh_sha256': digest(args.mesh),
        'python_summary_sha256': digest(args.python_run/'summary.public.json'),
        'actual_native_threads_requested_by_API': 8, 'affinity': sorted(os.sched_getaffinity(0)),
        'output_directory_mode': oct(stat.S_IMODE(args.output.stat().st_mode))}
    write(args.output/'summary.public.json', report)
    print(json.dumps({'status': report['status'], 'native_steps': len(rows),
        'native_stop_reason': reason, 'same_point_gates': {mode: {step: item['gate']
        for step, item in points.items()} for mode, points in same_point.items()}}))


if __name__ == '__main__':
    main()
