"""Isolated native scoring of the private CPU raw-prior initial/first/third states.

Reuse the validated native37 trajectory; do not execute another optimizer.
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
    parser.add_argument('--actual-native-trajectory', type=Path, required=True)
    parser.add_argument('--previous-decomposition', type=Path, required=True)
    parser.add_argument('--prior-raw-gates',type=Path,required=True)
    args = parser.parse_args()
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
    if not all(python_report['initial_gate'].values()):
        raise RuntimeError('raw CPU initial shared-state gate failed')
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

    def fresh(points, check_topology=False, stiffness=None):
        collection = gems.KvlMeshCollection()
        collection.read(str(args.mesh))
        if not np.array_equal(collection.reference_mesh.can_moves, data['can_move']):
            raise RuntimeError('native mobility masks differ')
        collection.transform(transform)
        collection.set_positions(np.asfortranarray(data['reference'].astype(np.float64)),
                                 [np.asfortranarray(points.astype(np.float64))])
        collection.k = float(data['stiffness']) if stiffness is None else float(stiffness)
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
    previous_decomposition=json.loads((args.previous_decomposition/'summary.public.json').read_text())
    if (previous_decomposition['capture_sha256'] != digest(args.capture/'shared_input.npz')
            or previous_decomposition['native_binding_sha256'] != digest(gems.__file__)
            or not all(previous_decomposition['exact_native_topology_reference_gate'].values())):
        raise RuntimeError('reuse prior topology/binary proof only for exact same capture')
    topology_gate.update(previous_decomposition['exact_native_topology_reference_gate'])
    prior_raw=json.loads((args.prior_raw_gates/'summary.public.json').read_text())
    if (not prior_raw['all_scientific_gates_passed']
            or prior_raw['capture_sha256']!=digest(args.capture/'shared_input.npz')
            or prior_raw['native_binding_sha256']!=digest(gems.__file__)
            or prior_raw['raw_adapter_sha256']!=python_report['raw_adapter_sha256']
            or prior_raw['optimizer_candidate_sha256']!=python_report['candidate_sha256']):
        raise RuntimeError('already passed raw0/1/3 scientific gates have different bindings')
    actual=json.loads((args.actual_native_trajectory/'summary.public.json').read_text())
    if (actual['native_binding_sha256']!=digest(gems.__file__)
            or actual['capture_sha256']!=digest(args.capture/'shared_input.npz')
            or actual['native_options_source_sha256']!=digest(core_path)
            or actual['program_sha256']!='9caac6a50f9f1e5552bbf332ced4724d30016c06f733db7b96f7549275f3de02'):
        raise RuntimeError('reused native37 source/input/binary differs')
    rows=python_report['modes']['native_definition_cpu']['rows']
    points_to_score=[len(rows)]
    result={}
    directory=args.python_run/'native_definition_cpu'
    for step in sorted(set(points_to_score)):
        path=directory/('initial.private.npz' if step==0 else 'accepted-%03d.private.npz'%step)
        with np.load(path,allow_pickle=False) as loaded:
            points=loaded['points'].copy();gradient=loaded['gradient'].copy()
            priors=loaded['priors'].copy();coverage=loaded['coverage'].copy()
        expected_sha=python_report['raw_initial']['private_initial_sha256'] if step==0 else rows[step-1]['private_state_sha256']
        if digest(path)!=expected_sha:raise RuntimeError('raw CPU accepted state changed')
        full_collection,full_mesh=fresh(points)
        total,g=calculator.evaluate_mesh_position(full_mesh)
        values=np.ascontiguousarray(np.concatenate([np.ones((len(points),1)),data['alphas'].astype(np.float64)],axis=1))
        drawn=np.asarray(full_mesh.rasterize_values(list(data['image'].shape),values))
        mask=np.isfinite(data['image']) & (data['image']!=0)
        loading=drawn[mask];native_priors=loading[:,1:].T;native_coverage=loading[:,0]!=0
        prior_error=helper.difference(priors,native_priors)
        full_error=helper.difference(g,gradient)
        cost=python_report['raw_initial']['cost'] if step==0 else rows[step-1]['accepted_cost']
        item={'native_cost':float(total),'raw_cpu_cost':cost,'native_minus_raw_cost':float(total-cost),
            'same_point_projected_full_gradient':full_error,'raw_priors_vs_native_float_drawer':prior_error,
            'coverage_different':int(np.count_nonzero(coverage!=native_coverage)),
            'private_CPU_state_sha256':digest(path),
            'gate':{'absolute_cost_pass':abs(total-cost)<=.01,
                    'full_gradient_relative_l2_pass':full_error['relative_l2']<=1e-5,
                    'raw_prior_max_abs_pass':prior_error['max_abs']<=1e-6,
                    'coverage_exact':bool(np.array_equal(coverage,native_coverage))}}
        if step>0:
            native_row=actual['native_rows'][step-1]
            native_path=args.actual_native_trajectory/'native_trajectory'/('accepted-%03d.private.npz'%step)
            if digest(native_path)!=native_row['private_state_sha256']:raise RuntimeError('reused actual native accepted state changed')
            with np.load(native_path,allow_pickle=False) as native:
                native_points=native['points'].copy()
            item['trajectory_at_distinct_accepted_points']={'points':helper.difference(points,native_points),
                'native_points_rounded_to_FP32_exact':bool(np.array_equal(points,native_points.astype(np.float32))),
                'native_minus_raw_cost':float(native_row['accepted_cost']-cost),
                'native_deformation':native_row['returned_maximal_deformation'],
                'raw_theoretical_deformation':rows[step-1]['theoretical_maximal_deformation'],
                'raw_actual_FP32_deformation':rows[step-1]['actual_maximal_deformation']}
            def jacobian_stats(current):
                tet=expected_tetra
                def edges(x):
                    vertices=x[tet].astype(np.float64)
                    return np.stack([vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0],vertices[:,3]-vertices[:,0]],axis=-1)
                refdet=np.linalg.det(edges(data['reference']))
                jac=np.linalg.det(edges(current))/refdet
                return {'minimum':float(jac.min()),'p01':float(np.percentile(jac,1)),
                        'median':float(np.median(jac)),'maximum':float(jac.max()),
                        'nonpositive':int(np.count_nonzero(jac<=0)),
                        'nonfinite':int(np.count_nonzero(~np.isfinite(jac))),
                        'tets_below_0p1':int(np.count_nonzero(jac<.1)),
                        'tets_above_10':int(np.count_nonzero(jac>10))}
            item['Jacobian_at_distinct_accepted_points']={'private_raw_CPU':jacobian_stats(points),
                    'actual_native':jacobian_stats(native_points)}
        result[str(step)]=item
    report={'status':'completed_isolated_native_CPU_raw_prior_continuation_gate',
        'scope':'Private frozen CPU closure only; no EM/recipe/GPU/default change and no native optimizer updates',
        'same_points':result,'tolerances':{'absolute_cost':.01,'full_gradient_relative_l2':1e-5,'raw_prior_max_abs':1e-6,'coverage':'exact'},
        'all_scientific_gates_passed':all(all(x['gate'].values()) for x in result.values()),
        'program_sha256':digest(__file__),'native_binding_sha256':digest(gems.__file__),'capture_sha256':digest(args.capture/'shared_input.npz'),
        'python_summary_sha256':digest(args.python_run/'summary.public.json'),'reused_actual_native37_summary_sha256':digest(args.actual_native_trajectory/'summary.public.json'),
        'raw_adapter_sha256':python_report['raw_adapter_sha256'],'optimizer_candidate_sha256':python_report['candidate_sha256'],
        'reused_raw_initial_first_third_gate_summary_sha256':digest(args.prior_raw_gates/'summary.public.json'),
        'reused_raw_first_three_gates':{k:v['gate'] for k,v in prior_raw['same_points'].items()},
        'frozen_source_sha256':python_report['source_sha256'],'native_recipe_source_sha256':digest(core_path),
        'reused_exact_topology_reference_gate':topology_gate,'native_optimizer_calls':0,
        'affinity':sorted(os.sched_getaffinity(0)),'native_threads_requested_by_API':8,
        'output_directory_mode':oct(stat.S_IMODE(args.output.stat().st_mode))}
    write(args.output/'summary.public.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
