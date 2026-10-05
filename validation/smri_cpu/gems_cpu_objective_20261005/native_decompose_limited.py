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
    parser.add_argument('--baseline-python', type=Path, required=True)
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
    unprojected_calculator = gems.KvlCostAndGradientCalculator(typeName='AtlasMeshToIntensityImage',
        images=[image], boundaryCondition='None', transform=transform,
        means=np.asfortranarray(data['means']), variances=np.asfortranarray(data['variances']),
        mixtureWeights=np.ones(len(data['means']), dtype=np.float32),
        numberOfGaussiansPerClass=np.ones(len(data['means']), dtype=np.int32))
    baseline = json.loads((args.baseline_python/'summary.public.json').read_text())
    if baseline['capture_sha256'] != digest(args.capture/'shared_input.npz'):
        raise RuntimeError('baseline points use a different capture')
    states = [('existing_armijo_3',args.baseline_python,'existing_armijo',baseline),
              ('native_definition_cpu_v2_3',args.python_run,'native_definition_cpu',python_report)]
    result = {}
    for name,run,mode,report in states:
        info=report['modes'][mode]
        if info['steps']<3:raise RuntimeError('three accepted updates required')
        path=run/mode/'accepted-003.private.npz'
        with np.load(path,allow_pickle=False) as loaded:points=loaded['points'].copy()
        collection,mesh=fresh(points)
        total,total_gradient=calculator.evaluate_mesh_position(mesh)
        total_none,total_gradient_none=unprojected_calculator.evaluate_mesh_position(mesh)
        data_collection,data_mesh=fresh(points)
        data_collection.k=0.
        cost_data,gradient_data=calculator.evaluate_mesh_position(data_mesh)
        cost_data_none,gradient_data_none=unprojected_calculator.evaluate_mesh_position(data_mesh)
        if abs(total-total_none)>1e-8 or abs(cost_data-cost_data_none)>1e-8:
            raise RuntimeError('boundary projection changed scalar cost')
        values=np.ascontiguousarray(np.concatenate([np.ones((len(points),1)),data['alphas'].astype(np.float64)],axis=1))
        drawn=np.asarray(mesh.rasterize_values(list(data['image'].shape),values))
        mask=np.isfinite(data['image']) & (data['image']!=0)
        selected=drawn[mask]
        target=args.output/(name+'.private.npz')
        np.savez_compressed(target,points=points,native_total_gradient=total_gradient,
            native_total_gradient_unprojected=total_gradient_none,
            native_data_gradient=gradient_data,native_data_gradient_unprojected=gradient_data_none,
            native_prior_gradient=total_gradient-gradient_data,
            native_prior_gradient_unprojected=total_gradient_none-gradient_data_none,
            native_priors=selected[:,1:].T,native_coverage=selected[:,0]!=0,
            native_coverage_loading=selected[:,0])
        result[name]={'native_total_cost':float(total),'native_data_cost':float(cost_data),
            'native_prior_cost':float(total-cost_data),'points_sha256':hashlib.sha256(points.tobytes()).hexdigest(),
            'python_state_file_sha256':digest(path),'native_private_state_file_sha256':digest(target),
            'native_raster_priors_shape':list(selected[:,1:].T.shape),
            'native_covered_voxels':int(np.count_nonzero(selected[:,0]!=0)),
            'native_raster_value_dtype':str(drawn.dtype),'native_boundary_costs_exact_to_1e-8':True,
            'native_pixel_values_unquantized':True,'native_scope_same_FP32_points_promoted_to_FP64':True}
    report={'status':'completed_third_point_native_data_prior_and_unprojected_decomposition',
        'same_points':result,'exact_native_topology_reference_gate':topology_gate,
        'native_binding_sha256':digest(gems.__file__),'native_recipe_source_sha256':digest(core_path),
        'validated_native_helper_sha256':digest(args.helper),'capture_sha256':digest(args.capture/'shared_input.npz'),
        'program_sha256':digest(__file__),'python_summary_sha256':digest(args.python_run/'summary.public.json'),
        'baseline_python_summary_sha256':digest(args.baseline_python/'summary.public.json'),
        'native_threads_requested_through_API':8,'affinity':sorted(os.sched_getaffinity(0)),
        'no_native_optimizer_called':True,'no_full_recipe':True,
        'output_directory_mode':oct(stat.S_IMODE(args.output.stat().st_mode))}
    write(args.output/'summary.public.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
