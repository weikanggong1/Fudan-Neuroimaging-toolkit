"""Isolated installed GEMS first-step oracle on saved real points only."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from time import perf_counter

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_mesh_topology(path):
    """Read IDs/order directly; never normalize alpha values or coordinates."""
    opener=gzip.open if Path(path).suffix=='.gz' else open
    with opener(path,'rt') as stream:
        points=int(stream.readline().split(':')[1]);cells=int(stream.readline().split(':')[1])
        stream.readline();meshes=int(stream.readline().split(':')[1]);stream.readline()
        rows=[stream.readline().split() for _ in range(points)]
        ids=np.array([int(r[0]) for r in rows],dtype=np.int64)
        reference=np.array([[float(v) for v in r[1:4]] for r in rows])
        mapping={int(v):i for i,v in enumerate(ids)}
        stream.readline()
        for _ in range(meshes):
            stream.readline()
            for _ in range(points):stream.readline()
        if not stream.readline().strip().lower().startswith('cells'):raise RuntimeError('mesh cells section missing')
        cell_ids=[];tetra=[];compressed_cell_ids=[]
        for cell_number in range(cells):
            row=stream.readline().split()
            if row[1].upper() not in {'VERTEX','LINE','TRIANGLE'}:
                cell_ids.append(int(row[0]));compressed_cell_ids.append(cell_number);tetra.append([mapping[int(v)] for v in row[2:6]])
    return ids,np.array(cell_ids,dtype=np.int64),np.array(tetra,dtype=np.int64),reference,np.array(compressed_cell_ids,dtype=np.int64)


def orientation(reference,tetra):
    r=reference[tetra];e=np.stack([r[:,1]-r[:,0],r[:,2]-r[:,0],r[:,3]-r[:,0]],axis=-1)
    d=np.linalg.det(e)
    return {'positive_tetrahedra':int((d>0).sum()),'negative_tetrahedra':int((d<0).sum()),
        'zero_tetrahedra':int((d==0).sum()),'minimum_signed_volume':float(d.min()/6),
        'maximum_signed_volume':float(d.max()/6),'sum_signed_volume':float(d.sum()/6)},e


def difference(left, right):
    left, right = np.asarray(left), np.asarray(right)
    delta = left.astype(np.float64) - right.astype(np.float64)
    finite=bool(np.isfinite(left).all() and np.isfinite(right).all())
    if not finite:
        return {'shape':list(left.shape),'dtype':str(left.dtype),'reference_dtype':str(right.dtype),
            'all_finite':False,'native_nonfinite_values':int(np.count_nonzero(~np.isfinite(left))),
            'python_nonfinite_values':int(np.count_nonzero(~np.isfinite(right))),
            'numerical_errors':None}
    norm = np.linalg.norm(right.astype(np.float64))
    return {'shape':list(left.shape),'dtype':str(left.dtype),'reference_dtype':str(right.dtype),
        'different':int(np.count_nonzero(delta)),'max_abs':float(np.max(np.abs(delta))),
        'rmse':float(np.sqrt(np.mean(delta*delta))),
        'relative_l2':float(np.linalg.norm(delta)/norm) if norm else None,
        'p99_abs':float(np.quantile(np.abs(delta),.99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--mesh',type=Path,required=True)
    parser.add_argument('--python-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    if stat.S_IMODE(args.output.stat().st_mode)!=0o700:
        raise RuntimeError('real arrays require actual0700 output directory')
    sys.path.insert(0,'/public/software/apps/Freesurfer/8.2.0-1/python/lib/python3.8/site-packages/samseg/gems')
    import gemsbindings as gems
    gems.setGlobalDefaultNumberOfThreads(8)
    if digest(gems.__file__)!='8125a39cd9885ada66ac2945dc5d7e38737e3bb707bce26ad6d796340fadbf82':
        raise RuntimeError('installed native oracle binary changed')
    native_core=Path('/public/software/apps/Freesurfer/8.2.0-1/python/lib/python3.8/site-packages/samseg/subregions/core.py')
    if digest(native_core)!='beb64fa1f2a9d6b6946388fc6641b92ffc5a100bef47b96b55877d4df9d0d700':
        raise RuntimeError('installed native recipe options changed')
    original=json.loads((args.capture/'report.public.json').read_text())
    for name, expected in original['outputs'].items():
        if digest(args.capture/name)!=expected:
            raise RuntimeError('capture array changed: '+name)
    if digest(args.mesh)!=original['inputs']['mesh']:
        raise RuntimeError('native atlas mesh identity changed')
    with np.load(args.capture/'shared_input.npz',allow_pickle=False) as loaded:
        data={name:loaded[name].copy() for name in loaded.files}
    image=gems.KvlImage(np.asfortranarray(data['image']))
    if not np.array_equal(image.getImageBuffer(),data['image']):
        raise RuntimeError('native image roundtrip differs')
    matrix=np.eye(4)
    matrix[:3,:3]=data['boundary_transform'].astype(np.float32).astype(np.float64)
    transform=gems.KvlTransform(np.asfortranarray(matrix))
    transform_determinant=float(np.linalg.det(matrix[:3,:3]))
    if not np.isfinite(transform_determinant) or transform_determinant==0:
        raise RuntimeError('captured native recipe transform parity invalid')
    raw_ids,raw_cells,raw_tetra,raw_reference,compressed_cell_ids=read_mesh_topology(args.mesh)
    if not np.array_equal(raw_tetra,data['tetrahedra']):raise RuntimeError('captured/raw atlas cell ID order differs')
    expected_tetra=raw_tetra.copy()
    if transform_determinant<0:expected_tetra[:,[0,1]]=expected_tetra[:,[1,0]]
    raw_orientation,raw_edges=orientation(data['reference'].astype(np.float64),raw_tetra)
    expected_orientation,expected_edges=orientation(data['reference'].astype(np.float64),expected_tetra)
    topology_gate={'captured_tetrahedra_exact_to_raw_atlas':True,
        'transform_determinant':transform_determinant,'swap_p0_p1':transform_determinant<0,
        'raw_reference_orientation':raw_orientation,'recipe_reference_orientation':expected_orientation,
        'native_points_and_reference_are_still_captured_FP32_promoted':True,
        'source_binding_proof_sha256':'1c51420676b47d0ee40c13be867bf6dfcbb888fed25fa13bca5de1104f544544'}
    if expected_orientation['negative_tetrahedra'] or expected_orientation['zero_tetrahedra']:
        raise RuntimeError('native recipe cell normalization did not produce positive reference orientation')
    calculator=gems.KvlCostAndGradientCalculator(typeName='AtlasMeshToIntensityImage',
        images=[image],boundaryCondition='Sliding',transform=transform,
        means=np.asfortranarray(data['means']),variances=np.asfortranarray(data['variances']),
        mixtureWeights=np.ones(len(data['means']),dtype=np.float32),
        numberOfGaussiansPerClass=np.ones(len(data['means']),dtype=np.int32))
    def fresh_mesh(points,stiffness):
        collection=gems.KvlMeshCollection()
        collection.read(str(args.mesh))
        raw=collection.reference_mesh
        if not np.array_equal(raw.can_moves,data['can_move']):
            raise RuntimeError('native node mobility/order changed')
        # Installed recipe calls this wrapper. It swaps p0/p1 for a negative
        # affine determinant; plain set_positions from v1/v2 did not do so.
        # The following setters restore the exact shared coordinates, leaving
        # the actual native transform's cell permutation intact.
        collection.transform(transform)
        collection.set_positions(np.asfortranarray(data['reference'].astype(np.float64)),
                                [np.asfortranarray(points.astype(np.float64))])
        collection.k=float(stiffness)
        mesh=collection.get_mesh(0)
        mesh.alphas=np.asfortranarray(data['alphas'].astype(np.float32))
        if not np.array_equal(mesh.points,points) or not np.array_equal(mesh.alphas,data['alphas']):
            raise RuntimeError('native mesh setter changed shared points/alphas')
        if not topology_gate.get('actual_native_written_cells_exact_to_recipe_permutation'):
            output_prefix=args.output/'parity_normalized_atlas.private'
            collection.write(str(output_prefix))
            output_mesh=Path(str(output_prefix)+'.gz')
            saved_ids,saved_cells,saved_tetra,saved_reference,_=read_mesh_topology(output_mesh)
            topology_gate.update(actual_native_written_cells_exact_to_recipe_permutation=bool(np.array_equal(saved_tetra,expected_tetra)),
                actual_native_written_cell_ids_exact_to_read_compaction=bool(np.array_equal(saved_cells,compressed_cell_ids)),
                actual_native_written_point_ids_exact_to_read_compaction=bool(np.array_equal(saved_ids,np.arange(len(raw_ids)))),
                raw_cell_ids_equal_written=bool(np.array_equal(saved_cells,raw_cells)),
                raw_point_ids_equal_written=bool(np.array_equal(saved_ids,raw_ids)),
                raw_cell_ids_changed_by_static_read=int(np.count_nonzero(raw_cells!=saved_cells)),
                raw_point_ids_changed_by_static_read=int(np.count_nonzero(raw_ids!=saved_ids)),
                text_reference_coordinate_max_rounding_error=float(np.max(np.abs(saved_reference-data['reference']))),
                native_in_memory_reference_exact=bool(np.array_equal(collection.reference_position,data['reference'])),
                private_written_mesh_sha256=digest(output_mesh))
            (args.output/'topology_gate.public.json').write_text(json.dumps(topology_gate,indent=2)+'\n')
            if not all(topology_gate[k] for k in ['actual_native_written_cells_exact_to_recipe_permutation',
                    'actual_native_written_cell_ids_exact_to_read_compaction','actual_native_written_point_ids_exact_to_read_compaction','native_in_memory_reference_exact']):
                raise RuntimeError('native recipe topology/reference ID gate failed')
        return collection,mesh
    evaluated={}
    points_to_check={}
    for method in ['armijo','reference']:
        report=json.loads((args.python_run/method/'trace.public.json').read_text())
        candidates=[report['records'][0],report['records'][1]]
        selected=np.load(args.python_run/method/'result.private.npz')['accepted_points']
        selected_sha=hashlib.sha256(selected.tobytes()).hexdigest()
        candidates.extend([r for r in report['records'] if r['point_sha256']==selected_sha][-1:])
        for kind,record in zip(['initial','first_trial','selected'],candidates):
            file=args.python_run/method/record['private_arrays']
            arrays=np.load(file)
            points_to_check[method+'_'+kind]={'points':arrays['points'].copy(),
                'gradient':arrays['gradient'].copy(),'cost':record['cost'],'file_sha256':digest(file)}
    # Evaluate every selected point independently on a new native mesh. This
    # intentionally does not claim access to the native line-search internals.
    for name,record in points_to_check.items():
        started=perf_counter()
        total_collection,mesh=fresh_mesh(record['points'],float(data['stiffness']))
        total,gradient=calculator.evaluate_mesh_position(mesh)
        data_collection,data_mesh=fresh_mesh(record['points'],0.)
        cost_data,data_gradient=calculator.evaluate_mesh_position(data_mesh)
        np.savez_compressed(args.output/(name+'.private.npz'),points=record['points'],
            native_total_gradient=gradient,native_data_gradient=data_gradient)
        evaluated[name]={'total_cost':float(total),'data_cost':float(cost_data),
            'prior_cost':float(total-cost_data),'python_cost_finite':record['cost'] is not None,
            'cost_minus_python':float(total-record['cost']) if record['cost'] is not None else None,
            'complete_projected_gradient_vs_python':difference(gradient,record['gradient']),
            'python_array_file_sha256':record['file_sha256'],
            'native_observation_seconds':perf_counter()-started}
        # Swapping the reference and current corners together preserves J.
        raw_current,raw_current_edges=orientation(record['points'].astype(np.float64),raw_tetra)
        normalized_current,normalized_current_edges=orientation(record['points'].astype(np.float64),expected_tetra)
        raw_jacobian=np.linalg.det(raw_current_edges@np.linalg.inv(raw_edges))
        normalized_jacobian=np.linalg.det(normalized_current_edges@np.linalg.inv(expected_edges))
        evaluated[name]['relative_jacobian_corner_permutation_max_abs']=float(np.max(np.abs(raw_jacobian-normalized_jacobian)))
    # Exact options were read from this installed synthetic-stage recipe; this
    # executes only ONE call on an independent original-state mesh.
    native_options={'Verbose':False,'MaximalDeformationStopCriterion':1e-10,
        'LineSearchMaximalDeformationIntervalStopCriterion':1e-10,
        'MaximumNumberOfIterations':1000,'BFGS-MaximumMemoryLength':12}
    collection,mesh=fresh_mesh(data['vertices'],float(data['stiffness']))
    initial_cost,initial_gradient=calculator.evaluate_mesh_position(mesh)
    optimizer=gems.KvlOptimizer('L-BFGS',mesh,calculator,native_options)
    started=perf_counter()
    returned_cost,max_def=optimizer.step_optimizer_samseg()
    actual_points=np.asarray(mesh.points).copy()
    accepted_cost,accepted_gradient=calculator.evaluate_mesh_position(mesh)
    np.savez_compressed(args.output/'native_step.private.npz',initial_points=data['vertices'],
        initial_gradient=initial_gradient,accepted_points=actual_points,accepted_gradient=accepted_gradient)
    step={'optimizer':'L-BFGS','options':native_options,'calls':1,
        'initial_cost':float(initial_cost),'returned_cost':float(returned_cost),
        'returned_maximal_deformation':float(max_def),'accepted_cost':float(accepted_cost),
        'point_dtype':str(actual_points.dtype),'gradient_dtype':str(accepted_gradient.dtype),
        'max_actual_node_displacement':float(np.max(np.linalg.norm(actual_points-data['vertices'],axis=1))),
        'native_one_step_observation_seconds':perf_counter()-started,
        'native_internal_trial_sequence_available':False,
        'python_comparisons':{}}
    for method in ['armijo','reference']:
        arrays=np.load(args.python_run/method/'result.private.npz')
        step['python_comparisons'][method]={'points':difference(arrays['accepted_points'],actual_points),
            'gradient_at_distinct_accepted_points':difference(arrays['accepted_gradient'],accepted_gradient),
            'native_points_rounded_to_float32_exact_to_python':bool(np.array_equal(actual_points.astype(np.float32),arrays['accepted_points']))}
    report={'status':'completed_isolated_native_same_points_and_one_step',
        'scope':'Exact saved real synthetic stage1; actual native transform cell parity then exact captured shared reference/points; not full recipe benchmark or production adoption',
        'structure':original['structure'],'same_points':evaluated,'native_step':step,
        'native_recipe_reference_parity_gate':topology_gate,
        'native_binding_sha256':digest(gems.__file__),'native_recipe_options_source_sha256':digest(native_core),
        'shared_input_sha256':digest(args.capture/'shared_input.npz'),'mesh_sha256':digest(args.mesh),
        'program_sha256':digest(__file__),'affinity':sorted(os.sched_getaffinity(0)),
        'native_itk_default_threads_requested_through_actual_api':8,
        'output_directory_mode':oct(stat.S_IMODE(args.output.stat().st_mode))}
    # These tolerance constants are fixed before this revision is run. They
    # assess identical FP32 point values, not distinct native FP64 steps.
    report['same_point_tolerances']={'complete_gradient_relative_l2':1e-5,'absolute_total_cost':.01}
    report['same_point_finite_gate']={name:{'gradient_pass':record['complete_projected_gradient_vs_python'].get('relative_l2',float('inf'))<=1e-5,
        'cost_pass':abs(record['cost_minus_python'])<=.01} for name,record in evaluated.items()
        if record['python_cost_finite']}
    (args.output/'summary.public.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'same_points':evaluated,'native_step':step}))


if __name__=='__main__':
    main()
