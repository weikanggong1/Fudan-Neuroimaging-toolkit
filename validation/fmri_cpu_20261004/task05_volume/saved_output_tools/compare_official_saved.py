"""Compare saved full official preproc to final FNIT preproc; no new MRI API."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from collect_cpu16 import image_check, load_binding, read, save, sha
from volume_numeric import affine_world, compare_arrays, physical_pair


def proved_file(info, observed):
    path=Path(info['path'])
    if str(path) not in observed or sha(path)!=info['sha256']:
        raise ValueError('Selected original output not bound to completed derivative/node inventory')
    return path


def compare_one(binding,data,reference,backend):
    import nibabel as nib
    import numpy as np
    nodes=read(reference['nodes_report'],reference['nodes_sha256'])
    inventory=read(reference['inventory'],reference['inventory_sha256'])
    job=inventory['job'];report=read(Path(job['output_dir'])/'report.safe.json',nodes['execution_report_sha256'])
    controller=read(reference['controller_status'],reference['controller_status_sha256'])
    actual=next(x for x in controller['jobs'] if x['name']==job['name'])
    if (nodes['status']!='completed_original_saved_nodes_exported' or nodes['fmriprep_version']!='25.2.4'
            or actual['status']!='completed' or actual['exit_code']!=0
            or report['status']!='complete' or nodes['cpu_threads']!=data['job']['threads']):
        raise ValueError('Full successful original source/image/process proof not complete')
    official_inputs=read(reference['inputs'],nodes['inputs_configuration_sha256'])
    if sha(official_inputs['fmriprep_image'])!=nodes['actual_image_sha256']:
        raise ValueError('Actual original container image differs after export')
    inputs=read(binding['input_path'])['public180']
    for label,key in (('raw_BOLD','bold'),('raw_T1w','t1w')):
        if nodes['raw_input_hashes'][label]!=binding['input_metadata'][inputs[key]]:
            raise ValueError('Original and finalFNIT did not process the same complete raw data')
    observed=set(inventory['derivative_files'])
    for row in inventory['node_outputs']:observed.update(row['observed_files'])
    call=data['calls'][0];outputs=call['outputs'];comparisons={};roles=reference['roles']
    for role in ('preproc_t1w','preproc_mni','bold_reference','mask_mni','t1_brain'):
        spec=roles.get(role)
        if spec is None:
            comparisons[role]={'status':'not_retained_or_not_proven_in_original_saved_outputs'};continue
        original=proved_file(spec,observed);candidate=outputs[role]
        if sha(candidate['path'])!=candidate['sha256']:raise ValueError('Preserved finalFNIT output changed')
        if role=='preproc_mni' and (spec.get('space')!='MNI152NLin6Asym' or spec.get('resolution_mm')!=2):
            raise ValueError('Use original MNI6res2, not its additional anatomical MNI2009 registration')
        comparisons[role]=physical_pair(candidate['path'],original,
                                         frames=180 if role.startswith('preproc') else None,mask=role=='mask_mni')
    # FNIT's normal result does not retain pre-scaled motion-only native180.
    native=roles.get('native_preproc')
    if native:
        original=proved_file(native,observed);image=nib.load(str(original))
        if image.ndim!=4 or image.shape[3]!=180:raise ValueError('Original nativepreproc is not complete180')
        checks=image_check(original,'official_native_preproc',call['configuration']['TR'])
        comparisons['native_preproc']={'status':'official_complete_validated_FNIT_same_scope_output_not_retained',
                                      'official_sha256':sha(original),'official_checks':checks,
                                      'FNITclean_not_compared':True}
    else:comparisons['native_preproc']={'status':'native_preproc_not_identified_in_original_inventory','FNITclean_not_compared':True}
    matrices=reference.get('affines',{})
    if 'bbr' in matrices:
        spec=matrices['bbr'];proved_file(spec['official'],observed)
        first=spec['fnit'];normal=outputs['bbr_matrix']
        if first['path']!=normal['path'] or first['sha256']!=normal['sha256']:
            raise ValueError('Compare the actual FNIT FLIRT BBR affine')
        comparisons['bbr_world_ras']=compare_arrays(affine_world(first),affine_world(spec['official']))
        comparisons['bbr_world_ras']['scope']='Moving-reference to T1w push in mm-world-RAS; each FLIRT grid and ITK center/convention explicitly pinned. This measures independently estimated registration differences.'
    else:comparisons['bbr_world_ras']={'status':'coordinate_convention_or_native_affine_not_proven'}
    if 'motion' in matrices:
        spec=matrices['motion'];entries=spec['official_per_frame']
        if len(entries)!=180 or spec['direction']!='original_frame_to_bold_reference_push':
            raise ValueError('Every motion frame and transform direction must be explicit')
        world=[]
        for entry in entries:
            proved_file(entry,observed);world.append(np.linalg.inv(affine_world(entry)))
        info=outputs['motion_pull']
        if sha(info['path'])!=info['sha256']:raise ValueError('Preserved motion pull changed')
        comparisons['motion_world_ras_pull']=compare_arrays(np.load(info['path'],allow_pickle=False),np.stack(world))
        comparisons['motion_world_ras_pull']['scope']='All180 boldref-world-RAS to original-frame-world-RAS pull affines. Official pushes inverted after explicit ITK/FLIRT conversion; reference-image differences remain part of independent pipeline comparison.'
    else:comparisons['motion_world_ras_pull']={'status':'full_frame_transform_direction_or_coordinate_convention_not_proven'}
    comparisons['nonlinear_warp']={'status':'not_compared_different_representations_and_independent_estimation',
                                  'FNIT_pull_sha256':outputs['mni_pull']['sha256'],
                                  'scope':'No ANTs/FNIRT/SynthMorph warp-equivalence claim; no transform evaluation or resampling added.'}
    return {'backend':backend,'cpu_threads':job['threads'],'same_node':True,'same_thread_budget':True,
            'same_physical_cores':False,'original_process_seconds_including_separate_hashes':actual['process_seconds_including_separate_hashes'],
            'original_calls':nodes['calls'],'original_stage_groups':nodes['stage_groups'],
            'original_node_runtime_scope':nodes['node_time_scope'],'official_cache_scope':nodes['cache_scope'],
            'GNU_time_scope':nodes['GNU_time_scope'],'original_scope':nodes['whole_scope'],
            'FNIT_scope':'Complete cold raw-BIDS API also performs highpass,scaling,PICA/AROMA and saves clean outputs; whole time is a different mathematical/output scope.',
            'FNIT_cold_complete_API_seconds':call['API_seconds'],'comparisons':comparisons,
            'source_image_execution_evidence':{key:nodes[key] for key in ('actual_image_sha256','adapter_sha256','reference_wrapper_source_sha256',
                'wrapper_payload_code_sha256','execution_report_sha256','manifest_sha256','inputs_configuration_sha256')},
            'reference_export_sha256':reference['nodes_sha256'],'reference_inventory_sha256':reference['inventory_sha256']}


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();os.umask(0o077)
    if args.output.exists():raise FileExistsError('Preserve previous comparison')
    manifest=read(args.manifest,args.manifest_sha256)
    binding=read(manifest['cpu_binding'],manifest['cpu_binding_sha256'])
    pair=binding['cpu_pairs']['fnirt_cpu1'];os.sched_setaffinity(0,set(pair['affinity']))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    leases=[]
    for key,mode in (('queue_lock',fcntl.LOCK_EX),('shared_group_lease',fcntl.LOCK_SH)):
        lease=Path(pair[key])
        if not lease.is_file():raise FileNotFoundError('Reuse existing completed CPU1 lease')
        handle=lease.open('a+');fcntl.flock(handle,mode);leases.append(handle)
    binding=load_binding(manifest['cpu_binding'],manifest['cpu_binding_sha256'])
    cpu=read(manifest['cpu_inventory'],manifest['cpu_inventory_sha256'])
    if len(manifest['references'])!=2 or {x['cpu_threads'] for x in manifest['references']}!={1,8}:
        raise ValueError('Complete original CPU1 and CPU8 references are both required')
    results=[];started=time.perf_counter();args.output.mkdir(parents=True)
    save(args.output/'status.private.json',{'status':'comparing_saved_full_preproc','new_API_calls':0})
    for reference in manifest['references']:
        threads=reference['cpu_threads']
        for backend in ('fnirt','synthmorph'):
            data=cpu['jobs'][f'{backend}_cpu{threads}_candidate']
            results.append(compare_one(binding,data,reference,backend))
    load_binding(manifest['cpu_binding'],manifest['cpu_binding_sha256'])
    required=[result['comparisons'][role] for result in results for role in ('preproc_t1w','preproc_mni')]
    available=sum(row.get('status')=='compared_same_physical_lattice' for row in required)
    complete=available==len(required)==8
    report={'schema_version':1,
            'status':'complete_same_grid_preproc_comparisons' if complete else 'finished_readout_with_grid_or_mapping_gaps',
            'required_preproc_comparisons_available':complete,'required_preproc_pairs':8,
            'same_physical_grid_preproc_pairs_compared':available,'new_API_calls':0,
            'complete_frames':180,'results':results,'collector_sha256':sha(__file__),
            'cpu_binding_sha256':manifest['cpu_binding_sha256'],'cpu_inventory_sha256':manifest['cpu_inventory_sha256'],
            'manifest_sha256':args.manifest_sha256,'comparison_seconds_outside_all_API_clocks':time.perf_counter()-started,
            'scope':'Independent end-to-end protocols. Full preproc compared only on the same proven physical lattice,using lossless spatial-axis permutation/flip; no clean-vs-preproc comparison and no added registration/interpolation.',
            'privacy':'Only anonymous roles,hashes and aggregate metrics; no paths,subject IDs,license contents,raw images or new individual figures.'}
    save(args.output/'comparison.public.json',report)
    save(args.output/'status.private.json',{'status':report['status'],'new_API_calls':0,
         'required_preproc_comparisons_available':complete,'report_sha256':sha(args.output/'comparison.public.json')})
    print(json.dumps(read(args.output/'status.private.json')))


if __name__=='__main__':main()
