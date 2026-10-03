"""Bind completed original raw-BIDS outputs to their actual surface inputs.

This observer reads the final fMRIPrep execution report and completed-interface
manifest. It performs hashes and joins paths only; it never executes registration,
projection or image processing. Private path mappings and all arrays stay on the
server. Internal reference templates must already have a verified private copy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nodes-private', type=Path, required=True)
    p.add_argument('--full-run-public', type=Path, required=True)
    p.add_argument('--binding-map-private', type=Path, required=True)
    p.add_argument('--initial-geometry-public', type=Path, required=True)
    p.add_argument('--shared-subject', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    run = read(args.full_run_public)
    if run.get('exit_code') != 0 or run.get('validation_complete') is not True:
        raise ValueError('requires a completed successful whole original workflow')
    if (run.get('input_frames') != 490 or run.get('STC') is not False
            or run.get('run_msmsulc') is not True or run.get('cifti_requested') is not True):
        raise ValueError('this protocol requires all 490 STC-off frames and fresh MSMSulc/CIFTI')
    nodes = [x for x in read(args.nodes_private)['nodes'] if not x['mapnode_parent']]
    mappings = read(args.binding_map_private)
    def host(value):
        for source, target in sorted(mappings.items(), key=lambda x: -len(x[0])):
            if value == source or value.startswith(source.rstrip('/') + '/'):
                path = Path(target + value[len(source):])
                if not path.is_file():
                    raise FileNotFoundError(path)
                return str(path)
        raise ValueError('an actual original input/output lacks a verified host mapping')
    def one(interface, fragment):
        found = [n for n in nodes if n['interface'] == interface and fragment in n['result_file']]
        if len(found) != 1:
            raise ValueError(f'expected one completed {interface} in {fragment}, found {len(found)}')
        return found[0]
    initial = read(args.initial_geometry_public)['initial_14_geometry_sha256']
    shared = {name: sha256(args.shared_subject / 'surf' / name) for name in initial}
    if shared != initial or run['initial_geometry_sha256'] != initial:
        raise ValueError('initial reference copy and unchanged shared candidate geometry differ')
    args.output_root.mkdir(parents=True, exist_ok=False)
    prepared, msm, initial_spheres = {key: [] for key in (
        'white', 'pial', 'midthickness', 'midthickness_fsLR', 'sphere_reg_fsLR',
        'template_sphere', 'cortex_mask', 'template_roi')}, {}, []
    left_right = []
    t1w_inputs = []
    for i, hemi in enumerate(('L', 'R')):
        marker = f'/bold_fsLR_resampling_wf/_hemi_{hemi}/'
        select = one('KeySelect', marker + 'select_surfaces/')['output_fields']
        for key in prepared:
            prepared[key].append(host(select[key]))
        ribbon = one('VolumeToSurfaceMapping', marker)['input_fields']
        t1w_inputs.append(host(ribbon['volume_file']))
        if ribbon.get('method') != 'ribbon-constrained':
            raise ValueError('original projection method changed')
        final = one('MetricMask', marker + 'mask_fsLR/')['output_fields']['out_file']
        left_right.append(host(final))
        registration = one('MSM', f'/msm_sulc_wf/msmsulc/mapflow/_msmsulc{i}/')
        fields = registration['input_fields']
        msm[hemi] = {key: host(fields[value]) for key, value in (
            ('rotated_sphere', 'in_mesh'), ('native_sulc', 'in_data'),
            ('reference_sphere', 'reference_mesh'), ('reference_sulc', 'reference_data'))}
        initial_node = one('SurfaceSphereProjectUnproject', f'/fsLR_reg_wf/project_unproject/mapflow/_project_unproject{i}/')
        initial_spheres.append(host(initial_node['output_fields']['sphere_out']))
    if t1w_inputs[0] != t1w_inputs[1]:
        raise ValueError('original hemispheres used different T1w preproc series')
    cifti = one('GenerateCifti', '/bold_grayords_wf/gen_cifti/')
    print(json.dumps({'cifti_input_keys': list(cifti['input_fields']),
                      'cifti_output_keys': list(cifti['output_fields'])}))
    fields = cifti['input_fields']
    bold_std = host(fields['bold_file'])
    dtseries = host(cifti['output_fields']['out_file'])
    fixed = {'bold_file': t1w_inputs[0], 'bold_std': bold_std, 'volume_roi': None,
             **prepared, 'native_rois': prepared['cortex_mask'],
             'initial_spheres': initial_spheres,
             'area_surfaces': {'native': prepared['midthickness'], 'fsLR': prepared['midthickness_fsLR']},
             'expected_frames': 490, 'repetition_time': run['input_repetition_time_seconds'],
             'signal': 'preproc', 'geometry_space': 'T1w world RAS', 'sphere_kind': 'estimated_msmsulc'}
    fixed_path = args.output_root / 'projection_inputs.private.json'
    write(fixed_path, fixed)
    config = run['msm_binary_selection']
    provenance = {'original_command_exit_code': run['exit_code'],
                  'original_validation_complete': run['validation_complete'],
                  'shared_initial_geometry_equal': True, 'shared_original_geometry_unchanged': True,
                  'private_geometry_processing_in_whole_wall': True,
                  'original_execution_report_sha256': sha256(args.full_run_public),
                  'private_final_geometry_unchanged': run['existing_reconstruction_geometry_unchanged'],
                  'changed_private_geometry_files': run['changed_geometry_files'],
                  'initial_geometry_sha256': run['initial_geometry_sha256'],
                  'final_private_geometry_sha256': run['geometry_sha256']}
    manifest = {'left': left_right[0], 'right': left_right[1], 'dtseries': dtseries,
                'registered_spheres': prepared['sphere_reg_fsLR'],
                'projection_inputs_json': str(fixed_path), 'msm_inputs': msm,
                'startpoint_sha256': {'t1w_preproc': sha256(t1w_inputs[0]), 'mni_preproc': sha256(bold_std)},
                'msm_config': {'source_sha256': config['source_msm_config_sha256'],
                               'effective_sha256': config['effective_msm_config_sha256'],
                               'numthreads': config['newmsm_threads'],
                               'configuration_text': config['effective_msm_configuration_text']},
                'registration_estimated_here': True, 'geometry_processing_provenance': provenance}
    axis = nib.load(dtseries).header.get_axis(0)
    if (axis.size != 490 or axis.start != 0 or axis.unit != 'SECOND'
            or axis.step != run['input_repetition_time_seconds']):
        raise ValueError('original actual CIFTI axis does not preserve raw BIDS TR')
    manifest['original_intermediate_time_authority'] = {
        'whole_execution_report_sha256': sha256(args.full_run_public),
        'raw_bids_repetition_time_seconds': run['input_repetition_time_seconds'],
        'actual_intermediate_time_units': {
            field: nib.load(fixed[field]).header.get_xyzt_units()[1]
            for field in ('bold_file', 'bold_std')},
        'actual_cifti_step_seconds': axis.step,
        'definition': 'Original ResampleSeries intermediates may have unknown NIfTI time units; raw BIDS RepetitionTime and original GenerateCifti.TR preserve seconds. Actual intermediate files and hashes are unchanged.'}
    write(args.output_root / 'outputs.private.json', manifest)
    write(args.output_root / 'manifest_build.public.json', {
        'schema_version': 1, 'validation_complete': True, 'observer_script_sha256': sha256(__file__),
        'original_execution_report_sha256': sha256(args.full_run_public),
        'completed_node_manifest_sha256': sha256(args.nodes_private),
        'binding_map_sha256': sha256(args.binding_map_private),
        'shared_geometry_unchanged': shared == initial,
        'geometry_processing_provenance': provenance,
        'surface_output_sha256': {k: sha256(manifest[k]) for k in ('left', 'right', 'dtseries')},
        'startpoint_sha256': manifest['startpoint_sha256'],
        'scope': 'Read-only postprocessing of successfully completed continuous original raw-BIDS workflow. All actual surface projection and MSM inputs are recovered from its completed interfaces; no FNIT geometry or images are substituted.'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
