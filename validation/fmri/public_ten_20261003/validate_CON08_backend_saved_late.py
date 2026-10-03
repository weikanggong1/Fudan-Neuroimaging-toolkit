"""独立只读复核 CON08 已保存产物；保留原 Path reporter 失败及丢失的 API 时钟。"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

REVISION = '1128bc52c7a0233266e5b8a8d7dc0b382994e676'
RUNNER = 'a19408d5b928227aa458b2d9480772c1b3820836508046461fbe3f9b0c49fc54'
HELPER = '7cf5ef33c17af14e55a776ec1ff539718552e2c948a6e1575ba0b5f95a940db9'
RAW_T1 = '2cf6d2f2afc65cb7e87ef68ac95a645387ab5c397f93e080a6b25ac485b810b6'
RAW_BOLD = '24f4c4547182eb8267a455cfbee24593564e96ca97160da58037bbc7bbcbdd4b'
PHASE = 'save report/files.private.json after full API return and saved-output checks'
MESSAGE = 'Object of type PosixPath is not JSON serializable'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def hashes(paths):
    return {name: sha256(path) for name, path in paths.items()}


def source_hashes(source):
    return {path.relative_to(source).as_posix(): sha256(path)
            for path in sorted((source / 'src/fnit').rglob('*.py'))}


def import_fixed(path, name, expected):
    if sha256(path) != expected:
        raise ValueError('a fixed validation helper changed before import')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_failure(report, failure):
    if (report.get('status') != 'failed' or report.get('error_type') != 'TypeError'
            or report.get('subject') != 'CON08' or report.get('source_revision') != REVISION
            or report.get('runner_sha256') != RUNNER or report.get('backend') != 'freesurfer'
            or report.get('surface_device') != 'cpu' or report.get('cpu_threads') != 4
            or report.get('cuda_visible_devices') != '' or report.get('torch_cuda_available') is not False
            or report.get('volume_executed') is not False or report.get('cold_reconstruction_preexisting') is not False
            or report.get('raw_t1w_sha256') != RAW_T1 or report.get('raw_bold_sha256') != RAW_BOLD
            or any(report.get(key) is not True for key in ('readonly_input_guards_equal',
                   'frozen_source_guards_equal', 'binding_guard_equal'))):
        raise ValueError('the original reporter failure or its science/input scope differs')
    if ('line 292, in execute' not in failure
            or 'save(output / "files.private.json", {"result": dataclasses.asdict(result), "configuration": config})' not in failure
            or failure.rstrip().splitlines()[-1] != 'TypeError: ' + MESSAGE):
        raise ValueError('the preserved failure is not the specified post-return Path serializer failure')
    if any(key in report for key in ('full_api_seconds', 'timing_seconds', 'outputs')):
        raise ValueError('this recovery contract requires the original unsaved result fields')


def protect_output(root, output, protected):
    if output.exists() or output.is_symlink():
        raise FileExistsError('late validation requires a new independent output directory')
    resolved = output.resolve()
    runs = (root / 'runs').resolve()
    if resolved == runs or not resolved.is_relative_to(runs):
        raise ValueError('late output must be a new unified FNIT/runs child')
    for path in protected:
        path = Path(path).resolve()
        if resolved.is_relative_to(path) or path.is_relative_to(resolved):
            raise ValueError('late output overlaps an original protected entity')


def output_path(derivatives, relative):
    path = (derivatives / relative).resolve(strict=True)
    if not path.is_relative_to(derivatives.resolve()) or not path.is_file():
        raise ValueError('a saved output escapes the original derivatives or is not regular')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fnit-root', type=Path, required=True)
    parser.add_argument('--cold-run', type=Path, required=True)
    parser.add_argument('--original-runner', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('late read-only verification must hide CUDA before Python starts')
    root, cold = args.fnit_root.resolve(), args.cold_run.resolve()
    index = load(root / 'INDEX.json')['active_tasks']['fmri_surface_backends_20261003']
    if index['frozen_source_revision'] != REVISION:
        raise ValueError('the unified index source differs')
    cohort = root / 'workspaces/fnit_surface_ten_public_20261003'
    source = (cohort / 'source_1128bc52').resolve()
    original_report_path = cold / 'report/report.public.json'
    original = load(original_report_path)
    failure_path = cold / 'report/failure.private.txt'
    validate_failure(original, failure_path.read_text())
    if (cold / 'report/files.private.json').exists():
        raise ValueError('the original unsaved file map must remain absent')
    binding_path, config_path = cold / 'binding.private.json', cold / 'config.private.json'
    binding, config = load(binding_path), load(config_path)
    if (config['device'] != 'cpu' or config['cpu_threads'] != 4 or config['auto_volume'] is not False
            or config['recon_all_backend'] != 'freesurfer' or config.get('recon_all') is not None
            or sha256(config_path) != original['configuration_sha256']):
        raise ValueError('the original configuration differs from its failed driver binding')
    paths = {key: Path(path) for key, path in binding['paths'].items()}
    if (hashes(paths) != binding['hashes_before']
            or original['readonly_inputs_before'] != binding['hashes_before']
            or original['readonly_inputs_after'] != binding['hashes_before']):
        raise ValueError('original whole-run input/native/resource/configuration guards differ')
    source_before = source_hashes(source)
    if source_before != binding['source_hashes_before'] or len(source_before) != 460:
        raise ValueError('actual 460 scientific source files differ before imports')
    helper_path = source / 'validation/fmri/public_ten_20261003/run_fnit_subject.py'
    if sha256(args.original_runner) != RUNNER or paths['runner'].resolve() != args.original_runner.resolve():
        raise ValueError('the actual original runner identity differs')
    paths.update({'original_failed_report': original_report_path, 'original_failure_log': failure_path,
                  'original_binding': binding_path, 'late_validator': Path(__file__).resolve(),
                  'fixed_validation_helper': helper_path, 'unified_index': root / 'INDEX.json'})
    derivatives = Path(config['derivatives_root']).resolve(strict=True)
    if derivatives != cold / 'derivatives':
        raise ValueError('the original derivatives path differs from its cold run')
    prefix = Path('sub-CON08/ses-preop/func/sub-CON08_ses-preop_task-rest')
    suffixes = {'left': '_hemi-L_space-fsLR_den-32k_desc-preproc_bold.func.gii',
                'right': '_hemi-R_space-fsLR_den-32k_desc-preproc_bold.func.gii',
                'dtseries': '_space-fsLR_den-91k_desc-preproc_bold.dtseries.nii',
                'metadata': '_space-fsLR_den-91k_desc-preproc_bold.json',
                'qc_report': '_space-fsLR_den-91k_desc-preproc_report.json',
                'sphere_L': '_hemi-L_space-fsLR_desc-preprocReg_sphere.surf.gii',
                'sphere_R': '_hemi-R_space-fsLR_desc-preprocReg_sphere.surf.gii'}
    selected = {key: output_path(derivatives, str(prefix) + suffix) for key, suffix in suffixes.items()}
    paths.update({'saved_output/' + key: path for key, path in selected.items()})
    # Guard every actually saved surface publication file, including GIFTI sidecars.
    paths.update({'publication/' + path.relative_to(derivatives).as_posix(): path
                  for path in selected['metadata'].parent.iterdir() if path.is_file()})
    manifest_path = Path(config['recon_all_output_dir']) / 'fnit-surface-reconstruction.json'
    paths['reconstruction_manifest'] = manifest_path
    protect_output(root, args.output_root, [source, cohort / 'raw', cohort / 'candidate_v4', cold,
        config['hcp_assets_dir'], args.original_runner.parent, Path(__file__).resolve().parent,
        *[path.parent for key, path in paths.items() if key.startswith('native/')],
        *[path for key, path in paths.items() if key.startswith('resources/')]])
    before = hashes(paths)
    tick = time.perf_counter()
    sys.path.insert(0, str(source / 'src'))
    checker = import_fixed(args.original_runner, 'original_cold_checker', RUNNER)
    helper = import_fixed(helper_path, 'fixed_science_validation_helper', HELPER)
    import fnit
    import nibabel as nib
    import numpy as np
    import torch
    from fnit.fmri import surface_reconstruction as adapter
    if Path(fnit.__file__).resolve() != source / 'src/fnit/__init__.py' or torch.cuda.is_available():
        raise ValueError('actual package or hidden-CUDA execution differs')
    torch.set_num_threads(4)
    metadata, qc = load(selected['metadata']), load(selected['qc_report'])
    content = metadata['FNIT']
    recon, request = content['Reconstruction'], content['Reconstruction']['request']
    subject = Path(recon['subject_dir']).resolve(strict=True)
    if (subject != (cold / 'reconstruction/subject').resolve() or recon != load(manifest_path)
            or recon['status'] != 'complete' or recon.get('reused', False) is not False
            or request['backend'] != 'freesurfer' or request['device'] != 'cpu'
            or request['source_sha256'] != RAW_T1 or request['options'] != config['recon_all_options']
            or not adapter._cached_report(recon, request, subject)):
        raise ValueError('actual cold reconstruction manifest/request/closure differs')
    if content['VolumePrerequisite'] != {'InitialState': 'ready', 'Executed': False, 'Reused': True, 'Seconds': 0.0}:
        raise ValueError('saved surface metadata does not prove the ready-volume handoff')
    for key in ('Reconstruction', 'RegisteredSpheres', 'TimingSeconds', 'VolumePrerequisite', 'Geometry'):
        if qc[key] != content[key]:
            raise ValueError('saved product metadata and QC disagree')
    if content['RegistrationQC'] != 'bids::' + selected['qc_report'].relative_to(derivatives).as_posix():
        raise ValueError('saved metadata points to a different QC product')
    saved_timing = content['TimingSeconds']
    if ('total' in saved_timing or 'total_before_publication' not in saved_timing
            or any(not np.isfinite(value) or value < 0 for value in saved_timing.values())):
        raise ValueError('saved pre-publication timing scope differs')
    outputs = {'dtseries': helper.image_check(selected['dtseries'], 180, 2.4)}
    outputs['dtseries'].update(checker.check_cifti_axes(selected['dtseries'], config['hcp_assets_dir'], 2.4))
    cifti = nib.load(str(selected['dtseries']))
    data = np.asarray(cifti.dataobj)
    for key, hemisphere, structure in (('left', 'L', 'CIFTI_STRUCTURE_CORTEX_LEFT'),
                                       ('right', 'R', 'CIFTI_STRUCTURE_CORTEX_RIGHT')):
        gifti = nib.load(str(selected[key]))
        if len(gifti.darrays) != 180 or any(row.data.shape != (32492,) or not np.isfinite(row.data).all() for row in gifti.darrays):
            raise ValueError('saved GIFTI does not contain the complete finite fsLR32k run')
        values = np.stack([row.data for row in gifti.darrays])
        sections = [(selection, subset) for name, selection, subset in cifti.header.get_axis(1).iter_structures() if name == structure]
        if len(sections) != 1 or not np.array_equal(values[:, sections[0][1].vertex], data[:, sections[0][0]]):
            raise ValueError('saved GIFTI cortical values differ from its complete CIFTI')
        outputs[hemisphere] = {'shape': [180, 32492], 'sha256': sha256(selected[key]), 'all_finite': True}
        sphere_path = selected['sphere_' + hemisphere]
        sphere = nib.load(str(sphere_path))
        points = sphere.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
        faces = sphere.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
        native_points, native_faces = nib.freesurfer.read_geometry(str(subject / ('surf/' + hemisphere.lower() + 'h.sphere')))
        if (points.shape != native_points.shape or not np.isfinite(points).all()
                or not np.array_equal(faces, native_faces)
                or content['RegisteredSpheres'][hemisphere]['File'] != 'bids::' + sphere_path.relative_to(derivatives).as_posix()
                or content['RegisteredSpheres'][hemisphere]['SHA256'] != sha256(sphere_path)):
            raise ValueError('persistent registered sphere domain/faces/metadata binding differs')
    # Include the whole manifest closure in the independent before/after file guard.
    for relative, entry in recon['files'].items():
        path = output_path(subject, relative)
        if path.stat().st_size != entry['size'] or sha256(path) != entry['sha256']:
            raise ValueError('cold reconstruction closure entry changed')
        paths['reconstruction_closure/' + relative] = path
        before['reconstruction_closure/' + relative] = entry['sha256']
    after, source_after = hashes(paths), source_hashes(source)
    if before != after or source_before != source_after:
        raise ValueError('original inputs/products/native/source changed during late validation')
    seconds = time.perf_counter() - tick
    args.output_root.mkdir(mode=0o700, exist_ok=False)
    source_file = args.output_root / 'source.private.json'
    source_file.write_text(json.dumps(source_before, indent=2) + '\n')
    filemap = {'result': {key: str(selected[key]) for key in ('left', 'right', 'dtseries', 'metadata', 'qc_report')},
        'configuration': config, 'late_generated_filemap': True, 'original_private_filemap_saved': False,
        'original_failed_report_sha256': before['original_failed_report'], 'original_binding_sha256': before['original_binding'],
        'timing_scope': 'Saved pre-publication timings only; returned timing_seconds.total was not saved.'}
    filemap['result'].update(registered_spheres=[str(selected['sphere_L']), str(selected['sphere_R'])],
        recon_all=str(subject), volume_executed=False, timing_seconds=saved_timing)
    filemap_path = args.output_root / 'files.private.json'
    filemap_path.write_text(json.dumps(filemap, indent=2) + '\n')
    report = {key: original[key] for key in ('subject', 'source_revision', 'backend', 'surface_device',
        'requested_reconstruction_device', 'actual_reconstruction_device', 'cuda_visible_devices', 'torch_cuda_available',
        'cpu_threads', 'cpu_affinity', 'inherited_affinity_cpu_count', 'configuration_sha256',
        'raw_t1w_sha256', 'raw_bold_sha256', 'native_sha256_before', 'readonly_input_guards_equal',
        'frozen_source_guards_equal', 'binding_guard_equal', 'readonly_inputs_before', 'readonly_inputs_after')}
    report.update(status='complete_saved_outputs_verified_late', original_driver_status='failed',
        boundary='Independent late saved-output validation; original driver failed after scientific API return.',
        original_driver_failure={'error_type': 'TypeError', 'message': MESSAGE, 'phase': PHASE,
            'original_report_sha256': before['original_failed_report'], 'original_runner_sha256': RUNNER,
            'failure_log_sha256': before['original_failure_log']},
        full_api_seconds=None, returned_api_total_seconds=None,
        original_driver_failure_wall_seconds=original['driver_through_saved_output_validation_seconds'],
        saved_timing_seconds=saved_timing, saved_product_timing_scope=content['TimingScope'],
        timing_seconds=saved_timing, reconstruction_timing_seconds=recon['timing'], reconstruction_reused=False,
        original_t1_identity=content['Geometry']['OriginalT1Identity'], volume_reused=True, volume_executed=False,
        frame_count=180, tr_seconds=2.4, outputs=outputs, GIFTI_CIFTI_cortical_values_exact=True,
        registered_spheres_sha256={hemi: content['RegisteredSpheres'][hemi]['SHA256'] for hemi in ('L', 'R')},
        metadata_sha256=before['saved_output/metadata'], qc_sha256=before['saved_output/qc_report'],
        reconstruction_manifest_sha256=before['reconstruction_manifest'],
        reconstruction_commands=[{'program': Path(item['argv'][0]).name, 'sha256': item['sha256'],
            'seconds': item['seconds']} for item in recon['commands']],
        registration=content['Registration'], registration_details=content['RegistrationDetails'],
        late_validation_seconds=seconds, original_source_guard_passed=True,
        late_input_guards_equal=True, late_source_guards_equal=True,
        late_input_sha256_before=before, late_input_sha256_after=after,
        actual_source_hashes_before=source_before, actual_source_hashes_after=source_after,
        source_manifest_sha256=sha256(source_file), late_private_filemap_sha256=sha256(filemap_path),
        late_validator_sha256=before['late_validator'], original_private_filemap_saved=False,
        scope='Additional CON08 FS8.2 CPU cold reconstruction and saved complete surface products. Original failed driver and source are preserved. Full API clock and returned total are unavailable, not reconstructed. No MRI rerun; late validation clock is separate. This does not replace the ten-case baseline.')
    public = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if any(token in public for token in ('/cwStorage/', '/home/', 'license.txt')):
        raise ValueError('public late report contains a private path')
    (args.output_root / 'report.public.json').write_text(public)
    print('CON08_SAVED_OUTPUTS_VERIFIED_LATE', seconds, sha256(args.output_root / 'report.public.json'))


if __name__ == '__main__':
    main()
