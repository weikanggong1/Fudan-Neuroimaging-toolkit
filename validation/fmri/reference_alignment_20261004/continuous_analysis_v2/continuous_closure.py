"""Independent late provenance checks; never changes the frozen MRI runner."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

BINDINGS_SHA = 'c6043f2426cf3339c5352faa530cbf040754a51ff9c70fd27a007873df69e76c'
SOURCE_SHA = 'dc8ea8c5688ca06373088c6d4968c249dee6f3e24b1dcfb74a7c1a71d2be96b5'
REFERENCE_SHA = 'ac885355a286ff6799aaeafc9735de1d0c0264b8afba55041ea4e94b1ddc3484'
RUNNER_SHA = '7964a43f6c0e9ecf52a3751dfbebad93a1bf742b848c1c4912353d2019bd0a74'
QUEUE_SHA = 'e25016cd11d31aa565c1669bc63e109c332013b8e05bad7c1d38c83e02c7f9de'
RESOURCES_SHA = 'd9ba2d624ec45dd78cab11fea627af95f601b6dd9d1c1ce943e9a98040212182'
ORDER = [['CON01', 'middle'], ['CON01', 'robust'], ['CON06', 'robust'], ['CON06', 'middle']]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), 'JSON object required')
    return value


def finite_seconds(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def inside(path, root):
    return Path(path).resolve(strict=True).is_relative_to(Path(root).resolve(strict=True))


def fresh_output(path, protected):
    target = Path(path).resolve()
    require(not Path(path).is_symlink() and not target.exists(), 'output must be a fresh directory')
    for root in protected:
        root = Path(root).resolve()
        require(not target.is_relative_to(root) and not root.is_relative_to(target), 'output overlaps protected input/source/run directory')
    return target


class Audit:
    def __init__(self):
        self.files = {}

    def check(self, entry, role):
        path = Path(entry['path']).resolve(strict=True)
        require(path.is_file(), 'bound file must be regular')
        actual = sha256(path)
        require(actual == entry['sha256'], role + ': SHA mismatch')
        size = entry.get('bytes', entry.get('size'))
        require(size is None or path.stat().st_size == size, role + ': size mismatch')
        self.files[role] = (path, actual)
        return path

    def snapshot(self, path, role):
        path = Path(path).resolve(strict=True)
        return self.check({'path': str(path), 'sha256': sha256(path)}, role)

    def finish(self):
        before = {key: value[1] for key, value in self.files.items()}
        after = {key: sha256(value[0]) if value[0].is_file() else None for key, value in self.files.items()}
        require(before == after, 'late verification inputs changed')
        return before, after


def source_check(source, manifest, audit):
    require(manifest['sha256'] == SOURCE_SHA, 'wrong frozen source manifest')
    path = audit.check(manifest, 'frozen_source_manifest')
    expected = load(path)
    require(expected.get('src/fnit/fmri/reference.py') == REFERENCE_SHA, 'wrong BOLD reference implementation')
    actual_names = {p.relative_to(source).as_posix() for p in (source / 'src/fnit').rglob('*.py')}
    require(actual_names == set(expected), 'frozen Python source membership changed')
    for name, digest in expected.items():
        audit.check({'path': str(source / name), 'sha256': digest}, 'source/' + name)
    return expected


def checked_case(case, variant, runs, bindings, queue, audit):
    original = bindings['cases'][case]
    root = runs / case / variant
    report_path = root / 'report/report.public.json'
    if not report_path.is_file():
        return None
    report = load(audit.snapshot(report_path, case + '/' + variant + '/report'))
    require(report['case_id'] == case and report['variant'] == variant, 'wrong case/variant label')
    if report['status'] != 'scientific_complete':
        return {'status': report['status'], 'report': report, 'report_path': report_path}
    require(report.get('input_guards_equal') is True and report.get('source_guards_equal') is True,
            'original MRI input/source guards must pass')
    require(report.get('failure') is None and report['frames'] == 180, 'invalid completed run')
    require(report.get('driver_sha256') == RUNNER_SHA and report.get('source_sha256') == SOURCE_SHA,
            'wrong completed runner/source identity')
    require(report['raw_hashes'] == {k: original['raw'][k]['sha256'] for k in ('t1w', 'bold')}, 'raw MRI identity differs')
    require(report['tr_seconds'] == original['tr_seconds'], 'wrong original TR')
    require(finite_seconds(report['api_seconds']) and finite_seconds(report['api_plus_validation_seconds'])
            and report['api_plus_validation_seconds'] >= report['api_seconds'], 'invalid API timing')
    manifest_path = runs / 'manifests' / (case + '.' + variant + '.private.json')
    manifest = load(audit.check({'path': str(manifest_path), 'sha256': report['manifest_sha256']}, case + '/' + variant + '/run_manifest'))
    require(manifest['case_id'] == case and manifest['variant'] == variant and manifest['tr_seconds'] == original['tr_seconds'], 'manifest identity differs')
    require(manifest['baseline_commit'] == report['baseline_commit'], 'baseline identity differs')
    require(report['baseline_commit'] == 'cc9402734faeba93b3a13c29932fa1392eaccf62', 'wrong frozen baseline label')
    config = manifest['configuration']
    source = Path(manifest['source_root']).resolve(strict=True)
    expected = source_check(source, manifest['source_manifest'], audit)
    run_source = audit.check({'path': str(root / 'report/source.private.json'), 'sha256': SOURCE_SHA}, case + '/' + variant + '/run_source')
    require(load(run_source) == expected, 'run source snapshot differs')
    workspace = source.parent
    audit.check({'path': str(workspace / 'runner_v1_code/run_continuous.py'), 'sha256': RUNNER_SHA}, 'frozen_runner')
    audit.check({'path': str(workspace / 'runner_v1_code/queue_continuous.py'), 'sha256': QUEUE_SHA}, 'frozen_queue_runner')
    audit.check(manifest['validation_helper'], 'frozen_validation_helper')
    for key, entry in manifest['input_files'].items():
        audit.check(entry, case + '/' + variant + '/original_runner_subset/' + key)
    for key in ('t1w', 'bold', 't1w_json', 'bold_json'):
        require(manifest['input_files'][key]['sha256'] == original['raw'][key]['sha256'], 'wrong raw file in run manifest')
        audit.check(original['raw'][key], 'raw/' + case + '/' + key)
    require(Path(config['bids_root']).resolve() == Path(bindings['raw_bids_root']).resolve(), 'wrong BIDS root')
    require(config['subject'] == case and config['session'] == 'preop' and config['task'] == 'rest', 'wrong raw BIDS selection')
    require(Path(config['derivatives_root']).resolve() == (root / 'derivatives').resolve(), 'wrong fresh derivative root')
    require(config['signal'] == 'preproc' and config['recon_all_backend'] == 'provided' and config['device'] == 'cuda:0', 'wrong endpoint/backend')
    require(config['volume_options']['bold_reference_strategy'] == variant and config['volume_options']['slice_timing'] is False
            and config['volume_options']['reuse_anatomical'] is False, 'wrong reference/STC/anatomical configuration')
    completed = [x for x in queue['completed'] if x['case'] == case and x['variant'] == variant]
    require(len(completed) == 1 and completed[0]['exit_code'] == 0, 'unique queue exit0 record required')
    execution = completed[0]
    require(Path(execution['report']).resolve() == report_path.resolve(), 'queue report path differs')
    require(finite_seconds(execution['process_seconds']) and execution['process_seconds'] >= report['api_plus_validation_seconds'], 'invalid process clock')
    files_path = audit.snapshot(root / 'report/files.private.json', case + '/' + variant + '/files')
    files = load(files_path)
    for key, entry in files.items():
        require(inside(entry['path'], root / 'derivatives'), 'output outside the current fresh derivative root')
        audit.check(entry, case + '/' + variant + '/output/' + key)
    for key in ('preproc_mni', 'dtseries'):
        require(report['output_checks'][key]['sha256'] == files[key]['sha256'] and report['output_checks'][key]['all_finite'] is True, 'original output check differs')
    require(report['output_checks']['preproc_mni']['shape'] == [91, 109, 91, 180]
            and report['output_checks']['dtseries']['shape'] == [180, 91282], 'wrong full endpoint shape')
    surface = load(files['surface_metadata']['path'])['FNIT']
    volume = load(files['volume_metadata']['path'])['FNIT']
    require(volume['Source']['Version'] == '0.16.0', 'wrong recorded FNIT version')
    recorded_source = volume['Source']['SourceSHA256']
    require(recorded_source.get('fmri/reference.py') == REFERENCE_SHA, 'wrong recorded reference source')
    require(all(expected.get('src/fnit/' + name) == digest for name, digest in recorded_source.items()), 'volume recorded source differs')
    require(hashlib.sha256(json.dumps(recorded_source, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            == volume['Source']['SourceManifestSHA256'], 'volume source fingerprint differs')
    require(surface['Signal'] == 'preproc' and volume['Signal'] == 'preproc', 'denoised endpoint not permitted')
    require(surface['VolumePrerequisite']['Executed'] is True and surface['VolumePrerequisite']['Reused'] is False
            and surface['VolumePrerequisite']['InitialState'] == 'missing', 'volume was not fresh automatic execution')
    require(surface['VolumeProcessing']['SliceTimingCorrection'] is False and surface['VolumeProcessing']['SusceptibilityCorrection'] is False, 'wrong STC/SDC scope')
    subject = Path(original['fnit']['recon_all']).resolve(strict=True)
    require(Path(config['recon_all']).resolve() == subject, 'wrong original own reconstruction')
    old_recon_entry = original['fnit']['reconstruction_manifest']
    require(manifest['input_files']['reconstruction_manifest']['sha256'] == old_recon_entry['sha256'], 'wrong original recon manifest binding')
    old_recon = load(audit.check(old_recon_entry, 'original_reconstruction/' + case))
    reconstruction = surface['Reconstruction']
    request = reconstruction['request']
    require(reconstruction['status'] == 'complete' and reconstruction['commands'] == [] and request['backend'] == 'provided', 'reconstruction was not direct own provided input')
    require(request['producer'] == {'fmri/surface_reconstruction.py': expected['src/fnit/fmri/surface_reconstruction.py']}, 'wrong reconstruction adapter source')
    require(request['source_sha256'] == original['raw']['t1w']['sha256'] and Path(request['source_t1w']).resolve() == Path(original['raw']['t1w']['path']).resolve(), 'selected T1w is not the bound raw T1w')
    require(Path(request['provided']['path']).resolve() == subject, 'runtime subject differs')
    required = {'mri/orig.mgz', 'mri/orig/001.mgz'} | {'surf/' + h + '.' + n for h in ('lh', 'rh') for n in ('white', 'pial', 'sphere', 'sphere.reg', 'thickness', 'sulc', 'graymid')}
    closure = request['provided']['files']
    require(set(closure) == required, 'runtime provided closure must contain the exact16 fixed files')
    for name, entry in closure.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts and inside(subject / name, subject), 'unsafe subject closure path')
        require(entry == old_recon['files'].get(name), 'runtime closure differs from original formal reconstruction')
        audit.check({'path': str(subject / name), **entry}, 'late_reconstruction/' + case + '/' + name)
    assets_root = Path(config['hcp_assets_dir']).resolve(strict=True)
    resources_path = assets_root / 'assets_manifest.public.json'
    resources = load(audit.check({'path': str(resources_path), 'sha256': RESOURCES_SHA}, 'original_resources_manifest'))
    require(resources.get('complete') is True, 'original resources incomplete')
    required_assets = {'fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz',
                       'fmriprep/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz',
                       'fmriprep/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz'}
    for hemi in ('L', 'R'):
        prefix = 'global/templates/standard_mesh_atlases/'
        required_assets |= {prefix + hemi + '.sphere.32k_fs_LR.surf.gii', prefix + hemi + '.atlasroi.32k_fs_LR.shape.gii',
                            prefix + 'fs_' + hemi + '/fsaverage.' + hemi + '.sphere.164k_fs_' + hemi + '.surf.gii',
                            prefix + 'fs_' + hemi + '/fs_' + hemi + '-to-fs_LR_fsaverage.' + hemi + '_LR.spherical_std.164k_fs_' + hemi + '.surf.gii'}
    resource_map = {x['relative_path']: x for x in resources['resources']}
    require(required_assets.issubset(resource_map), 'consumed resource missing from original manifest')
    for name in required_assets:
        record = resource_map[name]
        require(inside(assets_root / name, assets_root), 'unsafe resource path')
        audit.check({'path': str(assets_root / name), 'sha256': record['sha256'], 'bytes': record['bytes']}, 'late_resource/' + name)
    for hemi, key, input_key in [('L', 'LeftROI', 'left_registered_sphere'), ('R', 'RightROI', 'right_registered_sphere')]:
        require(surface['RegisteredSpheres'][hemi]['EstimatedHere'] is False
                and surface['RegisteredSpheres'][hemi]['SHA256'] == manifest['input_files'][input_key]['sha256'], 'MSM estimation was not excluded')
        require(surface['SurfaceAssetsSHA256'][key] == resource_map['global/templates/standard_mesh_atlases/' + hemi + '.atlasroi.32k_fs_LR.shape.gii']['sha256'], 'runtime atlas ROI differs')
    require(surface['SurfaceAssetsSHA256']['HCPdseg'] == resource_map['fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz']['sha256'], 'runtime subcortex resource differs')
    require(surface['StandardTemplateSHA256'] == volume['StandardTemplateSHA256'] == resource_map['fmriprep/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz']['sha256'], 'runtime standard template differs')
    return {'status': 'ready', 'report': report, 'report_path': report_path, 'files': files, 'source_root': source,
            'manifest_path': manifest_path, 'execution': execution, 'original_subset_count': len(manifest['input_files']),
            'late_runtime_reconstruction_files': len(closure), 'late_resource_files': len(required_assets)}
