"""公开 CON08 真实 graymid 的两组完整时序脑图；仅后验展示几何，不重跑 MRI。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REVISION = '1128bc52c7a0233266e5b8a8d7dc0b382994e676'
RENDERER_SHA = '419e700795deb96bcc338a12bf0bcd3b89cf99d6a7b4c7788537691f61d3e332'
COMPARISON_SHA = 'c9b7d8c9af4b1f15e5112069ec9c820d214a1c0615dabc8339bd2cb25ca22ed1'
COLD_SHA = 'a19408d5b928227aa458b2d9480772c1b3820836508046461fbe3f9b0c49fc54'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def import_fixed(path, name, expected):
    if sha256(path) != expected:
        raise ValueError('a fixed read-only helper differs before import')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_reference_report(reference, expected_raw):
    if (reference.get('status') != 'complete' or reference.get('subject') != 'CON08'
            or reference.get('input_sha256') != expected_raw or reference.get('frames') != 180
            or reference.get('repetition_time') != 2.4 or reference.get('container_exit_code') != 0
            or reference.get('source_unchanged_during_run') is not True
            or reference.get('input_unchanged_during_run') is not True
            or reference.get('corrective_saved_file_guards_equal') is not True):
        raise ValueError('original corrected reference whole/raw/source guards differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('hide CUDA before starting posthoc geometry/plotting')
    config = load(args.config)
    paths = {key: Path(value) for key, value in config.items() if key != 'output_root'}
    root, output = paths['fnit_root'].resolve(), Path(config['output_root'])
    index = load(root / 'INDEX.json')['active_tasks']['fmri_surface_backends_20261003']
    if index['frozen_source_revision'] != REVISION:
        raise ValueError('the indexed scientific source differs')
    cohort = root / 'workspaces/fnit_surface_ten_public_20261003'
    source = (cohort / 'source_1128bc52').resolve()
    manifest = load(paths['public_data_manifest'])
    public_case = [row for row in manifest['subjects'] if row['subject'] == 'CON08']
    if manifest.get('license') != 'CC0' or len(public_case) != 1 or public_case[0]['complete_original_frames'] != 180:
        raise ValueError('only the exact public CC0 CON08 complete run can be plotted')
    paths.pop('fnit_root')
    formal = load(paths['formal_report'])
    formal_files = load(paths['formal_files'])
    original = load(paths['original_cold_report'])
    late, late_files = load(paths['late_report']), load(paths['late_files'])
    extra_comparison, pair = load(paths['extra_comparison']), load(paths['cohort_pair_report'])
    reference, reference_files = load(paths['reference_report']), load(paths['reference_files'])
    expected_raw = {'t1w': public_case[0]['T1w']['sha256'], 'bold': public_case[0]['BOLD']['sha256']}
    validate_reference_report(reference, expected_raw)
    if (formal['input_sha256'] != expected_raw or pair['raw_input_sha256'] != expected_raw
            or pair['status'] != 'complete' or pair['case_id'] != 'CON08'
            or pair['candidate_source_revision_bound'] != REVISION
            or pair['sources_unchanged_during_comparison'] is not True
            or sha256(paths['formal_report']) != pair['candidate']['report_sha256']
            or sha256(paths['formal_files']) != pair['candidate']['files_manifest_sha256']
            or sha256(paths['reference_report']) != pair['reference']['report_sha256']
            or sha256(paths['reference_files']) != pair['reference']['files_manifest_sha256']):
        raise ValueError('original complete cohort comparison report/file maps/public raw binding differs')
    if (extra_comparison['status'] != 'complete_saved_timeseries_comparison'
            or extra_comparison['input_guards_equal'] is not True
            or extra_comparison['all_original_scientific_source_hashes_equal'] is not True
            or extra_comparison['cold_late_report_sha256'] != sha256(paths['late_report'])):
        raise ValueError('extra saved-output comparison did not complete under the original late binding')
    source_manifest = load(paths['formal_source_manifest'])
    actual_source = {path.relative_to(source).as_posix(): sha256(path)
        for path in sorted((source / 'src/fnit').rglob('*.py'))}
    if (len(actual_source) != 460 or actual_source != source_manifest
            or sha256(paths['formal_source_manifest']) != formal['source_sha256']):
        raise ValueError('actual source differs before numerical/helper imports')
    checker = import_fixed(paths['comparison_helper'], 'fixed_CON08_comparison', COMPARISON_SHA)
    checker.verify_reports(formal, late)
    checker.verify_late_report(original, late, paths['original_cold_report'], paths['late_files'], paths['late_source_manifest'])
    formal_config = load(paths['formal_configuration'])
    if (formal_config['device'] != 'cuda:0' or sha256(paths['formal_configuration']) != formal['configuration_sha256']
            or late_files['configuration']['device'] != 'cpu'):
        raise ValueError('actual GPU/CPU configuration differs')
    cold_binding = load(paths['cold_binding'])
    paths.update({'original_input/' + key: Path(value) for key, value in cold_binding['paths'].items()})
    if {key: sha256(Path(value)) for key, value in cold_binding['paths'].items()} != cold_binding['hashes_before']:
        raise ValueError('cold original raw/native/resource/config inputs changed')
    subject = Path(formal_files['recon_all']).resolve()
    metadata = load(formal_files['metadata'])['FNIT']
    paths.update({'formal_metadata': Path(formal_files['metadata']), 'formal_orig': subject / 'mri/orig.mgz'})
    for hemi in ('L', 'R'):
        for kind in ('white', 'pial'):
            paths['native_geometry/' + hemi + '/' + kind] = subject / ('surf/' + hemi.lower() + 'h.' + kind)
        middle = subject / metadata['Geometry']['MidthicknessSource'][hemi]['File']
        if sha256(middle) != metadata['Geometry']['MidthicknessSource'][hemi]['SHA256']:
            raise ValueError('actual CON08 graymid differs from original surface metadata')
        paths['native_geometry/' + hemi + '/middle'] = middle
        sphere = Path(formal_files['metadata']).parent / Path(metadata['RegisteredSpheres'][hemi]['File'].removeprefix('bids::')).name
        if sha256(sphere) != metadata['RegisteredSpheres'][hemi]['SHA256']:
            raise ValueError('actual CON08 saved MSM sphere differs')
        paths['native_geometry/' + hemi + '/sphere'] = sphere
        paths['atlas/' + hemi] = Path(formal_config['hcp_assets_dir']) / ('global/templates/standard_mesh_atlases/' + hemi + '.sphere.32k_fs_LR.surf.gii')
    selected = {'formal': formal_files, 'extra_FS82_CPU': late_files['result'], 'reference_FS73_CPU': reference_files.get('result', reference_files)}
    for role, files in selected.items():
        paths[role + '/CIFTI'] = Path(files['dtseries'])
    if (sha256(paths['formal/CIFTI']) != formal['output_checks']['dtseries']['sha256']
            or sha256(paths['extra_FS82_CPU/CIFTI']) != late['outputs']['dtseries']['sha256']):
        raise ValueError('selected formal/extra CIFTI differs from original exact product binding')
    # The fixed cohort pair already binds the corrected reference and full saved files;
    # independently use the original reference output record, never an arbitrary path.
    reference_record = reference.get('output_checks', reference.get('outputs', {}))['dtseries']
    if sha256(paths['reference_FS73_CPU/CIFTI']) != reference_record['sha256']:
        raise ValueError('selected original reference CIFTI differs from its corrected completion report')
    paths.update({'renderer': Path(__file__).resolve(), 'configuration': args.config})
    protected = [source, cohort / 'raw', cohort / 'candidate_v4', paths['reference_report'].parent,
        paths['original_cold_report'].parent.parent, paths['late_report'].parent,
        formal_config['hcp_assets_dir'], paths['renderer_helper'].parent, paths['comparison_helper'].parent,
        *[path.parent for key, path in paths.items() if key.startswith('original_input/native/')]]
    checker.protect_output(root, output, protected)
    before = {name: sha256(path) for name, path in paths.items()}
    started = time.perf_counter()
    sys.path.insert(0, str(source / 'src'))
    renderer = import_fixed(paths['renderer_helper'], 'fixed_backend_mesh_renderer', RENDERER_SHA)
    axes_checker = import_fixed(paths['original_runner'], 'fixed_cold_axes_checker', COLD_SHA)
    import nibabel as nib
    import numpy as np
    from fnit.fmri.surface_prepare import prepare_t1w_surface_geometry
    from fnit._hemisphere_parallel import workbench_environment
    for role in selected:
        axes_checker.check_cifti_axes(paths[role + '/CIFTI'], formal_config['hcp_assets_dir'], 2.4)
    output.mkdir(mode=0o700, exist_ok=False)
    geometry_started = time.perf_counter()
    native = prepare_t1w_surface_geometry(subject, output / 'display/native',
        fsnative_to_t1w=metadata['Geometry']['FsnativeToT1wWorldAffine'], parallel=False, cpu_threads=4)
    geometries, commands = {}, []
    wb = str(paths['original_input/native/workbench'])
    for hemi, geometry in (('L', native.left), ('R', native.right)):
        if geometry.midthickness_source.resolve() != paths['native_geometry/' + hemi + '/middle'].resolve():
            raise ValueError('mature geometry conversion selected a different actual middle')
        sphere = nib.load(str(paths['native_geometry/' + hemi + '/sphere']))
        middle = nib.load(str(geometry.midthickness))
        middle_faces = middle.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
        if not np.array_equal(middle_faces, sphere.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data):
            raise ValueError('native graymid and saved registration sphere face order differ')
        target = output / ('display/' + hemi + '.CON08.graymid.32k.surf.gii')
        argv = [wb, '-surface-resample', str(geometry.midthickness), str(paths['native_geometry/' + hemi + '/sphere']),
            str(paths['atlas/' + hemi]), 'BARYCENTRIC', str(target)]
        tick = time.perf_counter()
        subprocess.run(argv, check=True, capture_output=True, text=True, env=workbench_environment(4))
        image, atlas = nib.load(str(target)), nib.load(str(paths['atlas/' + hemi]))
        points, faces = [image.get_arrays_from_intent(intent)[0].data for intent in ('NIFTI_INTENT_POINTSET', 'NIFTI_INTENT_TRIANGLE')]
        if (points.shape != (32492, 3) or not np.isfinite(points).all()
                or not np.array_equal(faces, atlas.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data)):
            raise ValueError('actual display geometry does not use the fixed ordered fsLR32k mesh')
        geometries['left' if hemi == 'L' else 'right'] = (points, faces)
        commands.append({'program': Path(wb).name, 'operation': '-surface-resample BARYCENTRIC',
            'program_sha256': before['original_input/native/workbench'], 'hemisphere': hemi,
            'seconds': time.perf_counter() - tick, 'output_sha256': sha256(target),
            'input_middle_sha256': sha256(geometry.midthickness), 'native_face_order_exact': True})
    geometry_seconds = time.perf_counter() - geometry_started
    arrays = {}
    common_axis = None
    for role in selected:
        image = nib.load(str(paths[role + '/CIFTI']))
        axes = (image.header.get_axis(0), image.header.get_axis(1))
        if common_axis is not None and any(a != b for a, b in zip(common_axis, axes)):
            raise ValueError('complete actual time or scalar brain-model axes differ')
        common_axis = axes
        data = np.asarray(image.dataobj, np.float64)
        if not np.isfinite(data).all():
            raise ValueError('complete saved comparison data contains nonfinite values')
        arrays[role] = {}
        for name, section, subset in axes[1].iter_structures():
            if name.startswith('CIFTI_STRUCTURE_CORTEX_'):
                arrays[role]['left' if name.endswith('LEFT') else 'right'] = (np.asarray(subset.vertex), data[:, section])
    pairs = [('FS73_CPU_to_FNIT_GPU', 'reference_FS73_CPU', 'formal',
        'Reference: fMRIPrep25.2.4 / FS7.3 CPU\nCandidate: FNIT1128 GPU'),
        ('FNIT_GPU_to_FS82_CPU', 'formal', 'extra_FS82_CPU',
        'Reference: FNIT1128 GPU\nCandidate: extra FS8.2 CPU\nSaved outputs verified; API clock null')]
    plt, Normalize = renderer.plt, renderer.Normalize
    figure = plt.figure(figsize=(16, 9.8), constrained_layout=True, facecolor='white')
    metrics, color_limits = {}, {}
    for row, (name, first, second, title) in enumerate(pairs):
        maps, metrics[name] = renderer.difference_maps(arrays[first], arrays[second])
        maximum = max(value['nrmse_maximum'] or 0 for value in metrics[name].values())
        finite_error = np.concatenate([maps[hemi]['nrmse'][np.isfinite(maps[hemi]['nrmse'])]
                                       for hemi in ('left', 'right')])
        color_top = float(np.quantile(finite_error, .99)) if len(finite_error) else 0.
        color_limits[name] = {'rule': 'pooled defined cortical vertex NRMSE 99th percentile; values above top saturate for display only',
            'top': color_top, 'actual_maximum': maximum, 'saturated_vertex_count': int(np.count_nonzero(finite_error > color_top)),
            'defined_vertex_count': int(len(finite_error))}
        error_norm, r_norm = Normalize(0, color_top or 1), Normalize(-1, 1)
        row_axes = []
        for column, (hemi, kind) in enumerate((('left', 'nrmse'), ('right', 'nrmse'), ('left', 'temporal_r'), ('right', 'temporal_r'))):
            ax = figure.add_subplot(2, 4, row * 4 + column + 1, projection='3d')
            row_axes.append(ax)
            renderer.draw_surface(ax, geometries[hemi], maps[hemi][kind],
                error_norm if kind == 'nrmse' else r_norm, plt.get_cmap('viridis' if kind == 'nrmse' else 'coolwarm'),
                180 if hemi == 'left' else 0)
            ax.set_title(title + '\n' + ('LH' if hemi == 'left' else 'RH') + ' ' + kind, fontsize=9)
        error_bar = figure.colorbar(plt.cm.ScalarMappable(norm=error_norm, cmap='viridis'), ax=row_axes[:2], shrink=.6, pad=.01,
            label='NRMSE / row reference RMS\n(top: cortical 99th percentile)')
        error_bar.ax.set_title(f'max={maximum:.3f}\n>top saturated', fontsize=8)
        figure.colorbar(plt.cm.ScalarMappable(norm=r_norm, cmap='coolwarm'), ax=row_axes[2:], shrink=.6, pad=.01, label='Temporal Pearson r')
    figure.suptitle('Public CC0 CON08: all 180 frames, TR2.4s\nShared actual CON08 FNIT1128 graymid via saved MSM -> fsLR32k; gray = medial wall/undefined\nCPU/GPU MSM differences confound sole version attribution; late output QC does not restore API clock', fontsize=11)
    image_path = output / 'CON08_backend_saved_cortical_comparisons.png'
    figure.savefig(image_path, dpi=170)
    plt.close(figure)
    after = {name: sha256(path) for name, path in paths.items()}
    actual_after = {path.relative_to(source).as_posix(): sha256(path) for path in sorted((source / 'src/fnit').rglob('*.py'))}
    if before != after or actual_source != actual_after:
        raise ValueError('original inputs/native/source changed during posthoc rendering')
    report = {'status': 'complete', 'subject': 'CON08', 'source_revision': REVISION,
        'dataset': 'OpenNeuro ds001226 v5.0.1', 'public_dataset_url': 'https://openneuro.org/datasets/ds001226/versions/5.0.1', 'license': 'CC0',
        'raw_input_sha256': expected_raw, 'frames': 180, 'tr_seconds': 2.4,
        'scope': 'Independent posthoc saved-array maps on actual CON08 formal FNIT graymid resampled via its saved registration sphere to fixed fsLR32k. No MRI/registration rerun, no fitting or relabeling of producers; CPU/GPU confounding remains.',
        'display_geometry': commands, 'display_geometry_seconds': geometry_seconds,
        'scalar_rule': 'Exact actual cortical CIFTI vertex indices, common 21 models and time axis. NRMSE uses each row reference RMS; gray marks medial wall or undefined scalar.',
        'metrics': metrics, 'color_limits': color_limits, 'CIFTI_axes_exact': True, 'input_sha256_before': before, 'input_sha256_after': after,
        'input_guards_equal': True, 'source_guards_equal': True, 'actual_source_hashes_before': actual_source,
        'actual_source_hashes_after': actual_after, 'renderer_sha256': before['renderer'],
        'mature_renderer_sha256': RENDERER_SHA, 'comparison_helper_sha256': COMPARISON_SHA,
        'plot_and_geometry_seconds_excluded_from_MRI': time.perf_counter() - started,
        'extra_original_driver_status': 'failed', 'extra_full_api_seconds': None,
        'extra_saved_outputs_late_verified': True, 'figure_sha256': sha256(image_path)}
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if any(token in text for token in ('/cwStorage/', '/home/', 'license.txt')):
        raise ValueError('public figure provenance contains a private path')
    (output / 'figure.public.json').write_text(text)
    print('CON08_BRAIN_FIGURE_COMPLETE', report['figure_sha256'])


if __name__ == '__main__':
    main()
