"""独立官方原始数据链：消费已完成官方 DWI/anatomy 合同，运行五次追踪和八 atlas 矩阵。

仅供独立 CPU benchmark；不在 FNIT 运行时调用。不会读取 FNIT tracking_inputs.pt、
FA、FOD 或 atlas。保留官方自产原文件，用软链接交给已验证的 MRtrix 命令规划器。
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from connectome_repeat_common import load_metadata, load_profiles, sha256

ATLASES = ('fs-aparc', 'aparc+tian-s1', 'aparc.a2009s+tian-s1',
           'glasser+tian-s1', 'glasser+tian-s4', 'schaefer200+tian-s1',
           'schaefer500+tian-s4', 'schaefer1000+tian-s4')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference_helper():
    path = Path(__file__).with_name('benchmark_connectome_repeats_official.py')
    spec = importlib.util.spec_from_file_location('official_raw_command_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record_file(record, label):
    require(isinstance(record, dict) and set(('path', 'size_bytes', 'sha256')).issubset(record),
            f'{label}: an actual producer file record is required')
    path = Path(record['path'])
    require(path.is_file() and path.stat().st_size == record['size_bytes'] and
            sha256(path) == record['sha256'], f'{label}: producer bytes/size changed')
    return path.resolve()


def spacing_metadata(values):
    """Keep spatial geometry strict; an undefined channel-axis spacing is metadata only."""
    spacing = np.asarray(values, dtype=np.float64)
    require(spacing.ndim == 1 and len(spacing) >= 3 and np.isfinite(spacing[:3]).all() and
            (spacing[:3] > 0).all(), 'invalid actual three-dimensional spatial spacing')
    require(np.all(np.isnan(spacing[3:]) | (np.isfinite(spacing[3:]) & (spacing[3:] > 0))),
            'invalid nonspatial spacing; only undefined NaN or positive finite values allowed')
    undefined = [{'axis': int(axis), 'source_value': 'NaN',
                  'meaning': 'undefined nonspatial channel-axis spacing; original file retained'}
                 for axis in np.flatnonzero(np.isnan(spacing))]
    return [None if np.isnan(value) else float(value) for value in spacing], undefined


def image_record(record, label):
    path = record_file(record, label)
    image = nib.load(path)
    values = np.asarray(image.dataobj)
    require(np.isfinite(image.affine).all() and np.linalg.det(image.affine[:3, :3]) != 0,
            f'{label}: invalid image geometry')
    declared = record.get('grid', record)
    if 'shape' in declared:
        require(tuple(declared['shape']) == image.shape, f'{label}: producer shape differs')
    if 'affine' in declared:
        require(np.array_equal(np.asarray(declared['affine'], dtype=np.float64), image.affine),
                f'{label}: producer affine differs from actual file')
    spacing, undefined_spacing = spacing_metadata(image.header.get_zooms())
    return path, image, {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': record['sha256'],
        'shape': list(map(int, image.shape)), 'affine': image.affine.tolist(),
        'spacing': spacing, 'undefined_nonspatial_spacing': undefined_spacing,
        'storage_dtype': str(image.get_data_dtype()),
        'nonfinite_count': int((~np.isfinite(values)).sum()),
        'source_policy': 'unmodified official producer image; nonfinite values retained and reported'}


def preflight(anatomy_path, dwi_path, case_id):
    anatomy_path, dwi_path = Path(anatomy_path), Path(dwi_path)
    anatomy = json.loads(anatomy_path.read_text())
    dwi = json.loads(dwi_path.read_text())
    for contract, scope in ((anatomy, 'official_self_produced_fresh_fs_anatomy_and_raw_dwi_atlases'),
                            (dwi, 'official_self_produced_raw_dwi_chain')):
        require(contract.get('state') == 'completed' and contract.get('scope') == scope and
                contract.get('case_id') == case_id, 'actual completed official case/scope contract required')
    linked = record_file(anatomy['official_dwi_contract'], 'anatomy upstream official DWI contract')
    require(linked == dwi_path.resolve(), 'anatomy refers to another DWI producer contract')
    for name in ('prepared_report', 'official_anatomy_report'):
        record_file(anatomy[name], name)
    images, paths = {}, {}
    mapping = {'wm_fod': dwi['files']['wm_fod_normalized'], 'fa': dwi['files']['FA'],
               'five_tissue_act': anatomy['files']['five_tissue_dwi_world'],
               'gmwmi': anatomy['files']['gmwmi_dwi_world']}
    for name, entry in mapping.items():
        path, image, checked = image_record(entry, name)
        paths[name], images[name] = path, checked
        if name == 'wm_fod':
            require(len(image.shape) == 4 and image.shape[-1] == 45,
                    'official normalized WM FOD must have actual lmax8/45 coefficients')
        if name == 'five_tissue_act':
            require(len(image.shape) == 4 and image.shape[-1] == 5, '5TT needs five compartments')
    require(len(images['fa']['shape']) == 3 and len(images['gmwmi']['shape']) == 3,
            'official FA and GMWMI must be three-dimensional')
    # Each official operator reads physical coordinates from its own header.
    # Independent producer files need not share literal voxel-axis ordering;
    # record relationships without rewriting images or forcing FNIT geometry.
    relationships = {
        'fa_fod_headers_equal': images['fa']['shape'] == images['wm_fod']['shape'][:3] and
                                images['fa']['affine'] == images['wm_fod']['affine'],
        'gmwmi_five_tissue_headers_equal': images['gmwmi']['shape'] == images['five_tissue_act']['shape'][:3] and
                                         images['gmwmi']['affine'] == images['five_tissue_act']['affine'],
        'policy': 'source metadata and actual reader geometry retained; sampling uses world coordinates'}
    images['five_tissue_sift2'] = {**images['five_tissue_act']}
    paths['five_tissue_sift2'] = paths['five_tissue_act']
    require(set(anatomy['atlases']) == set(ATLASES), 'all eight canonical anatomy profiles required')
    profiles = {}
    for name in ATLASES:
        entry = anatomy['atlases'][name]
        atlas, image, checked = image_record(entry['atlas_dwi'], name)
        nodes = record_file(entry['nodes'], name + ':nodes')
        count = int(entry['n_nodes'])
        meta = load_metadata(nodes.parent, count)
        require(meta['node_rows'] is not None and meta['nodes_tsv_sha256'] == entry['nodes']['sha256'],
                f'{name}: actual canonical nodes.tsv required')
        values = np.asarray(image.dataobj)
        require(len(image.shape) == 3 and np.isfinite(values).all() and np.array_equal(values, np.rint(values)) and
                values.min() >= 0 and values.max() <= count, f'{name}: actual atlas labels/dimensionality differ')
        profiles[name] = {'atlas': str(atlas), 'image': checked, 'nodes_path': str(nodes),
                          'nodes_sha256': entry['nodes']['sha256'], 'nodes': count,
                          'node_rows': meta['node_rows'], 'maximum_present_label': int(values.max())}
    return {'case_id': case_id, 'scope': 'independently produced official whole chain; not fixed FNIT operator inputs',
            'anatomy_contract': {'path': str(anatomy_path.resolve()), 'sha256': sha256(anatomy_path)},
            'official_dwi_contract': {'path': str(dwi_path.resolve()), 'sha256': sha256(dwi_path)},
            'raw_t1w': anatomy.get('raw_t1w'), 'fresh_fs_origin': anatomy.get('fresh_fs_origin'),
            'official_rawprep_lineage': {key: dwi.get(key) for key in
                ('upstream_report', 'upstream_official_rawprep_report', 'upstream_official_rawprep_report_sha256')},
            'images': images, 'grid_relationships': relationships, 'input_paths': {key: str(value) for key, value in paths.items()},
            'profiles': profiles}


def parameters(n_seeds):
    # Native reader spacing is resolved by actual mrinfo before the tracking
    # plan is executed. Explicit scientific flags equal the fixed-input profile.
    return {'n_seed_attempts': n_seeds, 'step_mm': None, 'min_length_mm': None,
            'max_length_mm': 250., 'max_angle_degrees': 45., 'cutoff': .1,
            'power': .5, 'samples': 3, 'trials': 1000, 'max_attempts_per_seed': 1000,
            'downsample': 2, 'tracking_threads': 0,
            'rng_policy': 'official MRTRIX_RNG_SEED; integers do not imply the PyTorch RNG stream',
            'act_options': 'no backtrack, crop_at_gmwmi or mask',
            'spacing_policy': 'pinned official C++ defaults: step=0.5*Header voxel geometric mean, ACT minimum=2*same voxel size; actual TCK properties recorded'}


def native_readback(record, source):
    actual = json.loads(Path(record['json']).read_text())
    dtype = np.dtype(source['storage_dtype'])
    expected_type = {'f': 'Float', 'i': 'Int', 'u': 'UInt'}.get(dtype.kind, '') + str(dtype.itemsize * 8)
    require(expected_type and actual['datatype'].startswith(expected_type), 'native reader changed producer dtype')
    require(Path(actual['name']).resolve() == Path(source['path']).resolve(), 'reader examined another producer image')
    shape = np.asarray(source['shape'])
    strides = np.asarray(actual['strides'])
    size = np.asarray(actual['size'])
    axes = np.abs(strides[:3]) - 1
    spacing = np.asarray(actual['spacing'], dtype=np.float64)
    serialized_spacing, undefined_spacing = spacing_metadata(spacing)
    transform = np.asarray(actual['transform'], dtype=np.float64)
    require(size.shape == shape.shape and sorted(axes.tolist()) == [0, 1, 2] and
            np.array_equal(size[:3], shape[axes]) and np.array_equal(size[3:], shape[3:]),
            'official native reader shape/axis mapping differs')
    require(spacing.shape == shape.shape and
            transform.shape == (4, 4) and np.isfinite(transform).all() and
            np.array_equal(transform[3], [0., 0., 0., 1.]) and
            float(actual['intensity_offset']) == 0 and float(actual['intensity_scale']) == 1,
            'official native reader invalid geometry/intensity scaling')
    # The original mrinfo JSON is not rewritten; its SHA binds NaN/null metadata.
    actual = {**actual, 'spacing': serialized_spacing}
    return {'status': 'native_source_header_verified_reader_geometry_recorded', 'mrinfo_json': actual,
            'undefined_nonspatial_spacing': undefined_spacing,
            'mrinfo_json_sha256': sha256(Path(record['json'])), 'source': source,
            'scope': 'original official file semantics; not a fixed FNIT affine bit identity claim'}


def atomic(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def raw_binding(manifest_path, manifest_sha256, case_id, anatomy, dwi):
    require(sha256(manifest_path) == manifest_sha256, 'canonical raw manifest changed')
    manifest = json.loads(Path(manifest_path).read_text())
    cases = [item for item in manifest['cases'] if item['case_id'] == case_id]
    require(len(cases) == 1, 'exactly one original raw case required')
    case = cases[0]
    files = case['input_files']
    require(files and len({item['path'] for item in files}) == len(files), 'canonical raw inputs missing/duplicated')
    for item in files:
        require(Path(item['path']).is_file() and sha256(Path(item['path'])) == item['sha256'],
                'original public raw input bytes changed')
    t1 = anatomy['raw_t1w']
    require(any(item['kind'] == 'raw_t1w' and Path(item['path']).resolve() == Path(t1['path']).resolve()
                and item['sha256'] == t1['sha256'] for item in files), 'fresh official anatomy not bound to canonical raw T1')
    record_file(t1, 'raw T1')
    upstream = dwi.get('upstream_official_rawprep_report')
    digest = dwi.get('upstream_official_rawprep_report_sha256')
    if isinstance(upstream, dict):
        upstream_path = record_file(upstream, 'official rawprep provenance')
    else:
        require(isinstance(upstream, str) and digest is not None, 'actual official rawprep lineage record required')
        upstream_path = Path(upstream)
        require(upstream_path.is_file() and sha256(upstream_path) == digest, 'official rawprep provenance changed')
    report = json.loads(upstream_path.read_text())
    require((report.get('execution_completed') is True or report.get('state') == 'completed' or
             report.get('status') == 'completed' or report.get('completed') is True) and not report.get('error'),
            'official rawprep report incomplete/failed')
    require(report.get('subject') == case['subject'] and report.get('session') == case['session'],
            'official rawprep report belongs to another raw subject/session')
    expected = {str(Path(item['path']).resolve()): item['sha256'] for item in files
                if item['kind'] != 'dataset_description'}
    actual = {str(Path(path).resolve()): digest for path, digest in report.get('input_sha256', {}).items()}
    require(actual == expected, 'official rawprep original AP/PA/gradient/JSON/T1 input provenance differs from canonical')
    require(report.get('commands') and all(item.get('returncode') == 0 for item in report['commands']),
            'official rawprep actual commands incomplete/nonzero')
    eddy = [item for item in report['commands'] if
            Path(item.get('command', [''])[0]).name.lower().startswith('eddy')]
    require(len(eddy) == 1, 'one actual official EDDY command/solver required')
    command = eddy[0]['command']
    program = Path(command[0]).name.lower()
    mode = 'cpu' if program in ('eddy_cpu', 'eddy_openmp') else 'gpu' if program.startswith('eddy_cuda') else None
    require(mode is not None, 'official EDDY solver must identify actual CPU or CUDA binary')
    solver = {'mode': mode, 'program': command[0], 'program_sha256': eddy[0]['program_sha256'],
              'host': eddy[0]['host'], 'wall_seconds': eddy[0]['wall_seconds'], 'returncode': 0,
              'gpu_uuid': report.get('GPU_UUID') if mode == 'gpu' else None,
              'timing_policy': 'actual official preprocessing command, distinct from this CPU tracking/downstream benchmark'}
    return {'manifest_path': str(Path(manifest_path).resolve()), 'manifest_sha256': manifest_sha256,
            'dataset': manifest['dataset'], 'snapshot': manifest['snapshot'], 'license': manifest['license'],
            'case_id': case_id, 'files': files, 'all_actual_canonical_raw_sha_verified': True,
            'anatomy_raw_t1_verified': True, 'rawprep_canonical_dwi_coverage_verified': True,
            'official_rawprep_report': str(upstream_path.resolve()),
            'official_rawprep_report_sha256': sha256(upstream_path),
            'rawprep_producer_identity': report, 'official_eddy_solver': solver,
            'raw_b0_selection': {'actual_selection': report.get('selection'),
                'producer_selection_origin': report.get('selection_origin'),
                'policy': 'raw frame IDs and any shared input adaptation are preserved from actual producer; no independent selection claim'},
            'scope': 'canonical public raw T1/AP/PA/gradient/JSON identities verified against actual rawprep producer input SHA; actual source bytes checked'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anatomy-contract', type=Path, required=True)
    parser.add_argument('--dwi-contract', type=Path, required=True)
    parser.add_argument('--raw-manifest', type=Path, required=True)
    parser.add_argument('--raw-manifest-sha256', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--verified-reference-manifest', type=Path, required=True,
                        help='already audited fixed-input official reference, used for binary/helper identity only')
    parser.add_argument('--verified-reference-manifest-sha256', required=True)
    parser.add_argument('--mrtrix-bin', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--n-seeds', type=int, default=100000)
    parser.add_argument('--downstream-threads', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true', help='actual CPU input/lineage checks and command plan only')
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.n_seeds < 1 or args.downstream_threads != 8:
        parser.error('fresh output, positive attempted seed count, and downstream CPU8 required')
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('at least three unique nonnegative official seed labels required')
    started = time.perf_counter()
    origin = preflight(args.anatomy_contract, args.dwi_contract, args.case_id)
    anatomy = json.loads(args.anatomy_contract.read_text())
    dwi = json.loads(args.dwi_contract.read_text())
    raw = raw_binding(args.raw_manifest, args.raw_manifest_sha256, args.case_id, anatomy, dwi)
    helper = reference_helper()
    programs = {}
    for name in helper.PROGRAMS:
        path = args.mrtrix_bin / name
        require(path.is_file() and os.access(path, os.X_OK), f'official executable unavailable: {name}')
        programs[name] = {'path': str(path.resolve()), 'invoked_path': str(path), 'sha256': sha256(path)}
    require(sha256(args.verified_reference_manifest) == args.verified_reference_manifest_sha256,
            'verified official reference identity changed')
    verified = json.loads(args.verified_reference_manifest.read_text())
    require(verified.get('execution_completed') is True and verified['programs'] == programs and
            sha256(Path(helper.__file__)) == verified['script_sha256'] and
            sha256(Path(__file__).resolve().parents[1] / 'connectome_repeat_common.py') == verified['helper_sha256'],
            'actual binaries/command helper differ from previously verified pinned reference')
    params = parameters(args.n_seeds)
    plan = helper.command_plan(args.mrtrix_bin, args.output_dir, origin['profiles'], args.seeds,
                               params, args.downstream_threads)
    # Let the pinned C++ solver compute its actual float voxel defaults from
    # its own reader, instead of rebuilding that expression in Python.
    for record in plan:
        if record['stage'] == 'tracking':
            command = record['argv']
            for name in ('-step', '-minlength'):
                index = command.index(name)
                del command[index:index + 2]
    manifest = {'schema_version': 1, 'case_id': args.case_id, 'dataset': raw['dataset'],
        'snapshot': raw['snapshot'], 'scope': 'independent official raw DWI plus fresh FS anatomy tracking/downstream',
        'state': 'preflight_completed', 'execution_completed': False, 'scientific_parity': 'not_assessed',
        'source': origin, 'raw_case_binding': raw, 'parameters': params, 'seeds': args.seeds,
        'verified_reference_identity': {'path': str(args.verified_reference_manifest.resolve()),
            'sha256': args.verified_reference_manifest_sha256, 'scope': 'binary/command helper identity only; inputs remain independently official'},
        'downstream_threads': args.downstream_threads, 'commands': plan, 'completed_commands': [],
        'input_readbacks': {}, 'programs': programs, 'script_sha256': sha256(Path(__file__)),
        'reference_command_helper_sha256': sha256(Path(helper.__file__)),
        'matrix_helper_sha256': sha256(Path(__file__).resolve().parents[1] / 'connectome_repeat_common.py'),
        'environment': {'host': platform.node(), 'python': platform.python_version(), 'numpy': np.__version__,
                        'nibabel': nib.__version__, 'cuda_visible_devices': ''},
        'matrix_definitions': {'count': 'number of assigned tracks', 'sift2_fbc': 'sum(w)',
            'mean_length': 'sum(w * length) / sum(w)', 'mean_fa': 'sum(w * precise_track_mean_fa) / sum(w)',
            'assignment_radial_search_mm': 4., 'unassigned': 'dropped', 'self_connections': 'retained'},
        'input_policy': 'direct unmodified independently produced official NIfTI files; no FNIT inputs or image export',
        'timing_scope': 'this tracking/downstream component only; producer rawprep/recon/anatomy/modeling times separate'}
    if args.dry_run:
        print(json.dumps(manifest, indent=2, allow_nan=False))
        return
    # Same original case lineage is verified before observing SC matrices.
    require(raw.get('rawprep_canonical_dwi_coverage_verified') is True,
            'official rawprep producer canonical AP/PA/gradient/JSON coverage not yet audited')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / 'reference_manifest.json'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '8',
           'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'}
    try:
        version = helper.subprocess.run([str(args.mrtrix_bin / 'tckgen'), '-version'],
            check=True, capture_output=True, text=True, env=env)
        manifest['mrtrix_version'] = (version.stdout + version.stderr).strip()
        require(manifest['mrtrix_version'] == verified['mrtrix_version'], 'official binary version changed')
        images = args.output_dir / 'inputs'
        images.mkdir()
        for name, path in origin['input_paths'].items():
            (images / (name + '.nii.gz')).symlink_to(path)
        for name, profile in origin['profiles'].items():
            target = images / 'atlases' / name
            target.mkdir(parents=True)
            (target / 'atlas_dwi.nii.gz').symlink_to(profile['atlas'])
            for seed in args.seeds:
                directory = args.output_dir / f'seed-{seed}' / 'atlases' / name
                directory.mkdir(parents=True)
                shutil.copyfile(profile['nodes_path'], directory / 'nodes.tsv')
                (directory / 'nodes.txt').write_text(str(profile['nodes']) + '\n')
                (directory / 'atlas.sha256').write_text(profile['image']['sha256'] + '\n')
        manifest['state'] = 'running'
        atomic(report_path, manifest)
        for record in plan:
            if record['stage'] == 'tracking':
                require(len(manifest['input_readbacks']) == 5 + len(ATLASES), 'native reader contracts incomplete')
            actual = helper._run(record, {**env, 'MRTRIX_RNG_SEED': str(record.get('seed', 0))})
            manifest['completed_commands'].append(actual)
            atomic(report_path, manifest)
            require(actual['returncode'] == 0, f'official command failed; preserved log: {record["log"]}')
            if record['stage'] == 'input_readback':
                name = record['input']
                source = (origin['profiles'][name.removeprefix('atlas:')]['image']
                          if name.startswith('atlas:') else origin['images'][name])
                manifest['input_readbacks'][name] = native_readback(record, source)
                atomic(report_path, manifest)
        manifest['outputs'] = {}
        for seed in args.seeds:
            directory = args.output_dir / f'seed-{seed}'
            for name, profile in origin['profiles'].items():
                helper.align_official_matrices(directory / 'atlases' / name, profile['nodes'],
                                                profile['maximum_present_label'])
            profiles = load_profiles(directory)
            require(set(profiles) == set(ATLASES), 'official output atlas set incomplete')
            count = int(nib.streamlines.load(directory / 'tracks.tck', lazy_load=True).header['count'])
            vectors = {}
            for name in ('sift2_weights.txt', 'lengths.txt', 'mean_fa.txt'):
                values = np.loadtxt(directory / name, ndmin=1)
                require(values.shape == (count,), 'actual per-track scalar count differs')
                vectors[name] = {'sha256': sha256(directory / name),
                                 'nonfinite_count': int((~np.isfinite(values)).sum())}
            manifest['outputs'][str(seed)] = {'accepted_tracks': count, 'attempted_seeds': args.n_seeds,
                'tracks_sha256': sha256(directory / 'tracks.tck'), 'scalars': vectors,
                'actual_tck_header': {str(key): str(value) for key, value in
                    nib.streamlines.load(directory / 'tracks.tck', lazy_load=True).header.items()},
                'profiles': {name: meta for name, (_, meta) in profiles.items()}}
        require(preflight(args.anatomy_contract, args.dwi_contract, args.case_id) == origin,
                'official producer inputs changed during tracking/downstream')
        require(raw_binding(args.raw_manifest, args.raw_manifest_sha256, args.case_id, anatomy, dwi) == raw,
                'canonical raw or bound rawprep provenance changed')
        for entry in programs.values():
            require(sha256(Path(entry['path'])) == entry['sha256'], 'executed official binary changed')
        manifest['state'], manifest['execution_completed'] = 'completed', True
    except Exception as error:
        manifest['state'] = 'failed'
        manifest['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        manifest['total_wall_seconds'] = time.perf_counter() - started
        atomic(report_path, manifest)


if __name__ == '__main__':
    main()
