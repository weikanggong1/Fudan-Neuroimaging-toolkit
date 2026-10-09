"""Offline RAS-mm projections and descriptive stored-polyline statistics.

Required labels and seed budget come from the caller. This tool does not infer
input version, dataset identity, or permission from a filename or TCK header.
The default source and license are unknown. A case-specific JSON manifest may
supply source/license metadata bound to all three actual input SHA-256 values.

5TT resampling occurs only for the offline display; tract coordinates and
stored-polyline length statistics are unchanged. No tracking runtime, GPU,
endpoint analysis, connectivity matrix analysis, or network access is used.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def distribution(values):
    import numpy as np
    return {'mean': float(np.mean(values, dtype=np.float64)),
            'median': float(np.median(values)),
            'percentiles': {str(percent): float(np.percentile(values, percent))
                            for percent in (5, 25, 75, 95)},
            'min': float(np.min(values)), 'max': float(np.max(values))}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('must be a nonnegative integer')
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument('--fnit-tck', type=Path, required=True, help='FNIT TCK: finite stored RAS+ world-mm polylines')
    result.add_argument('--mrtrix-tck', type=Path, required=True, help='MRtrix TCK: finite stored RAS+ world-mm polylines')
    result.add_argument('--five-tissue', type=Path, required=True, help='5TT NIfTI [X,Y,Z,5], spatial units mm; display background only')
    result.add_argument('--output-dir', type=Path, required=True, help='Write tracking_qc.png and tracking_stats.json here')
    result.add_argument('--n-seeds', type=positive_int, required=True, help='Configured seed budget per software, explicitly supplied by caller; never inferred from header')
    result.add_argument('--fnit-label', required=True, help='Explicit FNIT version/variant label; no assumed baseline or candidate')
    result.add_argument('--dataset-label', required=True, help='Caller-supplied dataset label; does not establish its source or license')
    result.add_argument('--sample-count', type=positive_int, default=2000, help='Maximum displayed streamlines per software (default: 2000)')
    result.add_argument('--sample-seed', type=nonnegative_int, default=20261009, help='Same independent sampling seed per software (default: 20261009)')
    result.add_argument('--metadata-manifest', type=Path, help='Optional case-specific JSON source/license metadata plus expected_input_sha256 for fnit_tck, mrtrix_tck and five_tissue; default source/license unknown')
    return result


def case_metadata(args, input_hashes):
    dataset = {'label': args.dataset_label, 'source': 'unknown', 'license': 'unknown',
               'metadata_provenance': 'no source or license manifest supplied'}
    record = None
    if args.metadata_manifest is not None:
        path = args.metadata_manifest.expanduser().resolve()
        manifest = json.loads(path.read_text())
        if manifest.get('dataset_label') != args.dataset_label:
            raise ValueError('metadata manifest dataset_label must match --dataset-label')
        if manifest.get('fnit_label') != args.fnit_label:
            raise ValueError('metadata manifest fnit_label must match --fnit-label')
        if manifest.get('expected_input_sha256') != input_hashes:
            raise ValueError('metadata manifest expected_input_sha256 must match all three actual input files')
        supplied = manifest.get('dataset', {})
        if not isinstance(supplied, dict):
            raise ValueError('metadata manifest dataset must be an object')
        dataset.update(supplied)
        dataset['label'] = args.dataset_label
        dataset['metadata_provenance'] = 'caller-supplied case-specific manifest; all input SHA-256 bindings checked offline'
        if not dataset.get('source'):
            dataset['source'] = 'unknown'
        if not dataset.get('license'):
            dataset['license'] = 'unknown'
        record = {'path': str(path), 'sha256': file_sha256(path), 'input_sha256_bindings_verified': True,
                  'source_report': manifest.get('source_report', 'unknown')}
    return dataset, record


def generate_qc(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import nibabel as nib
    from nibabel.processing import resample_to_output
    import numpy as np
    import scipy
    from scipy.stats import ks_2samp

    for name in ('fnit_tck', 'mrtrix_tck', 'five_tissue'):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'input file does not exist: {path}')
        setattr(args, name, path)
    args.output_dir = args.output_dir.expanduser().resolve()
    input_hashes = {name: file_sha256(getattr(args, name)) for name in ('fnit_tck', 'mrtrix_tck', 'five_tissue')}
    dataset_metadata, manifest_record = case_metadata(args, input_hashes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    PNG = args.output_dir / 'tracking_qc.png'
    STATS = args.output_dir / 'tracking_stats.json'
    FIVE_TISSUE = args.five_tissue
    INPUTS = (('FNIT', args.fnit_tck), ('MRtrix3', args.mrtrix_tck))
    PLOT_SAMPLE_COUNT = args.sample_count
    PLOT_SAMPLE_SEED = args.sample_seed
    dataset_footer = f'Dataset: {args.dataset_label}; license: {dataset_metadata["license"]} (caller manifest if supplied)'
    if dataset_metadata.get('acknowledgement_doi'):
        dataset_footer += '; acknowledgement doi:' + str(dataset_metadata['acknowledgement_doi'])
    datasets = []
    for software, path in INPUTS:
        expected_sha256 = input_hashes["fnit_tck" if software == "FNIT" else "mrtrix_tck"]
        loaded = nib.streamlines.load(str(path), lazy_load=False)
        # Nibabel's TCK reader provides RAS+ millimetres; verify its advertised map.
        if not np.array_equal(loaded.tractogram.affine_to_rasmm, np.eye(4)):
            raise ValueError(f'{software} TCK reader did not provide RAS-mm coordinates')
        streamlines = list(loaded.tractogram.streamlines)
        header_count = int(loaded.header['count']) if 'count' in loaded.header else None
        if header_count is not None and len(streamlines) != header_count:
            raise ValueError(f'{software} actual streamline count disagrees with TCK header')
        if not streamlines:
            raise ValueError(f'{software} TCK contains no streamlines')
        lengths = np.empty(len(streamlines), dtype=np.float64)
        point_counts = np.empty(len(streamlines), dtype=np.int64)
        for index, streamline in enumerate(streamlines):
            points = np.asarray(streamline, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
                raise ValueError(f'{software} streamline {index} is not a finite [N>=2,3] polyline')
            lengths[index] = np.linalg.norm(np.diff(points, axis=0), axis=1).sum(dtype=np.float64)
            point_counts[index] = len(points)
        selected = np.sort(np.random.default_rng(PLOT_SAMPLE_SEED).choice(
            len(streamlines), size=min(PLOT_SAMPLE_COUNT, len(streamlines)), replace=False))
        summary = {'software': software, 'tck_file': str(path), 'tck_sha256': expected_sha256,
                   'size_bytes': path.stat().st_size, 'streamline_count': len(streamlines),
                   'header_streamline_count': header_count,
                   'configured_seed_budget': args.n_seeds,
                   'seed_budget_provenance': 'caller-provided --n-seeds; not inferred from TCK header or stored streamlines',
                   'stored_polyline_length_mm': distribution(lengths),
                   'stored_points_per_streamline': distribution(point_counts),
                   'coordinates': 'RAS+ world millimetres',
                   'coordinate_dtype_on_disk': str(streamlines[0].dtype),
                   'length_accumulation_dtype': 'float64',
                   'plot_sample_count': int(len(selected)),
                   'plot_selected_index_sha256': hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),
                   'variant': (args.fnit_label if software == 'FNIT' else
                               'MRtrix3 ' + str(loaded.header.get('method', 'input; method unknown'))),
                   'mrtrix_version': str(loaded.header.get('mrtrix_version', 'unknown; not recorded in this TCK')),
                   'downsample_factor_header': loaded.header.get('downsample_factor'),
                   'total_count_header': loaded.header.get('total_count')}
        datasets.append({'summary': summary, 'streamlines': streamlines, 'lengths': lengths,
                         'selected': selected})

    ks = ks_2samp(datasets[0]['lengths'], datasets[1]['lengths'], alternative='two-sided', method='asymp')

    # Resample only the offline displayed
    # GM+WM background to an axis-aligned RAS+ grid so it shares world-mm axes
    # with the unchanged TCK coordinates. The plot is not a voxel-exact overlay.
    five_tissue_image = nib.load(str(FIVE_TISSUE))
    if len(five_tissue_image.shape) != 4 or five_tissue_image.shape[-1] != 5:
        raise ValueError('--five-tissue must have shape [X,Y,Z,5]')
    if five_tissue_image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('--five-tissue spatial units must be mm for a RAS-mm overlay')
    gm_wm = np.asarray(five_tissue_image.dataobj[..., :3], dtype=np.float32).sum(axis=-1, dtype=np.float32)
    display_image = resample_to_output(nib.Nifti1Image(gm_wm, five_tissue_image.affine),
                                       voxel_sizes=(1.5, 1.5, 1.5), order=1, mode='constant', cval=0.)
    display_volume = np.asarray(display_image.dataobj, dtype=np.float32)
    display_affine = display_image.affine
    assert nib.aff2axcodes(display_affine) == ('R', 'A', 'S')
    assert np.array_equal(display_affine[:3, :3], np.diag(np.diag(display_affine[:3, :3])))
    foreground = display_volume > .05
    if not np.isfinite(display_volume).all() or not foreground.any():
        raise ValueError('--five-tissue must provide a finite, nonempty GM+WM background')
    voxel_bounds = []
    for axis in range(3):
        marginal = foreground.any(axis=tuple(other for other in range(3) if other != axis))
        valid_indices = np.flatnonzero(marginal)
        voxel_bounds.append((int(valid_indices[0]), int(valid_indices[-1])))
    world_bounds = [(float(display_affine[axis, 3] + low * display_affine[axis, axis] - 5.),
                     float(display_affine[axis, 3] + high * display_affine[axis, axis] + 5.))
                    for axis, (low, high) in enumerate(voxel_bounds)]

    planes = (
        ('Axial projection', 0, 1, 2, 'R (+x, mm)', 'A (+y, mm)', ('L', 'R', 'P', 'A')),
        ('Coronal projection', 0, 2, 1, 'R (+x, mm)', 'S (+z, mm)', ('L', 'R', 'I', 'S')),
        ('Sagittal projection', 1, 2, 0, 'A (+y, mm)', 'S (+z, mm)', ('P', 'A', 'I', 'S')),
    )

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 10, 'savefig.facecolor': 'white'})
    figure, axes = plt.subplots(2, 3, figsize=(14.8, 10.5))
    for row, dataset in enumerate(datasets):
        selected_tracks = [dataset['streamlines'][index] for index in dataset['selected']]
        colors = []
        for track in selected_tracks:
            direction = np.abs(np.asarray(track[-1], dtype=np.float64) - track[0])
            colors.append(direction / max(float(np.linalg.norm(direction)), 1.e-12))
        colors = np.asarray(colors)
        for column, (title, horizontal, vertical, projected_axis, xlabel, ylabel, orientation) in enumerate(planes):
            axis = axes[row, column]
            projection = display_volume.max(axis=projected_axis)
            remaining_axes = [current for current in range(3) if current != projected_axis]
            image_horizontal = remaining_axes.index(horizontal)
            image_vertical = remaining_axes.index(vertical)
            projection = projection.transpose(image_vertical, image_horizontal)
            extent = (display_affine[horizontal, 3] - display_affine[horizontal, horizontal] / 2,
                      display_affine[horizontal, 3] + (display_volume.shape[horizontal] - .5) * display_affine[horizontal, horizontal],
                      display_affine[vertical, 3] - display_affine[vertical, vertical] / 2,
                      display_affine[vertical, 3] + (display_volume.shape[vertical] - .5) * display_affine[vertical, vertical])
            axis.imshow(projection, origin='lower', cmap='gray', vmin=0., vmax=1.,
                        extent=extent, interpolation='bilinear', alpha=.84)
            projected_tracks = [np.asarray(track)[:, (horizontal, vertical)] for track in selected_tracks]
            axis.add_collection(LineCollection(projected_tracks, colors=colors, linewidths=.42, alpha=.20))
            axis.set_xlim(world_bounds[horizontal])
            axis.set_ylim(world_bounds[vertical])
            axis.set_aspect('equal')
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.set_facecolor('#eeeeee')
            axis.grid(False)
            axis.tick_params(labelsize=8)
            axis.set_title(title, pad=12)
            left, right, bottom, top = orientation
            for label, x, y in ((left, .02, .5), (right, .98, .5), (bottom, .5, .02), (top, .5, .98)):
                axis.text(x, y, label, transform=axis.transAxes, ha='center', va='center',
                          fontsize=10, color='white', fontweight='bold',
                          bbox={'facecolor': '#222222', 'alpha': .80, 'edgecolor': 'none', 'pad': 2})
        summary = dataset['summary']
        length_summary = summary['stored_polyline_length_mm']
        figure.text(.015, .695 if row == 0 else .325,
                    ('FNIT\n' + args.fnit_label if row == 0 else
                     'MRtrix3\n' + str(dataset['summary'].get('variant', 'input')).removeprefix('MRtrix3 ')),
                    va='center', ha='center', rotation=90, fontsize=12, fontweight='bold')
        axes[row, 1].text(.5, -.24 if row == 0 else -.16,
                          f"Stored tracks: {summary['streamline_count']:,}   |   displayed: {summary['plot_sample_count']:,}\n"
                          f"Stored-polyline length: mean {length_summary['mean']:.2f} mm; median {length_summary['median']:.2f} mm",
                          transform=axes[row, 1].transAxes, ha='center', va='top', fontsize=10)

    figure.suptitle('Tractography QC | ' + args.dataset_label, x=.52, y=.985,
                   fontsize=17, fontweight='bold')
    figure.text(.52, .949,
                f'{args.n_seeds:,} configured seeds per software (caller-provided); same sampling rule (up to {PLOT_SAMPLE_COUNT:,} tracks)',
                ha='center', fontsize=11)
    figure.text(.52, .922,
                'RAS+ world-mm orthogonal projections; 5TT GM+WM background resampled for display; not voxel-exact',
                ha='center', fontsize=10)
    figure.subplots_adjust(left=.08, right=.98, bottom=.14, top=.875, wspace=.23, hspace=.49)
    figure.text(.5, .045,
                f'FNIT label supplied by caller: {args.fnit_label}. Different RNGs; no paired tracks.\n'
                'Colors: endpoint direction in RAS (R=red, A=green, S=blue). Descriptive QC; endpoint/connectome parity not assessed.',
                ha='center', va='center', fontsize=10)
    figure.text(.5, .015,
                dataset_footer,
                ha='center', fontsize=8, color='#555555')
    figure.savefig(PNG, dpi=180)
    plt.close(figure)

    report = {
        'kind': 'offline descriptive tractography QC; no equivalence claim',
        'dataset': dataset_metadata,
        'metadata_manifest': manifest_record,
        'tcks': [dataset['summary'] for dataset in datasets],
        'length_method': 'Sum of Euclidean distances between consecutive stored RAS-mm TCK points, float64 arithmetic; no streamline resampling',
        'length_limitations': 'Stored-polyline length depends on stored point spacing/downsampling; it is not each implementation internal integration length.',
        'comparison': {'measure': 'two-sample Kolmogorov-Smirnov D for stored-polyline lengths',
                       'ks_d': float(ks.statistic), 'alternative': 'two-sided', 'method': 'scipy.stats.ks_2samp(method=asymp)',
                       'interpretation': 'descriptive distribution distance only; RNGs differ; no software equivalence conclusion',
                       'mean_length_difference_fnit_minus_mrtrix_mm': float(datasets[0]['lengths'].mean() - datasets[1]['lengths'].mean()),
                       'streamline_count_difference_fnit_minus_mrtrix': int(len(datasets[0]['lengths']) - len(datasets[1]['lengths']))},
        'plot': {'path': str(PNG), 'sha256': file_sha256(PNG),
                 'sampling': f'For each TCK, independently reset numpy default_rng to the same seed; choose up to {PLOT_SAMPLE_COUNT} unique streamline indices uniformly, without replacement; sort indices',
                 'sample_seed': PLOT_SAMPLE_SEED, 'sample_count_per_software': PLOT_SAMPLE_COUNT,
                 'coordinate_system': 'RAS+ world millimetres; neurological left-to-right orientation',
                 'background': 'Maximum intensity projections of 5TT cGM+sGM+WM; resampled at 1.5 mm only for display',
                 'voxel_exact': False, 'color_rule': 'normalized absolute endpoint displacement, R=red, A=green, S=blue',
                 'fnit_variant': args.fnit_label},
        'five_tissue': {'path': str(FIVE_TISSUE), 'sha256': input_hashes['five_tissue'],
                        'shape': list(five_tissue_image.shape), 'affine': five_tissue_image.affine.tolist(),
                        'native_orientation': list(nib.aff2axcodes(five_tissue_image.affine)),
                        'display_orientation': list(nib.aff2axcodes(display_image.affine)),
                        'display_affine': display_image.affine.tolist()},
        'endpoint_parity_assessed': False, 'structural_connectivity_matrix_parity_assessed': False,
        'raw_inputs_uploaded': False, 'gpu_executed': False,
        'cli_arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'offline_only': True,
        'script': {'path': str(Path(__file__).resolve()), 'sha256': file_sha256(Path(__file__))},
        'runtime': {'python': __import__('sys').executable, 'nibabel': nib.__version__, 'matplotlib': matplotlib.__version__,
                    'numpy': np.__version__, 'scipy': scipy.__version__},
    }
    STATS.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'stats': str(STATS), 'plot': str(PNG),
                      'counts': [dataset['summary']['streamline_count'] for dataset in datasets],
                      'length_mean_mm': [dataset['summary']['stored_polyline_length_mm']['mean'] for dataset in datasets],
                      'length_median_mm': [dataset['summary']['stored_polyline_length_mm']['median'] for dataset in datasets],
                      'ks_d': float(ks.statistic)}, indent=2))


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if not args.fnit_label.strip() or not args.dataset_label.strip():
        argument_parser.error('--fnit-label and --dataset-label must be nonempty')
    # The temporary directory holds only Matplotlib cache; no input copies.
    previous_cache = os.environ.get('MPLCONFIGDIR')
    try:
        with tempfile.TemporaryDirectory(prefix='fnit-qc-mpl-cache-') as cache_directory:
            os.environ['MPLCONFIGDIR'] = cache_directory
            generate_qc(args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        argument_parser.error(str(error))
    finally:
        if previous_cache is None:
            os.environ.pop('MPLCONFIGDIR', None)
        else:
            os.environ['MPLCONFIGDIR'] = previous_cache


if __name__ == '__main__':
    main()
