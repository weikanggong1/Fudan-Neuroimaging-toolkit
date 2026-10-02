"""仅绘制已冻结展示数组；源 NIfTI 只做 canonical 轴置换/翻转和原始切片提取。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def extract(manifests, directory):
    import nibabel as nib
    import numpy as np
    if directory.exists():
        raise ValueError('fresh display extraction directory required')
    directory.mkdir(parents=True)
    metadata = {'scope': 'display-only exact slice extraction and existing official seed0 matrix; no fitting or resampling',
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': {}}
    for manifest in manifests:
        report = json.loads(manifest.read_text())
        if report['state'] != 'completed' or not report['execution_completed']:
            raise ValueError('actual completed official reference required')
        case = report['case_id']
        if case in metadata['cases']:
            raise ValueError('duplicate actual case')
        dwi = json.loads(Path(report['source']['official_dwi_contract']['path']).read_text())
        sources = {'mean_b0': dwi['files']['mean_b0'], 'fa': report['source']['images']['fa'],
                   'atlas': report['source']['profiles']['fs-aparc']['image']}
        arrays, entries = {}, {}
        for name, source in sources.items():
            path = Path(source['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
                raise ValueError('actual official image changed')
            image = nib.load(path)
            if len(image.shape) != 3:
                raise ValueError('three-dimensional official display image required')
            canonical = nib.as_closest_canonical(image)
            values = np.asarray(canonical.dataobj)
            z = values.shape[2] // 2
            arrays[name] = np.ascontiguousarray(values[:, :, z])
            entries[name] = {'path': str(path), 'sha256': source['sha256'], 'source_shape': list(image.shape),
                'source_affine': image.affine.tolist(), 'canonical_affine': canonical.affine.tolist(),
                'slice_index': z, 'slice_world_origin_z_mm': float((canonical.affine @ np.array([0, 0, z, 1]))[2]),
                'orientation': list(nib.aff2axcodes(canonical.affine)), 'slice_shape': list(arrays[name].shape),
                'slice_dtype': str(arrays[name].dtype),
                'extraction': 'canonical axis permutation/flip only; original oblique scanner plane; no interpolation'}
        source = report['outputs']['0']['profiles']['fs-aparc']
        path = Path(source['directory']) / 'count.csv'
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['matrix_sha256']['count']:
            raise ValueError('actual existing official matrix changed')
        arrays['count'] = np.loadtxt(path, delimiter=',')
        if arrays['count'].shape != (source['nodes'], source['nodes']):
            raise ValueError('actual matrix/node dimension differs')
        entries['count'] = {'path': str(path), 'sha256': source['matrix_sha256']['count'], 'nodes': source['nodes'],
            'nodes_tsv_sha256': source['nodes_tsv_sha256'], 'atlas_sha256': source['atlas_sha256'],
            'definition': report['matrix_definitions']['count'], 'seed': 0}
        path = directory / (case + '.npz')
        np.savez(path, **arrays)
        metadata['cases'][case] = {'actual_reference_manifest': {'path': str(manifest.resolve()),
            'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()},
            'display_arrays': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
            'sources': entries, 'scientific_parity': 'not_assessed'}
    path = directory / 'display_source.json'
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--display-source', type=Path)
    group.add_argument('--manifests', type=Path, nargs='+')
    parser.add_argument('--display-directory', type=Path, help='new exact display array directory for --manifests')
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.manifests:
        if args.display_directory is None:
            parser.error('--manifests requires a fresh --display-directory')
        args.display_source = extract(args.manifests, args.display_directory)
    if args.extract_only:
        print(json.dumps({'display_source': str(args.display_source), 'scope': 'display extraction only'}))
        return
    if args.output is None:
        parser.error('rendering requires --output')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    metadata = json.loads(args.display_source.read_text())
    cases = list(metadata['cases'])
    fig, axes = plt.subplots(len(cases), 4, figsize=(13, 4 * len(cases)), squeeze=False)
    for row, case in enumerate(cases):
        item = metadata['cases'][case]
        path = Path(item['display_arrays']['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['display_arrays']['sha256']:
            raise ValueError('actual immutable display arrays changed')
        arrays = np.load(path, allow_pickle=False)
        for col, name in enumerate(('mean_b0', 'fa', 'atlas', 'count')):
            ax = axes[row, col]
            values = arrays[name]
            if name == 'mean_b0':
                finite = values[np.isfinite(values)]
                vmin, vmax = np.percentile(finite, [2, 98])
                image = ax.imshow(values.T, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
            elif name == 'fa':
                image = ax.imshow(values.T, origin='lower', cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            elif name == 'atlas':
                ax.imshow(arrays['mean_b0'].T, origin='lower', cmap='gray', interpolation='nearest')
                image = ax.imshow(np.ma.masked_equal(values.T, 0), origin='lower', cmap='nipy_spectral',
                                  vmin=1, vmax=item['sources']['count']['nodes'], alpha=.85, interpolation='nearest')
            else:
                image = ax.imshow(np.ma.masked_equal(values, 0), cmap='magma',
                                  norm=LogNorm(vmin=1, vmax=max(2, values.max())), interpolation='nearest',
                                  extent=(.5, values.shape[1] + .5, values.shape[0] + .5, .5))
                ax.set_xlabel('Node index'); ax.set_ylabel('Node index')
                ax.set_xticks(np.arange(1, values.shape[1] + 1, 20))
                ax.set_yticks(np.arange(1, values.shape[0] + 1, 20))
                fig.colorbar(image, ax=ax, fraction=.046, pad=.04, label='Track count')
            if col < 3:
                ax.axis('off')
                ax.text(.03, .52, 'L', transform=ax.transAxes, color='white', weight='bold')
                ax.text(.93, .52, 'R', transform=ax.transAxes, color='white', weight='bold')
            ax.set_title(('Official mean b0', 'Official FA', f"Official fs-aparc ({item['sources']['count']['nodes']} nodes)", 'Official count, seed 0')[col], fontsize=11)
            if col == 0:
                ax.text(-.08, .5, case, transform=ax.transAxes, va='center', ha='right', rotation=90, fontsize=12)
    fig.suptitle('Independent official raw reference: ' + ', '.join(cases), fontsize=14)
    fig.text(.5, .015, 'Native scanner slice after axis flip/permutation only; no interpolation. Matrix from existing seed 0.\n'
             'Five official repeats completed per case. Cross-software parity is not assessed in this figure.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[.01, .065, 1, .955], h_pad=2.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor='white')
    plt.close(fig)
    print(json.dumps({'output': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'scope': 'display only; source arrays unchanged'}))


if __name__ == '__main__':
    main()
