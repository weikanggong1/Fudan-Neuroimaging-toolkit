"""Compare complete, already executed real PICA/AROMA/confound outputs.

No fitting, private image copies, or reduced-frame controls are made here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'fmri'))
from compare_ica_fixed_input import normalised_columns, distribution
from compare_matched_pipeline import correlations


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array(path):
    image = nib.load(str(path))
    values = np.asarray(image.dataobj, np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite comparison input')
    return image, values


def pica(candidate, reference, mask):
    cm = np.loadtxt(candidate / 'ica_mixing.tsv', ndmin=2)
    rm = np.loadtxt(reference / 'melodic_mix', ndmin=2)
    if cm.shape[0] != 490 or rm.shape[0] != 490:
        raise ValueError('This recorded real run must retain all 490 time points')
    correlation = normalised_columns(cm).T @ normalised_columns(rm)
    rows, columns = linear_sum_assignment(-np.abs(correlation))
    signed = correlation[rows, columns]
    sign = np.where(signed < 0, -1., 1.)
    ci, ca = array(candidate / 'ica_components_z.nii.gz')
    ri, ra = array(reference / 'melodic_IC.nii.gz')
    if ci.shape[:3] != mask.shape or ri.shape[:3] != mask.shape or not np.allclose(ci.affine, ri.affine, atol=1e-4, rtol=0):
        raise ValueError('PICA map grids differ')
    ca = ca[mask][:, rows] * sign
    ra = ra[mask][:, columns]
    spatial_r = np.sum(normalised_columns(ca) * normalised_columns(ra), axis=0)
    _, thresholded = array(candidate / 'ica_components_pica_thresholded.nii.gz')
    ct = thresholded[mask][:, rows] != 0
    rt = np.column_stack([array(reference / f'stats/thresh_zstat{col+1}.nii.gz')[1][mask] != 0 for col in columns])
    counts = ct.sum(0) + rt.sum(0)
    dice = np.divide(2 * (ct & rt).sum(0), counts, out=np.ones(len(rows)), where=counts > 0)
    cf = np.loadtxt(candidate / 'ica_frequency_power.tsv', ndmin=2)[:, rows]
    rf = np.loadtxt(reference / 'melodic_FTmix', ndmin=2)[:, columns]
    return {'time_points': 490, 'components': {'fnit': cm.shape[1], 'original': rm.shape[1]},
            'matching': 'Hungarian absolute temporal Pearson r; same assignment and sign for maps',
            'temporal_absolute_r': distribution(np.abs(signed)),
            'spatial_sign_aligned_r': distribution(spatial_r),
            'spatial_rmse': distribution(np.sqrt(np.mean((ca-ra)**2, axis=0))),
            'threshold_support_dice': distribution(dice),
            'frequency_relative_l2': float(np.linalg.norm(cf-rf)/np.linalg.norm(rf)),
            'same_grid': True, 'full_mask_voxels': int(mask.sum())}


def images(candidate, reference, mask):
    ci, ca = array(candidate); ri, ra = array(reference)
    if ci.shape != ri.shape or ci.shape[-1] != 490 or not np.allclose(ci.affine, ri.affine, rtol=0, atol=1e-4):
        raise ValueError('Full output axes/grids differ')
    return {'shape': list(ci.shape), 'same_grid': True,
            'storage_dtypes': [str(ci.get_data_dtype()), str(ri.get_data_dtype())],
            'full_grid': correlations(ca, ra, temporal=True),
            'brain': correlations(ca[mask], ra[mask], temporal=True),
            'output_sha256': {'fnit': sha256(candidate), 'original': sha256(reference)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())['datasets']['real_run_01']
    _, mask_values = array(config['brain_mask']); mask = mask_values > 0
    baseline = args.run_root / 'baseline_v1'; official = args.run_root / 'reference_v2'
    report = {'schema_version': 1, 'dataset_alias': 'real_run_01', 'frames': 490,
              'driver_sha256': sha256(__file__), 'results': {}, 'unavailable': []}
    for threads in (1, 8):
        suffix = f'_t{threads}'
        for case in ('pica_auto', 'pica_fixed95'):
            report['results'][case+suffix] = pica(baseline/(case+suffix+'_fnit')/'ica', baseline/(case+suffix+'_official')/'melodic.ica', mask)
        for mode in ('nonaggr', 'aggr'):
            case = 'denoise_'+mode
            report['results'][case+suffix] = images(baseline/(case+suffix+'_fnit')/'denoised.nii.gz',baseline/(case+suffix+'_official')/'denoised.nii.gz',mask)
        cpath = baseline/('aroma_features'+suffix+'_fnit')/'features.npz'
        rpath = official/('aroma_features'+suffix+'_official')/'features.npz'
        if rpath.exists():
            left, right = np.load(cpath), np.load(rpath)
            report['results']['aroma_features'+suffix] = {
                key: {'max_abs_error': float(np.max(np.abs(left[key]-right[key]))),
                      'all_equal': bool(np.array_equal(left[key],right[key]))}
                for key in ('max_rp_corr','edge_fraction','high_freq_content','csf_fraction')}
            # Both adapters expose zero-based component IDs.
            report['results']['aroma_features'+suffix]['noise_equal'] = bool(np.array_equal(np.sort(left['noise_indices']),np.sort(right['noise_indices'])))
        else: report['unavailable'].append('aroma_features'+suffix)
        for case in ('drift','motion6','motion12','motion24','tissue','global','all','all_bandpass'):
            name='confounds_'+case+suffix
            rp=official/(name+'_afni')/'cleaned.nii.gz'
            if rp.exists():report['results'][name] = images(baseline/(name+'_fnit')/'cleaned.nii.gz',rp,mask)
            else:report['unavailable'].append(name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'compared':len(report['results']),'unavailable':report['unavailable']}))


if __name__ == '__main__':
    main()
