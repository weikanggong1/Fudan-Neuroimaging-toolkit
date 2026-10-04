"""Compare frozen real-input CUDA regression records and NIfTI headers."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np


def metadata(path):
    image = nib.load(path)
    return {'shape': image.shape, 'dtype': str(image.get_data_dtype()),
            'affine': image.affine, 'pixdim': image.header['pixdim'],
            'qform': image.get_qform(), 'sform': image.get_sform(),
            'qform_code': int(image.header['qform_code']),
            'sform_code': int(image.header['sform_code']),
            'units': image.header.get_xyzt_units()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.runs)
    rows = []
    for model in ('rigid', 'affine', 'deform', 'joint'):
        baseline = root / 'task3_gpu_v1' / (model + '_0_baseline') / 'artifacts'
        # The first v3 plan had an artifact suffix typo. Its source hash is v3.
        candidate = (root / 'task3_gpu_v3/rigid_candidate_v2/artifacts'
                     if model == 'rigid' else root / 'task3_gpu_v3_resume' /
                     (model + '_candidate_v3_resume') / 'artifacts')
        b = json.loads((baseline / 'report.private.json').read_text())
        c = json.loads((candidate / 'report.private.json').read_text())
        arrays = {key: item['sha256'] == c['array_files'][key]['sha256']
                  for key, item in b['array_files'].items()}
        headers = {}
        names = ('moved', 'fixed_moved') + (('transform', 'inverse')
                                          if model in ('deform', 'joint') else ())
        for name in names:
            bm = metadata(baseline / (name + '.nii.gz'))
            cm = metadata(candidate / (name + '.nii.gz'))
            headers[name] = {key: bool(np.array_equal(bm[key], cm[key]))
                             for key in bm}
        rows.append({'model': model, 'arrays_equal': arrays,
                     'metadata_equal': headers,
                     'same_inputs': b['input_sha256'] == c['input_sha256'],
                     'baseline_api_seconds': b['api_seconds'],
                     'candidate_v3_api_seconds': c['api_seconds'],
                     'peak_cuda_reserved_bytes': c['peak_cuda_reserved_bytes'],
                     'under20GB': c['peak_cuda_reserved_bytes'] < 20_000_000_000,
                     'source_sha256': c['source_sha256'],
                     'candidate_artifacts': str(candidate.relative_to(root))})
    report = {'scope': 'same-device full real input regression v3 against frozen original GPU paths; allocator reserved only; single v3 timing is not a paired speed estimate',
              'rows': rows,
              'all_equal': all(row['same_inputs'] and all(row['arrays_equal'].values())
                               and all(all(h.values()) for h in row['metadata_equal'].values())
                               for row in rows)}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
