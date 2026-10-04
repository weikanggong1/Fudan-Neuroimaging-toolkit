"""Rescore already saved function outputs at the NIfTI serialization boundary.

The v1 harness compared a pre-save float64 affine to a saved NIfTI float32
sform. Its raw report remains unchanged. This posthoc correction runs no CNN.
"""
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
    workspace = root/'workspaces/smri_cpu_20261004'
    output = root/'runs/smri_cpu_20261004/remaining_20261004/synth/sr_functions_node7_v1'
    raw = output/'report.public.json'
    report = json.loads(raw.read_text())
    jobs = json.loads((workspace/'task1/paired_baseline_v2.private.json').read_text())['jobs']
    report['raw_harness_report_sha256'] = digest(raw)
    report['posthoc_collector_sha256'] = digest(__file__)
    report['scoring_correction'] = 'Compare saved candidate/reference NIfTI affine/header at the same serialization boundary. Raw v1 pre-save float64 affine flags retained separately; no repeated inference.'
    for item in report['cases']:
        case = item['case']
        reference_path = next(job for job in jobs if job['id']==case+'_1_reference')['expected_outputs'][0]
        candidate = nib.load(str(output/(case+'.nii.gz'))); reference = nib.load(reference_path)
        values = np.asarray(candidate.dataobj); ref = np.asarray(reference.dataobj)
        delta = np.abs(values.astype(np.int16)-ref.astype(np.int16))
        item['raw_pre_save_affine_exact'] = item['affine_exact']
        item['affine_exact'] = bool(np.array_equal(candidate.affine,reference.affine))
        item['header_exact'] = bool(candidate.header.binaryblock==reference.header.binaryblock)
        item['different'] = int(np.count_nonzero(delta)); item['max_abs'] = int(delta.max())
        assert digest(reference_path) == item['reference_file_sha256']
        item['fixed_quantized_gate']['passes'] = bool(1-item['different']/delta.size >= .9999
            and item['max_abs'] <= 1 and float(delta.mean()) <= 1e-4
            and item['affine_exact'] and values.dtype == ref.dtype)
    report['all_passed'] = all(item['fixed_quantized_gate']['passes'] for item in report['cases'])
    (output/'rescored.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
