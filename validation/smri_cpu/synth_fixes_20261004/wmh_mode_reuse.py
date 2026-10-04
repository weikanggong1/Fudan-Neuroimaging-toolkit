"""Uncapped numerical control for one public model reused across GPU modes.

This preserves the full-view legacy CUDA path; it is not a 20 GB gate.
Run after the isolated CUDA initialization probe, under the common GPU flock.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from fnit.wmh_synthseg import WMHSynthSeg
from fnit.wmh_synthseg.pipeline import _write_volumes_csv


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('preserve previous attempts; choose a fresh directory')
    args.output_dir.mkdir(parents=True)
    assert args.weights.stat().st_size == 790531383
    assert digest(args.weights) == '0ece39dd651357aa95222fc4d45fa32d00f11e763d2583cae3f869989ce35988'
    torch.set_num_threads(8)
    model = WMHSynthSeg(weights=args.weights, device='cuda:0', threads=8)
    report = {'purpose': 'uncapped numerical control; legacy full-view is not within 20 GB',
              'budget_bytes': None, 'source_revision': args.source_revision,
              'input_sha256': digest(args.input), 'weight_sha256': digest(args.weights),
              'driver_sha256': digest(__file__), 'torch_version': torch.__version__, 'calls': []}
    for name in ('model', 'pipeline', 'spatial'):
        module = __import__('fnit.wmh_synthseg.'+name, fromlist=[''])
        report[name+'_sha256'] = digest(module.__file__)
    expected = {
        False: ['815ea52a6f01cc4ab950c03b7ecd4e7e150b31c9d3e50b34d8200358125fef08',
                '8fe276eb3fcd9b7dd835f7b3a69738fbbb6cee0972689e4c9168cc9e141166b6',
                '7eff80e5263257f6a4bc32d4d8bef2c9c36089755a29c27b1e5ea8cb602a70ec'],
        True: ['ca94a9caf118205a426a1dc7ad468b013716bd9b452b8a441f1258a053d1672f',
               'cc4cc54d9bcd0d20162f154b7e6b4c7216c723d8eed5d1537c654d75137cc893',
               'c26284e2fe67368f58e1d771e8a8f83fd21db9c18552b8835fc4efaecfbd6cf7']}
    try:
        for index, crop in enumerate((False, True, False)):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            before = model.model._memory_efficient_inference
            start = time.perf_counter()
            result = model(args.input, crop=crop, save_lesion_probabilities=True)
            torch.cuda.synchronize(0)
            call = {'crop': crop, 'api_seconds': time.perf_counter()-start,
                    'mode_before': before, 'mode_after': model.model._memory_efficient_inference,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(0)}
            output = args.output_dir/str(index)
            output.mkdir()
            result.segmentation.save(output/'seg.nii.gz')
            result.lesion_probability.save(output/'lesion.nii.gz')
            _write_volumes_csv(result.volumes_mm3, 'anonymous_seg.nii.gz', output/'volumes.csv')
            names = ('seg.nii.gz', 'lesion.nii.gz', 'volumes.csv')
            call['output_sha256'] = {name: digest(output/name) for name in names}
            call['matches_original_gpu'] = list(call['output_sha256'].values()) == expected[crop]
            report['calls'].append(call)
            assert call['mode_before'] == call['mode_after']
            assert call['matches_original_gpu']
            del result
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', exception_type=type(error).__name__, exception=str(error))
        raise
    finally:
        (args.output_dir/'report.public.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
