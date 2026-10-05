"""Candidate-only real CPU function gates against immutable official files.

No production imports this script. Reference software is never executed here.
The original per-function uint8 tolerance is retained; no missing official
floating reference is manufactured from the candidate.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import nibabel as nib
import numpy as np
import torch

from fnit.synthsr import SynthSR
from fnit.synthsr import _cpu_inference
from fnit.weights import WEIGHT_FILES


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    root = Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
    workspace = root/'workspaces/smri_cpu_20261004'
    runs = root/'runs/smri_cpu_20261004'
    output = runs/'remaining_20261004/synth/sr_functions_node7_v1'
    output.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((workspace/'task1/paired_baseline_v2.private.json').read_text())['jobs']
    cases = ('case01_sr_v1', 'case01_sr_lowfield_model',
             'case01_sr_no_flip', 'case01_sr_no_sharpen')
    original = json.loads((runs/'task1/sr_function_public_v2/summary.public.json').read_text())
    old_cases = {item['case']: item for item in original['cases']}
    torch.set_num_threads(8)
    report = {'scope': 'candidate-only complete real function API, original saved uint8 oracle; no reference rerun',
              'host': 'nodecw7', 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'torch_threads': torch.get_num_threads(), 'torch_version': torch.__version__,
              'source_sha256': {path.name: digest(path) for path in
                  (Path(_cpu_inference.__file__).parent/'model.py',
                   Path(_cpu_inference.__file__), Path(_cpu_inference.__file__).parent/'_cpu_math.py')},
              'driver_sha256': digest(__file__), 'cases': []}
    for case in cases:
        job = next(item for item in jobs if item['id'] == case+'_2_baseline')
        argv = job['argv']; input_path = Path(argv[argv.index('--i')+1])
        ref_job = next(item for item in jobs if item['id'] == case+'_1_reference')
        reference_path = Path(ref_job['expected_outputs'][0])
        old = old_cases[case]['comparisons'][0]['result']['outputs']['image']
        assert digest(reference_path) == old['reference_sha256']
        v1 = '--v1' in argv; lowfield = '--lowfield' in argv
        weight_name = ('synthsr_v10_210712.h5' if v1 else
                       'synthsr_lowfield_v20_230130.h5' if lowfield else 'synthsr_v20_230130.h5')
        checkpoint = workspace/'assets/weights'/weight_name
        expected_size, expected_hash = WEIGHT_FILES[weight_name][1:]
        assert checkpoint.stat().st_size == expected_size and digest(checkpoint) == expected_hash
        started = time.perf_counter()
        model = SynthSR(weights=checkpoint, device='cpu', threads=8, v1=v1, lowfield=lowfield)
        constructor = time.perf_counter()-started
        with torch.inference_mode():
            assert _cpu_inference._eligible(model.model, torch.zeros(1,1,16,16,16))
        started = time.perf_counter()
        result = model(input_path, disable_flipping='--disable_flipping' in argv,
                       disable_sharpening='--disable_sharpening' in argv)
        api = time.perf_counter()-started
        destination = output/(case+'.nii.gz')
        result.image.save(destination)
        reference = nib.load(str(reference_path)); ref = np.asarray(reference.dataobj)
        diff = np.abs(result.image.data.astype(np.int16)-ref.astype(np.int16))
        different = int(np.count_nonzero(diff)); maximum = int(diff.max())
        mae = float(diff.mean()); exact = 1-different/diff.size
        geometry = bool(np.array_equal(result.image.affine, reference.affine))
        dtype = bool(result.image.data.dtype == ref.dtype)
        passes = bool(exact >= .9999 and maximum <= 1 and mae <= 1e-4 and geometry and dtype)
        item = {'case': case, 'input_sha256': digest(input_path),
                'weight_size_bytes': expected_size, 'weight_sha256': expected_hash,
                'reference_file_sha256': old['reference_sha256'], 'output_file_sha256': digest(destination),
                'options': {'v1': v1, 'lowfield': lowfield,
                    'disable_flipping': '--disable_flipping' in argv,
                    'disable_sharpening': '--disable_sharpening' in argv},
                'shape': list(ref.shape), 'dtype': str(ref.dtype), 'count': diff.size,
                'different': different, 'max_abs': maximum, 'mae': mae,
                'affine_exact': geometry, 'dtype_exact': dtype,
                'old_different': old['whole_grid']['different_voxels'],
                'fixed_quantized_gate': {'exact_fraction_min': .9999, 'max_abs_max': 1,
                    'mae_max': 1e-4, 'passes': passes},
                'floating_gate': 'not_assessed: no saved original per-function floating oracle',
                'constructor_seconds': constructor, 'api_seconds': api,
                'load_after': list(os.getloadavg())}
        report['cases'].append(item)
        (output/'report.public.json').write_text(json.dumps(report, indent=2)+'\n')
        print(case, different, passes, flush=True)
        del model, result
    report['maximum_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['all_passed'] = all(item['fixed_quantized_gate']['passes'] for item in report['cases'])
    report['status'] = 'complete'
    (output/'report.public.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
