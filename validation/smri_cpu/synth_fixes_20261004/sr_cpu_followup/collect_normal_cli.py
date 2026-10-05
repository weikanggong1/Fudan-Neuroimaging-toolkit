"""Posthoc normal CLI comparison; never runs inside the CLI wall timer."""
import hashlib
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
    runs = root/'runs/smri_cpu_20261004'
    output = runs/'remaining_20261004/synth'
    reference_path = runs/'task1/sr_network_control_v2/reference/image.nii.gz'
    reference = nib.load(str(reference_path)); ref = np.asarray(reference.dataobj)
    records = []
    for arm in ('official1', 'old1', 'new1', 'new2', 'old2', 'official2'):
        folder = output/('sr_normal_cli_node7_v1_'+arm)
        metadata = json.loads((folder/'metadata.json').read_text())
        time_text = (folder/'time.txt').read_text()
        def field(name):
            return next(row.split(': ',1)[1] for row in time_text.splitlines() if name in row)
        clock = field('Elapsed (wall clock)').split(':')
        wall = sum(float(value)*60**index for index,value in enumerate(reversed(clock)))
        image = nib.load(str(folder/'image.nii.gz')); values = np.asarray(image.dataobj)
        diff = np.abs(values.astype(np.int16)-ref.astype(np.int16))
        item = {'arm': arm, **metadata, 'gnu_wall_seconds': wall,
                'gnu_user_seconds': float(field('User time (seconds)')),
                'gnu_system_seconds': float(field('System time (seconds)')),
                'gnu_maximum_rss_bytes': int(field('Maximum resident set size'))*1024,
                'exit_status': int(field('Exit status')),
                'reference_sha256': digest(reference_path), 'output_sha256': digest(folder/'image.nii.gz'),
                'count': diff.size, 'different': int(np.count_nonzero(diff)), 'max_abs': int(diff.max()),
                'affine_exact': bool(np.array_equal(image.affine,reference.affine)),
                'header_exact': bool(image.header.binaryblock == reference.header.binaryblock),
                'dtype': str(values.dtype)}
        if arm.startswith('official') or arm.startswith('new'):
            assert item['different'] == 0 and item['affine_exact'] and item['header_exact']
        records.append(item)
    source = root/'workspaces/smri_cpu_20261004/remaining_20261004/synth'
    report = {'schema':'fnit.synthsr.cpu.normal_cli.v1', 'scope':'normal uninstrumented public CLI, GNU wall includes cold process imports/loading, new empty JIT cache and one NIfTI save',
              'comparison_timing':'all hashes, original-array/geometry/header comparisons after all CLI processes completed; excluded from GNU wall',
              'input_sha256':digest(runs/'inputs/ds003138/case01_T1w.nii.gz'),
              'weight_sha256':digest(root/'workspaces/smri_cpu_20261004/assets/weights/synthsr_v20_230130.h5'),
              'source_sha256':{arm:{name:digest(source/folder/'src/fnit/synthsr'/name) for name in names} for arm,folder,names in
                  [('old','source_v10',['model.py']),('new','source_sr_v2',['model.py','_cpu_inference.py','_cpu_math.py'])]},
              'driver_sha256':digest(source/'sr_cli_v1/cpu_cli_abba.sh'), 'collector_sha256':digest(__file__),
              'records':records, 'status':'complete'}
    (output/'sr_normal_cli_node7_v1.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
