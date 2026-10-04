"""Posthoc GPU normal CLI artifact comparison; no inference or benchmark."""
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
    output=root/'runs/smri_cpu_20261004/remaining_20261004/synth'
    source=root/'workspaces/smri_cpu_20261004/remaining_20261004/synth'
    reference_path=output/'sr_normal_gpu_cli_v1_old1/image.nii.gz'
    reference=nib.load(str(reference_path)); ref=np.asarray(reference.dataobj)
    report={'schema':'fnit.synthsr.cuda.normal_cli.v1','scope':'20 GB bootstrap plus actual normal CLI, one NIfTI save, no trace/NPZ/hash/comparison inside inference',
            'posthoc_scope':'all scalar artifact comparisons here excluded from GNU wall',
            'input_sha256':digest(root/'runs/smri_cpu_20261004/inputs/ds003138/case01_T1w.nii.gz'),
            'weight_sha256':digest(root/'workspaces/smri_cpu_20261004/assets/weights/synthsr_v20_230130.h5'),
            'driver_sha256':digest(source/'sr_cli_v1/gpu_normal_cli.py'),
            'shell_sha256':digest(source/'sr_cli_v1/gpu_cli_abba.sh'),'collector_sha256':digest(__file__),
            'records':[],'status':'complete'}
    for arm in ('old1','new1','new2','old2'):
        folder=output/('sr_normal_gpu_cli_v1_'+arm)
        item=json.loads((folder/'report.public.json').read_text());time_text=(folder/'time.txt').read_text()
        def field(name):
            return next(line.split(': ',1)[1] for line in time_text.splitlines() if name in line)
        clock=field('Elapsed (wall clock)').split(':')
        item['arm']=arm
        item['gnu_wall_seconds']=sum(float(v)*60**i for i,v in enumerate(reversed(clock)))
        item['gnu_maximum_rss_bytes']=int(field('Maximum resident set size'))*1024
        item['gnu_user_seconds']=float(field('User time (seconds)'))
        item['gnu_system_seconds']=float(field('System time (seconds)'))
        item['exit_status']=int(field('Exit status'))
        image=nib.load(str(folder/'image.nii.gz'));values=np.asarray(image.dataobj)
        diff=np.abs(values.astype(np.int16)-ref.astype(np.int16))
        item['reference_sha256']=digest(reference_path);item['output_sha256']=digest(folder/'image.nii.gz')
        item['different']=int(np.count_nonzero(diff));item['max_abs']=int(diff.max())
        item['affine_exact']=bool(np.array_equal(image.affine,reference.affine))
        item['header_exact']=bool(image.header.binaryblock==reference.header.binaryblock)
        assert item['different']==0 and item['affine_exact'] and item['header_exact']
        assert item['peak_allocated_bytes']<=20_000_000_000 and item['peak_reserved_bytes']<=20_000_000_000
        assert not item['cpu_math_imported'] and not item['cpu_dispatch_imported']
        report['records'].append(item)
    (output/'sr_normal_gpu_cli_v1.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
