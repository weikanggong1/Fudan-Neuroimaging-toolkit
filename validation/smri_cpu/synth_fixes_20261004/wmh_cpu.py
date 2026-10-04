"""Complete real CPU WMH regression against a saved independent reference."""

import argparse
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from fnit.wmh_synthseg import WMHSynthSeg
from fnit.wmh_synthseg.pipeline import _write_volumes_csv
from wmh_gpu import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--reference-seg', type=Path, required=True)
    p.add_argument('--reference-lesion', type=Path, required=True)
    p.add_argument('--reference-csv', type=Path, required=True)
    p.add_argument('--crop', action='store_true')
    a = p.parse_args()
    if a.output_dir.exists():
        p.error('select a fresh directory')
    a.output_dir.mkdir(parents=True)
    if a.weights.stat().st_size != 790531383 or digest(a.weights) != '0ece39dd651357aa95222fc4d45fa32d00f11e763d2583cae3f869989ce35988':
        raise ValueError('checkpoint differs from fixed size/SHA manifest')
    start = time.perf_counter()
    model = WMHSynthSeg(weights=a.weights, device='cpu', threads=8)
    loaded = time.perf_counter()
    result = model(image=a.input, crop=a.crop, save_lesion_probabilities=True)
    completed = time.perf_counter()
    result.segmentation.save(a.output_dir/'seg.nii.gz')
    result.lesion_probability.save(a.output_dir/'lesion.nii.gz')
    _write_volumes_csv(result.volumes_mm3, 'anonymous_seg.nii.gz', a.output_dir/'volumes.csv')
    saved = time.perf_counter()
    report = {'schema': 'fnit.wmh.real.cpu.memory_fix.v1', 'crop': a.crop,
              'input_sha256': digest(a.input), 'weight_sha256': digest(a.weights),
              'driver_sha256': digest(__file__), 'constructor_seconds': loaded-start,
              'api_seconds': completed-loaded, 'save_seconds': saved-completed,
              'source_sha256': {name: digest(__import__('fnit.wmh_synthseg.'+name, fromlist=['']).__file__)
                                for name in ('model', 'pipeline', 'spatial')}, 'comparison': {}}
    for name, reference, candidate in [('seg', a.reference_seg, result.segmentation),
                                       ('lesion', a.reference_lesion, result.lesion_probability)]:
        expected = nib.load(str(reference))
        x = expected.get_fdata()
        y = candidate.get_fdata()
        if x.shape != y.shape:
            raise ValueError('grid shape differs')
        diff = np.abs(x-y)
        report['comparison'][name] = {'reference_sha256': digest(reference), 'shape': list(x.shape),
            'different': int(np.count_nonzero(diff)), 'max_abs': float(diff.max()),
            'affine_exact': bool(np.array_equal(expected.affine, candidate.affine)),
            'header_geometry_exact': bool(np.array_equal(expected.header['pixdim'],candidate.header['pixdim']))}
        if name == 'seg':
            report['comparison'][name]['label_dice'] = {str(int(label)): float(2*np.count_nonzero((x==label)&(y==label))/(np.count_nonzero(x==label)+np.count_nonzero(y==label)))
                                                       for label in np.union1d(x,y)}
    import csv
    with a.reference_csv.open() as stream:
        rows=list(csv.reader(stream))
    x=np.asarray([float(value) for value in rows[1][1:]])
    with (a.output_dir/'volumes.csv').open() as stream:
        rows=list(csv.reader(stream))
    y=np.asarray([float(value) for value in rows[1][1:]])
    report['comparison']['csv']={'columns':len(x),'different':int(np.count_nonzero(x-y)),
                                 'max_abs':float(np.abs(x-y).max()),'reference_sha256':digest(a.reference_csv)}
    report['output_sha256']={name:digest(a.output_dir/name) for name in ('seg.nii.gz','lesion.nii.gz','volumes.csv')}
    report['status']='complete'
    (a.output_dir/'report.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
