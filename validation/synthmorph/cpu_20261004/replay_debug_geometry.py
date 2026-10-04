"""Replay current debug writer on saved real network inputs, without rerunning NN."""
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from fnit.synthmorph import pipeline
from compare_registration import image_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--moving', required=True)
    parser.add_argument('--fixed', required=True)
    parser.add_argument('--saved-inputs', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    moving, fixed = nib.load(args.moving), nib.load(args.fixed)
    saved = Path(args.saved_inputs)
    old = [nib.load(saved/(name+'.nii.gz')) for name in ('inp_1','inp_2')]
    inputs = tuple(torch.from_numpy(np.asarray(image.dataobj, dtype=np.float32))[None,None]
                   for image in old)
    transforms = (np.linalg.inv(moving.affine) @ old[0].affine,
                  np.linalg.inv(fixed.affine) @ old[1].affine)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = {}
    for device in ('cpu','cuda:0'):
        images = pipeline._network_input_images(inputs, moving, fixed, *transforms,
                                                device, has_init=True)
        prefix = 'CPU_repaired' if device == 'cpu' else 'CUDA_legacy_metadata'
        rows[prefix] = {}
        for name, image in zip(('inp_1','inp_2'), images):
            path = output/(prefix+'_'+name+'.nii.gz')
            nib.save(image, path)
            target = Path(args.reference)/(name+'.nii.gz') if device=='cpu' else saved/(name+'.nii.gz')
            result = image_pair(path, target)
            rows[prefix][name] = result
    report = {
        'scope': 'saved actual normalized real network inputs from the measured deform192/init CLI; current helper executes only debug image construction; no repeat NN or new inference clock. CPU compared to original official debug, CUDA geometry compared to existing FNIT debug geometry without allocating GPU tensors.',
        'rows': rows,
        'pipeline_sha256': hashlib.sha256(Path(pipeline.__file__).read_bytes()).hexdigest(),
        'data_unmodified': True,
    }
    (output/'report.private.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
