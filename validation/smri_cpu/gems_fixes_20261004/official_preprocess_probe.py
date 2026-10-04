"""Isolated original-software preprocessing; never imported by FNIT runtime."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t1', required=True)
    parser.add_argument('--aseg', required=True)
    parser.add_argument('--wmparc', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be a new directory')
    from samseg.subregions.thalamus import ThalamicNuclei
    from samseg.subregions.hippocampus import HippoAmygdalaSubfields
    import samseg.subregions.core as core
    import samseg.subregions.thalamus as thalamus
    import samseg.subregions.hippocampus as hippocampus

    report = {'scope': 'original_preprocessing_only_no_mesh_fit', 'sources': {}, 'inputs': {}, 'structures': {}}
    for name, module in [('core', core), ('thalamus', thalamus), ('hippocampus', hippocampus)]:
        path = Path(module.__file__)
        report['sources'][name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in ['t1', 'aseg', 'wmparc']:
        path = Path(getattr(args, name))
        report['inputs'][name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in ['thalamus', 'hippo-amygdala-left', 'hippo-amygdala-right']:
        output = args.output / name
        options = dict(outDir=str(output), inputImageFileNames=[args.t1], inputSegFileName=args.aseg,
                       tempDir=str(output / 'temporary'), debug=True)
        model = (ThalamicNuclei(**options) if name == 'thalamus' else
                 HippoAmygdalaSubfields(side=name.rsplit('-', 1)[1], wmParcFileName=args.wmparc, **options))
        model.initialize()
        records = {}
        for field in ['processedImage', 'synthImage', 'maskDilated5mm', 'longMask']:
            image = getattr(model, field, None)
            if image is None:
                continue
            if isinstance(image, np.ndarray):
                image = model.processedImage.new(image)
            image.save(str(output / (field + '.mgz')))
            records[field] = {'shape': list(image.shape), 'vox2world': image.geom.vox2world.matrix.tolist(),
                              'dtype': str(image.data.dtype), 'nonzero': int(np.count_nonzero(image.data)),
                              'array_sha256': hashlib.sha256(np.ascontiguousarray(image.data).tobytes()).hexdigest()}
        report['structures'][name] = records
    (args.output / 'report.public.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
