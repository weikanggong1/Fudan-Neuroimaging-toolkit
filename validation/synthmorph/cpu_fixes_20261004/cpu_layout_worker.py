"""Real CPU precision diagnostic for the original feature tensor layout."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import torch
from fnit.synthmorph import models


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', required=True)
    args, remaining = p.parse_known_args()
    if remaining[remaining.index('--device') + 1] != 'cpu':
        raise ValueError('this isolated diagnostic only supports CPU')
    original_init = models.SynthMorphNetwork.__init__
    original_feature = models.FeatureDetector.forward
    original_deform = models.DeformNetwork.forward

    def initialize(self, *positional, **keywords):
        original_init(self, *positional, **keywords)
        for module in self.modules():
            if isinstance(module, torch.nn.Conv3d):
                module.to(memory_format=torch.channels_last_3d)

    def feature(self, tensor):
        return original_feature(self, tensor.contiguous(memory_format=torch.channels_last_3d))

    def deform(self, moving, fixed):
        return original_deform(self, moving.contiguous(memory_format=torch.channels_last_3d),
                               fixed.contiguous(memory_format=torch.channels_last_3d))

    models.SynthMorphNetwork.__init__ = initialize
    models.FeatureDetector.forward = feature
    models.DeformNetwork.forward = deform
    sys.argv = [args.worker, *remaining]
    runpy.run_path(args.worker, run_name='__main__')
    output = Path(remaining[remaining.index('--output') + 1])
    record = {'scope': 'isolated CPU channels_last_3d precision diagnostic; no production change',
              'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'worker_sha256': hashlib.sha256(Path(args.worker).read_bytes()).hexdigest()}
    (output / 'layout.private.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
