"""Run the benchmark with actual preloaded objects and compare its saved CLI outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

import nibabel as nib
import numpy as np
import fnit.synthmorph as synthmorph


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', required=True)
    p.add_argument('--cli', required=True)
    args, remaining = p.parse_known_args()
    started = time.perf_counter()
    images = []
    for name in ('moving', 'fixed'):
        source = nib.load(remaining[remaining.index('--' + name) + 1])
        image = nib.Nifti1Image(np.array(np.asanyarray(source.dataobj), copy=True),
                               source.affine.copy(), source.header.copy())
        if image.get_filename() is not None or not isinstance(image.dataobj, np.ndarray):
            raise RuntimeError('input was not materialized')
        images.append(image)
    input_seconds = time.perf_counter() - started
    before = [array_digest(image.dataobj) for image in images]
    original = synthmorph.SynthMorph

    class MaterializedMorph(original):
        def __call__(self, moving, fixed, **keywords):
            return super().__call__(*images, **keywords)

    synthmorph.SynthMorph = MaterializedMorph
    sys.argv = [args.worker, *remaining]
    try:
        runpy.run_path(args.worker, run_name='__main__')
    finally:
        synthmorph.SynthMorph = original
    output = Path(remaining[remaining.index('--output') + 1])
    cli = Path(args.cli)
    rows = {}
    for name, cli_name in [('moved', 'moved'), ('fixed_moved', 'fixed_moved'),
                           ('transform', 'forward'), ('inverse', 'inverse')]:
        left, right = nib.load(output / (name + '.nii.gz')), nib.load(cli / (cli_name + '.nii.gz'))
        rows[name] = {'same_array': bool(np.array_equal(np.asanyarray(left.dataobj), np.asanyarray(right.dataobj))),
                      'same_header': left.header.binaryblock == right.header.binaryblock,
                      'same_affine': bool(np.array_equal(left.affine, right.affine)),
                      'same_extensions': left.header.extensions == right.header.extensions}
    unchanged = before == [array_digest(image.dataobj) for image in images]
    report = {'scope': 'complete preloaded materialized-image API; decoding outside model/API timers',
              'input_load_and_materialize_seconds': input_seconds, 'input_unchanged': unchanged,
              'rows': rows, 'all_gates_passed': unchanged and all(all(row.values()) for row in rows.values()),
              'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'worker_sha256': hashlib.sha256(Path(args.worker).read_bytes()).hexdigest()}
    (output / 'materialized.private.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_gates_passed']:
        raise RuntimeError('complete materialized-image/CLI contract failed')


if __name__ == '__main__':
    main()
