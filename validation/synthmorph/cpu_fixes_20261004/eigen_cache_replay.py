"""Replay eight saved real affine matrices through old and guarded own adapters."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
from fnit.synthmorph import _cpu_eigen as candidate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'old-loader', 'old-cache', 'new-cache', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('fnit_old_cpu_eigen_adapter', args.old_loader)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    base = Path(args.root) / 'runs/smri_cpu_20261004/remaining_20261004/morph'
    matrices, sources = [], []
    for extent in (192, 256):
        path = base / 'nodecw7_fit_v14' / ('extent' + str(extent)) / 'artifacts/stages.npz'
        with np.load(path) as values:
            for name in ('fit_0', 'fit_1', 'average', 'inverse'):
                matrices.append(values[name])
                sources.append({'extent': extent, 'key': name, 'saved_stages_sha256': digest(path),
                                'input_array_sha256': hashlib.sha256(values[name].tobytes()).hexdigest()})
    inputs = np.concatenate(matrices)
    os.environ['FNIT_SYNTHMORPH_BUILD_CACHE'] = args.old_cache
    original = previous.affine_sqrt(inputs)
    os.environ['FNIT_SYNTHMORPH_BUILD_CACHE'] = args.new_cache
    start = time.monotonic()
    actual = candidate.affine_sqrt(inputs)
    build_seconds = time.monotonic() - start
    second_start = time.monotonic()
    repeated = candidate.affine_sqrt(inputs)
    repeated_seconds = time.monotonic() - second_start
    for index, row in enumerate(sources):
        row['old_array_sha256'] = hashlib.sha256(original[index].tobytes()).hexdigest()
        row['new_array_sha256'] = hashlib.sha256(actual[index].tobytes()).hexdigest()
        row['bytes_equal'] = row['old_array_sha256'] == row['new_array_sha256']
    manifests = list(Path(args.new_cache).glob('*/build.json'))
    if len(manifests) != 1:
        raise RuntimeError('ambiguous guarded build cache')
    build = json.loads(manifests[0].read_text())
    report = {'scope': __doc__, 'matrices': sources, 'first_build_and_call_seconds': build_seconds,
              'second_cached_call_seconds': repeated_seconds,
              'new_loader_sha256': digest(candidate.__file__), 'old_loader_sha256': digest(args.old_loader),
              'adapter_cpp_sha256': build['identity']['source_sha256'],
              'binary_sha256': build['binary_sha256'], 'compiler_sha256': build['identity']['compiler_sha256'],
              'compiler_version': build['identity']['compiler_version'], 'flags': build['identity']['flags'],
              'cache_root_mode': oct(Path(args.new_cache).stat().st_mode & 0o777),
              'cache_key_mode': oct(manifests[0].parent.stat().st_mode & 0o777),
              'cache_file_modes': {path.name: oct(path.stat().st_mode & 0o777)
                                   for path in manifests[0].parent.iterdir()},
              'repeated_call_bytes_equal': actual.tobytes() == repeated.tobytes(),
              'worker_sha256': digest(__file__)}
    report['all_gates_passed'] = (all(row['bytes_equal'] for row in sources)
                                  and report['repeated_call_bytes_equal']
                                  and report['cache_root_mode'] == '0o700'
                                  and report['cache_key_mode'] == '0o700'
                                  and all(mode == '0o600' for mode in report['cache_file_modes'].values()))
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_gates_passed']:
        raise RuntimeError('guarded adapter replay failed')


if __name__ == '__main__':
    main()
