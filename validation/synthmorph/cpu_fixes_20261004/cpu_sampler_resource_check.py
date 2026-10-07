"""Verify real paired inputs and declared external weights before CPU timing."""
import argparse
import hashlib
import json
from pathlib import Path

from fnit.weights import WEIGHT_FILES


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('moving', 'fixed', 'weights', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    expected = {'moving': 'afd1a20fe75fdea44313f0eda05020b916c87234e7a2045f7ccc6bb7c6e90b19',
                'fixed': '73e3866d4e54f9cb253868daab4bf90303a97bc193e8bda21e2e60c53a5dea21'}
    inputs = {name: digest(getattr(args, name)) for name in expected}
    if inputs != expected:
        raise RuntimeError('actual real inputs differ from the accepted same-input reference')
    weights = {}
    for name in ('synthmorph.affine.2.h5', 'synthmorph.deform.3.h5'):
        path = Path(args.weights) / name
        _, size, sha256 = WEIGHT_FILES[name]
        actual = digest(path)
        if path.stat().st_size != size or actual != sha256:
            raise RuntimeError('declared external weight changed: ' + name)
        weights[name] = {'size': size, 'sha256': actual}
    report = {'scope': __doc__, 'data': {'dataset': 'OpenNeuro ds003138 v1.0.1',
              'license': 'CC0', 'input_sha256': inputs}, 'weights': weights,
              'worker_sha256': digest(__file__), 'all_gates_passed': True}
    with Path(args.output).open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
