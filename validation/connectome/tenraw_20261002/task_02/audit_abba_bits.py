"""Audit every full ABBA checkpoint on CPU, including zero signs and NaN payloads."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    directory = Path(args.directory)
    formal = json.loads((directory / 'report.json').read_text())
    if not formal.get('formal_ABBA_complete') or formal.get('state') != 'completed':
        raise ValueError('Full ABBA and real-block diagnostic must finish before byte audit')
    torch.set_num_threads(8)
    reference = torch.load(args.reference, map_location='cpu', weights_only=True)
    results = []
    for run in formal['runs']:
        path = directory / f"{run['sequence']}_{run['version']}.pt"
        candidate = torch.load(path, map_location='cpu', weights_only=True)
        if candidate.keys() != reference.keys():
            raise ValueError(f'{path.name}: output keys differ')
        mismatches = {}
        for key, left in reference.items():
            right = candidate[key]
            if left.dtype != right.dtype or left.shape != right.shape:
                raise ValueError(f'{path.name}: {key} dtype or shape differs')
            left_bytes = left.reshape(-1).contiguous().view(torch.uint8)
            right_bytes = right.reshape(-1).contiguous().view(torch.uint8)
            different = (left_bytes != right_bytes).reshape(left.numel(), left.element_size()).any(dim=1)
            count = int(different.sum())
            if count:
                mismatches[key] = count
        results.append({'sequence': run['sequence'], 'version': run['version'],
                        'checkpoint_sha256': sha256(path), 'tensor_count': len(candidate),
                        'bitwise_mismatches': mismatches, 'passed': not mismatches})
        del candidate
    report = {'scope': 'full saved ABBA CPU tensor bytes; signed zeros and NaN payloads included',
              'harness_sha256': sha256(__file__), 'formal_report_sha256': sha256(directory / 'report.json'),
              'reference': str(Path(args.reference)), 'reference_sha256': sha256(args.reference),
              'runs': results, 'passed': all(run['passed'] for run in results)}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise RuntimeError('Bitwise mismatch; see saved audit')
    print(args.output)


if __name__ == '__main__':
    main()
