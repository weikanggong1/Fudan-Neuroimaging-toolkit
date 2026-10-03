"""Read source identities and completed producer receipts; never run solvers/tests."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


CONFIGURATION_SHA = 'f8eba1cbf3eab2baa549172b32be2c9d3b1014ad478f6e3702d7987378f52f8c'
SCIENCE_COMMIT = '1fe86ab8347b29d9c47be8109627736576222912'
BASELINE_COMMIT = '7af34e6d072e843fb2558c931bb2781f1d4b0be9'
CHANGED_FILES = ['src/fnit/cli.py', 'src/fnit/connectome/mtnormalise.py',
                 'src/fnit/connectome/pipeline.py', 'src/fnit/connectome/response.py',
                 'src/fnit/connectome/tracking.py']


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_inventory(directory):
    directory = Path(directory)
    files = sorted(path for path in (directory / 'src/fnit').rglob('*')
                   if path.is_file() and path.suffix not in {'.pyc', '.pyo'}
                   and '__pycache__' not in path.parts)
    files += [directory / name for name in ('pyproject.toml', 'environment.yml')
              if (directory / name).is_file()]
    hashes = {str(path.relative_to(directory)): digest(path.read_bytes()) for path in files}
    return hashes, digest(json.dumps(hashes, sort_keys=True).encode())


def source_check(directory, expected):
    hashes, fingerprint = source_inventory(directory)
    declared = expected['source_sha256']
    mismatches = [name for name in sorted(set(hashes) | set(declared))
                  if hashes.get(name) != declared.get(name)]
    require(not mismatches and fingerprint == expected['source_fingerprint'],
            'source bytes differ from frozen declaration: ' + str(mismatches))
    return {'directory': str(Path(directory).resolve()), 'file_count': len(hashes),
            'source_fingerprint': fingerprint, 'mismatches': mismatches}


def git(repository, *arguments):
    return subprocess.check_output(['git', *arguments], cwd=repository)


def ast_signature(raw):
    return digest(ast.dump(ast.parse(raw.decode()), include_attributes=False).encode())


def git_check(repository):
    source_paths = ['src/fnit', 'pyproject.toml', 'environment.yml']
    science_diff = git(repository, 'diff', '--name-only', SCIENCE_COMMIT, 'HEAD', '--', *source_paths).decode().splitlines()
    working_diff = git(repository, 'diff', 'HEAD', '--name-only', '--', *source_paths).decode().splitlines()
    require(not science_diff and not working_diff, 'repository science differs from frozen Git origin')
    changed = git(repository, 'diff', '--name-only', BASELINE_COMMIT, 'HEAD', '--', *source_paths).decode().splitlines()
    require(sorted(changed) == sorted(CHANGED_FILES), 'unexpected production/dependency change')
    rows = {}
    for name in CHANGED_FILES:
        current = (Path(repository) / name).read_bytes()
        origin = git(repository, 'show', SCIENCE_COMMIT + ':' + name)
        require(current == origin, 'science origin bytes changed: ' + name)
        rows[name] = {'current_sha256': digest(current), 'science_origin_sha256': digest(origin),
                      'current_ast_sha256': ast_signature(current), 'science_origin_ast_sha256': ast_signature(origin)}
    return {'head': git(repository, 'rev-parse', 'HEAD').decode().strip(),
            'science_origin_commit': SCIENCE_COMMIT, 'baseline_commit': BASELINE_COMMIT,
            'science_origin_diff': science_diff, 'working_science_diff': working_diff,
            'baseline_changed_files': changed, 'five_files_byte_and_ast_identity': rows,
            'dependency_files_unchanged': True}


def producer_check(config, require_complete):
    root = Path(config['run_root'])
    require((root / 'status.json').is_file(), 'actual phase status is unavailable')
    state_raw = (root / 'status.json').read_bytes()
    state = json.loads(state_raw)
    cases = {}
    for plan in config['execution_order']:
        version, case_id = plan['version'], plan['case_id']
        key = version + '/' + case_id
        path = root / version / case_id / 'gpu_report.json'
        if not path.is_file():
            cases[key] = {'status': 'not_started'}
            continue
        raw = path.read_bytes()
        actual = json.loads(raw)
        entry = {'status': actual.get('status'), 'path': str(path), 'sha256': digest(raw)}
        if actual.get('status') == 'completed':
            require(actual.get('version') == version and actual.get('case_id') == case_id,
                    'completed producer case/version differs: ' + key)
            expected = config['declared_source_manifests'][version]
            for when in ('source_before', 'source_after'):
                observed = actual[when]
                require(observed['source_fingerprint'] == expected['source_fingerprint'] and
                        observed['source_sha256'] == expected['source_sha256'],
                        'completed producer source binding differs: ' + key + '/' + when)
            require(path.read_bytes() == raw, 'completed report changed during audit: ' + key)
            entry.update(source_before=actual['source_before']['source_fingerprint'],
                         source_after=actual['source_after']['source_fingerprint'],
                         all_declared_file_sha256_before_after_match=True)
        cases[key] = entry
    completed = sum(item['status'] == 'completed' for item in cases.values())
    if require_complete:
        require(completed == len(config['execution_order']) and state['status'] == 'execution_completed',
                'final planned producer coverage is not completed')
    return {'phase_status': state['status'], 'phase_status_snapshot_sha256': digest(state_raw),
            'planned': len(config['execution_order']), 'completed': completed, 'cases': cases,
            'final_source_gate': 'completed' if completed == len(cases) and state['status'] == 'execution_completed'
            else 'not_assessed: producers incomplete',
            'scope': 'source identity only; precision, timing and memory acceptance are separate'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True, help='candidate source directory, read only')
    parser.add_argument('--configuration', type=Path, required=True, help='actual immutable configuration')
    parser.add_argument('--repository', type=Path, help='optional Git checkout for origin-byte/AST checks')
    parser.add_argument('--test-directory', type=Path, help='optional existing tested checkout; current identity only')
    parser.add_argument('--check-producers', action='store_true', help='read actual run_root reports without running work')
    parser.add_argument('--require-complete', action='store_true', help='require all planned completed producer bindings')
    parser.add_argument('--output', type=Path, required=True, help='new receipt; existing files are never replaced')
    args = parser.parse_args()
    require(not args.output.exists(), 'receipt already exists')
    require(not args.require_complete or args.check_producers, '--require-complete requires --check-producers')
    raw = args.configuration.read_bytes()
    require(digest(raw) == CONFIGURATION_SHA, 'frozen configuration bytes changed')
    config = json.loads(raw)
    expected = config['declared_source_manifests']['candidate']
    require(config['source_archives']['candidate']['git_origin_commit'] == SCIENCE_COMMIT,
            'declared science Git origin differs')
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'read-only source acceptance preparation; no solver/test execution or data edits',
              'script_sha256': digest(Path(__file__).read_bytes()), 'command': [sys.executable, *sys.argv],
              'configuration_sha256': digest(raw), 'candidate': source_check(args.source, expected)}
    if args.repository:
        result['git'] = git_check(args.repository)
    if args.test_directory:
        result['test_checkout_current'] = source_check(args.test_directory, expected)
        result['test_historical_source_before_after'] = 'not_recorded; current identity is not a historical snapshot'
    if args.check_producers:
        result['actual_producers'] = producer_check(config, args.require_complete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'receipt': str(args.output), 'candidate_source_fingerprint': expected['source_fingerprint'],
                      'producer_source_gate': result.get('actual_producers', {}).get('final_source_gate', 'not_checked')}))


if __name__ == '__main__':
    main()
