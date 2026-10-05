"""Inspect persisted recipe records and hashes after connection recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    base = args.root
    workspace = base / 'workspaces/smri_cpu_20261004/remaining_20261004/gems/cpu-epsilon-v1'
    run = base / 'runs/smri_cpu_20261004/remaining_20261004/gems/cpu-epsilon-v1'
    frozen = json.loads((workspace / 'source.public.json').read_text())
    binding = json.loads((run / 'recipe_binding.public.json').read_text())
    candidate = workspace / 'source/src'
    baseline = Path(frozen['baseline_source'])
    if not baseline.is_absolute():
        baseline = workspace.parent / baseline
    if not baseline.name == 'src':
        baseline /= 'src'
    verified = {}
    for tag, source in [('candidate', candidate), ('baseline', baseline)]:
        expected = binding['sources'][tag]
        actual = {name: digest(source / name) for name in expected}
        verified[tag] = {'file_count': len(expected), 'all_sha256_equal': actual == expected,
                         'differences': [name for name in expected if actual[name] != expected[name]],
                         'gems_core_sha256': actual['fnit/gems/core.py']}
    records = {}
    for path in sorted((run / 'recipes').glob('*/record.json')):
        record = json.loads(path.read_text())
        records[path.parent.name] = {key: record.get(key) for key in (
            'status', 'started_utc', 'finished_utc', 'wall_seconds', 'returncode',
            'maximum_sampled_tree_rss_bytes', 'maximum_sampled_tree_threads',
            'load_before', 'load_after', 'cpu_affinity')}
        records[path.parent.name]['record_sha256'] = digest(path)
    repo_commit = subprocess.check_output(['git', '-C', str(base / 'repo'),
                                           'rev-parse', 'HEAD'], text=True, timeout=30).strip()
    output = {'scope': 'read-only persistence/source inspection; no fit or source changes',
              'formal_repo_commit': repo_commit,
              'formal_repo_gems_core_sha256': digest(base / 'repo/src/fnit/gems/core.py'),
              'fixed_entry_sha256': {name: digest(base / name) for name in ('README.md', 'INDEX.json')},
              'source_verification': verified,
              'actual_gems_sources': {name: digest(candidate / name) for name in frozen['gems_source_files']},
              'helper_verification': {name: {'expected': checksum,
                                           'actual': digest(workspace / name),
                                           'exact': checksum == digest(workspace / name)}
                                      for name, checksum in binding['helpers_sha256'].items()},
              'recipe_records': records,
              'controller_tail': (run / 'recipe_controller.log').read_text().splitlines()[-12:],
              'collector': json.loads((run / 'posthoc_collector.public.json').read_text()),
              'posthoc_outputs': {path.name: {'sha256': digest(path), 'bytes': path.stat().st_size}
                                  for path in sorted(run.glob('*-*.public.json'))
                                  if path.name.startswith(('score-', 'state-'))}}
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
