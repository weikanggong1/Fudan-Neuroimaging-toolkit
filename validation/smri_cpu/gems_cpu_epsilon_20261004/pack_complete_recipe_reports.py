"""Preserve full recipe metrics with lossless local JSON references."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).parent
    encoding = module('encoding', source.parent / 'gems_fixes_20261004/report_encoding.py')
    derive = module('derive', source / 'derive_recipe_report.py')
    records = {}
    manifest = {}
    for family in ('thalamus', 'hippo-amygdala'):
        for kind in ('score', 'state'):
            name = f'{kind}-{family}.public.json'
            raw = (args.raw_dir / name.replace('.public.json', '.raw.json')).read_bytes()
            record = json.loads(raw)
            if kind == 'score':
                record = derive.derive(record)
                record['derived_soft_volume_deltas_from_original_report_sha256'] = hashlib.sha256(raw).hexdigest()
                record['derivation_source_sha256'] = hashlib.sha256((source / 'derive_recipe_report.py').read_bytes()).hexdigest()
            records[name] = record
            manifest[name] = {'original_server_sha256': hashlib.sha256(raw).hexdigest(),
                              'original_server_bytes': len(raw),
                              'decoded_canonical_sha256': encoding.digest(record)}
    raw = (args.raw_dir / 'recovery_queue.raw.json').read_bytes()
    records['recovery_queue.public.json'] = json.loads(raw)
    manifest['recovery_queue.public.json'] = {'original_recovery_sha256': hashlib.sha256(raw).hexdigest(),
                                            'decoded_canonical_sha256': encoding.digest(records['recovery_queue.public.json'])}

    counts = Counter()
    values = {}

    def visit(node):
        if isinstance(node, dict):
            token = encoding.canonical(node)
            if len(token) >= 300:
                counts[token] += 1
                values[token] = node
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    for record in records.values():
        visit(record)
    tokens = sorted(token for token, count in counts.items() if count > 1)
    names = {token: f'v{i}' for i, token in enumerate(tokens)}
    shared_name = 'complete_recipe_shared.public.json'

    def pack(node, allow_reference=True):
        if isinstance(node, dict):
            token = encoding.canonical(node)
            if allow_reference and token in names:
                return {'$ref': shared_name + '#/nodes/' + names[token]}
            return {key: pack(value) for key, value in node.items()}
        if isinstance(node, list):
            return [pack(value) for value in node]
        return node

    shared = {'encoding': 'fnit-validation-local-json-refs-v1',
              'scope': 'Identical repeated metadata and regional metric dictionaries; no values omitted',
              'nodes': {names[token]: pack(values[token], False) for token in tokens}}
    (args.output_dir / shared_name).write_text(encoding.compact(shared) + '\n')
    for name, record in records.items():
        path = args.output_dir / name
        if path.exists():
            raise FileExistsError(path)
        path.write_text(encoding.compact(pack(record)) + '\n')
        assert encoding.load_report(path) == record, name
        manifest[name].update(encoded_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              encoded_bytes=path.stat().st_size, lossless_decode_equal=True)

    # This earlier single-arm report contains the identical paired baseline.
    old_path = args.output_dir / 'score-thalamus-baseline.public.json'
    old = encoding.load_report(old_path)
    assert old['candidate'] == records['score-thalamus.public.json']['baseline']
    before = encoding.digest(old)
    old['candidate'] = {'$ref': 'score-thalamus.public.json#/baseline'}
    old_path.write_text(encoding.compact(old) + '\n')
    assert encoding.digest(encoding.load_report(old_path)) == before
    manifest[old_path.name] = {'decoded_canonical_sha256': before,
                               'encoded_sha256': hashlib.sha256(old_path.read_bytes()).hexdigest(),
                               'encoded_bytes': old_path.stat().st_size, 'lossless_decode_equal': True}
    manifest[shared_name] = {'encoded_sha256': hashlib.sha256((args.output_dir / shared_name).read_bytes()).hexdigest(),
                             'encoded_bytes': (args.output_dir / shared_name).stat().st_size,
                             'shared_dictionary_count': len(tokens)}
    result = {'scope': 'Complete old/new/official metrics and states; all values/source hashes preserved',
              'packer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reports': manifest}
    (args.output_dir / 'complete_recipe_encoding.public.json').write_text(encoding.compact(result) + '\n')
    print(json.dumps({'all_lossless_equal': True, 'shared_dictionary_count': len(tokens),
                      'encoded_bytes': sum(row['encoded_bytes'] for row in manifest.values())}))


if __name__ == '__main__':
    main()
