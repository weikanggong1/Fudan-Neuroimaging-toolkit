"""发布GCA JSON收据副本：替换私有路径/主机，保留全部数值和原始SHA。

输入目录只能包含待发布报告，不包含影像/许可证/凭据；不覆盖原报告。
"""
import argparse
import hashlib
import json
from pathlib import Path


def numeric_leaves(value):
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in numeric_leaves(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in numeric_leaves(item)]
    return [value] if isinstance(value, (int, float, bool)) or value is None else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--private-root', required=True)
    parser.add_argument('--private-host', required=True)
    parser.add_argument('--host-alias', default='A100-8')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest = {'scope': 'public derived JSON receipts; raw originals retained privately',
                'numeric_boolean_null_leaves_preserved': True, 'reports': {}}

    def sanitize(value):
        if isinstance(value, str):
            return value.replace(args.private_root, '[PRIVATE_FNIT]').replace(args.private_host, args.host_alias)
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, dict):
            return {sanitize(key): sanitize(item) for key, item in value.items()}
        return value

    for path in sorted(args.input_dir.rglob('*.json')):
        raw = path.read_bytes()
        original = json.loads(raw)
        result = sanitize(original)
        if numeric_leaves(original) != numeric_leaves(result):
            raise RuntimeError('numeric report altered: ' + str(path))
        relative = path.relative_to(args.input_dir)
        public = args.output_dir/relative
        public.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(result, indent=2)+'\n').encode()
        public.write_bytes(encoded)
        manifest['reports'][str(relative)] = {
            'raw_sha256': hashlib.sha256(raw).hexdigest(),
            'public_sha256': hashlib.sha256(encoded).hexdigest(),
            'numeric_boolean_null_leaf_count': len(numeric_leaves(original))}
    (args.output_dir/'PUBLIC_EXPORT_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
