"""导出完整注册收据，复用现有发布器；不复制影像或改写原收据。

input-dir 是私有只读收据目录，output-dir 必须不存在；private-root 和
private-host 仅用于替换机器路径/主机。JSONL审计转换成JSON事件列表，
同时保留原字节SHA；所有数值、布尔及null与原件核对。输出是完整JSON、
审计事件、原始/公开哈希清单，不声明OS隔离或整体等效。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--private-root', required=True)
    parser.add_argument('--private-host', required=True)
    args = parser.parse_args(argv)
    exporter = Path(__file__).resolve().parents[4]/'tools/export_recon_gca_receipts.py'
    if not exporter.is_file():
        raise FileNotFoundError(exporter)
    with TemporaryDirectory(prefix='fnit-reg-public-json-') as directory:
        temporary = Path(directory)
        audit_manifest = {}
        for path in sorted(args.input_dir.rglob('*.json')):
            output = temporary/path.relative_to(args.input_dir)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, output)
        for path in sorted(args.input_dir.rglob('*.jsonl')):
            raw = path.read_bytes()
            rows = [json.loads(line) for line in raw.decode().splitlines()]
            relative = path.relative_to(args.input_dir).with_suffix('.events.json')
            output = temporary/relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(rows, indent=2)+'\n')
            audit_manifest[str(relative)] = {
                'original_jsonl_sha256': hashlib.sha256(raw).hexdigest(),
                'event_count': len(rows),
                'conversion': 'JSONL rows to JSON array; event values unchanged'}
        (temporary/'AUDIT_CONVERSION.json').write_text(json.dumps(audit_manifest, indent=2)+'\n')
        subprocess.run([sys.executable, str(exporter), '--input-dir', str(temporary),
                        '--output-dir', str(args.output_dir), '--private-root', args.private_root,
                        '--private-host', args.private_host], check=True)
        # 清理机器信息后的公开副本使用public命名；原收据及原SHA不变。
        manifest_path = args.output_dir/'PUBLIC_EXPORT_MANIFEST.json'
        manifest = json.loads(manifest_path.read_text())
        reports = {}
        for relative, entry in manifest['reports'].items():
            public_relative = relative.removesuffix('.private.json')+'.public.json' \
                if relative.endswith('.private.json') else relative
            if public_relative != relative:
                (args.output_dir/relative).rename(args.output_dir/public_relative)
                entry['original_relative_path'] = relative
            reports[public_relative] = entry
        manifest['reports'] = reports
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
