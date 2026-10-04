"""只读校验冻结10例原始T1w；不下载、不修补、不替换失败被试。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from download_public_t1w import digest, image_metadata, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cases = manifest['cases']
    if len(cases) != 10 or len({(c['dataset'], c['subject']) for c in cases}) != 10:
        raise ValueError('requires the ten frozen distinct subjects')
    rows = []
    for case in cases:
        row = {'id': case['id'], 'server_input': case['server_input']}
        try:
            path = Path(case['server_input'])
            row.update(bytes=path.stat().st_size, sha256=digest(path), md5=digest(path, 'md5'))
            if row['bytes'] != case['bytes'] or row['sha256'] != case['sha256'] or row['md5'] != case['expected_md5']:
                raise ValueError('frozen size/SHA256/snapshot MD5 mismatch')
            row.update(nifti=image_metadata(path), status='verified')
        except Exception as error:
            row.update(status='failed', error=repr(error))
        rows.append(row)
    result = {'schema': 'fnit-cohort-readonly-verification-v1',
              'manifest_sha256': digest(args.manifest), 'script_sha256': digest(Path(__file__)),
              'verified_utc': datetime.now(timezone.utc).isoformat(),
              'cases': rows, 'completed': sum(r['status'] == 'verified' for r in rows)}
    result['status'] = 'complete' if result['completed'] == 10 else 'failed'
    write(args.output, result)
    print(json.dumps({'status': result['status'], 'completed': result['completed']}))
    raise SystemExit(result['status'] != 'complete')


if __name__ == '__main__':
    main()
