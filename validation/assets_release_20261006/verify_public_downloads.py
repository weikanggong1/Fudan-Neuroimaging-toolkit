"""Verify complete public Release responses; never use installed credentials."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

BASE = 'https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/download/assets-v1/'


def verify(record):
    start = time.perf_counter()
    for attempt in range(3):
        try:
            size = 0
            digest = hashlib.sha256()
            with urlopen(Request(BASE + record['name'], headers={
                    'User-Agent': 'FNIT-public-resource-verification/1.0'}), timeout=120) as response:
                if response.status != 200:
                    raise ValueError('unexpected status')
                final_host = urlsplit(response.geturl()).hostname
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b''):
                    size += len(chunk)
                    digest.update(chunk)
            if size != record['size'] or digest.hexdigest() != record['sha256']:
                raise ValueError('public GET size or SHA-256 mismatch')
            return dict(name=record['name'], size=size, sha256=digest.hexdigest(),
                        status='verified', final_host=final_host,
                        elapsed_seconds=time.perf_counter() - start)
        except Exception as error:
            if attempt == 2:
                return dict(name=record['name'], status='failed',
                            error_type=type(error).__name__, error=str(error)[:300])
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = manifest['newly_published_assets']
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for task in as_completed([pool.submit(verify, record) for record in records]):
            result = task.result()
            results.append(result)
            args.output.write_text(json.dumps(dict(
                authentication='anonymous; no credential headers',
                scope='entire response, exact byte count and SHA-256',
                results=results), indent=2) + '\n')
            print(result['status'], result['name'], flush=True)
    raise SystemExit(0 if all(r['status'] == 'verified' for r in results) else 2)


if __name__ == '__main__':
    main()
