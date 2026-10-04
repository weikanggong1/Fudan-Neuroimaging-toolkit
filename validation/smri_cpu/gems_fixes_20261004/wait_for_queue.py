"""Run a post-timing audit only after all recorded queue arms complete."""

import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, action='append', required=True)
    parser.add_argument('--timeout', type=float, default=14400)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a post-timing command is required')
    started = time.monotonic()
    while time.monotonic() - started < args.timeout:
        records = [json.loads(path.read_text()) if path.exists() else None for path in args.record]
        if any(row and row['status'] in ('failed', 'timeout') for row in records):
            raise SystemExit('Queue arm failed; audit not run')
        if all(row and row['status'] == 'complete' for row in records):
            raise SystemExit(subprocess.call(command))
        time.sleep(5)
    raise SystemExit('Timed out waiting for the complete queue')


if __name__ == '__main__':
    main()
