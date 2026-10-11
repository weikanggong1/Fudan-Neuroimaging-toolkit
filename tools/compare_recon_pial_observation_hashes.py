"""比较自产修复前后四轮观测的全部数组哈希和接受决定；无需加载影像。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='修复前 events.json')
    parser.add_argument('--candidate', type=Path, required=True, help='修复后 events.json')
    parser.add_argument('--output', type=Path, required=True, help='新的纯统计 JSON')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    kinds = {'border', 'step', 'pin'}
    before = [event for event in baseline if event['kind'] in kinds]
    after = [event for event in candidate if event['kind'] in kinds]
    differences = []
    for index, (first, second) in enumerate(zip(before, after)):
        # 私有文件名不参与算法比较；数组摘要包含 dtype、shape 和完整字节 SHA。
        first_fields = {key: value for key, value in first.items() if key != 'file'}
        second_fields = {key: value for key, value in second.items() if key != 'file'}
        for key in sorted(first_fields.keys() | second_fields.keys()):
            if first_fields.get(key) != second_fields.get(key):
                differences.append(dict(event=index, kind=first['kind'], step=first.get('step'), field=key))
    report = dict(scope='all recorded border, step arrays and decisions, and pin state; cleanup excluded',
                  baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  baseline_events=len(before), candidate_events=len(after),
                  baseline_steps=sum(event['kind'] == 'step' for event in before),
                  candidate_steps=sum(event['kind'] == 'step' for event in after),
                  differences=differences, exact=len(before) == len(after) and not differences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
