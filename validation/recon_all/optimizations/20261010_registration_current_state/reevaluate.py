"""只读评估已完成四次阶段，保留原生产/质量失败与完整全部行。

summary必需，指向原ABBA完整JSON；output必需且不能存在。不重算MRI，
不修改原报告或表面。仅重复精确几何/轨迹/审计比较，输出独立JSON，
绑定原summary、原driver、本评估器SHA。质量失败不会改成成功。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--comparison-tool', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    value = json.loads(args.summary.read_text())
    value['original_driver_status'] = value['status']
    value['original_driver_error'] = value.get('error')
    value['original_summary_sha256'] = sha(args.summary)
    value['read_only_comparison_tool_sha256'] = sha(args.comparison_tool)
    value['reevaluator_sha256'] = sha(Path(__file__))
    module_spec = importlib.util.spec_from_file_location('registration_comparison', args.comparison_tool)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    try:
        valid = module.evaluate_completed_reports(value, args.summary.parent)
        value['read_only_evaluation_status'] = 'complete'
        value['status'] = 'complete' if valid else 'failed_mesh_quality'
    except BaseException as error:
        value.update(status='failed', read_only_evaluation_status='failed',
                     read_only_error=repr(error))
        raise
    finally:
        value['read_only_comparison_seconds'] = time.perf_counter()-started
        value['read_only_comparison_is_production_runtime'] = False
        args.output.write_text(json.dumps(value, indent=2)+'\n')


if __name__ == '__main__':
    main()
