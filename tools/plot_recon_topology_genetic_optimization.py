"""已完成首缺陷优化的完整墙钟、逐候选耗时与共享负载图；不做GA计算。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*, metrics, candidate_timings, shared_load, output):
    """三份已导出白名单JSON→新PNG及哈希回执；全部参数具名必填。

    metrics/candidate_timings/shared_load是同一完整默认首缺陷回归的指标
    白名单，output必须是不存在的PNG。不加载MRI、参考网格或赢家，
    评分逐位门与退出码未通过时拒绝画成功结果。两个实际冷CLI和
    相同完整搜索/最终API分别展示；不同affinity/共享时段不称ABBA。
    候选耗时按实际序号含冷JIT；不把单轮或中位数外推完整recon-all。
    """
    output = Path(output)
    if output.exists() or output.suffix.lower() != '.png':
        raise ValueError('fresh PNG required')
    report = json.loads(Path(metrics).read_text())
    rows = json.loads(Path(candidate_timings).read_text())
    loads = json.loads(Path(shared_load).read_text())
    if (not report['strict_optimization_pair_passed'] or report['execution']['exit_code'] != 0
            or not rows['scores_all_byte_equal']):
        raise ValueError('completed exact full first-default-GA metrics required')
    new, old = rows['candidate'], rows['baseline']
    a = np.asarray([r['fitness'] for r in new], np.float64)
    b = np.asarray([r['fitness'] for r in old], np.float64)
    if (a.shape != b.shape or a.tobytes() != b.tobytes()
            or len(new) != report['search_evaluations'] + 1):
        raise ValueError('complete search plus final score required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    labels = ['Complete search\n+ final API', 'Fresh child CLI\nwith validation/IO']
    old_times = [report['baseline_stage_seconds']['connected_search_and_finalization_seconds'],
                 report['baseline_execution']['complete_fresh_cli_seconds']]
    new_times = [report['stage_seconds']['connected_search_and_finalization_seconds'],
                 report['execution']['complete_fresh_cli_seconds']]
    x = np.arange(2)
    axes[0].bar(x-.18, np.asarray(old_times)/60, width=.36, label='Previous complete')
    axes[0].bar(x+.18, np.asarray(new_times)/60, width=.36, label='Optimized complete')
    axes[0].set(xticks=x, xticklabels=labels, ylabel='minutes',
                title='Observed full first-defect wall time')
    axes[0].legend(fontsize=8)
    for xx, values in ((x-.18, old_times), (x+.18, new_times)):
        for position, value in zip(xx, values):
            axes[0].text(position, value/60, f'{value/60:.2f}', ha='center', va='bottom', fontsize=8)
    for values, label in ((old, 'Previous'), (new, 'Optimized')):
        axes[1].plot([r['candidate'] for r in values], [r['candidate_seconds'] for r in values],
                     linewidth=.7, alpha=.8, label=label)
    axes[1].set(xlabel='score index; last is final', ylabel='complete candidate API (s)',
                title=f'{len(new)} scores: every fitness byte equal')
    axes[1].legend(fontsize=8)
    samples = loads['samples']
    axes[2].plot([r['seconds']/60 for r in samples], [r['load_1m'] for r in samples], linewidth=.8)
    axes[2].set(xlabel='new driver elapsed (minutes)', ylabel='host load average (1 min)',
                title='Shared load; no baseline load curve')
    affinity_note = ('same CPU affinity' if report['current_cpu_affinity'] == report['baseline_cpu_affinity']
                     else 'different CPU affinity')
    figure.suptitle('Same four-thread budget, '+affinity_note+'/different time; no controlled ABBA throughput claim', fontsize=10)
    figure.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160); plt.close(figure)
    return dict(scope='completed full first-defect GA only; not whole recon-all',
        scores=len(new), scored_fitness_all_byte_equal=True,
        input_sha256=dict(metrics=sha(metrics), candidate_timings=sha(candidate_timings), shared_load=sha(shared_load)),
        output_sha256=sha(output), tool_sha256=sha(__file__),
        original_whole_GA_reference='not_assessed', whole_recon_all='not_assessed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metrics', 'candidate-timings', 'shared-load', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
