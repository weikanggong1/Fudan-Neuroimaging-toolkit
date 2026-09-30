"""汇总官方 DeepPrep 完整运行、阶段计时和显存采样。"""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path


def timestamp_ms(value):
    try:
        return float(value)
    except ValueError:
        return dt.datetime.fromisoformat(value).timestamp() * 1000


def summarize(root):
    result = {'software': json.loads((root / 'software_manifest.public.json').read_text(encoding='utf-8')),
              'input': json.loads((root / 'input_manifest.public.json').read_text(encoding='utf-8')),
              'timing_notes': ['每种输出均从原始 T1w 和完整 490 帧 BOLD 开始，未复用解剖重建和 Nextflow 结果。',
                               'pipeline_wall_seconds 包含容器启动、处理、写出和 QC；下载、配置、缓存复制和输出验证另行完成。',
                               'branch_span_seconds 是该分支首次任务开始至最后任务完成的时间，分支可能重叠，不能相加。',
                               'GPU 显存按 2 秒采样，最高采样值不是连续峰值；整卡用量包含共享任务。'], 'runs': []}
    for mode in ['volume', 'surface']:
        output = root / 'runs' / mode
        run = json.loads((output / 'benchmark.json').read_text(encoding='utf-8'))
        if run['status'] != 'complete' or run['exit_code'] != 0:
            raise ValueError(f'{mode} has not completed and passed output validation')
        record = {'mode': mode, 'pipeline_wall_seconds': run['wall_seconds'], 'started_utc': run['started_utc'],
                  'finished_utc': run['finished_utc'], 'exit_code': run['exit_code'], 'validated_outputs': run['outputs'],
                  'configuration': {key: run[key] for key in ['input_frames', 'gpu_index', 'cpus', 'host_memory_gb', 'gpu_tasks_serialized', 'cold_output', 'anatomical_reconstruction_reused', 'precision', 'TF_FORCE_GPU_ALLOW_GROWTH']}}
        with (output / 'QC/trace.tsv').open() as stream:
            tasks = list(csv.DictReader(stream, delimiter='\t'))
        cached = [task for task in tasks if task['status'] == 'CACHED']
        failures = [task for task in tasks if task['status'] not in ['COMPLETED', 'CACHED'] or task['exit'] not in ['0', '']]
        if cached or failures:
            raise ValueError(f'{mode}: {len(cached)} cached tasks, {len(failures)} failed/incomplete tasks')
        branch = {}
        for label in ['anat_wf', 'bold_wf']:
            selected = [task for task in tasks if label + ':' in task['name']]
            if selected:
                branch[label] = (max(timestamp_ms(task['complete']) for task in selected) - min(timestamp_ms(task['start']) for task in selected)) / 1000
        record['branch_span_seconds'] = branch
        record['completed_task_count'] = len(tasks)
        record['slowest_tasks'] = sorted([{'name': task['name'], 'realtime_seconds': float(task['realtime']) / 1000} for task in tasks], key=lambda task: -task['realtime_seconds'])[:12]
        with (output / 'gpu_samples.csv').open() as stream:
            samples = list(csv.DictReader(stream))
        samples = [s for s in samples if s['gpu_memory_mib'] and s['benchmark_memory_mib']]
        if not samples or not any(float(s['benchmark_memory_mib']) > 0 for s in samples):
            raise ValueError(f'{mode}: no valid GPU process memory samples')
        record['gpu_samples'] = {'count': len(samples),
                                 'benchmark_highest_sample_mib': max(float(s['benchmark_memory_mib']) for s in samples),
                                 'whole_card_memory_range_mib': [min(float(s['gpu_memory_mib']) for s in samples), max(float(s['gpu_memory_mib']) for s in samples)],
                                 'whole_card_utilization_range_percent': [min(float(s['gpu_utilization_percent']) for s in samples), max(float(s['gpu_utilization_percent']) for s in samples)]}
        result['runs'].append(record)
    (root / 'benchmark_results.public.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (root / 'benchmark_timings.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['mode', 'pipeline_wall_seconds', 'pipeline_wall_minutes', 'anat_branch_span_seconds', 'bold_branch_span_seconds', 'benchmark_highest_gpu_sample_mib'])
        for run in result['runs']:
            writer.writerow([run['mode'], run['pipeline_wall_seconds'], run['pipeline_wall_seconds'] / 60,
                             run['branch_span_seconds'].get('anat_wf'), run['branch_span_seconds'].get('bold_wf'), run['gpu_samples']['benchmark_highest_sample_mib']])
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='服务器上的 benchmark 根目录')
    summarize(parser.parse_args().root)
