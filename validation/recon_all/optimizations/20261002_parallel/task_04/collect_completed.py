"""等待所有实际命令完成后生成汇总；缺少/失败的结果不能变成验收通过。"""
import argparse, hashlib, json, os, pathlib, subprocess, sys, time


def wait_reports(paths):
    while True:
        missing = 0
        for path in paths:
            if not path.exists():
                missing += 1
                continue
            try:
                report = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                missing += 1
                continue
            if report['status'] == 'failed':
                raise RuntimeError(f'Benchmark failed: {path}')
            if report['status'] != 'complete':
                missing += 1
        if not missing:
            return
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    task = args.task_root.resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', NUMBA_NUM_THREADS='4', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONPATH=str(task/'candidate/src'))
    subjects, hemis, stages = ('sub01','sub02'), ('lh','rh'), ('remesh','sphere','register')
    paired = [task/'cold_pairs'/s/h/st/v/'report.json' for s in subjects for h in hemis for st in stages for v in ('baseline','candidate')]
    reference = [task/'reference'/s/h/st/'report.json' for s in subjects for h in hemis for st in stages]
    chains = [task/'chains'/s/h/'output/report.json' for s in subjects for h in hemis]
    wait_reports(paired)
    summary = task/'completed_summary'
    summary.mkdir(exist_ok=True)
    subprocess.run([sys.executable,str(task/'compare_stages.py'),'--pairs',str(task/'cold_pairs'),'--output',str(summary),'--quality','--quality-helper',str(task/'surface_quality_helper.py')],check=True)
    wait_reports(chains)
    subprocess.run([sys.executable,str(task/'compare_chains.py'),'--task-root',str(task),'--output',str(summary/'chains.json')],check=True)
    wait_reports(reference)
    subprocess.run([sys.executable,str(task/'compare_reference.py'),'--pairs',str(task/'cold_pairs'),'--reference',str(task/'reference'),'--distance-helper',str(task/'compare_surface_chain.py'),'--output',str(summary/'official.json')],check=True)
    pair = json.loads((summary/'comparison.json').read_text())
    chain = json.loads((summary/'chains.json').read_text())
    official = json.loads((summary/'official.json').read_text())
    checks = {'paired_count':len(pair['rows']), 'chain_count':len(chain['rows']), 'official_count':len(official['rows']), 'all_paired_exact':all(row['strict_stage_reproduction'] for row in pair['rows']), 'all_chains_propagated':all(row['strict_chain_propagation'] for row in chain['rows']), 'overall_equivalence':'not_assessed'}
    checks['metadata_sha256']={str(path.relative_to(task)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paired+reference+chains}
    (summary/'completion.json').write_text(json.dumps(checks,indent=2)+'\n')
    if (checks['paired_count'],checks['chain_count'],checks['official_count']) != (12,4,12) or not checks['all_paired_exact'] or not checks['all_chains_propagated']:
        raise RuntimeError('Required measured acceptance incomplete or failed')
    print(json.dumps(checks,indent=2),flush=True)


if __name__ == '__main__':
    main()
