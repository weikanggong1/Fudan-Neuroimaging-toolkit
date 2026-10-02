"""真实自产 sphere→register 连续链；只读取已完成配对sphere，不修改冻结检查点。"""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys, time


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--lock', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    task = pathlib.Path.cwd()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, NUMBA_NUM_THREADS='4', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', FNIT_OUTER_MONITOR='1')
    commit = '9f7bb7a5af72e7572fcb8de617575645f29f51d7'
    pending = [(subject, hemi) for subject in ('sub01', 'sub02') for hemi in ('lh', 'rh')]
    while pending:
        for subject, hemi in list(pending):
            sphere_dir = task/'cold_pairs'/subject/hemi/'sphere'/'candidate'
            report_path = sphere_dir/'report.json'
            if not report_path.exists():
                continue
            sphere_report = json.loads(report_path.read_text())
            if sphere_report['status'] != 'complete':
                raise RuntimeError(f'Upstream sphere failed: {report_path}')
            if sphere_report['args']['commit'] != commit:
                raise RuntimeError('Upstream candidate source mismatch')
            chain = task/'chains'/subject/hemi
            output = chain/'output'
            if (output/'report.json').exists():
                existing = json.loads((output/'report.json').read_text())
                if existing['status'] != 'complete':
                    raise RuntimeError(f'Existing chain failure: {output}')
                pending.remove((subject, hemi))
                continue
            surf = chain/'subject'/'surf'
            surf.mkdir(parents=True, exist_ok=True)
            checkpoint = args.root/'serial_20261001'/('whole_sub01_candidate_retry1' if subject == 'sub01' else 'whole_sub02_candidate_retry2')
            inputs = {'sphere': sphere_dir/(hemi+'.sphere'), 'smoothwm': checkpoint/'surf'/(hemi+'.smoothwm'), 'sulc': checkpoint/'surf'/(hemi+'.sulc')}
            for name, source in inputs.items():
                shutil.copyfile(source, surf/(hemi+'.'+name))
            provenance = {'sphere_produced_by': str(report_path), 'sphere_source_commit': commit, 'sphere_producer_script_sha256': sphere_report['benchmark_script_sha256'], 'input_sha256': {key:sha(path) for key,path in inputs.items()}, 'overall_equivalence':'not_assessed'}
            if provenance['input_sha256']['sphere'] != sphere_report['output_sha256']:
                raise RuntimeError('Sphere provenance hash mismatch')
            (chain/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
            command = ['flock', '-x', args.lock, sys.executable, str(task/'run_monitored.py'), '--gpu-uuid', args.gpu_uuid, '--output', str(output/'monitor'), '--interval', '.25', '--', sys.executable, str(task/'run_stage.py'), '--source-root', str(task/'candidate'), '--checkpoint', str(chain/'subject'), '--output-root', str(output), '--assets', str(args.root/'assets'), '--commit', commit, '--stage', 'register', '--hemisphere', hemi, '--device', 'cuda:0', '--gpu-uuid', args.gpu_uuid]
            print(f'Starting continuous chain {subject} {hemi}', flush=True)
            subprocess.run(command, check=True)
            pending.remove((subject, hemi))
        if pending:
            time.sleep(30)
    print('Four continuous chains complete', flush=True)


if __name__ == '__main__':
    main()
