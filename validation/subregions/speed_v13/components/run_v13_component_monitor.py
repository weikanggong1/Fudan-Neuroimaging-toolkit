import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_subregions_unified_20260930')
PYTHON = '/home1/gongwk/anaconda3/bin/python'
COMMIT = '3ea8163966ef922c9cee67018095ab39d1f6bfa4'
PREPARED = Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_subregions_plus_20260928/samseg_native_build_20260929')
STAGES = {'thalamus': 'port_thalamus_debug', 'hippo-amygdala-left': 'port_hippo_integrated_fullfix_left_tmp'}


def sample(pid):
    now = time.time()
    card_text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    cards = []
    gpu_indices = {}
    for line in card_text.strip().splitlines():
        index, uuid, total, used, free, utilization = [item.strip() for item in line.split(',')]
        gpu_indices[uuid] = int(index)
        cards.append({'gpu_index': int(index), 'gpu_uuid': uuid, 'total_memory_mib': int(total), 'used_memory_mib': int(used), 'free_memory_mib': int(free), 'utilization_gpu_percent': int(utilization)})
    process_text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory,gpu_uuid', '--format=csv,noheader,nounits'], text=True)
    own = []
    if pid is not None:
        for line in process_text.strip().splitlines():
            fields = [item.strip() for item in line.split(',')]
            if len(fields) == 3 and fields[0] == str(pid):
                own.append({'pid': pid, 'gpu_index': gpu_indices.get(fields[2]), 'gpu_uuid': fields[2], 'used_memory_mib': int(fields[1])})
    return {'unix_time': now, 'utc_time': datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat(), 'own_pid': pid, 'own_process_memory_mib': sum(row['used_memory_mib'] for row in own), 'own_processes': own, 'gpus': cards}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('structure', choices=STAGES)
    args = parser.parse_args()
    tag = 'compact_' + args.structure.replace('-', '_') + '_tf32_v13'
    output = ROOT / (tag + '.json')
    log_path = ROOT / (tag + '.log')
    load_path = ROOT / (tag + '_gpu_load.jsonl')
    status_path = ROOT / (tag + '_monitor.json')
    for path in (output, log_path, load_path, status_path):
        if path.exists():
            raise RuntimeError('Refusing to overwrite existing result: ' + str(path))
    before = sample(None)
    free = next(row['free_memory_mib'] for row in before['gpus'] if row['gpu_index'] == 0)
    if free < 9216:
        print(json.dumps({'status': 'waiting_for_gpu0', 'free_memory_mib': free}), flush=True)
        return
    command = [PYTHON, str(ROOT / 'source_v13/validation/subregions/benchmark_compact_mesh.py'), '--prepared-stage', str(PREPARED / STAGES[args.structure]), '--lut', str(ROOT / 'atlases' / args.structure / 'compressionLookupTable.txt'), '--structure', args.structure, '--device', 'cuda:0', '--output', str(output), '--shared-geometry', '--analytic-prior', '--tf32', '--repeats', '3', '--source-commit', COMMIT, '--memory-fraction', '.10']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(ROOT / 'source_v13/src'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    started = time.time()
    with log_path.open('x') as log, load_path.open('x') as load:
        load.write(json.dumps(before) + '\n')
        load.flush()
        process = subprocess.Popen(command, cwd=ROOT / 'source_v13', env=env, stdout=log, stderr=subprocess.STDOUT)
        status = {'structure': args.structure, 'pid': process.pid, 'source_commit': COMMIT, 'command': command, 'started_unix_time': started, 'physical_gpu': 0, 'memory_fraction': .10, 'own_memory_cap_mib': 19073, 'gpu_poll_seconds': 5, 'start_gpu_sample': before}
        status_path.write_text(json.dumps(status, indent=2) + '\n')
        print(json.dumps({'status': 'started', 'structure': args.structure, 'pid': process.pid, 'output': str(output), 'gpu0_free_memory_mib': free}), flush=True)
        cap_exceeded = False
        last_message = started
        while True:
            row = sample(process.pid)
            load.write(json.dumps(row) + '\n')
            load.flush()
            if row['own_process_memory_mib'] > 19073:
                cap_exceeded = True
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.poll() is not None:
                break
            if time.time() - last_message >= 30:
                print(json.dumps({'status': 'running', 'structure': args.structure, 'pid': process.pid, 'elapsed_seconds': time.time() - started, 'own_process_memory_mib': row['own_process_memory_mib']}), flush=True)
                last_message = time.time()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        after = sample(process.pid)
        load.write(json.dumps(after) + '\n')
    status.update(ended_unix_time=time.time(), elapsed_seconds=time.time() - started, returncode=process.returncode, memory_cap_exceeded=cap_exceeded, output_exists=output.exists(), end_gpu_sample=after)
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps({'status': 'completed', 'structure': args.structure, 'returncode': process.returncode, 'elapsed_seconds': status['elapsed_seconds'], 'output_exists': output.exists(), 'memory_cap_exceeded': cap_exceeded}), flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
