import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_subregions_unified_20260930')
source = root / 'source_v16_c5'
old = root / 'source_v15'
prepared = root.parent / 'fnit_subregions_plus_20260928/samseg_native_build_20260929'
python = '/home1/gongwk/anaconda3/bin/python'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(source / 'src'),
           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
           NUMEXPR_NUM_THREADS='4', PYTHONUNBUFFERED='1')
while True:
    previous = json.loads((root / 'v16_c4_components_status.json').read_text())
    if previous['state'] != 'running':
        assert previous['state'] == 'completed', previous
        break
    time.sleep(5)
commands = []
for name, directory in [('thalamus', 'port_thalamus_debug'),
                        ('hippo-amygdala-left', 'port_hippo_integrated_fullfix_left_tmp')]:
    tag = 'v16_c5_dense_layout_' + name
    commands.append((tag, [python, str(source / 'validation/subregions/benchmark_dense_layout.py'),
        '--prepared-stage', str(prepared / directory), '--lut', str(root / 'atlases' / name / 'compressionLookupTable.txt'),
        '--structure', name, '--output', str(root / (tag + '.json')), '--repeats', '3', '--memory-fraction', '.10']))
status_path = root / 'v16_c5_components_status.json'
assert not status_path.exists()
status = {'state': 'running', 'source': str(source), 'started_unix': time.time(), 'steps': []}
def save():
    status_path.write_text(json.dumps(status, indent=2) + '\n')
save()
for tag, command in commands:
    log_path = root / (tag + '.log')
    assert not log_path.exists()
    row = {'label': tag, 'command': command, 'started_unix': time.time()}
    status['steps'].append(row)
    with log_path.open('x') as log, (root / (tag + '_gpu_load.jsonl')).open('x') as load:
        process = subprocess.Popen(command, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        row['pid'] = process.pid
        save()
        while process.poll() is None:
            cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
            own = max([int(line.split(',')[1]) for line in apps.splitlines() if line.split(',')[0].strip() == str(process.pid)] or [0])
            row['max_own_process_memory_mib'] = max(row.get('max_own_process_memory_mib', 0), own)
            load.write(json.dumps({'unix_time': time.time(), 'own_pid': process.pid, 'own_process_memory_mib': own, 'gpus': cards.strip().splitlines()}) + '\n')
            load.flush()
            if own > 19073:
                row['memory_limit_exceeded'] = True
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            save()
        row.update(exit_code=process.returncode, finished_unix=time.time())
        save()
    if process.returncode:
        status.update(state='failed', finished_unix=time.time())
        save()
        raise SystemExit(process.returncode)
status.update(state='completed', finished_unix=time.time())
save()
