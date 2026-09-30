"""从空目录测量官方 DeepPrep 的 volume 或 surface 完整处理耗时。"""
import argparse
import csv
import datetime as dt
import json
import pathlib
import subprocess
import threading
import time


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def monitor_gpu(process, gpu_index, destination, stop):
    """每 2 秒采样整卡负载和本次进程树显存；采样值不是连续峰值。"""
    with destination.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['timestamp_utc', 'gpu_index', 'gpu_memory_mib', 'gpu_utilization_percent', 'benchmark_memory_mib', 'benchmark_gpu_process_count'])
        while not stop.is_set():
            try:
                pairs = subprocess.check_output(['ps', '-e', '-o', 'pid=', '-o', 'ppid='], text=True).splitlines()
                descendants = {process.pid}
                parents = [tuple(map(int, line.split())) for line in pairs]
                while True:
                    expanded = descendants | {pid for pid, parent in parents if parent in descendants}
                    if expanded == descendants:
                        break
                    descendants = expanded
                device = subprocess.check_output(['nvidia-smi', '-i', str(gpu_index), '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')
                applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).splitlines()
                own = [float(memory) for line in applications for pid, memory in [line.split(',')] if int(pid) in descendants and memory.strip().isdigit()]
                writer.writerow([dt.datetime.now(dt.timezone.utc).isoformat(), gpu_index, float(device[0]), float(device[1]), sum(own), len(own)])
                stream.flush()
            except (subprocess.SubprocessError, ValueError) as error:
                writer.writerow([dt.datetime.now(dt.timezone.utc).isoformat(), gpu_index, '', '', '', str(error)])
                stream.flush()
            stop.wait(2)


def validate_outputs(output, mode):
    """用 nibabel 加载输出，核实完整时间序列和双侧网格。"""
    import nibabel as nib
    import numpy as np
    records = []
    func = output / 'BOLD/sub-benchmark/func'
    if mode == 'volume':
        files = list(func.glob('*space-MNI152NLin6Asym_res-*_desc-preproc_bold.nii.gz'))
        if len(files) != 1:
            raise ValueError(f'Expected one MNI BOLD output, found {len(files)}')
        image = nib.load(files[0])
        if image.shape[-1] != 490 or not np.allclose(image.header.get_zooms()[:3], 2):
            raise ValueError(f'Unexpected MNI output geometry: {image.shape}, {image.header.get_zooms()}')
        if not np.isfinite(np.asarray(image.dataobj)).all():
            raise ValueError('Nonfinite MNI BOLD output')
        records.append({'file': str(files[0].relative_to(output)), 'shape': list(image.shape), 'finite': True})
    else:
        for hemisphere in ['L', 'R']:
            files = list(func.glob(f'*hemi-{hemisphere}*space-fsaverage*bold*')) + list(func.glob(f'*space-fsaverage*hemi-{hemisphere}*bold*'))
            files = sorted(set(files))
            files = [p for p in files if p.suffix in ['.mgh', '.mgz', '.gii'] or p.name.endswith('.nii.gz')]
            # 兼容不同容器的最终表面格式；本次 25.1.0 输出为 GIFTI。
            files = [p for p in files if p.name.endswith('_bold.func.nii.gz')] or files
            if len(files) != 1:
                raise ValueError(f'Expected one {hemisphere} fsaverage6 BOLD output, found {[p.name for p in files]}')
            image = nib.load(files[0])
            data = np.asarray(image.agg_data()) if isinstance(image, nib.gifti.GiftiImage) else np.asarray(image.dataobj).squeeze()
            if data.shape == (490, 40962):
                data = data.T
            if data.shape != (40962, 490) or not np.isfinite(data).all():
                raise ValueError(f'Unexpected surface data: {files[0].name}, {data.shape}')
            records.append({'file': str(files[0].relative_to(output)), 'shape': list(data.shape), 'finite': True})
    for hemisphere in ['lh', 'rh']:
        for surface in ['white', 'pial', 'sphere.reg']:
            path = output / 'Recon/sub-benchmark/surf' / f'{hemisphere}.{surface}'
            vertices, faces = nib.freesurfer.read_geometry(path)
            if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
                raise ValueError(f'Invalid surface geometry: {path}')
            records.append({'file': str(path.relative_to(output)), 'vertices': len(vertices), 'faces': len(faces), 'finite': True})
    confounds = list(func.glob('*desc-confounds_timeseries.tsv'))
    if len(confounds) != 1:
        raise ValueError(f'Expected one confounds TSV, found {len(confounds)}')
    with confounds[0].open() as stream:
        if sum(1 for _ in stream) - 1 != 490:
            raise ValueError('Confounds rows do not match 490 BOLD frames')
    records.append({'file': str(confounds[0].relative_to(output)), 'rows': 490})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, required=True, help='服务器上的 benchmark 根目录')
    parser.add_argument('--mode', choices=['volume', 'surface'], required=True, help='最终 BOLD 输出类型')
    parser.add_argument('--gpu', type=int, default=1, help='物理 GPU 序号')
    parser.add_argument('--cpus', type=int, default=10, help='Nextflow 可用 CPU 调度额度')
    parser.add_argument('--memory', type=int, default=32, help='Nextflow 可用主机内存，GB；不是显存上限')
    parser.add_argument('--singularity', default='singularity', help='Singularity 可执行文件名或绝对路径')
    parser.add_argument('--fs-license', type=pathlib.Path, required=True, help='用户已获许可的 FreeSurfer license.txt；只读挂载')
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'runs' / args.mode
    if output.exists():
        raise SystemExit(f'Cold benchmark requires an absent output directory: {output}')
    output.mkdir(parents=True)
    runtime_home = output / 'runtime_home'
    runtime_home.mkdir()
    singularity = args.singularity
    image = root / 'containers/deepprep_25.1.0.sif'
    license_file = str(args.fs_license.resolve())
    common = [singularity, 'exec', '--cleanenv', '--home', str(runtime_home) + ':/home/benchmark', '-B', str(root) + ':/benchmark', str(image)]
    # 使用官方镜像中的离线 Nextflow 依赖，写到本轮独立的 home。
    subprocess.run(common + ['bash', '-lc', 'mkdir -p "$HOME/.nextflow" "$HOME/.cache"; cp -a /home/deepprep/.nextflow/. "$HOME/.nextflow/"; cp -a /home/deepprep/.cache/. "$HOME/.cache/"'], check=True)
    config = root / f'{args.mode}.config'
    config.write_text(f"executor {{ name = 'local'; cpus = {args.cpus}; memory = '{args.memory} GB' }}\n" + """trace {
    enabled = true
    file = '/output/QC/trace.tsv'
    raw = true
    fields = 'task_id,hash,native_id,name,status,exit,submit,start,complete,duration,realtime,%cpu,peak_rss,peak_vmem,cpus,attempt'
}
process {
    withLabel: with_gpu {
        maxForks = 1
    }
}
""".replace("maxForks = 1", f"maxForks = 1\n        cpus = {args.cpus}"))
    command = [singularity, 'run', '--cleanenv', '--nv', '--home', str(runtime_home) + ':/home/benchmark',
               '--env', 'TF_FORCE_GPU_ALLOW_GROWTH=true,CUDA_DEVICE_ORDER=PCI_BUS_ID',
               '-B', str(root / 'bids') + ':/input:ro', '-B', str(output) + ':/output',
               '-B', str(config) + ':/benchmark.config:ro', '-B', license_file + ':/fs_license.txt:ro',
               str(image), '/input', '/output', 'participant', '--bold_task_type', 'rest',
               '--participant_label', 'benchmark', '--fs_license_file', '/fs_license.txt',
               '--config_file', '/benchmark.config', '--device', str(args.gpu),
               '--cpus', str(args.cpus), '--memory', str(args.memory),
               '--bold_sdc', 'False', '--bold_confounds', 'True', '--bold_skip_frame', '0',
               '--bold_volume_space', 'MNI152NLin6Asym' if args.mode == 'volume' else 'None',
               '--bold_volume_res', '02', '--bold_surface_spaces', 'None' if args.mode == 'volume' else 'fsaverage6']
    result = {'status': 'running', 'mode': args.mode, 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'command': command, 'cold_output': True, 'anatomical_reconstruction_reused': False,
              'input_frames': 490, 'gpu_index': args.gpu, 'cpus': args.cpus, 'host_memory_gb': args.memory, 'gpu_tasks_serialized': True,
              'precision': 'Official DeepPrep model defaults; no added precision changes', 'TF_FORCE_GPU_ALLOW_GROWTH': True}
    write_json(output / 'benchmark.json', result)
    start = time.perf_counter()
    with (output / 'pipeline.log').open('w') as log:
        process = subprocess.Popen(command, cwd=output, stdout=log, stderr=subprocess.STDOUT)
        stop = threading.Event()
        monitor = threading.Thread(target=monitor_gpu, args=(process, args.gpu, output / 'gpu_samples.csv', stop), daemon=True)
        monitor.start()
        exit_code = process.wait()
        seconds = time.perf_counter() - start
        stop.set()
        monitor.join(timeout=10)
    result.update(exit_code=exit_code, wall_seconds=seconds, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), status='failed' if exit_code else 'validation_pending')
    write_json(output / 'benchmark.json', result)
    if exit_code:
        raise SystemExit(exit_code)
    try:
        result['outputs'] = validate_outputs(output, args.mode)
        result['status'] = 'complete'
    except Exception as error:
        result['status'] = 'output_validation_failed'
        result['validation_error'] = str(error)
        write_json(output / 'benchmark.json', result)
        raise
    write_json(output / 'benchmark.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
