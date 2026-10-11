"""本轮自产输入的独立GCA缓存诊断；不改生产源码或读取官方图。"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import threading
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:5')
    parser.add_argument('--inverse-backend', choices=['cpu', 'torch'])
    parser.add_argument('--candidate-chunk', type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    parent_run = args.run / (args.subject + '-candidate')
    parent_benchmark = parent_run / 'benchmark.json'
    benchmark = json.loads(parent_benchmark.read_text())
    version = json.loads((args.source / 'SOURCE_MANIFEST.json').read_text())
    # Frozen production settings, not the historical large-chunk recipe.
    inverse = args.inverse_backend or benchmark['gca_inverse_backend']
    chunk = args.candidate_chunk or benchmark['gca_candidate_chunk']
    assert benchmark['gca_execution'] == 'in-process'
    inp = parent_run / 'subject/mri'
    local_subject = args.out / 'subject'
    mri = local_subject / 'mri'
    (mri / 'transforms').mkdir(parents=True)
    (local_subject / 'scripts').mkdir()
    inputs = {}
    for name in ['nu.mgz', 'brainmask.mgz']:
        source = inp / name
        target = mri / name
        before = sha(source)
        shutil.copy2(source, target)
        after = sha(target)
        assert before == after == sha(source)
        inputs[name] = {'producer_path': str(source), 'sha256': after}
    atlas = args.root / 'assets/recon_resources_20261009_v1/assets/average/RB_all_2020-01-02.gca'
    receipt = {'status': 'running', 'scope': 'same current self-produced nu/mask; GCA isolation only; not whole-pipeline speed or metric equivalence',
               'producer_benchmark_sha256': sha(parent_benchmark),
               'producer_source_manifest_sha256': sha(args.source / 'SOURCE_MANIFEST.json'),
               'producer_code_version': benchmark.get('code_version'),
               'producer_source_version': version,
               'inputs': inputs, 'atlas_sha256': sha(atlas),
               'host': socket.gethostname(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
               'threads': 4, 'logical_device': args.device,
               'candidate_chunk': chunk, 'inverse_backend': inverse,
               'producer_gca_configuration': {key: benchmark[key] for key in
                    ['gca_inverse_backend', 'gca_candidate_chunk', 'gca_execution']},
               'execution': 'isolated', 'native_binary': None,
               'source_sha256': {}, 'script_sha256': sha(__file__)}
    report_path = args.out / 'diagnosis.private.json'
    report_path.write_text(json.dumps(receipt, indent=2))
    import torch
    from fnit.recon_all.profiling import configure_cuda_allocator, ProcessTreeDeviceSampler, autocast_state
    from fnit.recon_all.native_free import _run_native_em_register
    from fnit.recon_all.thread_budget import thread_budget
    for name in ['fnit.recon_all.native_free', 'fnit.recon_all.gca_torch_worker',
                 'fnit.recon_all.mri_em_register_python', 'fnit.recon_all.mri_em_register_score_gpu']:
        module = importlib.import_module(name)
        receipt['source_sha256'][name] = sha(module.__file__)
    allocator = configure_cuda_allocator(device=args.device, policy='disabled')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(args.device)
    live = torch.ones(1 << 20, device=args.device, dtype=torch.float32)
    torch.cuda.synchronize(args.device)
    def parent_state():
        return {'initialized': torch.cuda.is_initialized(), 'current_device': torch.cuda.current_device(),
                'allocator_environment': os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING'),
                'precision': {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                              'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                              'autocast': autocast_state('cuda')},
                'live_shape': list(live.shape), 'live_dtype': str(live.dtype)}
    receipt['parent_allocator'] = allocator
    receipt['parent_before'] = parent_state()
    receipt['gpu_name'] = torch.cuda.get_device_name(args.device)
    receipt['gpu_uuid'] = str(torch.cuda.get_device_properties(args.device).uuid)
    sampler = ProcessTreeDeviceSampler(device=args.device, parent_pid=os.getpid(), interval=.5)
    stopped = threading.Event()
    def sample():
        while not stopped.is_set():
            sampler.sample_if_due(force=True)
            stopped.wait(.5)
    monitoring = threading.Thread(target=sample, daemon=True)
    monitoring.start()
    tick = time.perf_counter()
    try:
        with thread_budget(threads=4):
            result = _run_native_em_register(
                binary=None, mri=mri, atlas=atlas,
                assets=args.root / 'assets/recon_resources_20261009_v1/assets',
                backend='torch', binary_sha256=None, threads=4,
                device=args.device, inverse_backend=inverse, candidate_chunk=chunk,
                execution='isolated')
        torch.cuda.synchronize(args.device)
        receipt.update(status='complete', full_call_seconds=time.perf_counter() - tick,
                       worker=result, output_lta_sha256=sha(mri / 'transforms/talairach.lta'))
        receipt['parent_after'] = parent_state()
        receipt['parent_state_unchanged'] = receipt['parent_after'] == receipt['parent_before']
        receipt['parent_live_tensor_unchanged'] = bool(torch.all(live == 1).item())
    except BaseException as error:
        receipt.update(status='failed', full_call_seconds=time.perf_counter() - tick,
                       error=repr(error), parent_after=parent_state())
        raise
    finally:
        stopped.set()
        monitoring.join(timeout=5)
        sampler.sample_if_due(force=True)
        receipt['memory'] = sampler.report()
        receipt['parent_torch_memory_status'] = 'unavailable_allocator_disabled; not zero GPU memory'
        report_path.write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
