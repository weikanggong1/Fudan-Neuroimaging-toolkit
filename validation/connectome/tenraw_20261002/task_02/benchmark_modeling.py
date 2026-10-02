"""Frozen same-checkpoint DTI/response/CSD/mtnormalise AB/BA benchmark.

Input JSON must name newly downloaded subject, provenance manifest and corrected
DWI/gradient/masks. This is a component chain, not raw end-to-end benchmark.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
import threading
import time
import types
import nibabel as nib
import numpy as np
import torch
from tensor_comparison import compare_tensor_dicts


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): digest.update(block)
    return digest.hexdigest()


def package(directory, name):
    root = types.ModuleType(name)
    root.__path__ = [str(Path(directory).resolve())]
    sys.modules[name] = root
    modules = {}
    for key in ('fod', 'response', 'mtnormalise'):
        spec = importlib.util.spec_from_file_location(f'{name}.{key}', Path(directory) / f'{key}.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[key] = module
    return modules


class MemorySampler:
    """nvidia-smi NVML-backed whole-process sampling, MiB measurement precision."""
    def __init__(self):
        self.stop = threading.Event(); self.records = []; self.failures = 0; self.phase = "chain_boundary"
    def run(self):
        while not self.stop.is_set():
            start = time.perf_counter(); phase = self.phase
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory,gpu_uuid', '--format=csv,noheader,nounits'], text=True, timeout=5)
                own = [line.split(',') for line in output.splitlines() if line.split(',')[0].strip() == str(os.getpid())]
                self.records.append({'t': start, 'dispatch_phase':phase, 'bytes': sum(int(row[1]) * 1024**2 for row in own), 'uuid': [row[2].strip() for row in own], 'pid':os.getpid()})
            except Exception: self.failures += 1
            self.stop.wait(.2)
    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True); self.thread.start(); return self
    def __exit__(self, *args):
        self.stop.set(); self.thread.join()
    def result(self):
        times = [r['t'] for r in self.records]
        return {'peak_bytes': max((r['bytes'] for r in self.records), default=None), 'peak_sample':max(self.records,key=lambda r:r['bytes']) if self.records else None, 'raw_samples':self.records, 'samples': len(times), 'failed_samples': self.failures, 'max_interval_s': max(np.diff(times), default=None), 'uuid': sorted({u for r in self.records for u in r['uuid']}), 'dispatch_phase_sampled_peaks':{phase:max(r['bytes'] for r in self.records if r['dispatch_phase']==phase) for phase in {r['dispatch_phase'] for r in self.records}}, 'stage_label_limitation':'CPU enqueue boundaries; asynchronous kernels can overlap the next dispatch phase', 'measurement_resolution_bytes': 1024**2, 'child_processes': 'none; nvidia-smi helper does not allocate GPU memory'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Only frozen baseline; export modeling checkpoints, not AB/BA performance')
    parser.add_argument('--single-version',choices=('baseline','candidate'),default='baseline',help='Diagnostic prepare-only version; never an ABBA speed result')
    parser.add_argument('--reference-saved',help='Frozen same-checkpoint CPU tensors for strict single-version diagnostic')
    parser.add_argument('--csd-batch-size',type=int,choices=(2048,3072,4096),default=4096,help='Real full-mask memory diagnostic only; DTI/response remain4096')
    parser.add_argument('--allow-frozen-baseline-over-budget',action='store_true',help='Preserve baseline budget failure while continuing ABBA; candidate budget remains mandatory')
    parser.add_argument('--rounds',type=int,choices=(1,2),default=2,help='Number of complete ABBA rounds, prepare-only stays single run')
    args = parser.parse_args()
    if args.csd_batch_size != 4096 and not (args.prepare_only and args.reference_saved): parser.error('CSD batch diagnostics require prepare-only and frozen reference')
    if not args.prepare_only and (args.single_version != 'baseline' or args.reference_saved): parser.error('single-version/reference-saved require prepare-only')
    config = json.loads(Path(args.checkpoint).read_text())
    required = ('subject', 'manifest', 'dwi', 'gradient', 'brain_mask', 'response_mask', 'fod_mask', 'normalise_mask')
    if any(key not in config for key in required): raise ValueError(f'checkpoint requires {required}')
    if config.get('new_download_20261002') is not True: raise ValueError('new-download provenance must be explicitly verified')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    environment={'python_version':sys.version,'python_executable':sys.executable,'python_binary_sha256':sha(sys.executable),'torch_version':torch.__version__,'torch_cuda_version':torch.version.cuda,'torch_init_sha256':sha(torch.__file__),'numpy_version':np.__version__,'nibabel_version':nib.__version__,'conda_package_metadata_sha256':{p.name:sha(p) for p in sorted((Path(sys.prefix)/'conda-meta').glob('*.json'))},'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'allocator_config':os.environ.get('PYTORCH_CUDA_ALLOC_CONF','default'),'nvidia_driver_gpu':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).splitlines()}
    environment_sha256=hashlib.sha256(json.dumps(environment,sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(8); device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    files = {key: {'path': config[key], 'sha256': sha(config[key])} for key in required if key != 'subject'}
    if 'preproc_report' in config: files['preproc_report']={'path':config['preproc_report'],'sha256':sha(config['preproc_report'])}
    start = time.perf_counter()
    image = nib.load(config['dwi']); array = image.get_fdata(dtype=np.float32)
    arrays = {'signal': array, 'gradient': np.loadtxt(config['gradient']), 'affine': image.affine}
    for key in ('brain_mask', 'response_mask', 'fod_mask', 'normalise_mask'):
        mask_image = nib.load(config[key])
        if mask_image.shape != image.shape[:3] or not np.allclose(mask_image.affine, image.affine, rtol=0, atol=1e-6): raise ValueError(f'{key} geometry differs from DWI')
        arrays[key] = np.asarray(mask_image.dataobj) > 0
    read_s = time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats()
    with MemorySampler() as setup_sampler:
        setup_sampler.phase='setup_h2d'
        start = time.perf_counter()
        tensors = {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}
        torch.cuda.synchronize(); h2d_s = time.perf_counter()-start
        versions = {key: package(path, f'task02_{key}') for key, path in [('baseline',args.baseline),('candidate',args.candidate)]}
        setup_sampler.phase='setup_shells'
        shells = versions['baseline']['response'].mrtrix_shell_centres(tensors['gradient'])[2]
        torch.cuda.synchronize()
    setup_memory={'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),'nvml_process':setup_sampler.result()}
    layout_records=[];layout_instrumentation={}
    def record_csd_layout(version,signal,flat,phase):
        record={'version':version,'phase':phase,'signal_shape':list(signal.shape),'signal_stride':list(signal.stride()),'signal_is_contiguous':signal.is_contiguous(),'signal_storage_pointer':signal.untyped_storage().data_ptr(),'signal_storage_bytes':signal.untyped_storage().nbytes(),'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}
        if flat is not None:record.update({'flat_shape':list(flat.shape),'flat_stride':list(flat.stride()),'flat_is_contiguous':flat.is_contiguous(),'flat_storage_pointer':flat.untyped_storage().data_ptr(),'flat_storage_bytes':flat.untyped_storage().nbytes(),'shares_storage':flat.untyped_storage().data_ptr()==signal.untyped_storage().data_ptr()})
        layout_records.append(record)
    # Metadata-only probes around the actual reshape; never create a second
    # flat copy or change layout/shape/arithmetic to perform this audit.
    for version,modules in versions.items():
        text=inspect.getsource(modules['fod'].fit_mrtrix_msmt_csd)
        anchor_line='    flat_signal = signal.reshape(-1, signal.shape[-1])'
        assert text.count(anchor_line)==1
        observed_text=text.replace(anchor_line,f'    _record_csd_layout({version!r},signal,None,"before_reshape")\n'+anchor_line+f'\n    _record_csd_layout({version!r},signal,flat_signal,"after_reshape")')
        modules['fod'].__dict__['_record_csd_layout']=record_csd_layout
        exec(compile(observed_text,'<metadata-only CSD reshape audit>','exec'),modules['fod'].__dict__)
        layout_instrumentation[version]=hashlib.sha256(observed_text.encode()).hexdigest()
    setup_peak=setup_memory['nvml_process']['peak_bytes']
    setup_memory['budget_passed']=setup_peak is not None and 0 < setup_peak < 20_000_000_000 and setup_memory['allocated_peak_bytes'] < 20_000_000_000 and setup_memory['reserved_peak_bytes'] < 20_000_000_000
    report = {'kind': 'same corrected checkpoint component chain; not raw end-to-end', 'subject': config['subject'], 'hostname':socket.gethostname(), 'cpu_affinity':sorted(os.sched_getaffinity(0)), 'harness_sha256':sha(__file__), 'comparison_harness_sha256':sha(Path(__file__).with_name('tensor_comparison.py')), 'checkpoint_json_sha256':sha(args.checkpoint), 'files': files, 'declared_tolerance': {'neq': 0, 'max': 0, 'discrete_neq': 0}, 'batch_size': 4096, 'ABBA_rounds':args.rounds if not args.prepare_only else None, 'csd_batch_size':args.csd_batch_size, 'allocator_config':os.environ.get('PYTORCH_CUDA_ALLOC_CONF','default'), 'dtype': 'float32 input/output; float64 solvers', 'tf32': True, 'torch': torch.__version__, 'environment':environment, 'environment_sha256':environment_sha256, 'read_s':read_s, 'h2d_s':h2d_s, 'setup_memory':setup_memory,'csd_layout_records':layout_records,'layout_instrumentation_sha256':layout_instrumentation,'layout_policy':'metadata-only probes on actual reshape; no new layout conversion or duplicate reshape', 'allocator_round_policy':'synchronize and empty_cache before every measured chain; reset cost reported separately and included in run_total_wall_s', 'source': {key:{name:sha(Path(path)/f'{name}.py') for name in ('response','fod','mtnormalise')} for key,path in [('baseline',args.baseline),('candidate',args.candidate)]}, 'timing_policy': 'formal ABBA synchronises only whole-chain boundaries; stage CPU dispatch includes existing internal synchronisations, not GPU stage wall; optional profile stage-wall is separate', 'runs': []}
    def chain(version, instrument_stages=False, memory_sampler=None):
        modules = versions[version]; timing = {}
        def stage(name, function):
            if memory_sampler is not None: memory_sampler.phase=name
            torch.cuda.reset_peak_memory_stats()
            if instrument_stages: torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                value = function()
                if instrument_stages: torch.cuda.synchronize()
            except Exception as error:
                report['failed_run']={'version':version,'stage':name,'partial_timings':timing,'failed_stage_wall_s':time.perf_counter()-start,'error':repr(error)}
                (out/'report.json').write_text(json.dumps(report,indent=2))
                raise
            timing[name + ('_wall_s' if instrument_stages else '_cpu_dispatch_s')] = time.perf_counter()-start
            timing[name + '_memory_snapshot'] = {'allocated_bytes':torch.cuda.memory_allocated(), 'reserved_bytes':torch.cuda.memory_reserved(), 'allocated_stage_peak_bytes':torch.cuda.max_memory_allocated(), 'reserved_stage_peak_bytes':torch.cuda.max_memory_reserved()}
            return value
        torch.cuda.synchronize()
        start = time.perf_counter()
        fa, direction = stage('full_mask_tensor', lambda: modules['response'].fit_mrtrix_dhollander_tensor(tensors['signal'], tensors['gradient'], tensors['brain_mask']))
        shell, wmrf, gmrf, csfrf, selection = stage('dhollander', lambda: modules['response'].estimate_mrtrix_dhollander(tensors['signal'],tensors['gradient'],shells,tensors['response_mask']))
        wm, gm, csf = stage('msmt_csd',lambda: modules['fod'].fit_mrtrix_msmt_csd(tensors['signal'],tensors['gradient'],shell,wmrf,gmrf,csfrf,tensors['fod_mask'],batch_size=args.csd_batch_size))
        norm = stage('mtnormalise', lambda: modules['mtnormalise'].normalise_mrtrix_three_tissue(wm,gm,csf,tensors['normalise_mask'],tensors['affine']))
        torch.cuda.synchronize()
        timing['component_chain_wall_s'] = time.perf_counter()-start
        results = {'fa':fa,'direction':direction,'wmrf':wmrf,'gmrf':gmrf,'csfrf':csfrf,'wm':wm,'gm':gm,'csf':csf,'wm_norm':norm.wm,'gm_norm':norm.gm,'csf_norm':norm.csf,'field':norm.field,'accepted_mask':norm.accepted_mask,'factors':norm.balance_factors,**{f'selection_{k}':v for k,v in selection.items()}}
        return results,timing
    reference = torch.load(args.reference_saved,map_location='cpu',weights_only=True) if args.reference_saved else None
    if args.reference_saved: report['external_frozen_reference']={'path':args.reference_saved,'sha256':sha(args.reference_saved)}
    order = (args.single_version,) if args.prepare_only else ('baseline','candidate','candidate','baseline')*args.rounds
    for sequence,version in enumerate(order):
        run_start=time.perf_counter()
        reset_start=time.perf_counter();torch.cuda.synchronize();torch.cuda.empty_cache();allocator_reset_s=time.perf_counter()-reset_start
        torch.cuda.reset_peak_memory_stats()
        sampler=MemorySampler()
        try:
            with sampler: outputs,timing = chain(version,memory_sampler=sampler)
        except Exception:
            report['failure_nvml_process']=sampler.result()
            report.setdefault('failed_run',{'version':version,'stage':'chain_boundary'}).update({'allocated_stage_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_stage_peak_bytes':torch.cuda.max_memory_reserved()})
            (out/'report.json').write_text(json.dumps(report,indent=2))
            raise
        timing.update({'allocator_reset_s':allocator_reset_s,'version':version,'sequence':sequence,'abba_round':sequence//4 if not args.prepare_only else None,'within_round':sequence%4 if not args.prepare_only else None,'allocated_peak_bytes':max(v['allocated_stage_peak_bytes'] for k,v in timing.items() if k.endswith('_memory_snapshot')),'reserved_peak_bytes':max(v['reserved_stage_peak_bytes'] for k,v in timing.items() if k.endswith('_memory_snapshot')),'nvml_process':sampler.result()})
        start = time.perf_counter(); cpu = {key:value.detach().cpu() for key,value in outputs.items()}; timing['d2h_s'] = time.perf_counter()-start
        del outputs
        start = time.perf_counter(); torch.save(cpu,out/f'{sequence}_{version}.pt'); timing['write_s'] = time.perf_counter()-start
        report['runs'].append(timing)
        peak = timing['nvml_process']['peak_bytes']
        timing['memory_budget_passed'] = setup_memory['budget_passed'] and peak is not None and 0 < peak < 20_000_000_000 and timing['allocated_peak_bytes'] < 20_000_000_000 and timing['reserved_peak_bytes'] < 20_000_000_000
        timing['memory_acceptance_scope']='NVML sampled full-process chain plus exact Torch allocated/reserved stage peaks; setup separately reported; no guessed cap'
        (out/'report.json').write_text(json.dumps(report,indent=2))
        if sequence == 0 and reference is None: reference = cpu
        else:
            errors = compare_tensor_dicts(reference,cpu)
            timing['error_vs_first_baseline'] = errors
        export_start = time.perf_counter()
        if sequence <= 1:
            for key in ('fa','direction','wm_norm','gm_norm','csf_norm','wm','gm','csf','field','accepted_mask'):
                values = cpu[key].numpy()
                if values.dtype == np.bool_: values = values.astype(np.uint8)
                nib.save(nib.Nifti1Image(values, image.affine), out/f'{version}_{key}.nii.gz')
            for key in ('wmrf','gmrf','csfrf'):
                np.savetxt(out/f'{version}_{key}.txt', cpu[key].numpy(), fmt='%.17g')
        timing['checkpoint_export_s'] = time.perf_counter()-export_start
        del cpu
        timing['run_total_wall_s']=time.perf_counter()-run_start
        (out/'report.json').write_text(json.dumps(report,indent=2))
        baseline_budget_exception = args.allow_frozen_baseline_over_budget and version == 'baseline' and peak is not None and peak > 0
        if not timing['memory_budget_passed'] and baseline_budget_exception:
            timing['baseline_budget_exception'] = 'explicit root instruction: frozen baseline overshoot remains a failed budget observation; continue measuring candidate'
            (out/'report.json').write_text(json.dumps(report,indent=2))
        if not timing['memory_budget_passed'] and not baseline_budget_exception:
            report['acceptance_failed'] = 'whole-process GPU memory unavailable or >=20e9 bytes; outputs retained for diagnosis only'
            (out/'report.json').write_text(json.dumps(report,indent=2))
            raise RuntimeError(report['acceptance_failed'])
        if 'error_vs_first_baseline' in timing and any(value['neq'] or value.get('nonfinite_mismatch', 0) for value in errors.values()):
            raise RuntimeError('strict same-input equality failed; see report.json')
    report['formal_ABBA_complete']=not args.prepare_only and len(report['runs']) == 4*args.rounds
    (out/'report.json').write_text(json.dumps(report,indent=2))
    # Profile one real 4096-voxel CSD block after formal full-mask ABBA.
    # This diagnostic never replaces full-mask timing, equality or budget checks.
    if not args.prepare_only:
        block=torch.nonzero(tensors['fod_mask'].reshape(-1)).flatten()[:4096]
        real_data=tensors['signal'].reshape(-1,tensors['signal'].shape[-1])[block].to(torch.float64)
        response=[reference[key].to(device) for key in ('wmrf','gmrf','csfrf')]
        profile_report={'kind':'real first 4096-mask-voxel ICLS diagnostic, not a reduced MRI benchmark','voxels':int(block.numel()),'voxel_indices_sha256':hashlib.sha256(block.cpu().numpy().tobytes()).hexdigest(),'parent_harness_sha256':sha(__file__),'profiles':{},'outputs':{},'hardware_dram_bandwidth':'not measured; no hardware bandwidth counter in this harness'}
        profile_outputs={}
        for version in versions:
            design,constraints=versions[version]['fod']._mrtrix_msmt_design(tensors['gradient'],shells,*response)
            prepared=versions[version]['fod']._mrtrix_icls_design_matrices(design,constraints)
            torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            with MemorySampler() as sampled:
                sampled.phase='instrumented_real_icls_block'
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    solved=versions[version]['fod']._mrtrix_icls_batch(real_data,design,constraints,prepared=prepared)
                    torch.cuda.synchronize()
            profile_outputs[version]=solved.cpu()
            del solved
            trace=out/f'{version}.real_block.trace.json';prof.export_chrome_trace(str(trace))
            profile_report['profiles'][version]={'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),'nvml_process':sampled.result(),'trace':str(trace),'operators':[{'name':e.key,'calls':e.count,'cpu_self_us':e.self_cpu_time_total,'cuda_self_us':e.self_device_time_total} for e in prof.key_averages()]}
            del design,constraints,prepared
        left,right=profile_outputs['baseline'],profile_outputs['candidate'];delta=(left-right).abs()
        profile_report['strict_float64_error']={'neq':int((left!=right).sum()),'max':float(delta.max()) if delta.numel() else 0,'p99':float(torch.quantile(delta.reshape(-1),.99)) if delta.numel() else 0,'rmse':float(delta.square().mean().sqrt()) if delta.numel() else 0}
        (out/'real_block_profile.json').write_text(json.dumps(profile_report,indent=2))
        report['real_block_profile']=str(out/'real_block_profile.json')
        (out/'report.json').write_text(json.dumps(report,indent=2))
        if profile_report['strict_float64_error']['neq']:
            report['diagnostic_failed']='real ICLS block Float64 mismatch';(out/'report.json').write_text(json.dumps(report,indent=2));raise RuntimeError(report['diagnostic_failed'])
        del profile_outputs,real_data,response
    # Profiles are diagnostic and excluded from timing summaries; nested GPU times are not summed.
    if args.profile:
        for version in versions:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                outputs,stage_timing = chain(version, instrument_stages=True)
            del outputs
            (out/f'{version}.instrumented_stage_wall.json').write_text(json.dumps(stage_timing,indent=2))
            prof.export_chrome_trace(str(out/f'{version}.trace.json'))
            entries = [{'name':e.key,'calls':e.count,'cpu_self_us':e.self_cpu_time_total,'cuda_self_us':e.self_device_time_total} for e in prof.key_averages()]
            (out/f'{version}.profile.json').write_text(json.dumps(entries,indent=2))
    report['state']='completed';report['candidate_memory_passed']=all(r['memory_budget_passed'] for r in report['runs'] if r['version']=='candidate') if any(r['version']=='candidate' for r in report['runs']) else None
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(out/'report.json')

if __name__ == '__main__': main()
