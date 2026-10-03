"""只读 CPU 复核已完成 FNIT 五种子后处理的冻结源、实际输入/输出及旧seed0。"""
from __future__ import annotations
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def audit(config_path, config_sha256, output_root, cpu_preparation_root):
    started = time.perf_counter()
    worker = module(config_path.parent/'tools/reference/benchmark_connectome_fnit_repeats.py', 'frozen_five_post_audit')
    worker.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU output audit must hide GPUs')
    worker.require(worker.sha256(config_path) == config_sha256, 'frozen postprocessing config changed')
    config = json.loads(config_path.read_text()); source = worker.verify_source(config)
    sys.path.insert(0, str(config_path.parent/'tools'))
    common = module(config_path.parent/'tools/connectome_repeat_common.py', 'frozen_post_matrix_common')
    official_audit = module(config_path.parent/'tools/reference/audit_connectome_repeats.py', 'frozen_official_post_audit')
    reference = official_audit.audit(Path(config['official_manifest']), Path(config['checkpoint_dir']),
        Path(config['fnit_dir']), config_path.parent/'tools/reference/benchmark_connectome_repeats_official.py')
    import torch
    import numpy as np
    import nibabel as nib
    worker.require(not torch.cuda.is_initialized(), 'CPU source/reference audit initialized CUDA')
    controller_path = output_root/'controller.json'; controller = json.loads(controller_path.read_text())
    worker.require(controller['status'] == 'five_seed_postprocessing_completed' and
        controller['config_sha256'] == config_sha256, 'actual five-seed controller not complete or config differs')
    seed_rows = {}
    for seed in range(5):
        directory = output_root/f'seed-{seed}'; path = directory/'report.json'; report = json.loads(path.read_text())
        worker.require(report['status'] == 'completed' and report['memory_gate'] == 'passed' and
            report['config_sha256'] == config_sha256 and controller['seeds'][str(seed)]['returncode'] == 0 and
            controller['seeds'][str(seed)]['report_sha256'] == worker.sha256(path), 'actual seed completion/source differs')
        worker.require(report['script_sha256'] == worker.sha256(config_path.parent/'tools/reference/benchmark_connectome_fnit_repeats.py') and
            report['source'] == source and report['device_identity']['canonical_uuid'] == config['gpu_uuid'],
            'actual seed source/device identity differs')
        peaks = [report['cuda_memory']['peak_allocated_bytes'], report['cuda_memory']['peak_reserved_bytes'],
                 report['process_memory']['peak_process_tree_bytes']]
        worker.require(worker.memory_gate(peaks, report['process_memory']['failed_samples']) == 'passed', 'actual memory gate fails')
        tracking = Path(report['inputs']['tracking_directory'])
        for name,digest in report['inputs']['tracking_array_sha256'].items():
            worker.require(worker.sha256(tracking/f'{name}.npy') == digest, 'actual original packed input changed')
        worker.require(worker.sha256(tracking/'report.json') == report['inputs']['tracking_report_sha256'], 'actual tracking report changed')
        for name,digest in report['output_sha256'].items():
            worker.require(worker.sha256(directory/name) == digest, 'actual scalar/TCK output changed')
        arrays = {name: np.load(tracking/f'{name}.npy',allow_pickle=False) for name in worker.ARRAY_NAMES}
        count = worker.array_contract(arrays,np)
        original = [arrays['points'][int(a):int(b)] for a,b in zip(arrays['offsets'][:-1],arrays['offsets'][1:])]
        actual = list(nib.streamlines.load(directory/'tracks.tck',lazy_load=False).streamlines)
        worker.require(len(original)==len(actual)==count and all(a.dtype==b.dtype==np.float32 and
            np.array_equal(a.view(np.uint8),b.view(np.uint8)) for a,b in zip(original,actual)), 'actual TCK differs from original points')
        worker.require(worker.sha256(directory/'tracks.tck') == worker.sha256(cpu_preparation_root/f'seed-{seed}/tracks.tck'),
            'GPU-post CPU-serialized TCK differs from independent CPU population input')
        metrics = np.load(directory/'track_metrics.npz',allow_pickle=False)
        for key,source_key in [('lengths','lengths_mm'),('endpoints','endpoints')]:
            worker.require(metrics[key].dtype == arrays[source_key].dtype and
                np.array_equal(metrics[key].view(np.uint8), arrays[source_key].view(np.uint8)), 'stored original tracking metric changed')
        scalar_rows = {}
        for filename,key in [('sift2_weights.txt','weights'),('mean_fa.txt','mean_fa'),('lengths.txt','lengths')]:
            original_scalar = metrics[key]; scalar = np.loadtxt(directory/filename,ndmin=1).astype(original_scalar.dtype)
            worker.require(original_scalar.shape == scalar.shape and
                np.array_equal(original_scalar.view(np.uint8),scalar.view(np.uint8)), 'scalar text/NPZ readback lost bits')
            scalar_rows[key] = {'dtype':str(original_scalar.dtype),'shape':list(original_scalar.shape),
                'nonfinite_count':int((~np.isfinite(original_scalar)).sum()),'text_npz_readback_bits_equal':True}
        profiles = common.load_profiles(directory)
        worker.require(set(profiles)==set(config['atlases']), 'actual atlas set changed')
        for name,(matrices,meta) in profiles.items():
            source_meta = report['inputs']['source_profiles'][name]
            worker.require(meta['matrix_sha256']==report['atlas_timings_seconds'][name]['matrix_sha256'] and
                meta['atlas_sha256']==source_meta['atlas_sha256'] and meta['node_rows']==source_meta['node_rows'],
                'actual matrix/atlas/node output changed')
        seed_rows[str(seed)] = {'report_path':str(path),'report_sha256':worker.sha256(path),
            'tck_original_and_cpu_population_bits_equal':True,'tracks':count,'scalars':scalar_rows,
            'matrix_csv_count':len(profiles)*4,'matrix_atlas_nodes_sha_verified':True,
            'cuda_memory':report['cuda_memory'],'process_memory':report['process_memory'],
            'timings_seconds':report['timings_seconds'],'atlas_timings_seconds':report['atlas_timings_seconds'],
            'actual_gpu_step_size_hex':report['actual_gpu_step_size_hex'],'device_identity':report['device_identity'],
            'input_source':report['inputs'],'output_sha256':report['output_sha256']}
    old = common.load_profiles(Path(config['fnit_dir'])); new = common.load_profiles(output_root/'seed-0')
    seed0 = {}
    for name in old:
        common.check_metadata([old[name],new[name]],name)
        seed0[name] = {'metrics': common._compare(old[name][0],new[name][0]),'full_matrix_difference':{}}
        for kind in common.NAMES:
            a,b=old[name][0][kind],new[name][0][kind]
            delta=b-a
            seed0[name]['full_matrix_difference'][kind]={'array_equal':bool(np.array_equal(a,b)),
                'changed_entries':int((a!=b).sum()),'max_absolute_error':float(np.abs(delta).max()),
                'rmse':float(np.sqrt(np.mean(delta*delta))), 'old_csv_sha256':old[name][1]['matrix_sha256'][kind],
                'new_csv_sha256':new[name][1]['matrix_sha256'][kind]}
            # Frozen assignment emits int64 count and Float32 final floating matrices;
            # the production CLI writes %d / %.9g while this benchmark writes %.18e.
            dtype = np.int64 if kind == 'count' else np.float32
            old_tensor, new_tensor = a.astype(dtype), b.astype(dtype)
            formatted = io.StringIO()
            np.savetxt(formatted, new_tensor, delimiter=',', fmt='%d' if kind == 'count' else '%.9g')
            old_path = Path(old[name][1]['directory']) / f'connectome_{kind}.csv'
            seed0[name]['full_matrix_difference'][kind]['frozen_output_dtype'] = str(np.dtype(dtype))
            seed0[name]['full_matrix_difference'][kind]['original_dtype_readback_bits_equal'] = bool(np.array_equal(old_tensor.view(np.uint8),new_tensor.view(np.uint8)))
            seed0[name]['full_matrix_difference'][kind]['original_cli_format_replay_bytes_equal'] = formatted.getvalue().encode('ascii') == old_path.read_bytes()
    old_scalar_path = Path(config['checkpoint_dir'])/'track_metrics.npz'
    old_scalar, new_scalar = np.load(old_scalar_path,allow_pickle=False), np.load(output_root/'seed-0/track_metrics.npz',allow_pickle=False)
    scalar_diff = {}
    for kind in ('weights','lengths','mean_fa','endpoints'):
        a,b=old_scalar[kind],new_scalar[kind]
        worker.require(a.shape==b.shape and a.dtype==b.dtype, 'original seed0 scalar shape/dtype differs')
        scalar_diff[kind]={'dtype':str(a.dtype),'shape':list(a.shape),'array_equal':bool(np.array_equal(a,b)),
            'changed_entries':int((a!=b).sum()),'max_absolute_error':float(np.abs(a-b).max())}
    worker.require(not torch.cuda.is_initialized(), 'CPU post audit initialized CUDA')
    return {'schema_version':1,'state':'actual_five_seed_source_input_output_audit_completed',
        'scientific_parity':'not_assessed_by_audit; inspect separate empirical matrix/population envelope',
        'scope':'CPU read-only frozen-source/actual-output audit; no GPU/tracking/SIFT2/matrix recomputation',
        'script_sha256':worker.sha256(Path(__file__)),'config_sha256':config_sha256,'frozen_source':source,
        'controller_sha256':worker.sha256(controller_path),'official_reference_audit':reference,'seeds':seed_rows,
        'seed0_original_diagnostic_vs_reprocessed':seed0,
        'seed0_original_scalar_npz_sha256':worker.sha256(old_scalar_path),
        'seed0_original_vs_reprocessed_scalars':scalar_diff,
        'seed0_numeric_policy':'exact count, actual frozen output dtype bits, CLI format replay and observed Float64 SIFT2 differences are reported separately; no new tolerance or cause attribution',
        'cuda_initialized':False,'seconds':time.perf_counter()-started}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--cpu-preparation-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():raise ValueError('actual audit output must be new')
    report=audit(args.config,args.config_sha256,args.output_root,args.cpu_preparation_root)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':report['state'],'seeds':len(report['seeds']),'seconds':report['seconds']}))


if __name__=='__main__':main()
