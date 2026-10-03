"""真实固定TCK逐轨赋值＋严格相同长度文件：原矩阵函数CPU/CUDA对照。"""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import socket
import time

import nibabel as nib
import numpy as np
import torch

from benchmark_assignment_weights import MemorySampler, NAMES, checked, metrics, module, sha


def capture_nodes(mod):
    source=inspect.getsource(mod.build_connectomes)
    source=source.replace('    for start in range(0, count_tracks, batch_size):',
        '    assigned_nodes = torch.zeros((count_tracks, 2), device=device, dtype=torch.int64)\n    for start in range(0, count_tracks, batch_size):',1)
    anchor='        nodes = nearest_labels(endpoints[start:stop].reshape(-1, 3)).reshape(-1, 2)'
    if source.count(anchor)!=1:raise ValueError('frozen assignment node capture anchor changed')
    source=source.replace(anchor,anchor+'\n        assigned_nodes[start:stop] = nodes',1)
    source=source.replace('    return result','    result["assigned_nodes"] = assigned_nodes\n    return result',1)
    namespace=dict(vars(mod));exec(compile(source,'assignment_nodes_diagnostic','exec'),namespace)
    return namespace['build_connectomes'],sha_source(source)


def sha_source(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--abba-rounds',default=4,type=int)
    args=p.parse_args()
    if args.output.exists():raise ValueError('fresh output required')
    args.output.mkdir(parents=True)
    reference_sha=sha(args.reference);ref=json.loads(args.reference.read_text())
    if ref['state']!='completed' or ref['execution_completed'] is not True or len(ref['commands'])!=16 or any(c['returncode']!=0 for c in ref['commands']):raise ValueError('all 16 common-reference commands required')
    source_sha=sha(args.source);mod=module(args.source,'matrix_current');capture,capture_sha=capture_nodes(mod)
    producer_path=checked(ref['producer']['path'],ref['producer']['sha256'])
    producer=json.loads(producer_path.read_text())
    actual_seed=producer['outputs'][str(ref['seed'])]
    tracks=checked(ref['inputs']['tracks.tck']['path'],ref['inputs']['tracks.tck']['sha256'])
    endpoints_np=np.stack([(p[0],p[-1]) for p in nib.streamlines.load(str(tracks)).tractogram.streamlines])
    device=torch.device(args.device);torch.set_num_threads(8)
    if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=True
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    inputs={'endpoints':torch.as_tensor(endpoints_np,device=device,dtype=torch.float32)}
    for name,file,dtype in [('weights','sift2_weights.txt',torch.float64),('lengths','lengths.txt',torch.float32)]:
        path=checked(ref['inputs'][file]['path'],ref['inputs'][file]['sha256']);inputs[name]=torch.as_tensor(np.loadtxt(path),device=device,dtype=dtype)
    fa_path=checked(tracks.parent/'mean_fa.txt',actual_seed['scalars']['mean_fa.txt']['sha256'])
    inputs['fa']=torch.as_tensor(np.loadtxt(fa_path),device=device,dtype=torch.float32)
    report={'scope':'same actual official fixed TCK and same external lengths file; unchanged production source; component only',
        'host':socket.gethostname(),'device':str(device),'torch':torch.__version__, 'harness_sha256':sha(__file__),
        'helper_sha256':sha(Path(__file__).with_name('benchmark_assignment_weights.py')),
        'source':{'path':str(args.source),'sha256':source_sha},'diagnostic_capture_sha256':capture_sha,
        'reference':{'path':str(args.reference),'sha256':reference_sha},'streamlines':len(endpoints_np),
        'timing_policy':'uninstrumented source; resident inputs, warmup; 4 ABBA rounds alternating batch1024/256; synchronization boundaries; diagnostic capture separately',
        'profiles':{}}
    sampler=MemorySampler() if device.type=='cuda' else None
    if sampler:sampler.__enter__()
    try:
        for profile,info in ref['profiles'].items():
            image=nib.load(checked(info['atlas']['path'],info['atlas']['sha256']))
            kwargs={**inputs,'atlas':torch.as_tensor(np.asarray(image.dataobj).astype(np.int32),device=device),
                'affine':torch.as_tensor(image.affine,device=device,dtype=torch.float64)}
            matrices={name:np.loadtxt(checked(item['path'],item['sha256']),delimiter=',') for name,item in info['outputs'].items() if name!='assignments'}
            previous=actual_seed['profiles'][profile]
            for name in ['sift2_fbc','mean_fa']:
                matrices[name]=np.loadtxt(checked(Path(previous['directory'])/(name+'.csv'),previous['matrix_sha256'][name]),delimiter=',')
            native_nodes=np.loadtxt(checked(info['outputs']['assignments']['path'],info['outputs']['assignments']['sha256']),dtype=np.int64)
            if native_nodes.shape!=(len(endpoints_np),2):raise ValueError('native assignment count/order shape mismatch')
            sync();diagnostic=capture(**kwargs);sync()
            actual_nodes=diagnostic.pop('assigned_nodes').cpu().numpy();actual_nodes.sort(axis=1);native_nodes.sort(axis=1)
            arrays={name:value.cpu().numpy() for name,value in diagnostic.items()}
            for batch in [1024,256]:mod.build_connectomes(**kwargs,batch_size=batch);sync()
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            runs=[];batched={}
            for position,batch in enumerate([1024,256,256,1024]*args.abba_rounds):
                sync();started=time.perf_counter();result=mod.build_connectomes(**kwargs,batch_size=batch);sync()
                runs.append({'sequence':position,'batch_size':batch,'wall_s':time.perf_counter()-started})
                if batch not in batched:batched[batch]={name:value.cpu().numpy() for name,value in result.items()}
            profile_result={'atlas':info['atlas'],'actual_nodes_different_streamlines':int(np.count_nonzero(np.any(actual_nodes!=native_nodes,axis=1))),
                'accuracy':{name:metrics(actual,matrices[name]) for name,actual in arrays.items()}, 'runs':runs,
                'batch_parity':{name:bool(np.array_equal(batched[1024][name],batched[256][name])) for name in arrays},
                'median_wall_s':{str(batch):float(np.median([r['wall_s'] for r in runs if r['batch_size']==batch])) for batch in [1024,256]}}
            if device.type=='cuda':profile_result.update(allocated_peak_bytes=torch.cuda.max_memory_allocated(device),reserved_peak_bytes=torch.cuda.max_memory_reserved(device))
            np.savez_compressed(args.output/(profile+'.npz'),assigned_nodes=actual_nodes,native_nodes=native_nodes,**arrays)
            report['profiles'][profile]=profile_result
            checked(info['atlas']['path'],info['atlas']['sha256'])
            print(profile,'nodes',profile_result['actual_nodes_different_streamlines'],'lengthmax',profile_result['accuracy']['mean_length']['max_abs_error'],flush=True)
    finally:
        if sampler:sampler.__exit__()
    if sampler:
        report['nvml_process']=sampler.report();peaks=[v[k] for v in report['profiles'].values() for k in ['allocated_peak_bytes','reserved_peak_bytes']]
        report['sampled_memory_budget_passed']=bool(max(peaks)<20_000_000_000 and report['nvml_process']['peak_bytes'] is not None and report['nvml_process']['peak_bytes']<20_000_000_000)
    checked(args.source,source_sha);checked(args.reference,reference_sha)
    checked(producer_path,ref['producer']['sha256'])
    for item in ref['inputs'].values():checked(item['path'],item['sha256'])
    checked(fa_path,actual_seed['scalars']['mean_fa.txt']['sha256'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
