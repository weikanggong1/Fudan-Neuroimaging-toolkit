"""One saved real up3.conv0 old/new call; no remaining decoder or full CNN."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()


def tensor_sha(value):
    array=value.detach().numpy()
    assert array.shape[0]==1 and array.flags.c_contiguous
    h=hashlib.sha256()
    for channel in range(array.shape[1]):h.update(memoryview(array[0,channel]).cast('B'))
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','plan','checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=('baseline','candidate'),required=True)
    p.add_argument('--reference',type=Path)
    args=p.parse_args();os.umask(0o077);args.output.mkdir(mode=0o700)
    plan=json.loads(args.plan.read_text());started=time.perf_counter()
    before={n:sha(args.source/'fnit/synthseg_parc'/n) for n in plan['source_files']}
    assert before==plan['source_files']
    for name,info in plan['checkpoint_files'].items():
        assert sha(args.checkpoint/name)==info['sha256'] and (args.checkpoint/name).stat().st_size==info['bytes']
    manifest=json.loads((args.checkpoint/'manifest.private.json').read_text())
    assert manifest['status']=='partial_capture_complete' and manifest['source_files']==plan['producer_files']
    assert manifest['input_sha256']==plan['input_sha256'] and manifest['prepared_values_sha256']==plan['prepared_values_sha256']
    weight=args.root/plan['weight_fnit_relative_path']
    assert sha(weight)==plan['weight']['sha256'] and weight.stat().st_size==plan['weight']['bytes']
    sys.path.insert(0,str(args.source))
    import numpy as np
    import torch
    import fnit
    import candidate
    from fnit.synthseg_parc.segment import SegmentUNet
    from fnit.synthseg_parc.cpu_join import cpu_join_allowed,join_nearest_cpu
    assert Path(fnit.__file__).resolve()==(args.source/'fnit/__init__.py').resolve()
    assert sha(candidate.__file__)==plan['prototype_files']['candidate.py']
    torch.set_num_threads(8);torch.set_num_interop_threads(8)
    assert sorted(os.sched_getaffinity(0))==plan['cpu_affinity']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    def flags():
        return {'matmul_tf32':bool(torch.backends.cuda.matmul.allow_tf32),
                'cudnn_tf32':bool(torch.backends.cudnn.allow_tf32),'mkldnn':bool(torch.backends.mkldnn.enabled),
                'cpu_autocast':bool(torch.is_autocast_enabled('cpu')),'grad':bool(torch.is_grad_enabled()),
                'cuda_initialized':bool(torch.cuda.is_initialized())}
    initial=flags();model=SegmentUNet().load_h5(weight).eval();layer=model.up[3].conv0
    skip=torch.from_numpy(np.load(args.checkpoint/'skip.npy'))
    value=torch.from_numpy(np.load(args.checkpoint/'value.npy'))
    assert tuple(skip.shape)==(1,24,192,224,256) and tuple(value.shape)==(1,48,96,112,128)
    assert skip.dtype==value.dtype==torch.float32
    reference_file=None
    if args.reference is not None:
        reference_meta=json.loads((args.reference.parent/'report.private.json').read_text())
        assert reference_meta['mode']=='baseline' and reference_meta['reference_generation_only']
        reference_file={'sha256':sha(args.reference),'bytes':args.reference.stat().st_size,
                        'value_sha256':reference_meta['output']['value_sha256']}
        assert reference_file==reference_meta['saved_reference']
    with torch.inference_mode(),torch.backends.mkldnn.flags(enabled=False):
        assert cpu_join_allowed(model,skip,value)
        image=join_nearest_cpu(skip,value);del skip,value
        input_hash=tensor_sha(image)
        assert input_hash==plan['prior_join_bit_identity_sha256']
        assert candidate.eligible(layer,image)
        weight_hash=tensor_sha(layer.weight.view(1,*layer.weight.shape))
        bias_hash=tensor_sha(layer.bias.view(1,24,1,1,1))
        usage_before=resource.getrusage(resource.RUSAGE_SELF);clock=time.perf_counter()
        actual=(layer(image) if args.mode=='baseline' else candidate.forward(layer,image))
        operation=time.perf_counter()-clock;usage_after=resource.getrusage(resource.RUSAGE_SELF)
        assert tensor_sha(image)==input_hash
        assert tensor_sha(layer.weight.view(1,*layer.weight.shape))==weight_hash
        assert tensor_sha(layer.bias.view(1,24,1,1,1))==bias_hash
    assert tuple(actual.shape)==(1,24,192,224,256) and actual.dtype==torch.float32
    assert actual.is_contiguous() and actual.device.type=='cpu'
    assert actual.untyped_storage().data_ptr()!=image.untyped_storage().data_ptr()
    del image,model,layer
    array=actual.numpy();different=0;maximum=0.;finite=True
    reference=np.load(args.reference,mmap_mode='r') if args.reference else None
    if reference is not None:assert array.shape==reference.shape and array.dtype==reference.dtype
    for channel in range(24):
        for depth in range(0,192,8):
            part=array[:,channel,depth:depth+8];finite=finite and bool(np.isfinite(part).all())
            if reference is not None:
                expected=reference[:,channel,depth:depth+8]
                different+=int(np.count_nonzero(part.view(np.uint32)!=expected.view(np.uint32)))
                maximum=max(maximum,float(np.max(np.abs(part-expected))))
    output={'value_sha256':tensor_sha(actual),'shape':list(actual.shape),'stride':list(actual.stride()),'dtype':str(actual.dtype)}
    saved_reference=None
    if reference is None:
        assert args.mode=='baseline';file=args.output/'conv0_baseline.private.npy';np.save(file,array)
        saved_reference={'sha256':sha(file),'bytes':file.stat().st_size,'value_sha256':output['value_sha256']}
    else:
        assert sha(args.reference)==reference_file['sha256']
    after={n:sha(args.source/'fnit/synthseg_parc'/n) for n in before}
    report={'schema':'fnit_seg_CPU_single_layer_trial/v1','mode':args.mode,
        'scope':'Saved decoder input, one up3.conv0 pre-ELU call only; not a complete MRI pipeline or benchmark',
        'status':'single_layer_bit_gate_passed' if finite and different==0 else 'single_layer_bit_gate_failed',
        'worker_sha256':sha(__file__),'helper_sha256':sha(candidate.__file__),'plan_sha256':sha(args.plan),
        'source_before':before,'source_after':after,'source_unchanged':before==after,
        'input_sha256':plan['input_sha256'],'checkpoint_files':plan['checkpoint_files'],'joined_input_value_sha256':input_hash,
        'joined_input_unchanged':True,'weight_file_sha256_before_and_after':sha(weight),
        'weight_bias_values_unchanged':True,'weight_values_sha256':weight_hash,'bias_values_sha256':bias_hash,
        'reference_generation_only':reference is None,'saved_reference':saved_reference,'comparison_reference':reference_file,
        'output':output,'bit_gate':{'comparison_executed':reference is not None,'bit_different_values':different,
            'max_absolute':maximum,'all_finite':finite,'output_independent':True,
            'shape_stride_dtype_same':reference is None or list(reference.strides)==list(array.strides)},
        'operation_seconds':operation,'process_user_seconds':usage_after.ru_utime-usage_before.ru_utime,
        'process_system_seconds':usage_after.ru_stime-usage_before.ru_stime,
        'process_minor_faults':usage_after.ru_minflt-usage_before.ru_minflt,
        'process_major_faults':usage_after.ru_majflt-usage_before.ru_majflt,
        'maximum_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'initial_flags':initial,'final_flags':flags(),'flags_unchanged':initial==flags(),
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'threads':torch.get_num_threads(),
        'interop_threads':torch.get_num_interop_threads(),'torch_version':torch.__version__,
        'torch_parallel_info':torch.__config__.parallel_info(),'maximum_slab_bytes':plan['caps'][args.mode],
        'full_CNN_executed':False,'ELU_BN_head_softmax_executed':False,'whole_map_CSV_assessed':False,
        'worker_observed_seconds_with_load_save_hash_compare':time.perf_counter()-started}
    (args.output/'report.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    assert report['weight_file_sha256_before_and_after']==plan['weight']['sha256']
    assert report['source_unchanged'] and report['flags_unchanged']
    assert report['maximum_RSS_bytes']<=plan['gates']['maximum_RSS_bytes'] and report['bit_gate']['shape_stride_dtype_same']
    print(json.dumps({'status':report['status'],'mode':args.mode,'operation_seconds':operation}))
    if report['status']=='single_layer_bit_gate_failed':raise SystemExit(2)


if __name__=='__main__':main()
