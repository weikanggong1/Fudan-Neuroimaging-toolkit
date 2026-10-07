"""Bounded actual-layer-parameter contracts; stop before MRI on any bit change."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from unittest.mock import patch


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(part)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','plan','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();os.umask(0o077)
    plan=json.loads(args.plan.read_text())
    sources={n:sha(args.source/'fnit/synthseg_parc'/n) for n in plan['source_files']}
    assert sources==plan['source_files']
    weight=args.root/plan['weight_fnit_relative_path']
    assert sha(weight)==plan['weight']['sha256'] and weight.stat().st_size==plan['weight']['bytes']
    sys.path.insert(0,str(args.source))
    import torch
    import candidate
    from fnit.synthseg_parc.segment import SegmentUNet
    from fnit.synthseg_parc.cpu_conv import convolution_slabs
    assert sha(candidate.__file__)==plan['prototype_files']['candidate.py']
    torch.set_num_threads(8);torch.set_num_interop_threads(8)
    assert sorted(os.sched_getaffinity(0))==plan['cpu_affinity']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    def flags():
        return {'matmul_tf32':bool(torch.backends.cuda.matmul.allow_tf32),
                'cudnn_tf32':bool(torch.backends.cudnn.allow_tf32),
                'mkldnn':bool(torch.backends.mkldnn.enabled),
                'cpu_autocast':bool(torch.is_autocast_enabled('cpu')),
                'grad':bool(torch.is_grad_enabled()),'cuda_initialized':bool(torch.cuda.is_initialized())}
    initial=flags()
    model=SegmentUNet().load_h5(weight).eval();layer=model.up[3].conv0
    assert tuple(layer.weight.shape)==(24,72,3,3,3) and tuple(layer.bias.shape)==(24,)
    generator=torch.Generator().manual_seed(20261006)
    rows=[];failed=False;fallback_count=0
    with torch.inference_mode(),torch.backends.mkldnn.flags(enabled=False):
        for name,depth,strided,cancellation in (
                ('single_depth',1,False,False),('two_depth',2,False,False),
                ('boundary_internal',7,False,False),('many_slabs',31,False,False),
                ('strided_odd_depth',33,True,False),('cancellation',31,False,True)):
            image=torch.randn((1,72,depth,13,17),generator=generator)
            image[:,0].zero_();image[:,1].fill_(-0.)
            if cancellation:
                image[:,2::2].mul_(1e4);image[:,3::2].mul_(-1e4)
            if strided:
                storage=torch.empty((1,72,depth,13,34));storage[...,::2].copy_(image);image=storage[...,::2]
            input_bits=image.clone().view(torch.int32)
            weight_bits=layer.weight.clone().view(torch.int32);bias_bits=layer.bias.clone().view(torch.int32)
            plane=72*13*17*4
            old=convolution_slabs(image,layer.weight,layer.bias,padding=1,maximum_slab_bytes=16*plane)
            actual=candidate.forward(layer,image,maximum_slab_bytes=4*plane)
            different=int(torch.count_nonzero(old.view(torch.int32)!=actual.view(torch.int32)))
            row={'name':name,'shape':list(image.shape),'strided':strided,'actual_weight_and_bias':True,
                 'old_slab_depth':14,'new_slab_depth':2,'bit_different_values':different,
                 'max_absolute':float((old-actual).abs().max()),'finite':bool(torch.isfinite(actual).all()),
                 'shape_stride_dtype_equal':old.shape==actual.shape and old.stride()==actual.stride() and old.dtype==actual.dtype,
                 'input_unchanged':bool(torch.equal(image.clone().view(torch.int32),input_bits)),
                 'weight_bias_unchanged':bool(torch.equal(layer.weight.view(torch.int32),weight_bits) and torch.equal(layer.bias.view(torch.int32),bias_bits)),
                 'output_independent':actual.untyped_storage().data_ptr()!=image.untyped_storage().data_ptr()}
            rows.append(row)
            assert all(row[key] for key in ('finite','shape_stride_dtype_equal','input_unchanged','weight_bias_unchanged','output_independent'))
            if different:
                failed=True;break
    # Only exact arithmetic contracts allow later guards and MRI arms.
    if not failed:
        sample=torch.zeros((1,72,2,3,4))
        sentinel=object()
        def old_forward(image):
            nonlocal fallback_count
            fallback_count+=1;return sentinel
        with patch.object(layer,'forward',side_effect=old_forward):
            assert candidate.forward(layer,sample) is sentinel  # grad-enabled
            with torch.inference_mode(),torch.backends.mkldnn.flags(enabled=True):
                assert candidate.forward(layer,sample) is sentinel
            with torch.inference_mode(),torch.backends.mkldnn.flags(enabled=False):
                layer.train();assert candidate.forward(layer,sample) is sentinel;layer.eval()
                assert candidate.forward(layer,torch.zeros_like(sample,dtype=torch.float64)) is sentinel
                assert candidate.forward(layer,torch.empty(sample.shape,device='meta')) is sentinel
                with patch('fnit.synthseg_parc.cpu_conv.cpu_autocast_enabled',return_value=True):
                    assert candidate.forward(layer,sample) is sentinel
                hook=layer.register_forward_hook(lambda *_:None)
                try:assert candidate.forward(layer,sample) is sentinel
                finally:hook.remove()
        assert fallback_count==7
        with torch.inference_mode(),torch.backends.mkldnn.flags(enabled=False):
            with patch('fnit.synthseg_parc.cpu_conv.convolution_slabs',side_effect=RuntimeError('declared failure')):
                try:candidate.forward(layer,sample)
                except RuntimeError as exc:assert str(exc)=='declared failure'
                else:raise AssertionError('candidate swallowed exception')
    after={n:sha(args.source/'fnit/synthseg_parc'/n) for n in sources}
    report={'schema':'fnit_seg_CPU_slab_contracts/v1','status':'bit_gate_failed_stop_before_MRI' if failed else 'passed',
        'scope':'Synthetic FP32 inputs with actual C72->24 model weight/bias; no MRI layer or benchmark',
        'rows':rows,'fallback_calls':fallback_count,'remaining_contracts_skipped_on_first_nonexact':failed,
        'torch_version':torch.__version__,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_parallel_info':torch.__config__.parallel_info(),
        'initial_flags':initial,'final_flags':flags(),'source_before':sources,'source_after':after,
        'weight_sha256':sha(weight),'helper_sha256':sha(candidate.__file__),'worker_sha256':sha(__file__),
        'plan_sha256':sha(args.plan),'maximum_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'real_MRI_executed':False,'production_changed':False}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    assert after==sources and report['final_flags']==initial and not torch.cuda.is_initialized()
    print(json.dumps({'status':report['status'],'completed_contract_rows':len(rows)}))
    if failed:raise SystemExit(2)


if __name__=='__main__':main()
