"""Real-image SynthMorph API/observer worker for independent benchmarks."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch
from fnit.synthmorph import SynthMorph, convert_warp_to_fsl
from fnit.synthmorph import models
import fnit


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def arrays(result,model):
    fields={'forward':result.transform.matrix if model in ('affine','rigid') else np.asarray(result.transform.dataobj)}
    if result.inverse is not None:fields['inverse']=result.inverse.matrix if model in ('affine','rigid') else np.asarray(result.inverse.dataobj)
    for name in ('moved','fixed_moved'):
        image=getattr(result,name)
        if image is not None:fields[name]=np.asarray(image.dataobj)
    return fields


def main():
    p=argparse.ArgumentParser()
    for name in ('moving','fixed','weights','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--model',default='joint',choices=('rigid','affine','deform','joint'))
    p.add_argument('--extent',type=int,default=256);p.add_argument('--hyper',type=float,default=.5);p.add_argument('--steps',type=int,default=7)
    p.add_argument('--init');p.add_argument('--mid-space',action='store_true');p.add_argument('--header-only',action='store_true');p.add_argument('--debug',action='store_true')
    p.add_argument('--transform-only',action='store_true');p.add_argument('--no-inverse',action='store_true');p.add_argument('--observer',action='store_true');p.add_argument('--functional',action='store_true')
    args=p.parse_args();torch.set_num_threads(8);device=torch.device(args.device)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    report={'scope':'real-image API, load and saving separate; observer durations excluded from official speed comparisons','source_root':fnit.__file__,'model':args.model,'device':str(device),'extent':args.extent,'hyper':args.hyper,'steps':args.steps,'input_sha256':{n:digest(getattr(args,n)) for n in ('moving','fixed')},'flags_before':{'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32}}
    if device.type=='cuda':
        torch.cuda.set_device(device);total=torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(19_000_000_000/total,device);torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter();register=SynthMorph(weights=args.weights,device=device,model=args.model,extent=args.extent,hyper=args.hyper,steps=args.steps);synchronize(device)
    report['model_load_seconds']=time.perf_counter()-start
    report['flags_after']={'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32}
    times=[];handles=[];active=[];phase_times=[];original_functions={}
    if args.observer:
        import fnit.synthmorph.pipeline as pipeline
        for name in ('_load', 'network_space', 'transform', 'compose', '_resampled_image', 'voxel_displacement_to_ras'):
            original_functions[name]=getattr(pipeline,name)
            def wrap(function,name):
                @functools.wraps(function)
                def timed(*positional,**keywords):
                    started=time.perf_counter()
                    try:return function(*positional,**keywords)
                    finally:phase_times.append({'function':name,'inclusive_seconds':time.perf_counter()-started})
                return timed
            setattr(pipeline,name,wrap(original_functions[name],name))
    if args.observer:
        for name,module in register.network.named_modules():
            if isinstance(module,(models.FeatureDetector,models.DeformNetwork,models.AffineNetwork,models.SynthMorphNetwork)):
                def pre(mod,inp,name=name):active.append((name,time.perf_counter()))
                def post(mod,inp,out,name=name):
                    item=active.pop();assert item[0]==name
                    times.append({'module':name,'seconds':time.perf_counter()-item[1]})
                handles.extend((module.register_forward_pre_hook(pre),module.register_forward_hook(post)))
    precision=[];call={'init':args.init,'mid_space':args.mid_space,'header_only':args.header_only,'output_dir':output/'debug' if args.debug else None,'transform_only':args.transform_only,'compute_inverse':not args.no_inverse,'precision_report':precision}
    synchronize(device);start=time.perf_counter();result=register(args.moving,args.fixed,**call);synchronize(device)
    report['api_seconds']=time.perf_counter()-start
    for handle in handles:handle.remove()
    for name,function in original_functions.items():setattr(pipeline,name,function)
    report['nested_module_observer']=times;report['phase_observer']=phase_times;report['precision_report']=precision
    report['worker_sha256']=digest(__file__)
    fields=arrays(result,args.model);report['array_files']={}
    for key,data in fields.items():
        path=output/(key+'.npy');np.save(path,data)
        report['array_files'][key]={'sha256':digest(path),'shape':list(data.shape),'dtype':str(data.dtype),'finite':bool(np.isfinite(data).all())}
    start=time.perf_counter()
    for name in ('moved','fixed_moved','transform','inverse'):
        item=getattr(result,name)
        if item is not None:item.save(output/(name+('.lta' if name in ('transform','inverse') and args.model in ('affine','rigid') else '.nii.gz')))
    report['save_seconds']=time.perf_counter()-start
    if args.functional and not args.header_only and not args.transform_only:
        checks={}
        start=time.perf_counter();only=register(args.moving,args.fixed,transform_only=True,init=args.init,mid_space=args.mid_space);synchronize(device)
        checks['transform_only_seconds']=time.perf_counter()-start;checks['transform_only_no_images']=only.moved is None and only.fixed_moved is None
        checks['transform_only_arrays_equal']={key:bool(np.array_equal(value,arrays(only,args.model)[key])) for key,value in fields.items() if key in ('forward','inverse')}
        if args.model in ('joint','deform'):
            start=time.perf_counter();forward=register(args.moving,args.fixed,transform_only=True,compute_inverse=False,init=args.init,mid_space=args.mid_space);synchronize(device)
            checks['forward_only_seconds']=time.perf_counter()-start;checks['forward_only_inverse_none']=forward.inverse is None
            checks['forward_only_array_equal']=bool(np.array_equal(fields['forward'],arrays(forward,args.model)['forward']))
            start=time.perf_counter();fsl=convert_warp_to_fsl(result.transform,moving=args.moving,fixed=args.fixed)
            checks['convert_to_fsl_seconds']=time.perf_counter()-start;fsl.save(output/'fsl_warp.nii.gz');checks['fsl_intent_code']=int(fsl.header['intent_code'])
        report['functional']=checks
    if device.type=='cuda':report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(device);report['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved(device)
    source=Path(fnit.__file__).parent
    report['source_sha256']={str(path.relative_to(source)):digest(path) for path in [source/'_nib.py',source/'_transforms.py',*(source/'synthmorph').glob('*.py')]}
    (output/'report.private.json').write_text(json.dumps(report,indent=2,default=str)+'\n')


if __name__=='__main__':main()
