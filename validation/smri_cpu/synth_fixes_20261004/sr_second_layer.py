"""Find the next SynthSR divergence using the exact real first ELU output.

The reference is isolated TensorFlow 2.13. Candidate convolution and BN use
PyTorch; only scalar diagnostics leave the server. No full inference benchmark
or production layout change is performed by this stage probe.
"""

import argparse
import copy
import json
from pathlib import Path
import runpy
import time

import numpy as np

from sr_first_layer import digest, metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['reference','candidate'],required=True)
    p.add_argument('--first-layer-dir',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--official-script',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():p.error('preserve previous stage records')
    a.output_dir.mkdir(parents=True)
    if a.weights.stat().st_size!=106163752 or digest(a.weights)!='a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b':raise ValueError('weight identity differs')
    report={'schema':'fnit.synthsr.second_layer.real.v1','arm':a.arm,
            'driver_sha256':digest(__file__),'first_elu_sha256':digest(a.first_layer_dir/'elu.npy'),
            'results':{},'scope':'second convolution and first BN only, full actual first ELU input'}
    source=np.load(a.first_layer_dir/'elu.npy',mmap_mode='r')
    if a.arm=='reference':
        if digest(a.official_script)!='917b20e90b00c1b2807ffcc1e8e2db123d3a63d52c8d395b8ba5a7f078c8b098':raise ValueError('reference source differs')
        ns=runpy.run_path(str(a.official_script),run_name='_isolated_reference_');tf=ns['tf']
        tf.config.threading.set_inter_op_parallelism_threads(8);tf.config.threading.set_intra_op_parallelism_threads(8)
        original=ns['build_model'](str(a.weights));conv=original.get_layer('unet_conv_downarm_0_1');norm=original.get_layer('unet_bn_down_0')
        conv.activation=tf.keras.activations.linear
        x=tf.keras.Input(shape=source.shape[2:]+(source.shape[1],))
        raw=conv(x);elu=tf.nn.elu(raw);bn=norm(elu,training=False)
        model=tf.keras.Model(x,[raw,elu,bn])
        started=time.perf_counter();values=model.predict(np.ascontiguousarray(source.transpose(0,2,3,4,1)),verbose=0)
        report['seconds']=time.perf_counter()-started
        for name,value in zip(['raw','elu','bn'],values):np.save(a.output_dir/(name+'.npy'),value.transpose(0,4,1,2,3))
        report['tensorflow_version']=tf.__version__
    else:
        import torch
        from fnit.synthsr.model import SynthSRUNet,load_h5_weights
        from sr_cpu_elu_prototype import _cpu_reference_elu
        torch.set_num_threads(8);model=load_h5_weights(SynthSRUNet(),a.weights).eval()
        x=torch.from_numpy(np.ascontiguousarray(source))
        ref={name:np.load(a.reference_dir/(name+'.npy'),mmap_mode='r') for name in ['raw','elu','bn']}
        with torch.inference_mode():
            for mode in ['contiguous','channels_last']:
                conv=copy.deepcopy(model.down[0][1]);tensor=x
                if mode=='channels_last':
                    conv=conv.to(memory_format=torch.channels_last_3d);tensor=x.to(memory_format=torch.channels_last_3d)
                started=time.perf_counter();raw=conv(tensor).contiguous()
                row={'convolution_seconds':time.perf_counter()-started,'raw':metrics(ref['raw'],raw.numpy()),
                     'native_elu':metrics(ref['elu'],torch.nn.functional.elu(raw).numpy()),
                     'reference_elu':metrics(ref['elu'],_cpu_reference_elu(raw).numpy())}
                report['results'][mode]=row
                del raw,tensor,conv
            norm=model.down_bn[0]
            exact_elu=torch.from_numpy(np.ascontiguousarray(ref['elu']))
            shape=(1,-1,1,1,1);mean=norm.running_mean.view(shape);var=norm.running_var.view(shape)
            weight=norm.weight.view(shape);bias=norm.bias.view(shape)
            scale=weight/torch.sqrt(var+norm.eps)
            for name,operation in [('native',lambda:norm(exact_elu)),
                                   ('subtract_first',lambda:(exact_elu-mean)*scale+bias),
                                   ('scale_bias',lambda:exact_elu*scale+(bias-mean*scale))]:
                report['results']['bn_'+name]=metrics(ref['bn'],operation().numpy())
    (a.output_dir/'report.public.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
