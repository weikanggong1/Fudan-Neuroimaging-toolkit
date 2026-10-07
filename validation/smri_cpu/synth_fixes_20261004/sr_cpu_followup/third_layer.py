"""Find the next difference after the exact first BN, using its full real output."""
import argparse
import copy
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sr_first_layer import digest,metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['reference','candidate'],required=True)
    p.add_argument('--second-dir',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--official-script',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():p.error('keep earlier attempts')
    a.output_dir.mkdir(parents=True)
    assert a.weights.stat().st_size==106163752 and digest(a.weights)=='a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b'
    source=np.load(a.second_dir/'bn.npy',mmap_mode='r')
    report={'arm':a.arm,'driver_sha256':digest(__file__),'input_bn_sha256':digest(a.second_dir/'bn.npy'),'scope':'full first maxpool and next 24-to-48 convolution; not complete inference timing','results':{}}
    if a.arm=='reference':
        assert digest(a.official_script)=='917b20e90b00c1b2807ffcc1e8e2db123d3a63d52c8d395b8ba5a7f078c8b098'
        ns=runpy.run_path(str(a.official_script),run_name='_isolated_reference_');tf=ns['tf']
        tf.config.threading.set_inter_op_parallelism_threads(8);tf.config.threading.set_intra_op_parallelism_threads(8)
        original=ns['build_model'](str(a.weights));conv=original.get_layer('unet_conv_downarm_1_0')
        conv.activation=tf.keras.activations.linear
        x=tf.keras.Input(shape=source.shape[2:]+(source.shape[1],))
        pool=tf.nn.max_pool3d(x,ksize=2,strides=2,padding='VALID');raw=conv(pool);elu=tf.nn.elu(raw)
        model=tf.keras.Model(x,[pool,raw,elu])
        started=time.perf_counter();values=model.predict(np.ascontiguousarray(source.transpose(0,2,3,4,1)),verbose=0)
        report['stage_seconds']=time.perf_counter()-started
        for name,value in zip(['pool','raw','elu'],values):np.save(a.output_dir/(name+'.npy'),value.transpose(0,4,1,2,3))
        report['tensorflow_version']=tf.__version__
    else:
        import torch
        from fnit.synthsr.model import SynthSRUNet,load_h5_weights
        from numba import set_num_threads
        from elu_numba import elu_torch
        torch.set_num_threads(8);set_num_threads(8)
        model=load_h5_weights(SynthSRUNet(),a.weights).eval()
        x=torch.from_numpy(np.array(source,copy=True,order='C'))
        ref={name:np.load(a.reference_dir/(name+'.npy'),mmap_mode='r') for name in ['pool','raw','elu']}
        with torch.no_grad():
            for mode in ['contiguous','channels_last','channels_last_no_mkldnn']:
                conv=copy.deepcopy(model.down[1][0]);tensor=x
                if mode.startswith('channels_last'):
                    conv=conv.to(memory_format=torch.channels_last_3d);tensor=x.to(memory_format=torch.channels_last_3d)
                tensor=torch.nn.functional.max_pool3d(tensor,2)
                row={'pool':metrics(ref['pool'],tensor.numpy())}
                with torch.backends.mkldnn.flags(enabled=not mode.endswith('no_mkldnn')):
                    started=time.perf_counter();raw=conv(tensor)
                    row['convolution_seconds']=time.perf_counter()-started
                    row['raw']=metrics(ref['raw'],raw.numpy())
                    row['elu']=metrics(ref['elu'],elu_torch(raw).numpy())
                report['results'][mode]=row
                del conv,tensor,raw
    report['status']='complete'
    (a.output_dir/'report.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
