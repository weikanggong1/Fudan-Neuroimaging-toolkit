"""Trace both actual flip inputs to the first remaining down-path difference."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sr_first_layer import digest,metrics


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['reference','candidate'],required=True)
    for name in ('network-input-dir','weights','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-dir',type=Path)
    p.add_argument('--official-script',type=Path)
    a=p.parse_args()
    if a.output_dir.exists():p.error('preserve previous attempts')
    a.output_dir.mkdir(parents=True)
    assert a.weights.stat().st_size==106163752 and digest(a.weights)=='a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b'
    report={'arm':a.arm,'driver_sha256':digest(__file__),'branches':{},'scope':'two exact actual saved CNN inputs, down-path intermediate stages; not API performance'}
    if a.arm=='reference':
        assert digest(a.official_script)=='917b20e90b00c1b2807ffcc1e8e2db123d3a63d52c8d395b8ba5a7f078c8b098'
        ns=runpy.run_path(str(a.official_script),run_name='_isolated_reference_');tf=ns['tf']
        tf.config.threading.set_inter_op_parallelism_threads(8);tf.config.threading.set_intra_op_parallelism_threads(8)
        original=ns['build_model'](str(a.weights))
        names=[name for level in range(5) for name in [f'unet_conv_downarm_{level}_0',f'unet_conv_downarm_{level}_1',f'unet_bn_down_{level}']]
        model=tf.keras.Model(original.inputs,[original.get_layer(name).output for name in names])
        report['tensorflow_version']=tf.__version__
        for index in (0,1):
            path=a.network_input_dir/f'network_{index}_input.npy'
            source=np.load(path)
            values=model.predict(np.ascontiguousarray(source.transpose(0,2,3,4,1)),verbose=0)
            output=a.output_dir/str(index);output.mkdir()
            row={'input_file_sha256':digest(path),'input_array_sha256':array_sha(source),'stages':{}}
            for name,value in zip(names,values):
                value=value.transpose(0,4,1,2,3)
                row['stages'][name]={'shape':list(value.shape),'array_sha256':array_sha(value)}
                if 'downarm_0' not in name:
                    np.save(output/(name+'.npy'),value)
            report['branches'][str(index)]=row
            del values,source
    else:
        import torch
        from numba import set_num_threads
        from fnit.synthsr.model import SynthSRUNet,load_h5_weights
        from elu_numba import elu_torch
        from bn_numba import batch_norm_torch
        torch.set_num_threads(8);set_num_threads(8)
        model=load_h5_weights(SynthSRUNet(),a.weights).eval().to(memory_format=torch.channels_last_3d)
        reference=json.loads((a.reference_dir/'report.public.json').read_text())
        with torch.no_grad():
            for index in (0,1):
                path=a.network_input_dir/f'network_{index}_input.npy'
                source=np.load(path)
                value=torch.from_numpy(source).to(memory_format=torch.channels_last_3d)
                row={'input_file_sha256':digest(path),'input_array_sha256':array_sha(value.numpy()),'input_hash_matches_reference':array_sha(value.numpy())==reference['branches'][str(index)]['input_array_sha256'],'stages':{},'first_difference':None}
                output=a.output_dir/str(index);output.mkdir()
                for level,(convs,norm) in enumerate(zip(model.down,model.down_bn)):
                    for part,conv in enumerate(convs):
                        raw=conv(value);value=elu_torch(raw)
                        name=f'unet_conv_downarm_{level}_{part}'
                        sha=array_sha(value.numpy());target=reference['branches'][str(index)]['stages'][name]
                        row['stages'][name]={'array_sha256':sha,'exact':sha==target['array_sha256']}
                        if sha!=target['array_sha256'] and row['first_difference'] is None:
                            row['first_difference']=name
                            np.save(output/'first_raw.npy',raw.numpy());np.save(output/'first_elu.npy',value.numpy())
                            refpath=a.reference_dir/str(index)/(name+'.npy')
                            if refpath.exists():row['stages'][name]['metrics']=metrics(np.load(refpath,mmap_mode='r'),value.numpy())
                        del raw
                    value=batch_norm_torch(value,norm)
                    name=f'unet_bn_down_{level}'
                    sha=array_sha(value.numpy());target=reference['branches'][str(index)]['stages'][name]
                    row['stages'][name]={'array_sha256':sha,'exact':sha==target['array_sha256']}
                    if sha!=target['array_sha256'] and row['first_difference'] is None:
                        row['first_difference']=name
                        np.save(output/'first_bn.npy',value.numpy())
                        row['stages'][name]['metrics']=metrics(np.load(a.reference_dir/str(index)/(name+'.npy'),mmap_mode='r'),value.numpy())
                    if level<len(model.down)-1:value=torch.nn.functional.max_pool3d(value,2)
                report['branches'][str(index)]=row
                del value,source
    report['status']='complete'
    (a.output_dir/'report.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
