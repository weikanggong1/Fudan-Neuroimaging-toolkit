"""Isolated TF fusion control; the unchanged whole-graph oracle stays frozen."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('official-script','weights','network-input-dir','whole-reference-dir','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():p.error('keep prior control')
    a.output_dir.mkdir(parents=True)
    assert digest(a.official_script)=='917b20e90b00c1b2807ffcc1e8e2db123d3a63d52c8d395b8ba5a7f078c8b098'
    assert a.weights.stat().st_size==106163752 and digest(a.weights)=='a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b'
    ns=runpy.run_path(str(a.official_script),run_name='_isolated_reference_');tf=ns['tf']
    tf.config.threading.set_inter_op_parallelism_threads(8);tf.config.threading.set_intra_op_parallelism_threads(8)
    original=ns['build_model'](str(a.weights))
    oracle=json.loads((a.whole_reference_dir/'report.public.json').read_text())
    report={'driver_sha256':digest(__file__),'tensorflow_version':tf.__version__,'scope':'prefix-only fusion controls; does not replace original full inference oracle or benchmark','controls':{}}
    for mode in ('original','remapping_disabled'):
        tf.config.optimizer.set_experimental_options({} if mode=='original' else {'remapping':False})
        model=tf.keras.Model(original.inputs,original.get_layer('unet_conv_downarm_0_1').output)
        output=a.output_dir/mode;output.mkdir()
        row={}
        profile=output/'profile.private'
        tf.profiler.experimental.start(str(profile),tf.profiler.experimental.ProfilerOptions(host_tracer_level=2,python_tracer_level=0,device_tracer_level=0))
        try:
            for index in (0,1):
                path=a.network_input_dir/f'network_{index}_input.npy';x=np.load(path)
                result=model.predict(np.ascontiguousarray(x.transpose(0,2,3,4,1)),verbose=0).transpose(0,4,1,2,3)
                np.save(output/f'branch{index}_elu.npy',result)
                sha=array_sha(result)
                row[str(index)]={'input_file_sha256':digest(path),'output_array_sha256':sha,
                    'matches_frozen_whole_graph':sha==oracle['branches'][str(index)]['stages']['unet_conv_downarm_0_1']['array_sha256']}
        finally:tf.profiler.experimental.stop()
        from tensorflow.core.profiler.protobuf import xplane_pb2
        names=[]
        for path in profile.rglob('*.xplane.pb'):
            space=xplane_pb2.XSpace();space.ParseFromString(path.read_bytes())
            for plane in space.planes:
                names.extend(metadata.name for metadata in plane.event_metadata.values() if any(k in metadata.name.lower() for k in ('conv','elu','bias')))
        row['actual_execution_event_names']=sorted(set(names))
        row['optimizer_options']=tf.config.optimizer.get_experimental_options()
        report['controls'][mode]=row
    (a.output_dir/'report.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
