"""Profile unchanged original CNN and verify both outputs against the frozen oracle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sr_first_layer import digest,metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('official-script','weights','network-dir','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():p.error('preserve prior result')
    a.output_dir.mkdir(parents=True)
    assert digest(a.official_script)=='917b20e90b00c1b2807ffcc1e8e2db123d3a63d52c8d395b8ba5a7f078c8b098'
    assert a.weights.stat().st_size==106163752 and digest(a.weights)=='a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b'
    ns=runpy.run_path(str(a.official_script),run_name='_isolated_reference_');tf=ns['tf']
    tf.config.threading.set_inter_op_parallelism_threads(8);tf.config.threading.set_intra_op_parallelism_threads(8)
    model=ns['build_model'](str(a.weights))
    report={'scope':'unchanged original full CNN, profiler execution evidence; not an end-to-end benchmark',
            'driver_sha256':digest(__file__),'tensorflow_version':tf.__version__,
            'optimizer_options':tf.config.optimizer.get_experimental_options(),
            'TF_ENABLE_ONEDNN_OPTS':os.environ.get('TF_ENABLE_ONEDNN_OPTS'),'branches':{}}
    profile=a.output_dir/'profile.private'
    tf.profiler.experimental.start(str(profile),tf.profiler.experimental.ProfilerOptions(host_tracer_level=2,python_tracer_level=0,device_tracer_level=0))
    try:
        for index in (0,1):
            path=a.network_dir/f'network_{index}_input.npy';source=np.load(path)
            result=model.predict(np.ascontiguousarray(source.transpose(0,2,3,4,1)),verbose=0).transpose(0,4,1,2,3)
            reference=np.load(a.network_dir/f'network_{index}_output.npy',mmap_mode='r')
            row={'input_sha256':digest(path),'frozen_output_file_sha256':digest(a.network_dir/f'network_{index}_output.npy'),
                 'metrics':metrics(reference,result)}
            row['result_array_sha256']=hashlib.sha256(np.ascontiguousarray(result).tobytes()).hexdigest()
            report['branches'][str(index)]=row
            assert row['metrics']['different']==0
    finally:tf.profiler.experimental.stop()
    from tensorflow.core.profiler.protobuf import xplane_pb2
    names=[]
    for path in profile.rglob('*.xplane.pb'):
        space=xplane_pb2.XSpace();space.ParseFromString(path.read_bytes())
        for plane in space.planes:
            names.extend(metadata.name for metadata in plane.event_metadata.values() if any(k in metadata.name.lower() for k in ('conv','elu','bias')))
    report['actual_execution_event_names']=sorted(set(names))
    report['status']='complete'
    (a.output_dir/'report.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
