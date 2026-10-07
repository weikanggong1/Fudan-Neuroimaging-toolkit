"""Record the actual official BN inference graph and small numeric factors."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-script',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():p.error('preserve prior evidence')
    a.output_dir.mkdir(parents=True)
    assert digest(a.official_script)=='917b20e90b00c1b2807ffcc1e8e2db123d3a63d52c8d395b8ba5a7f078c8b098'
    assert a.weights.stat().st_size==106163752 and digest(a.weights)=='a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b'
    ns=runpy.run_path(str(a.official_script),run_name='_isolated_reference_')
    tf=ns['tf'];tf.config.threading.set_inter_op_parallelism_threads(8);tf.config.threading.set_intra_op_parallelism_threads(8)
    norm=ns['build_model'](str(a.weights)).get_layer('unet_bn_down_0')
    traced=tf.function(lambda value:norm(value,training=False)).get_concrete_function(tf.TensorSpec((1,192,224,256,24),tf.float32))
    nodes=[]
    for operation in traced.graph.get_operations():
        row={'name':operation.name,'type':operation.type,'inputs':[x.name for x in operation.inputs]}
        if operation.type.startswith('FusedBatchNorm'):
            row['attrs']={k:operation.get_attr(k).decode() if isinstance(operation.get_attr(k),bytes) else operation.get_attr(k) for k in ('epsilon','data_format','is_training')}
        nodes.append(row)
    gamma,beta,mean,variance=norm.get_weights()
    inv=tf.math.rsqrt(tf.convert_to_tensor(variance)+np.float32(norm.epsilon)).numpy()
    scale=(tf.convert_to_tensor(gamma)*tf.convert_to_tensor(inv)).numpy()
    offset=(tf.convert_to_tensor(beta)-tf.convert_to_tensor(mean)*tf.convert_to_tensor(scale)).numpy()
    np.savez(a.output_dir/'factors.private.npz',gamma=gamma,beta=beta,mean=mean,variance=variance,inverse=inv,scale=scale,offset=offset)
    report={'schema':'fnit.synthsr.cpu.bn.actual_graph.v1','tensorflow_version':tf.__version__,'bn_fused':norm.fused,'bn_epsilon':norm.epsilon,'nodes':nodes,'driver_sha256':digest(__file__),'factor_sha256':digest(a.output_dir/'factors.private.npz'),'scope':'graph and 24-channel factors only; no full network execution'}
    (a.output_dir/'report.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
