"""Independent TensorFlow ELU control on the captured real convolution plane."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sr_first_layer import metrics, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    if a.report.exists():p.error('preserve existing record')
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    raw=np.load(a.reference_dir/'raw.npy',mmap_mode='r')
    expected=np.load(a.reference_dir/'elu.npy',mmap_mode='r')
    index=raw.shape[2]//2
    x=np.ascontiguousarray(raw[:,:,index:index+1].transpose(0,2,3,4,1))
    standalone=tf.nn.elu(tf.convert_to_tensor(x)).numpy().transpose(0,4,1,2,3)
    @tf.function
    def compiled(value):return tf.nn.elu(value)
    graph=compiled(tf.convert_to_tensor(x)).numpy().transpose(0,4,1,2,3)
    y=np.asarray(expected[:,:,index:index+1])
    report={'schema':'fnit.synthsr.tf.elu.control.real.v1','driver_sha256':digest(__file__),
            'tensorflow_version':tf.__version__,'tf_include':tf.sysconfig.get_include(),
            'subset':{'axis':2,'index':index,'shape':list(y.shape)},
            'standalone_vs_original_conv_elu':metrics(y,standalone),
            'compiled_vs_original_conv_elu':metrics(y,graph)}
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
