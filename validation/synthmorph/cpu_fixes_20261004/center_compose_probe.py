"""Original center composition on bound identical half-affine matrices; no CNN."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    import tensorflow as tf
    import voxelmorph as vxm
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    inputs=np.load(args.input);arrays={};rows={}
    for label,half in zip(inputs['labels'],inputs['roots']):
        extent,scale,direction=str(label).split('_');shape=int(extent)*float(scale)
        center=np.eye(4,dtype=np.float32);center[:3,3]=-(shape-1)*.5
        uncenter=center.copy();uncenter[:3,3]*=-1
        # The original layer maps compose across the batch, one 3x4 matrix at a time.
        original=vxm.utils.compose(tuple(tf.constant(value[:3]) for value in (uncenter,half,center)),shift_center=False)
        direct=tf.constant(uncenter)@(tf.constant(half)@tf.constant(center))
        original=vxm.utils.make_square_affine(original).numpy()
        rows[str(label)]={'direct_right_association_equal':bool(np.array_equal(original,direct.numpy()))}
        for name,value in [('half',half),('center',center),('uncenter',uncenter),('composed',original)]:arrays[str(label)+'_'+name]=value
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    np.savez(output/'center_stages.npz',**arrays)
    report={'scope':'original compose on eight identical official half-affines; no CNN',
            'rows':rows,'input_sha256':digest(args.input),'worker_sha256':digest(__file__),
            'original_compose_sha256':digest(inspect.getsourcefile(vxm.utils.compose)),
            'stages_sha256':digest(output/'center_stages.npz')}
    (output/'report.private.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
