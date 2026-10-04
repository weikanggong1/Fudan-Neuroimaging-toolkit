"""Real-input CPU layout diagnostic; does not alter production precision."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from fnit.synthmorph import SynthMorph
from fnit.synthmorph import models


def main():
    p=argparse.ArgumentParser()
    for name in ('moving','fixed','weights','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--model',choices=('rigid','affine','deform','joint'),default='joint')
    args=p.parse_args();torch.set_num_threads(8)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    load=time.perf_counter();register=SynthMorph(weights=args.weights,device='cpu',model=args.model)
    rows=[];load=time.perf_counter()-load
    old_feature=models.FeatureDetector.forward;old_deform=models.DeformNetwork.forward
    for index,policy in enumerate(('contiguous','channels_last_3d','channels_last_3d','contiguous')):
        layout=torch.channels_last_3d if policy=='channels_last_3d' else torch.contiguous_format
        for module in register.network.modules():
            if isinstance(module,nn.Conv3d):module.to(memory_format=layout)
        if policy=='channels_last_3d':
            models.FeatureDetector.forward=lambda self,x:old_feature(self,x.contiguous(memory_format=torch.channels_last_3d))
            models.DeformNetwork.forward=lambda self,moving,fixed:old_deform(self,moving.contiguous(memory_format=torch.channels_last_3d),fixed.contiguous(memory_format=torch.channels_last_3d))
        else:
            models.FeatureDetector.forward=old_feature;models.DeformNetwork.forward=old_deform
        started=time.perf_counter();r=register(args.moving,args.fixed,transform_only=True)
        elapsed=time.perf_counter()-started
        if args.model in ('rigid','affine'):
            arrays={'forward':r.transform.matrix,'inverse':r.inverse.matrix}
        else:arrays={'forward':np.asarray(r.transform.dataobj),'inverse':np.asarray(r.inverse.dataobj)}
        hashes={}
        for key,data in arrays.items():
            filename=output/(str(index)+'_'+key+'.npy');np.save(filename,data)
            hashes[key]=hashlib.sha256(filename.read_bytes()).hexdigest()
        rows.append({'iteration':index,'policy':policy,'api_seconds':elapsed,'arrays_sha256':hashes})
        (output/'progress.json').write_text(json.dumps({'scope':'loaded model CPU API transform_only; layout diagnostic not official e2e','load_seconds':load,'runs':rows},indent=2)+'\n')
    (output/'report.json').write_text(json.dumps({'scope':'loaded model CPU API transform_only; complete two forward networks and both integrals retained','load_seconds':load,'runs':rows},indent=2)+'\n')


if __name__=='__main__':main()
