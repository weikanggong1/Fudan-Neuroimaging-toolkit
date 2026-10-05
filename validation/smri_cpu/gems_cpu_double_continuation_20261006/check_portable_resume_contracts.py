"""Serialization-only contracts; no MRI, native calculator or optimizer runs."""
import argparse
import copy
from dataclasses import dataclass,field
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
import torch


@dataclass(frozen=True)
class BlockIndex:
    shape:tuple
    block_size:int
    candidates:tuple
    _device_cache:dict=field(default_factory=dict,compare=False)


class Closure:
    def __init__(self):
        self.start=torch.zeros((4,3),dtype=torch.float64)
        self.shape=(16,8,8);self.tetra=torch.zeros((3,4),dtype=torch.int64)
        self.precision_policy={'points':'torch.float64','owner':'torch.float32'}
        self.anchor=self.start.clone();self.anchor[1,1]=.125
        self.index=BlockIndex(self.shape,8,(np.array([0,2],dtype=np.int64),np.array([1],dtype=np.int64)))
        self.index._device_cache['derived-plan']=('immutable',)
        self.evaluations=12;self.rebuilds=2;self.last={'data_cost':7.,'prior_cost':1.}
    def static_signature(self):return {'start':{'dtype':str(self.start.dtype),'shape':list(self.start.shape)}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helper',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(8)
    spec=importlib.util.spec_from_file_location('portable_helper',args.helper)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    results={}
    with tempfile.TemporaryDirectory() as tmp:
        directory=Path(tmp);(directory/'native_definition_cpu').mkdir()
        accepted=directory/'native_definition_cpu'/'accepted-004.private.npz'
        np.savez_compressed(accepted,points=np.zeros((4,3),dtype=np.float64))
        closure=Closure();saved=helper.save_portable(closure,directory/'closure_state.private.npz')
        prior={'closure_checkpoint':saved,'actual_private_CPU_precision_policy':closure.precision_policy,
            'modes':{'native_definition_cpu':{'rows':[{'step':4,'private_state_sha256':helper.digest(accepted),'total_evaluations':16}]}}}
        target=Closure();target.index._device_cache.clear();target.anchor.zero_()
        target.evaluations=0;target.rebuilds=0;target.last=None
        info=helper.reconstruct(target,type('Raster',(),{'BlockIndex':BlockIndex}),directory,prior)
        before=saved['state_fingerprint'];after=helper.fingerprint(target)
        results['portable_arrays_counters_last_static_exact_and_dynamic_plans_empty']=(
            {k:v for k,v in before.items() if k!='cache_keys'}=={k:v for k,v in after.items() if k!='cache_keys'}
            and info['dynamic_cache_initially_empty'])
        def rejected(name,changed):
            try:helper.reconstruct(Closure(),type('Raster',(),{'BlockIndex':BlockIndex}),directory,changed)
            except RuntimeError:results[name]=True
            else:results[name]=False
        changed=copy.deepcopy(prior);changed['closure_checkpoint']['private_checkpoint_sha256']='0'*64
        rejected('changed_checkpoint_bytes_rejected',changed)
        changed=copy.deepcopy(prior);changed['closure_checkpoint']['state_fingerprint']['anchor_sha256']='0'*64
        rejected('changed_array_binding_rejected',changed)
        changed=copy.deepcopy(prior);changed['actual_private_CPU_precision_policy']['points']='torch.float32'
        rejected('changed_precision_rejected',changed)
        changed=copy.deepcopy(prior);changed['closure_checkpoint']['state_fingerprint']['closure_evaluations']=13
        rejected('changed_saved_counter_rejected',changed)
        changed=copy.deepcopy(prior);changed['closure_checkpoint']['state_fingerprint']['static_signature']['start']['dtype']='torch.float32'
        rejected('changed_static_signature_rejected',changed)
    report={'scope':'serialization-only units, not MRI benchmark','contracts':results,'all_passed':all(results.values()),
            'helper_sha256':helper.digest(args.helper),'program_sha256':helper.digest(__file__)}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_passed']:raise RuntimeError('portable restore contract failed')
    print(json.dumps(report))


if __name__=='__main__':main()
