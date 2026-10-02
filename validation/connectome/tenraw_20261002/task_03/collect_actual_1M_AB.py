"""Collect only completed actual two-pilot1M A/B capacity+strict output evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def collect(root, subjects):
    evidence={'scope':'Actual listed pilot1M A/B capacity+bitwise equality; not ABBA or stable1M speedup',
              'collector_source_sha256':sha(Path(__file__)),
              'baseline_commit':'f436de588647a0de80735e4a98d53df5d88e502d',
              'candidate_commit':'c4811b4b192014cd59e1031385e359cd992ef9e5',
              'not_executed_planned_arms':['B2','A2'],'pilots':{}}
    for subject in subjects:
        directory=root/f'sub-{subject}_gpu0_1M_AB_after_five_post_v3'
        controller=json.loads((directory/'controller.json').read_text())
        assert controller['state']=='AB_strict_completed'
        strict=json.loads((directory/'strict_A1_B1.json').read_text())
        assert strict['strict_pass']
        assert all(v['neq']==0 and v['max_abs']==v['rmse']==v['p99']==0 for k,v in strict.items() if k!='strict_pass')
        pilot={'controller':controller,'controller_sha256':sha(directory/'controller.json'),
               'strict_comparison':strict,'strict_comparison_sha256':sha(directory/'strict_A1_B1.json'),'runs':{}}
        reference=None
        for label in ['A1','B1']:
            output=directory/'AB'/label;report=json.loads((output/'report.json').read_text())
            assert report['checkpoint_sha256']==controller['checkpoint_sha256'] and report['manifest_sha256']==controller['manifest_sha256']
            assert report['harness_sha256']=='a2218bd70353cb0fb4124aa95bdf317593d0d489e5da4164ea31bb52b091599e'
            assert report['source_commit']==evidence['baseline_commit' if label=='A1' else 'candidate_commit']
            assert report['n_seeds']==1000000 and report['seed']==0 and report['batch_size']==8192
            assert report['tf32'] and not report['compile_arc'] and report['budget_pass'] and not report['has_fa']
            assert report['nvml_gpu_uuids']==[controller['execution_gpu_uuid']]
            assert all(0<report[k]<20_000_000_000 for k in ['allocated_peak_bytes','reserved_peak_bytes','nvml_peak_bytes'])
            if reference is None:reference=report['tracking_kwargs']
            assert report['tracking_kwargs']==reference
            arrays={}
            for name in ['points','offsets','endpoints','lengths_mm','accepted_seeds']:
                p=output/(name+'.npy');a=np.load(p,mmap_mode='r',allow_pickle=False);assert np.isfinite(a).all()
                arrays[name]={'sha256':sha(p),'dtype':str(a.dtype),'shape':list(a.shape),'bytes':p.stat().st_size}
            pilot['runs'][label]={'report':report,'report_sha256':sha(output/'report.json'),'arrays':arrays}
        pilot['accepted_equal']=pilot['runs']['A1']['report']['accepted']==pilot['runs']['B1']['report']['accepted']
        assert pilot['accepted_equal']
        evidence['pilots'][subject]=pilot
    evidence['completed_pilots']=subjects
    evidence['all_requested_pilot_strict_and_memory_pass']=True
    evidence['all_two_pilot_strict_and_memory_pass']=set(subjects)=={'CON03','CON01'}
    return evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--subjects',nargs='+',choices=['CON03','CON01'],default=['CON03','CON01']);args=parser.parse_args();result=collect(args.results_root,args.subjects);args.output.write_text(json.dumps(result,indent=2)+'\n');print('Actual completed pilots strict+memory all pass:',args.subjects)
