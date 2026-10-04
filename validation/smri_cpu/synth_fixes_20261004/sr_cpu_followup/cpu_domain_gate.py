"""Candidate-only unique real input/domain gates with frozen official outputs."""
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import nibabel as nib
import numpy as np
import torch

from fnit.synthsr import SynthSR, _cpu_inference
from fnit.weights import WEIGHT_FILES


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    root=Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
    workspace=root/'workspaces/smri_cpu_20261004';runs=root/'runs/smri_cpu_20261004'
    output=runs/'remaining_20261004/synth/sr_domains_node7_v1';output.mkdir(exist_ok=True)
    jobs=[];definitions={};old_cases={}
    for name in ('paired_baseline_v2.private.json','domain_format_jobs.private.json'):
        manifest=json.loads((workspace/'task1'/name).read_text());jobs.extend(manifest['jobs'])
        definitions.update({item['id']:item for item in manifest['cases']})
    for name in ('sr_function_public_v2','sr_domain_public_v2'):
        old=json.loads((runs/'task1'/name/'summary.public.json').read_text())
        old_cases.update({item['case']:item for item in old['cases']})
    cases=('case02_synthsr_default','flair_sr_default','true_ct_sr','true_lowfield_sr',
           'real_epi_first_channel_sr','real_epi_npz_sr')
    torch.set_num_threads(8)
    report={'scope':'candidate-only complete real API on six unique inputs; original saved reference, no original rerun',
            'host':'nodecw7','cpu_affinity':sorted(os.sched_getaffinity(0)),
            'torch_threads':torch.get_num_threads(),'torch_version':torch.__version__,
            'source_sha256':{p.name:digest(p) for p in (Path(_cpu_inference.__file__).parent/'model.py',
                Path(_cpu_inference.__file__),Path(_cpu_inference.__file__).parent/'_cpu_math.py')},
            'driver_sha256':digest(__file__),'cases':[]}
    for case in cases:
        info=definitions[case];options=info['candidate_options'];input_path=Path(info['image'])
        reference_job=next(item for item in jobs if item['id']==case+'_1_reference')
        reference_path=Path(reference_job['expected_outputs'][0])
        old_result=old_cases[case]['comparisons'][0]['result']['outputs']['image']
        assert digest(reference_path)==old_result['reference_sha256']
        v1='--v1' in options;lowfield='--lowfield' in options
        name=('synthsr_v10_210712.h5' if v1 else 'synthsr_lowfield_v20_230130.h5' if lowfield else 'synthsr_v20_230130.h5')
        checkpoint=workspace/'assets/weights'/name;size,sha=WEIGHT_FILES[name][1:]
        assert checkpoint.stat().st_size==size and digest(checkpoint)==sha
        started=time.perf_counter();model=SynthSR(weights=checkpoint,device='cpu',threads=8,v1=v1,lowfield=lowfield)
        constructor=time.perf_counter()-started
        with torch.inference_mode():
            assert _cpu_inference._eligible(model.model,torch.zeros(1,1,16,16,16))
        started=time.perf_counter();result=model(input_path,ct='--ct' in options,
            disable_flipping='--disable_flipping' in options,disable_sharpening='--disable_sharpening' in options)
        api=time.perf_counter()-started
        destination=output/(case+info.get('output_suffix','.nii.gz'));result.image.save(destination)
        is_float=destination.suffix=='.npz'
        if is_float:
            ref=np.load(reference_path)['vol_data'];values=np.load(destination)['vol_data'];geometry=None;header=None
            fail=int(np.count_nonzero(~np.isclose(values,ref,rtol=1e-5,atol=1e-3)))
            gate={'rtol':1e-5,'atol':1e-3,'failed_values':fail,'passes':fail==0}
        else:
            original=nib.load(str(reference_path));candidate=nib.load(str(destination))
            ref=np.asarray(original.dataobj);values=np.asarray(candidate.dataobj)
            geometry=bool(np.array_equal(candidate.affine,original.affine))
            header=bool(candidate.header.binaryblock==original.header.binaryblock)
            delta=np.abs(values.astype(np.int16)-ref.astype(np.int16))
            gate={'exact_fraction_min':.9999,'max_abs_max':1,'mae_max':1e-4,
                  'passes':bool(1-np.count_nonzero(delta)/delta.size>=.9999 and delta.max()<=1
                                and delta.mean()<=1e-4 and geometry and values.dtype==ref.dtype)}
        delta=np.abs(values.astype(np.float64)-ref.astype(np.float64))
        item={'case':case,'description':info['data_description'],'input_sha256':digest(input_path),
              'options':options,'weight_size_bytes':size,'weight_sha256':sha,
              'reference_sha256':old_result['reference_sha256'],'output_sha256':digest(destination),
              'shape':list(ref.shape),'dtype':str(values.dtype),'count':delta.size,
              'different':int(np.count_nonzero(delta)),'max_abs':float(delta.max()),
              'rmse':float(np.sqrt(np.mean(delta**2))),'mae':float(delta.mean()),
              'affine_exact':geometry,'header_exact':header,'fixed_original_gate':gate,
              'old_metrics':old_result['whole_grid'],
              'constructor_seconds':constructor,'api_seconds':api,'load_after':list(os.getloadavg())}
        report['cases'].append(item);(output/'report.public.json').write_text(json.dumps(report,indent=2)+'\n')
        print(case,item['different'],gate['passes'],flush=True)
        del model,result;gc.collect()
    report['all_passed']=all(item['fixed_original_gate']['passes'] for item in report['cases'])
    report['maximum_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['status']='complete';(output/'report.public.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
